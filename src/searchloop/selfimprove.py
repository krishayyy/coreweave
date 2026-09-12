"""The outer-outer loop: the agent reads its own failures and rewrites its own
instructions.

The inner loop searches. The outer loop notices its premise is wrong and
proposes a new one. Neither of them gets better over time -- the hundredth
search is no wiser than the first, because nothing survives the end of a run.

This is what persists. After a batch of searches, every nomination the agent
made is scored against the withheld truth, the failures are summarised into
statistics and concrete worked examples, and the agent is shown its own record
and asked what instruction would have helped.

The part that makes this more than "rewrite your prompt and hope" is the gate.
A proposed lesson is a candidate, not an improvement. It is measured on a fold
the lesson was not derived from, and it is accepted only if it actually helps.
That is the same rule the rest of the system runs on: the model may nominate,
the evidence decides.

Folds are strict. Lessons are learned on a training fold and validated on a
separate one, and neither is ever the suite the headline result is reported on.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import llm, tracing
from .grid import SearchGrid
from .revision import nominate_llm, should_revise
from .scenario import Scenario

LESSON_BOOK = Path(__file__).resolve().parents[2] / "data" / "lessons.json"

# A lesson must clear this margin on the validation fold to be accepted. Set
# above zero deliberately: a change that merely fails to hurt is not an
# improvement, and accepting those is how a prompt fills with noise.
MIN_BEARING_GAIN_DEG = 5.0


@dataclass
class Lesson:
    """An instruction the agent wrote for itself, with its provenance."""

    id: str
    text: str
    learned_at: str
    rationale: str
    evidence: dict[str, Any]
    validation: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class LessonBook:
    """Everything the agent has learned, including what it tried and rejected."""

    lessons: list[Lesson] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = LESSON_BOOK) -> "LessonBook":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls([Lesson(**item) for item in raw.get("lessons", [])])

    def save(self, path: Path = LESSON_BOOK) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"lessons": [lesson.to_dict() for lesson in self.lessons]}, indent=2))

    @property
    def accepted(self) -> list[Lesson]:
        return [lesson for lesson in self.lessons if lesson.accepted]

    @property
    def rejected(self) -> list[Lesson]:
        return [lesson for lesson in self.lessons if not lesson.accepted]

    def prompt_section(self) -> str:
        """Accepted lessons, rendered for injection into the nomination prompt.

        Empty when nothing has been learned, so an untrained agent's prompt is
        exactly the prompt it always was -- which is what makes the before and
        after comparable.
        """
        if not self.accepted:
            return ""
        lines = [
            "LESSONS FROM PREVIOUS SEARCHES. These were derived by reviewing "
            "your own past proposals against what turned out to be true, and "
            "each one measurably improved accuracy when tested:",
        ]
        lines += [f"  {i}. {lesson.text}" for i, lesson in enumerate(self.accepted, 1)]
        return "\n".join(lines)


# -- scoring the agent's own record ---------------------------------------


def _bearing_and_distance(scenario: Scenario, grid: SearchGrid) -> tuple[float, float]:
    dr = scenario.true_anchor_rc[0] - scenario.ipp_rc[0]
    dc = scenario.true_anchor_rc[1] - scenario.ipp_rc[1]
    bearing = math.degrees(math.atan2(dc, -dr)) % 360.0
    distance = float(np.hypot(dr, dc) * grid.cell_m / 1000.0)
    return bearing, distance


@dataclass
class FailureReport:
    """What the agent's own record says about how it fails."""

    n_nominations: int
    bearing_median: float
    bearing_within_20: float
    distance_median: float
    by_cue: dict[str, dict[str, float]]
    examples: list[dict[str, Any]]

    def render(self) -> str:
        """The report as the agent will read it."""
        out = [
            f"You made {self.n_nominations} proposals across these searches.",
            "",
            "ACCURACY OF THE STARTING POINT YOU PROPOSED",
            f"  bearing  : median error {self.bearing_median:.0f} degrees; "
            f"{100 * self.bearing_within_20:.0f}% were within 20 degrees",
            f"  distance : median error {self.distance_median:+.1f} km",
            "",
            "BROKEN DOWN BY WHAT THE CASE FILE TOLD YOU",
        ]
        for cue, stats in sorted(self.by_cue.items(), key=lambda kv: -kv[1]["bearing_median"]):
            out.append(
                f"  {cue:<22} bearing error {stats['bearing_median']:3.0f} deg, "
                f"{100 * stats['within_20']:3.0f}% within 20 deg  "
                f"(n={int(stats['n'])})")
        out += ["", "SPECIFIC PROPOSALS THAT MISSED"]
        for ex in self.examples:
            out += [
                f"  Case file said: \"{ex['cue']}\"",
                f"    you proposed : bearing {ex['proposed_bearing']:.0f} deg, "
                f"{ex['proposed_distance']:.1f} km  ({ex['label']})",
                f"    truth was    : bearing {ex['true_bearing']:.0f} deg, "
                f"{ex['true_distance']:.1f} km   -> off by "
                f"{ex['bearing_error']:.0f} degrees",
            ]
        return "\n".join(out)


def analyse(runs: list[dict], suite: dict[str, Scenario], grid: SearchGrid,
            max_examples: int = 6) -> FailureReport:
    """Score every nomination in a batch of runs against the withheld truth."""
    bearings: list[float] = []
    distances: list[float] = []
    by_cue: dict[str, list[float]] = {}
    examples: list[dict[str, Any]] = []

    for run in runs:
        if run.get("arm") != "llm":
            continue
        scenario = suite.get(run["scenario_id"])
        if scenario is None or scenario.kind != "B":
            continue
        true_bearing, true_distance = _bearing_and_distance(scenario, grid)
        cue = scenario.late_evidence[0].text if scenario.late_evidence else scenario.case_file

        for trace in run["trace"]:
            for nom in trace["nominations"]:
                err = abs((nom["bearing_deg"] - true_bearing + 180) % 360 - 180)
                bearings.append(err)
                distances.append(nom["distance_km"] - true_distance)
                by_cue.setdefault(scenario.subkind, []).append(err)
                if err > 45:
                    examples.append({
                        "cue": cue[:200], "label": nom["label"],
                        "proposed_bearing": nom["bearing_deg"],
                        "proposed_distance": nom["distance_km"],
                        "true_bearing": true_bearing, "true_distance": true_distance,
                        "bearing_error": err,
                    })

    b = np.array(bearings) if bearings else np.array([0.0])
    d = np.array(distances) if distances else np.array([0.0])
    # Worst misses first: those are the ones with a lesson in them.
    examples.sort(key=lambda e: -e["bearing_error"])

    return FailureReport(
        n_nominations=len(bearings),
        bearing_median=float(np.median(b)),
        bearing_within_20=float((b <= 20).mean()),
        distance_median=float(np.median(d)),
        by_cue={
            k: {"bearing_median": float(np.median(v)),
                "within_20": float((np.array(v) <= 20).mean()),
                "n": float(len(v))}
            for k, v in by_cue.items()
        },
        examples=examples[:max_examples],
    )


# -- proposing an amendment -----------------------------------------------

_SYSTEM = """You are reviewing your own performance as a search and rescue \
planning assistant.

In previous searches you were given a case file and asked to propose where a \
missing subject actually started. You are now being shown how those proposals \
compared with the truth, which you did not have at the time.

Your task is to write ONE additional instruction for yourself -- a short, \
general rule that would have prevented the errors you are about to read.

It must be:
  GENERAL. A rule about how to read evidence, not a fact about any particular \
case. It will be applied to searches you have never seen.
  ACTIONABLE. Something you can follow while writing a proposal.
  SPECIFIC TO THE OBSERVED FAILURE. Look at which categories you do well on and \
which you do badly on, and ask what the bad ones have in common. The difference \
between them is where the lesson is.
  NOT A RESTATEMENT of instructions you already follow.

Do not propose a rule about trying harder or being more careful. Those are not \
actionable. Find the specific reasoning step that went wrong.

Return ONLY JSON:
{"lesson": "the instruction, one or two sentences, addressed to yourself",
 "rationale": "which pattern in the data this addresses",
 "predicted_effect": "what should improve if this is right"}"""


@tracing.op
def propose_lessons(report: FailureReport, book: LessonBook,
                    current_instructions: str, k: int = 3) -> list[Lesson]:
    """Several candidate instructions, to be validated against each other.

    One candidate per round is a poor search over instruction space: the first
    thing a model says is not reliably its best, and a single rejection then
    reads as "nothing can help" when it only means "that one did not". Asking
    for several distinct hypotheses and measuring each gives the gate something
    to choose between.
    """
    out: list[Lesson] = []
    for attempt in range(k):
        lesson = propose_lesson(report, book, current_instructions,
                                already=[x.text for x in out], variant=attempt)
        if lesson is not None:
            out.append(lesson)
    return out


@tracing.op
def propose_lesson(report: FailureReport, book: LessonBook,
                   current_instructions: str,
                   already: list[str] | None = None,
                   variant: int = 0) -> Lesson | None:
    """Ask the agent what instruction would have prevented its own errors."""
    existing = "\n".join(f"  - {lesson.text}" for lesson in book.accepted) or "  (none yet)"
    tried = "\n".join(
        f"  - {lesson.text[:120]} (rejected: {lesson.validation.get('verdict', 'no gain')})"
        for lesson in book.rejected) or "  (none yet)"

    siblings = "\n".join(f"  - {t[:140]}" for t in (already or [])) or "  (none)"
    angles = [
        "Focus on the single largest source of error.",
        "Focus on a DIFFERENT mechanism from the obvious one. If the obvious "
        "reading is that a cue was ignored, consider instead that it was read "
        "but applied backwards, or applied to the wrong reference point.",
        "Focus on what the categories you do WELL on have in common, and write "
        "the rule that would extend that behaviour to the others.",
    ]
    user = (
        f"YOUR RECORD\n{report.render()}\n\n"
        f"INSTRUCTIONS YOU ALREADY HAVE\n{current_instructions}\n\n"
        f"LESSONS ALREADY IN FORCE\n{existing}\n\n"
        f"LESSONS ALREADY TRIED AND REJECTED — do not propose these again\n{tried}\n\n"
        f"CANDIDATES ALREADY WRITTEN THIS ROUND — propose something materially "
        f"different\n{siblings}\n\n"
        f"{angles[variant % len(angles)]}\n\n"
        f"Write one new instruction for yourself."
    )
    raw = llm.complete(_SYSTEM, user, max_tokens=700, temperature=0.7)
    parsed = llm.extract_json(raw)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else None
    if not isinstance(parsed, dict) or not parsed.get("lesson"):
        return None

    text = str(parsed["lesson"]).strip()
    return Lesson(
        id=f"lesson-{len(book.lessons) + 1:03d}",
        text=text[:400],
        learned_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        rationale=str(parsed.get("rationale", ""))[:400],
        evidence={
            "n_nominations": report.n_nominations,
            "bearing_median": report.bearing_median,
            "bearing_within_20": report.bearing_within_20,
            "by_cue": report.by_cue,
            "predicted_effect": str(parsed.get("predicted_effect", ""))[:300],
        },
    )


# -- validating a proposed lesson -----------------------------------------

@dataclass
class Validation:
    """What a candidate lesson actually did on a fold it was not derived from."""

    n: int
    bearing_before: float
    bearing_after: float
    gain_deg: float
    within20_before: float
    within20_after: float
    improved: int
    worsened: int
    verdict: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@tracing.op
def probe_bearing_error(
    scenario: Scenario, grid: SearchGrid, pod, cfg, lessons: str
) -> list[float]:
    """Bearing error of the proposals made on one scenario.

    Runs the deterministic search until the premise fails, then asks for
    proposals once and scores them. A single nomination rather than a full
    search: the lesson is about proposal quality, measuring it directly is far
    cheaper than measuring it through detection rolls, and the daily token
    budget is the binding constraint on how much validation is possible at all.
    """
    from .belief import Belief, SearchRecord
    from .hypotheses import library_hypotheses
    from .planner import plan_sortie, segment_capacity, sortie_pos

    true_bearing, _ = _bearing_and_distance(scenario, grid)
    belief = Belief.from_hypotheses(library_hypotheses(grid, scenario.ipp_rc))
    capacity = segment_capacity(grid, cfg.period_track_km, cfg.altitude_m)
    position = scenario.ipp_rc

    for period in range(cfg.max_periods):
        trigger = should_revise(belief, cfg, period, None)
        if trigger.fired:
            noms = nominate_llm(belief, grid, scenario.briefing(period), trigger,
                                scenario.ipp_rc, lessons=lessons)
            return [abs((n.anchor_bearing_deg - true_bearing + 180) % 360 - 180)
                    for n in noms]
        segment = plan_sortie(grid, belief.joint, pod, position, budget_cells=capacity)
        belief.observe(SearchRecord(
            period, segment, [float(pod[c]) for c in segment], False,
            sortie_pos(belief.joint, pod, segment), 0.0))
        position = segment[len(segment) // 2]
    return []


def validate(lesson: Lesson, book: LessonBook, scenarios: list[Scenario],
             grid: SearchGrid, pod, cfg) -> Validation:
    """Measure the candidate against the same fold without it.

    The comparison is paired: identical scenarios, identical search history up
    to the moment of proposal. The only difference is whether the lesson is in
    the prompt.
    """
    base_lessons = book.prompt_section()
    trial = LessonBook(book.accepted + [lesson])
    trial_lessons = trial.prompt_section()

    # Paired per scenario. Every proposal in a response is admitted to the
    # mixture, so what matters is whether the response contained a good account
    # at all -- the best proposal's error. Taking a median across all proposals
    # from all scenarios instead mixes between-scenario variance into the
    # comparison and buries an effect this size in noise.
    before: list[float] = []
    after: list[float] = []
    all_before: list[float] = []
    all_after: list[float] = []

    for scenario in scenarios:
        b_errs = probe_bearing_error(scenario, grid, pod, cfg, base_lessons)
        a_errs = probe_bearing_error(scenario, grid, pod, cfg, trial_lessons)
        if not b_errs or not a_errs:
            continue
        before.append(min(b_errs))
        after.append(min(a_errs))
        all_before += b_errs
        all_after += a_errs

    if not before:
        return Validation(0, 0, 0, 0, 0, 0, 0, 0, "no data")

    b = np.array(before)
    a = np.array(after)
    gain = float(np.mean(b) - np.mean(a))
    improved = int((a < b - 1.0).sum())
    worsened = int((a > b + 1.0).sum())

    # Two conditions, because a mean can be carried by one scenario: the gain
    # must clear the bar AND more scenarios must improve than worsen.
    ok = gain >= MIN_BEARING_GAIN_DEG and improved > worsened
    if ok:
        verdict = "accepted"
    elif gain < MIN_BEARING_GAIN_DEG:
        verdict = f"gain {gain:+.0f} deg below the {MIN_BEARING_GAIN_DEG:.0f} deg bar"
    else:
        verdict = f"gain came from too few scenarios ({improved} better, {worsened} worse)"

    return Validation(
        n=len(before),
        bearing_before=float(np.mean(b)), bearing_after=float(np.mean(a)),
        gain_deg=gain,
        within20_before=float((np.array(all_before) <= 20).mean()),
        within20_after=float((np.array(all_after) <= 20).mean()),
        improved=improved, worsened=worsened,
        verdict=verdict,
    )
