"""The loop that improves the loop.

After a batch of searches, the agent is shown how it performed, what
configuration produced that performance, and what it has already tried. It
proposes one change to its own configuration. The change is measured on
scenarios it was not tuned on, and kept only if it actually helps.

This replaces an earlier design that asked the agent to write itself a better
instruction. That failed four times -- twice from a human reading the same
statistics, twice from the agent -- and all four failures arrived at the same
plausible, wrong rule. Meanwhile every change that did improve this system was
a configuration change: sixteen compass points rather than eight, committing
when confident rather than always hedging, conditioning distance on ground
already swept. Sentences were never the lever. Configuration is.

The gate is unchanged in spirit and stricter in practice. A proposal must
improve the objective by more than a fixed bar AND by more than the noise the
improvement was measured through AND not regress the find rate, on a fold it
was not derived from. Anything else is recorded and refused.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import llm, tracing, tunable

TUNING_LOG = Path(__file__).resolve().parents[2] / "data" / "tuning.json"

# A proposal must clear this on the objective to be worth the churn.
MIN_SCORE_GAIN = 0.015
# ...and may not cost more than this on the raw find rate. Set from the noise
# rather than from taste: with twelve scenarios at six repeats the standard
# error on a find rate is near six points, so a guard tighter than that rejects
# signal for being noise -- which it did, blocking a real objective gain on a
# six-point drop that was inside one standard error.
MAX_FIND_REGRESSION = 0.08


@dataclass
class Score:
    """How a configuration performed on a fold.

    `objective` is what the gate judges, and it is deliberately not the mean
    ranking of the true location. Ranking measures reasoning, and an earlier
    version of this gate judged on it -- which would have preferred an untuned
    configuration that ranked the subject better while finding fewer of them.
    Reasoning is the means; finding someone before they die is the end.

    So the objective rewards finding, and finding sooner: a search that
    resolves in period 3 of 16 scores near 1, one that resolves at the buzzer
    scores near 0, and one that never resolves scores 0. Continuous, so twelve
    validation cases can separate configurations that a binary find rate could
    not.
    """

    find_rate: float
    localise_rate: float
    mean_peak: float
    objective: float
    n: int
    repeats: int = 1
    # Per-run objective, in a fixed (scenario, repeat) order so that two folds
    # scored on the same scenarios and seeds can be differenced run by run. A
    # paired difference is what lets the gate know whether a gain it likes is
    # larger than the noise it is measured through; without it the gate is
    # comparing two point estimates to a fixed bar and calling that rigour.
    runs: list[float] = field(default_factory=list, repr=False)

    @property
    def n_runs(self) -> int:
        return self.n * self.repeats

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("runs")          # too large for the log, and derivable
        return d


@dataclass
class Trial:
    """One proposed change, and what happened to it."""

    id: str
    parameter: str
    was: float
    now: float
    rationale: str
    predicted: str
    at: str
    before: dict[str, Any] = field(default_factory=dict)
    after: dict[str, Any] = field(default_factory=dict)
    accepted: bool = False
    verdict: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TuningLog:
    settings: dict[str, float] = field(default_factory=lambda: dict(tunable.DEFAULTS))
    trials: list[Trial] = field(default_factory=list)
    # Where this run started. Recorded rather than inferred: `--from-naive`
    # writes the naive configuration into this file, so a later run without the
    # flag would load naive settings, diff them against DEFAULTS, and report
    # eight spurious changes as if the loop had made them.
    baseline: dict[str, float] = field(
        default_factory=lambda: dict(tunable.DEFAULTS))

    @classmethod
    def load(cls, path: Path = TUNING_LOG) -> "TuningLog":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        settings = {**tunable.DEFAULTS, **raw.get("settings", {})}
        return cls(
            settings=settings,
            trials=[Trial(**t) for t in raw.get("trials", [])],
            # An older log predates this field. Its starting point is
            # unrecoverable, so assume it started where it now sits rather than
            # inventing a diff against DEFAULTS that never happened.
            baseline={**tunable.DEFAULTS, **raw.get("baseline", settings)},
        )

    def save(self, path: Path = TUNING_LOG) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"settings": self.settings,
             "baseline": self.baseline,
             "trials": [t.to_dict() for t in self.trials]}, indent=2))

    @property
    def accepted(self) -> list[Trial]:
        return [t for t in self.trials if t.accepted]


def score_fold(grid, pod, scenarios, settings: dict[str, float], base_cfg,
               arm: str, repeats: int = 6) -> Score:
    """Run a fold and summarise it.

    Repeats matter more here than anywhere else in the project. A single pass
    over twelve scenarios gives a find rate with a standard error near fourteen
    points, while the configuration changes being judged move the objective by
    one or two. Without repeats the gate is not strict, it is blind: it refuses
    everything because it can see nothing.

    Repeats re-roll detection only. The scenarios are identical, so this
    averages out the sensor's luck without pretending to more cases than exist.

    The settings are installed here rather than by the caller. Half of this
    configuration travels through a Config object and half through module
    globals, and when the caller owned both channels it was possible -- and
    happened -- to measure a Config that did not match the globals in force.
    One function now owns both.
    """
    from .loop import run_scenario
    from .scenario import stable_seed

    tunable.apply(settings)
    cfg = tunable.as_config(base_cfg, settings)

    found = localised = total = 0
    peaks: list[float] = []
    speed: list[float] = []
    for repeat in range(repeats):
        for scenario in scenarios:
            result = run_scenario(grid, pod, scenario, arm, cfg,
                                  np.random.default_rng(stable_seed(scenario.id, repeat)))
            found += result.found
            localised += result.periods_to_localize is not None
            total += 1
            peaks.append(result.peak_truth_percentile)
            speed.append((cfg.max_periods - result.periods_to_find + 1) / cfg.max_periods
                         if result.found else 0.0)
    return Score(found / total, localised / total, float(np.mean(peaks)),
                 float(np.mean(speed)), len(scenarios), repeats, speed)


_SYSTEM = """You are tuning a search and rescue planning system that you are \
part of.

The system searches for a missing person. When the ground it has swept rules \
out the account it was working from, it proposes new accounts of what happened \
and searches those instead. You are shown how it performed, the configuration \
that produced that performance, and every change already tried.

Propose ONE change to one parameter.

Reason from the numbers you are shown, not from general principle. If the \
system reconsiders its premise too late to act on the new account, the trigger \
is too high. If it thrashes between accounts without committing, it is \
revising too often or hedging across too many directions. If it commits to one \
direction and that direction is wrong, it is hedging too little.

Do not propose a change already tried and rejected -- the same change will be \
measured the same way and rejected again. If you are told some parameters have \
never been tried, prefer one of those: a parameter you have not measured is \
worth more information than another value of one you have.

Return ONLY JSON:
{"parameter": "exact name from the list",
 "value": the new value,
 "rationale": "which number in the report motivates this",
 "predicted_effect": "what should improve, and what might get worse"}"""


@tracing.op
def propose(before: Score, log: TuningLog, breakdown: str,
            attempt: int = 0, debug: bool = False) -> Trial | None:
    """Ask for one configuration change.

    `attempt` varies the prompt. Responses are cached by prompt digest, so
    without it a retry returns the identical cached answer and a retry loop is
    decoration -- which is exactly how a run once produced no proposals at all
    across eight rounds.
    """
    tried = "\n".join(
        f"  - {t.parameter} {t.was} -> {t.now}: {t.verdict}"
        for t in log.trials) or "  (nothing yet)"
    # Left to itself the model returns to the same two or three parameters and
    # the duplicate guard then burns the round. Naming the unexplored ones is
    # the cheapest way to widen the search.
    untouched = [k for k in tunable.KNOBS if not any(t.parameter == k for t in log.trials)]
    untried = ", ".join(untouched) if untouched else "(all have been tried at least once)"
    current = "\n".join(f"  {k} = {v}" for k, v in sorted(log.settings.items()))

    user = (
        f"CURRENT PERFORMANCE on {before.n} cases where the initial premise was "
        f"wrong, each searched {before.repeats} times with different sensor luck "
        f"({before.n_runs} runs)\n"
        f"  subject located            {100 * before.find_rate:.0f}%\n"
        f"  true location reached the top decile of belief   "
        f"{100 * before.localise_rate:.0f}%\n"
        f"  mean best ranking of the true location           "
        f"{before.mean_peak:.0f}%\n"
        f"  speed score (rewards finding, and finding sooner)  "
        f"{before.objective:.3f}\n\n"
        f"{breakdown}\n\n"
        f"CURRENT CONFIGURATION\n{current}\n\n"
        f"PARAMETERS YOU MAY CHANGE\n{tunable.describe(log.settings)}\n\n"
        f"ALREADY TRIED\n{tried}\n\n"
        f"NEVER TRIED — prefer one of these\n  {untried}\n\n"
        f"Propose one change."
        + (f"\n\n(attempt {attempt + 1}: your previous answer could not be used; "
           f"pick a different parameter)" if attempt else "")
    )
    raw = llm.complete(_SYSTEM, user, max_tokens=700, temperature=0.7)
    parsed = llm.extract_json(raw)
    if isinstance(parsed, list):
        parsed = parsed[0] if parsed else None
    if not isinstance(parsed, dict):
        return None

    checked = tunable.validate(parsed.get("parameter"), parsed.get("value"))
    if checked is None:
        if debug:
            print(f"      [rejected: unknown parameter "
                  f"{parsed.get('parameter')!r}]")
        return None
    name, value = checked
    was = log.settings[name]
    if abs(value - was) < 1e-9:
        if debug:
            print(f"      [rejected: {name} is already {was}]")
        return None       # proposing the current value is not a proposal

    # Listing prior trials in the prompt is advisory and the model re-proposed a
    # rejected change anyway. Enforce it: an identical (parameter, value) pair
    # would be measured the same way and rejected the same way, and a round
    # spent re-measuring it is a round not spent searching.
    # Exact equality was not enough: nothing stops the model answering 0.499
    # where 0.5 was refused, and a difference that small is far inside the
    # measurement noise, so the round buys nothing. Reject anything within 5% of
    # the knob's range of a value already measured.
    knob = tunable.KNOBS[name]
    tolerance = max(1e-9, 0.05 * (knob.hi - knob.lo))
    for prior in log.trials:
        if prior.parameter == name and abs(prior.now - value) <= tolerance:
            if debug:
                print(f"      [rejected: {name} -> {value} is indistinguishable "
                      f"from {prior.now}, already measured]")
            return None

    return Trial(
        id=f"trial-{len(log.trials) + 1:03d}",
        parameter=name, was=was, now=value,
        rationale=str(parsed.get("rationale", ""))[:300],
        predicted=str(parsed.get("predicted_effect", ""))[:300],
        at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


def paired_stderr(before: Score, after: Score) -> float:
    """Standard error of the objective difference, run by run.

    The two folds are the same scenarios under the same seeds, so the runs pair
    up and most of the scenario-to-scenario variance -- which is large, and
    which neither configuration is responsible for -- cancels in the difference.
    Returns 0.0 when the vectors are missing or do not pair, which makes the
    gate fall back to the fixed bar alone rather than silently inventing
    confidence it does not have.
    """
    if len(before.runs) != len(after.runs) or len(before.runs) < 2:
        return 0.0
    diff = np.asarray(after.runs) - np.asarray(before.runs)
    return float(np.std(diff, ddof=1) / np.sqrt(diff.size))


def judge(trial: Trial, before: Score, after: Score) -> None:
    """Accept on the objective, and never at the cost of finding people.

    Three conditions, in order of how often they bite:

    1. the gain clears a fixed bar, so a change too small to matter is not kept
       for the churn;
    2. the gain clears its own measurement noise, so a change that is really
       zero is not kept because the dice fell well. Repeats were added to give
       the gate something to see; this is the part that actually looks;
    3. the find rate does not fall further than the noise on the find rate,
       because a configuration that reasons beautifully and recovers fewer
       people is a worse configuration.
    """
    gain = after.objective - before.objective
    regression = before.find_rate - after.find_rate
    stderr = paired_stderr(before, after)
    trial.before = before.to_dict()
    trial.after = after.to_dict()
    margin = f"objective {gain:+.3f} ± {stderr:.3f}" if stderr else f"objective {gain:+.3f}"

    if gain < MIN_SCORE_GAIN:
        trial.verdict = f"{margin}, below the {MIN_SCORE_GAIN:.3f} bar"
    elif stderr and gain < stderr:
        trial.verdict = (f"{margin}, inside one standard error -- "
                         f"indistinguishable from no change")
    elif regression > MAX_FIND_REGRESSION:
        trial.verdict = (f"{margin} but find rate "
                         f"{-100 * regression:+.0f} pp -- not worth it")
    else:
        trial.verdict = (f"{margin}, find rate "
                         f"{100 * (after.find_rate - before.find_rate):+.0f} pp")
        trial.accepted = True
