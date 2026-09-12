"""The outer loop: noticing the premise is wrong, and proposing a new one.

Three nominators, which are the three arms of the experiment:

  none       -- never revise. Pure Bayesian search inside the published library.
                The baseline every claim is measured against.

  heuristic  -- when stuck, relocate to wherever unsearched mass remains. No
                language model, no reading of the case file. This arm exists to
                separate "the model understood the evidence" from the much
                weaker "trying somewhere else eventually helps".

  llm        -- read the case file and the search history, and propose accounts
                of what happened.

A nomination is a *typed parameter set*, never a probability field. The model
names a behaviour profile and where to anchor it; this module converts that into
a prior by the identical function the published library uses. The model cannot
hand-draw a distribution, cannot weight itself beyond a capped prior, and cannot
touch the aircraft.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import llm, tracing
from .belief import Belief
from .config import Config
from .grid import SearchGrid
from .hypotheses import PROFILES, Hypothesis, build_prior_field

MAX_NOMINATION_PRIOR = 0.35


@dataclass
class Nomination:
    label: str
    narrative: str
    profile_key: str
    anchor_bearing_deg: float
    anchor_distance_km: float
    prior: float
    # Angular uncertainty on the bearing. A proposal like "they were driven
    # somewhere north-east" is an arc, not a point, and committing the whole
    # prior to one exact bearing throws away the fact that the account itself
    # is only directional.
    anchor_bearing_sigma_deg: float = 0.0
    rationale: str = ""
    evidence_cited: tuple[str, ...] = ()
    origin: str = "llm"


@dataclass
class RevisionTrigger:
    fired: bool
    reason: str
    leader_label: str = ""
    leader_exhaustion: float = 0.0


@tracing.op
def should_revise(
    belief: Belief, cfg: Config, period: int, last_revision_period: int | None
) -> RevisionTrigger:
    """Fire when the leading account has been substantially ruled out.

    Never on an iteration count. The quantity used is P(no contact so far | H)
    for the current leader -- literally "how much of what this story predicted
    have we now covered and come up empty".
    """
    leader, _ = belief.leader
    idx = belief.hypotheses.index(leader)
    exhaustion = float(belief.exhaustion[idx])

    if last_revision_period is not None and period - last_revision_period < cfg.min_periods_between_revisions:
        return RevisionTrigger(False, "cooling down since last revision", leader.label, exhaustion)
    if exhaustion < cfg.exhaustion_trigger:
        return RevisionTrigger(
            False, f"leading account still {100 * (1 - exhaustion):.0f}% unexplored",
            leader.label, exhaustion,
        )
    return RevisionTrigger(
        True,
        f"{100 * exhaustion:.0f}% of the mass predicted by '{leader.label}' has been "
        f"covered with no contact",
        leader.label, exhaustion,
    )


def _anchor_at(grid: SearchGrid, ipp_rc: tuple[int, int],
               bearing_deg: float, distance_km: float) -> tuple[int, int]:
    rows, cols = grid.shape
    cells = distance_km * 1000.0 / grid.cell_m
    theta = math.radians(bearing_deg)
    r = int(round(ipp_rc[0] - cells * math.cos(theta)))   # bearing 0 = north = -row
    c = int(round(ipp_rc[1] + cells * math.sin(theta)))
    return int(np.clip(r, 0, rows - 1)), int(np.clip(c, 0, cols - 1))


def to_hypothesis(
    nom: Nomination, grid: SearchGrid, ipp_rc: tuple[int, int], period: int
) -> Hypothesis | None:
    """Convert a typed nomination into a scored hypothesis.

    This is the only path from a proposal into the belief, and it runs the same
    `build_prior_field` the published library uses. A malformed or unknown
    profile is rejected here rather than being coerced into something plausible.

    When the nomination carries angular uncertainty, the prior is spread over an
    arc of anchors rather than committed to one bearing.
    """
    profile = PROFILES.get(nom.profile_key)
    if profile is None:
        return None

    anchor = _anchor_at(grid, ipp_rc, float(nom.anchor_bearing_deg),
                        float(nom.anchor_distance_km))

    sigma = float(nom.anchor_bearing_sigma_deg)
    if sigma > 1.0:
        offsets = np.array([-1.5, -0.75, 0.0, 0.75, 1.5])
        weights = np.exp(-0.5 * offsets**2)
        weights /= weights.sum()
        field = np.zeros(grid.shape)
        for off, w in zip(offsets, weights):
            a = _anchor_at(grid, ipp_rc, nom.anchor_bearing_deg + off * sigma,
                           nom.anchor_distance_km)
            field += w * build_prior_field(grid, profile, a)
        field /= field.sum()
        return Hypothesis(
            id=f"nom-p{period}-{nom.profile_key}-{int(nom.anchor_bearing_deg)}",
            label=nom.label, narrative=nom.narrative, profile=profile,
            anchor_rc=anchor, prior_field=field, origin="nominated",
            born_iteration=period, rationale=nom.rationale,
        )

    return Hypothesis(
        id=f"nom-p{period}-{nom.profile_key}-{int(nom.anchor_bearing_deg)}",
        label=nom.label,
        narrative=nom.narrative,
        profile=profile,
        anchor_rc=anchor,
        prior_field=build_prior_field(grid, profile, anchor),
        origin="nominated",
        born_iteration=period,
        rationale=nom.rationale,
    )


# -- heuristic arm ---------------------------------------------------------

@tracing.op
def nominate_heuristic(
    belief: Belief, grid: SearchGrid, ipp_rc: tuple[int, int], rng: np.random.Generator
) -> list[Nomination]:
    """Relocate toward whatever mass is left. Reads no evidence whatsoever.

    Isolates the value of "try somewhere else" from the value of understanding
    why somewhere else.
    """
    residual = belief.joint.copy()
    # Suppress a radius around the IPP: that region is what has just failed.
    rows, cols = grid.shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    near = np.hypot(rr - ipp_rc[0], cc - ipp_rc[1]) * grid.cell_m / 1000.0 < 2.0
    residual[near] = 0.0

    target = np.unravel_index(int(np.argmax(residual)), residual.shape)
    dr, dc = target[0] - ipp_rc[0], target[1] - ipp_rc[1]
    bearing = (math.degrees(math.atan2(dc, -dr))) % 360.0
    distance = float(np.hypot(dr, dc) * grid.cell_m / 1000.0)

    return [Nomination(
        label=f"Relocated search, {distance:.1f} km on bearing {bearing:.0f}",
        narrative="Unexplained residual probability remains in this direction.",
        profile_key="hiker",
        anchor_bearing_deg=bearing,
        anchor_distance_km=distance,
        prior=0.25,
        rationale="Heuristic relocation toward remaining unsearched mass.",
        origin="heuristic",
    )]


# -- llm arm ---------------------------------------------------------------

_SYSTEM = """You are assisting a search and rescue incident commander.

A search has been running for several operational periods and has found nothing. \
Every account considered at the outset has now been substantially ruled out by \
that lack of contact. Propose DIFFERENT accounts of what happened.

You are not estimating where the subject is now. You are proposing what \
occurred, and specifically WHERE THE SUBJECT ACTUALLY BEGAN. The planning \
system takes your starting point and applies the behaviour profile's own \
distance model to it, so you name an origin, not a destination.

THE CRITICAL CONSTRAINT. Every hypothesis already under consideration is \
anchored at the planning point, and between them they cover the ground within \
roughly %(reach).0f km of it -- which is the area that has just been searched \
without contact. An account that puts the subject's start within a few \
kilometres of the planning point therefore predicts ground already ruled out \
and adds nothing. To be worth anything, a new account must place the start \
somewhere the existing ones cannot reach. In practice that means \
%(near).0f-%(far).0f km from the planning point.

Scale: the operating area is %(extent).0f km across, so a displacement of \
1-2 km is negligible here and 6-10 km is the difference between the near and \
far side of the massif. Read distance cues in the case file literally -- \
"the back side", "a different access point", "a lift" all imply crossing the \
terrain, not stepping off the trail.

Ground every proposal in specific details of the case file, and do not restate \
an account that has already been ruled out. The documented ways a search goes \
wrong: the subject was transported away from the planning point; deliberately \
went somewhere other than what they told people; their behaviour category was \
misjudged; or the planning point itself rests on a false premise.

Bearings are compass degrees: 0 = north, 90 = east, 180 = south, 270 = west.

Available behaviour profiles (choose the one whose movement pattern fits):
%(profiles)s

Return ONLY a JSON array of 2-3 objects:
[{"label": "short name for this account",
  "narrative": "one or two sentences: what you think happened",
  "profile_key": "one of the keys above",
  "anchor_bearing_deg": compass bearing from the planning point to where this \
account says the subject ACTUALLY STARTED,
  "anchor_distance_km": how far from the planning point that start was,
  "prior": 0.05-0.35 how much belief this account deserves,
  "rationale": "which specific detail of the case file supports this",
  "evidence_cited": ["quoted fragment from the case file"]}]"""


def _profile_menu() -> str:
    return "\n".join(
        f"  {k}: {p.label} -- {p.narrative}" for k, p in PROFILES.items()
    )


def _terrain_summary(grid: SearchGrid, ipp_rc: tuple[int, int]) -> str:
    """The map an incident commander would already be holding.

    Phrases like "the back side" or "the far side of the divide" are only
    interpretable against terrain. Without this the model is guessing a compass
    bearing from prose, which is where its proposals were going wrong.
    """
    rows, cols = grid.shape
    elev = grid.elevation
    summit = np.unravel_index(int(np.argmax(elev)), elev.shape)

    def bearing_to(target) -> float:
        return math.degrees(math.atan2(target[1] - ipp_rc[1],
                                       -(target[0] - ipp_rc[0]))) % 360.0

    def compass(deg: float) -> str:
        points = ["north", "north-east", "east", "south-east",
                  "south", "south-west", "west", "north-west"]
        return points[int((deg + 22.5) % 360 // 45)]

    summit_km = float(np.hypot(summit[0] - ipp_rc[0],
                               summit[1] - ipp_rc[1]) * grid.cell_m / 1000.0)
    summit_bearing = bearing_to(summit)
    # The far side of the massif: continue past the summit on the same line.
    far_bearing = (summit_bearing) % 360.0

    # Where the low ground lies, which is where drainages run out to.
    low = np.unravel_index(int(np.argmin(np.where(elev > 0, elev, 1e9))), elev.shape)
    low_bearing = bearing_to(low)

    return (
        f"TERRAIN. The planning point sits at {elev[ipp_rc]:.0f} m. The high point "
        f"of the massif is {summit_km:.1f} km away to the {compass(summit_bearing)} "
        f"(bearing {summit_bearing:.0f}). The far side of the massif -- what a "
        f"witness would call 'the back side' or 'the other side of the divide' -- "
        f"is reached by continuing on bearing {far_bearing:.0f} past the summit, "
        f"roughly {summit_km * 2:.0f} km from the planning point. The lowest ground "
        f"in the operating area lies to the {compass(low_bearing)} "
        f"(bearing {low_bearing:.0f}), which is where the drainages run out."
    )


def _coverage_summary(belief: Belief, grid: SearchGrid, ipp_rc: tuple[int, int]) -> str:
    """Where the search has actually been, in terms the model can reason about."""
    if not belief.history:
        return "No ground swept yet."
    dists = []
    for record in belief.history:
        for r, c in record.cells:
            dists.append(np.hypot(r - ipp_rc[0], c - ipp_rc[1]))
    km = np.asarray(dists) * grid.cell_m / 1000.0
    return (
        f"Ground swept so far lies between {km.min():.1f} and {km.max():.1f} km "
        f"of the planning point, concentrated around {np.median(km):.1f} km. "
        f"All of it came up empty."
    )


def _history_summary(belief: Belief, grid: SearchGrid) -> str:
    lines = []
    for h, w, e in belief.ranked():
        lines.append(
            f"  - '{h.label}' ({h.origin}): posterior {w:.2f}, "
            f"{100 * e:.0f}% of its predicted area covered with no contact"
        )
    swept = sum(len(r.cells) for r in belief.history)
    area = swept * (grid.cell_m / 1000.0) ** 2
    return (
        f"Operational periods completed: {len(belief.history)}\n"
        f"Area swept: {area:.1f} sq km\n"
        f"Overall probability the subject would have been found by now if any "
        f"current account were correct: {100 * belief.cumulative_pos:.0f}%\n"
        f"Accounts considered so far:\n" + "\n".join(lines)
    )


@tracing.op
def nominate_llm(
    belief: Belief,
    grid: SearchGrid,
    briefing: str,
    trigger: RevisionTrigger,
    ipp_rc: tuple[int, int] | None = None,
) -> list[Nomination]:
    """Ask the model for new accounts. Raises NoProviderError without a key."""
    extent_km = grid.shape[0] * grid.cell_m / 1000.0
    # The reach of the existing mixture: the widest distance model in the
    # library is what bounds where the current hypotheses can put mass.
    reach_km = max(p.d50_km for p in PROFILES.values()) * 2.0

    system = _SYSTEM % {
        "profiles": _profile_menu(),
        "extent": extent_km,
        "reach": reach_km,
        "near": max(reach_km, extent_km * 0.28),
        "far": extent_km * 0.55,
    }
    coverage = _coverage_summary(belief, grid, ipp_rc) if ipp_rc else ""
    terrain = _terrain_summary(grid, ipp_rc) if ipp_rc else ""
    user = (
        f"CASE FILE\n{briefing}\n\n"
        f"{terrain}\n\n"
        f"SEARCH TO DATE\n{_history_summary(belief, grid)}\n{coverage}\n\n"
        f"WHY YOU ARE BEING ASKED\n{trigger.reason}.\n\n"
        f"Propose 2-3 different accounts of what happened."
    )
    raw = llm.complete(system, user, max_tokens=1800, temperature=0.8)
    parsed = llm.extract_json(raw)
    if isinstance(parsed, dict):
        parsed = [parsed]

    out: list[Nomination] = []
    for item in parsed[:3]:
        try:
            out.append(Nomination(
                label=str(item["label"])[:80],
                narrative=str(item.get("narrative", ""))[:400],
                profile_key=str(item["profile_key"]),
                anchor_bearing_deg=float(item["anchor_bearing_deg"]) % 360.0,
                anchor_distance_km=float(np.clip(float(item["anchor_distance_km"]), 0.0, 12.0)),
                prior=float(np.clip(float(item.get("prior", 0.15)), 0.01, MAX_NOMINATION_PRIOR)),
                rationale=str(item.get("rationale", ""))[:300],
                evidence_cited=tuple(str(e)[:200] for e in item.get("evidence_cited", []))[:3],
                origin="llm",
            ))
        except (KeyError, TypeError, ValueError):
            continue      # A malformed proposal is dropped, not repaired.
    return out
