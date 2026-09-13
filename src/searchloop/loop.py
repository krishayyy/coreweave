"""The orchestrator: run one scenario to resolution and record the trajectory.

Detection is simulated honestly. Sweeping the cell the subject is actually in
does not guarantee contact -- it succeeds with probability POD for that cell, so
a search can pass over the subject and miss, which is the single most common way
real searches lose time.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import numpy as np

from .belief import Belief, SearchRecord
from .config import Config
from .grid import SearchGrid
from .hypotheses import library_hypotheses
from .planner import plan_sortie, segment_capacity, sortie_pos
from .revision import (
    Nomination,
    RevisionTrigger,
    nominate_heuristic,
    nominate_jev,
    nominate_llm,
    should_revise,
    to_hypothesis,
)
from .scenario import Scenario
from . import tracing

Arm = Literal["none", "heuristic", "llm", "jev"]

# Instructions the agent has written for itself and that survived validation.
# Set by the self-improvement driver; empty by default, so an untrained run is
# exactly the run it always was.
ACTIVE_LESSONS = ""

# Resolved searches the agent can draw on. Set by the learning-curve driver;
# None means no experience, which is how every run behaved before this existed.
ACTIVE_MEMORY = None


@dataclass
class PeriodTrace:
    """Everything that happened in one operational period."""

    period: int
    leader_label: str
    leader_weight: float
    leader_exhaustion: float
    cumulative_pos: float
    entropy: float
    cells_swept: int
    track_km: float
    found: bool
    # Where the true location sits within the belief field, as a percentile of
    # cells carrying less mass. This measures whether the agent worked out where
    # the subject was, independently of whether the sensor happened to catch
    # them -- find-rate alone is capped by POD and mostly reports detection luck.
    truth_percentile: float
    trigger_fired: bool
    trigger_reason: str
    nominations: list[dict[str, Any]] = field(default_factory=list)
    rejected_nominations: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RunResult:
    scenario_id: str
    scenario_kind: str
    scenario_subkind: str
    arm: str
    found: bool
    periods_to_find: int | None
    periods_run: int
    area_swept_km2: float
    revisions: int
    # First period at which the true location entered the top decile of belief.
    # None if it never did. This is the reasoning metric: it moves when the agent
    # reallocates belief correctly, and is unaffected by detection rolls.
    periods_to_localize: int | None
    peak_truth_percentile: float
    final_leader: str
    final_leader_origin: str
    true_distance_km: float
    trace: list[PeriodTrace] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["trace"] = [asdict(t) for t in self.trace]
        return d


@tracing.op
def run_scenario(
    grid: SearchGrid,
    pod: np.ndarray,
    scenario: Scenario,
    arm: Arm,
    cfg: Config,
    rng: np.random.Generator,
    on_period=None,
    county=None,
) -> RunResult:
    """Search until the subject is found or the period budget is exhausted.

    `county` is the shared local model of the county being flown, or None for
    a county with no resolved cases yet.
    """
    belief = Belief.from_hypotheses(
        library_hypotheses(grid, scenario.ipp_rc, county))
    capacity = segment_capacity(grid, cfg.period_track_km, cfg.altitude_m)
    position = scenario.ipp_rc
    last_revision: int | None = None
    revisions = 0
    found_at: int | None = None
    trace: list[PeriodTrace] = []

    for period in range(cfg.max_periods):
        # --- outer loop: is the premise still holding up? ---
        # Evidence that became available this period, if any.
        arriving = next((e.text for e in scenario.late_evidence
                         if e.period == period), None)
        trigger = should_revise(belief, cfg, period, last_revision, arriving)
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []

        if trigger.fired and arm != "none" and revisions < cfg.max_revisions:
            try:
                if arm == "heuristic":
                    noms = nominate_heuristic(belief, grid, scenario.ipp_rc, rng)
                elif arm == "jev":
                    precedent = ""
                    if ACTIVE_MEMORY is not None:
                        evidence = (scenario.late_evidence[0].text
                                    if scenario.late_evidence else scenario.case_file)
                        precedent = ACTIVE_MEMORY.prompt_section(
                            evidence, scenario.case_file)
                    noms = nominate_jev(belief, grid, scenario.briefing(period),
                                        trigger, scenario.ipp_rc, precedent)
                else:
                    precedent = ""
                    if ACTIVE_MEMORY is not None:
                        evidence = (scenario.late_evidence[0].text
                                    if scenario.late_evidence else scenario.case_file)
                        precedent = ACTIVE_MEMORY.prompt_section(
                            evidence, scenario.case_file)
                    noms = nominate_llm(belief, grid, scenario.briefing(period), trigger,
                                        scenario.ipp_rc, lessons=ACTIVE_LESSONS,
                                        precedent=precedent)
            except Exception as exc:            # a failed nomination must not end the search
                noms = []
                rejected.append({"error": f"{type(exc).__name__}: {exc}"})

            for nom in noms:
                hypothesis = to_hypothesis(nom, grid, scenario.ipp_rc, period)
                if hypothesis is None:
                    rejected.append({**_nom_dict(nom), "reason": "unknown profile_key"})
                    continue
                belief.add_hypothesis(hypothesis, nom.prior)
                accepted.append(_nom_dict(nom))
            if accepted:
                revisions += 1
                last_revision = period
            tracing.log("revision", {
                "scenario": scenario.id, "period": period + 1, "arm": arm,
                "trigger": trigger.reason,
                "leader_ruled_out": trigger.leader_exhaustion,
                "accepted": accepted, "rejected": rejected,
            })

        # --- inner loop: sweep a segment ---
        segment = plan_sortie(grid, belief.joint, pod, position, budget_cells=capacity)
        pos = sortie_pos(belief.joint, pod, segment)
        cell_pods = [float(pod[c]) for c in segment]

        found = False
        if scenario.true_rc in set(segment):
            found = bool(rng.random() < float(pod[scenario.true_rc]))

        belief.observe(SearchRecord(
            iteration=period, cells=segment, pod=cell_pods, found=found, pos=pos,
            track_km=len(segment) * grid.cell_m / 1000.0,
        ))
        position = segment[len(segment) // 2]
        belief.prune(cfg.prune_floor)

        joint = belief.joint
        truth_pct = float(100.0 * (joint < joint[scenario.true_rc]).mean())

        leader, weight = belief.leader
        trace.append(PeriodTrace(
            period=period + 1,
            leader_label=leader.label,
            leader_weight=weight,
            leader_exhaustion=float(belief.exhaustion[belief.hypotheses.index(leader)]),
            cumulative_pos=belief.cumulative_pos,
            entropy=belief.entropy(),
            cells_swept=len(segment),
            track_km=len(segment) * grid.cell_m / 1000.0,
            found=found,
            truth_percentile=truth_pct,
            trigger_fired=trigger.fired,
            trigger_reason=trigger.reason,
            nominations=accepted,
            rejected_nominations=rejected,
        ))
        if on_period is not None:
            on_period(trace[-1], belief, segment)

        if found:
            found_at = period + 1
            break

    leader, _ = belief.leader
    swept = sum(t.cells_swept for t in trace)
    localized = next((t.period for t in trace if t.truth_percentile >= 90.0), None)
    peak_pct = max((t.truth_percentile for t in trace), default=0.0)
    dist = float(np.hypot(
        scenario.true_rc[0] - scenario.ipp_rc[0],
        scenario.true_rc[1] - scenario.ipp_rc[1],
    ) * grid.cell_m / 1000.0)

    return RunResult(
        scenario_id=scenario.id,
        scenario_kind=scenario.kind,
        scenario_subkind=scenario.subkind,
        arm=arm,
        found=found_at is not None,
        periods_to_find=found_at,
        periods_run=len(trace),
        area_swept_km2=swept * (grid.cell_m / 1000.0) ** 2,
        revisions=revisions,
        periods_to_localize=localized,
        peak_truth_percentile=peak_pct,
        final_leader=leader.label,
        final_leader_origin=leader.origin,
        true_distance_km=dist,
        trace=trace,
    )


def _nom_dict(nom: Nomination) -> dict[str, Any]:
    return {
        "label": nom.label, "narrative": nom.narrative, "profile_key": nom.profile_key,
        "bearing_deg": nom.anchor_bearing_deg, "distance_km": nom.anchor_distance_km,
        "prior": nom.prior, "rationale": nom.rationale,
        "evidence_cited": list(nom.evidence_cited), "origin": nom.origin,
    }
