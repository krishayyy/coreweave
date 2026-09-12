"""Diagnostic: is the ceiling reachable at all?

Feeds the loop a nomination derived from the scenario's withheld ground truth.
This is NOT an experimental arm and its numbers are never reported as a result.
It answers one engineering question: if the right account were proposed, would
the machinery actually convert that into a find? If the oracle cannot find the
subject, no amount of reasoning quality would have helped and the bottleneck is
elsewhere.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from searchloop import revision                       # noqa: E402
from searchloop.config import DEFAULT as CFG          # noqa: E402
from searchloop.grid import build_grid                # noqa: E402
from searchloop.loop import run_scenario              # noqa: E402
from searchloop.pod import pod_field                  # noqa: E402
from searchloop.revision import Nomination            # noqa: E402
from searchloop.scenario import generate_suite        # noqa: E402
from searchloop.terrain import load_terrain           # noqa: E402


def oracle_factory(scenario, grid):
    """A nomination pointing at the true account -- diagnostic only."""
    def nominate(belief, g, ipp_rc, rng):
        # The true STARTING point, not the subject's location: a hypothesis
        # names where someone began, and the profile's distance model supplies
        # the spread from there.
        dr = scenario.true_anchor_rc[0] - ipp_rc[0]
        dc = scenario.true_anchor_rc[1] - ipp_rc[1]
        bearing = math.degrees(math.atan2(dc, -dr)) % 360.0
        distance = float(np.hypot(dr, dc) * grid.cell_m / 1000.0)
        return [Nomination(
            label="[oracle] true account",
            narrative=scenario.true_account,
            profile_key=scenario.true_profile_key,
            anchor_bearing_deg=bearing,
            anchor_distance_km=distance,
            prior=0.35,
            rationale="diagnostic: derived from withheld ground truth",
            origin="oracle",
        )]
    return nominate


def main() -> int:
    grid = build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor, CFG.treeline_m)
    pod = pod_field(grid, CFG.altitude_m)
    suite = generate_suite(grid, 30, 20, seed=7)
    type_b = [s for s in suite if s.kind == "B"]

    original = revision.nominate_heuristic
    found = 0
    periods = []
    try:
        for s in type_b:
            revision.nominate_heuristic = oracle_factory(s, grid)
            import searchloop.loop as loop_mod
            loop_mod.nominate_heuristic = revision.nominate_heuristic
            rng = np.random.default_rng(abs(hash(s.id)) % 2**32)
            r = run_scenario(grid, pod, s, "heuristic", CFG, rng)
            found += int(r.found)
            periods.append(r.periods_to_find if r.found else CFG.max_periods)
    finally:
        revision.nominate_heuristic = original
        import searchloop.loop as loop_mod
        loop_mod.nominate_heuristic = original

    print(f"DIAGNOSTIC ONLY -- not an experimental arm, never reported as a result.\n")
    print(f"oracle nomination on {len(type_b)} type B scenarios:")
    print(f"  found {found}/{len(type_b)} ({100 * found / len(type_b):.0f}%)")
    print(f"  mean periods (censored at {CFG.max_periods}): {np.mean(periods):.1f}")
    print(f"\nbaseline for comparison was 40% -- the reachable ceiling is the "
          f"number above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
