"""How many times should the loop be allowed to change its mind?

Each revision admits new hypotheses, and an uncapped mixture dilutes: the joint
spreads over a dozen competing accounts and the planner stops committing to any
of them. Swept on the tuning suite (seed 99) and applied unchanged to the
reported suite.

Scored with the heuristic nominator and the oracle, neither of which needs
credentials -- so the cap is chosen without reference to the model arm's
numbers.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import searchloop.loop as loop_mod                    # noqa: E402
from ceiling_check import oracle_factory              # noqa: E402
from searchloop.config import DEFAULT as CFG          # noqa: E402
from searchloop.grid import build_grid                # noqa: E402
from searchloop.loop import run_scenario              # noqa: E402
from searchloop.pod import pod_field                  # noqa: E402
from searchloop.revision import nominate_heuristic    # noqa: E402
from searchloop.scenario import generate_suite, stable_seed  # noqa: E402
from searchloop.terrain import load_terrain           # noqa: E402


def score(grid, pod, suite, cfg, arm, oracle=False, repeats=3):
    found = loc = n = 0
    try:
        for repeat in range(repeats):
            for s in suite:
                loop_mod.nominate_heuristic = (
                    oracle_factory(s, grid) if oracle else nominate_heuristic)
                r = run_scenario(grid, pod, s, arm, cfg,
                                 np.random.default_rng(stable_seed(s.id, repeat)))
                found += r.found
                loc += r.periods_to_localize is not None
                n += 1
    finally:
        loop_mod.nominate_heuristic = nominate_heuristic
    return found / n, loc / n


def main() -> int:
    grid = build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor, CFG.treeline_m)
    pod = pod_field(grid, CFG.altitude_m)
    suite = generate_suite(grid, 0, 24, 0, seed=99)

    base_f, base_l = score(grid, pod, suite, CFG, "none")
    print(f"tuning suite: {len(suite)} type B, seed 99")
    print(f"baseline (no revision): find {100 * base_f:.0f}%  localise {100 * base_l:.0f}%\n")
    print(f"{'cap':>4} | {'heuristic find':>15} {'localise':>9} | {'oracle find':>12} {'localise':>9}")
    print("-" * 62)
    for cap in (1, 2, 3, 99):
        cfg = replace(CFG, max_revisions=cap)
        hf, hl = score(grid, pod, suite, cfg, "heuristic")
        of, ol = score(grid, pod, suite, cfg, "heuristic", oracle=True)
        print(f"{cap:>4} | {100 * hf:>14.0f}% {100 * hl:>8.0f}% | "
              f"{100 * of:>11.0f}% {100 * ol:>8.0f}%")
    print("\nThe oracle column is the ceiling: how much a correct account is worth")
    print("at each cap. A cap that strangles the oracle would strangle the model too.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
