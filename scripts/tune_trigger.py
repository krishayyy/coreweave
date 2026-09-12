"""Choose the revision trigger threshold on a TUNING suite, never the test suite.

The threshold is a design parameter. Selecting it on the same scenarios the
result is reported on would be fitting the test set, so this sweeps against a
separate seed and the chosen value is then applied unchanged to the held-out
suite.

Scored against the oracle nominator, which upper-bounds what any reasoning
quality could achieve: a threshold that the oracle cannot exploit is a threshold
no model could exploit either.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from searchloop import revision                       # noqa: E402
import searchloop.loop as loop_mod                    # noqa: E402
from searchloop.config import DEFAULT as CFG          # noqa: E402
from searchloop.grid import build_grid                # noqa: E402
from searchloop.loop import run_scenario              # noqa: E402
from searchloop.pod import pod_field                  # noqa: E402
from searchloop.scenario import generate_suite        # noqa: E402
from searchloop.terrain import load_terrain           # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ceiling_check import oracle_factory              # noqa: E402

TUNING_SEED = 99


def score(grid, pod, suite, cfg, arm, oracle=False):
    original = revision.nominate_heuristic
    found, periods = 0, []
    try:
        for s in suite:
            if oracle:
                loop_mod.nominate_heuristic = oracle_factory(s, grid)
            rng = np.random.default_rng(abs(hash(s.id)) % 2**32)
            r = run_scenario(grid, pod, s, arm, cfg, rng)
            found += int(r.found)
            periods.append(r.periods_to_find if r.found else cfg.max_periods)
    finally:
        loop_mod.nominate_heuristic = original
    return found / len(suite), float(np.mean(periods))


def main() -> int:
    grid = build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor, CFG.treeline_m)
    pod = pod_field(grid, CFG.altitude_m)
    suite = [s for s in generate_suite(grid, 0, 24, seed=TUNING_SEED)]

    print(f"tuning suite: {len(suite)} type B scenarios, seed {TUNING_SEED}\n")
    print(f"{'trigger':>8} {'periods':>8} | {'baseline':>9} | {'oracle':>9} | {'headroom':>9}")
    print("-" * 56)

    best = None
    for max_periods in (16, 20, 24):
        for trigger in (0.30, 0.40, 0.50, 0.55):
            cfg = replace(CFG, exhaustion_trigger=trigger, max_periods=max_periods)
            base, _ = score(grid, pod, suite, cfg, "none")
            orac, _ = score(grid, pod, suite, cfg, "heuristic", oracle=True)
            head = orac - base
            flag = ""
            if best is None or head > best[0]:
                best, flag = (head, trigger, max_periods), "  <-"
            print(f"{trigger:>8.2f} {max_periods:>8} | {100 * base:>8.0f}% | "
                  f"{100 * orac:>8.0f}% | {100 * head:>+8.0f}pp{flag}")

    print(f"\nchosen: trigger={best[1]}, max_periods={best[2]} "
          f"(headroom {100 * best[0]:+.0f}pp on the tuning suite)")
    print("This value is now applied unchanged to the held-out suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
