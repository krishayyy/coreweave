"""Find rate against search budget.

The headline number is quoted at a sixteen-period budget, which is eight days of
twelve-hour operational periods. That budget is a choice, and quoting one point
off a curve invites the obvious question: is the method better, or just faster?

Both, and the curve separates them. Every arm improves with more time, since
enough sweeping eventually covers any finite area. What differs is how much time
each needs to get there -- and in search and rescue the budget is not really
aircraft hours, it is how long a person survives.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import searchloop.loop as loop_mod                    # noqa: E402
from ceiling_check import oracle_factory              # noqa: E402
from searchloop.config import DEFAULT as CFG          # noqa: E402
from searchloop.grid import build_grid                # noqa: E402
from searchloop.loop import run_scenario              # noqa: E402
from searchloop.pod import pod_field                  # noqa: E402
from searchloop.revision import nominate_heuristic    # noqa: E402
from searchloop.scenario import generate_suite, stable_seed  # noqa: E402
from searchloop.terrain import load_terrain           # noqa: E402

ARMS = [("none", "conventional search", False),
        ("jev", "this system", False),
        ("heuristic", "oracle (told the answer)", True)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", default="8,12,16,20,24,28,32")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", default=str(ROOT / "runs" / "time_curve.json"))
    args = ap.parse_args()

    grid = build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor, CFG.treeline_m)
    pod = pod_field(grid, CFG.altitude_m)
    suite = [s for s in generate_suite(grid, 30, 24, 12, seed=7) if s.kind == "B"]
    budgets = [int(b) for b in args.budgets.split(",")]

    print(f"{len(suite)} type B scenarios x {args.repeats} repeats per point\n")
    print(f"{'periods':>8} {'days':>6} | " + " | ".join(f"{l:>22}" for _, l, _ in ARMS))
    print("-" * 84)

    curve: dict[str, list] = {arm: [] for arm, _, _ in ARMS}
    for budget in budgets:
        cfg = replace(CFG, max_periods=budget)
        cells = []
        for arm, _, oracle in ARMS:
            found = total = 0
            for repeat in range(args.repeats):
                for scenario in suite:
                    if oracle:
                        loop_mod.nominate_heuristic = oracle_factory(scenario, grid)
                    result = run_scenario(grid, pod, scenario, arm, cfg,
                                          np.random.default_rng(stable_seed(scenario.id, repeat)))
                    found += result.found
                    total += 1
            loop_mod.nominate_heuristic = nominate_heuristic
            rate = found / total
            curve[arm].append({"periods": budget, "days": budget * 12 / 24, "rate": rate})
            cells.append(f"{100 * rate:>21.0f}%")
        print(f"{budget:>8} {budget * 12 / 24:>6.1f} | " + " | ".join(cells), flush=True)

    Path(args.out).write_text(json.dumps(curve, indent=2))

    # How long conventional search needs to reach what this system reaches in 8 days.
    ours8 = next(p["rate"] for p in curve["jev"] if p["periods"] == 16)
    conv = curve["none"]
    matched = next((p for p in conv if p["rate"] >= ours8), None)
    print()
    if matched:
        print(f"Conventional search needs {matched['days']:.0f} days to reach what this "
              f"system reaches in 8 ({100 * ours8:.0f}%).")
    else:
        print(f"Conventional search does not reach {100 * ours8:.0f}% within "
              f"{conv[-1]['days']:.0f} days.")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
