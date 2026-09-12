"""Run the experiment: every scenario against every arm, and report honestly.

Usage:
    python scripts/experiment.py [--n-a 30] [--n-b 20] [--arms none,heuristic,llm]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from searchloop import llm                                   # noqa: E402
from searchloop.config import DEFAULT as CFG                 # noqa: E402
from searchloop.grid import build_grid                       # noqa: E402
from searchloop.loop import run_scenario                     # noqa: E402
from searchloop.pod import pod_field                         # noqa: E402
from searchloop.scenario import generate_suite               # noqa: E402
from searchloop.terrain import load_terrain                  # noqa: E402

ARM_LABELS = {
    "none": "library only (no revision)",
    "heuristic": "blind relocation (no case file)",
    "llm": "nomination from the case file",
}


def summarise(results: list, arm: str) -> dict:
    out = {"arm": arm}
    for kind in ("A", "B"):
        sub = [r for r in results if r.scenario_kind == kind]
        if not sub:
            continue
        found = [r for r in sub if r.found]
        periods = [r.periods_to_find for r in found]
        # Unfound runs are censored, not excluded: reporting the median over
        # found runs only would make a worse arm look faster.
        censored = [r.periods_to_find if r.found else CFG.max_periods for r in sub]
        out[kind] = {
            "n": len(sub),
            "found": len(found),
            "find_rate": len(found) / len(sub),
            "median_periods_found_only": float(np.median(periods)) if periods else None,
            "mean_periods_censored": float(np.mean(censored)),
            "mean_area_km2": float(np.mean([r.area_swept_km2 for r in sub])),
            "revisions": float(np.mean([r.revisions for r in sub])),
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-a", type=int, default=30)
    ap.add_argument("--n-b", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arms", default="none,heuristic,llm")
    ap.add_argument("--out", default="runs/experiment.json")
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if "llm" in arms and not llm.available():
        print("! no LLM credentials found -- skipping the 'llm' arm.")
        print("  set ANTHROPIC_API_KEY / OPENAI_API_KEY / WANDB_API_KEY / TYPESAFE_API_KEY\n")
        arms = [a for a in arms if a != "llm"]

    grid = build_grid(
        load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom, CFG.radius_tiles),
        CFG.grid_factor, CFG.treeline_m,
    )
    pod = pod_field(grid, CFG.altitude_m)
    suite = generate_suite(grid, args.n_a, args.n_b, seed=args.seed)

    print(f"operating area {grid.shape[0] * grid.cell_m / 1000:.1f} km across, "
          f"{grid.cell_m:.0f} m cells, mean POD {pod.mean():.2f}")
    print(f"{len(suite)} scenarios ({args.n_a} type A, {args.n_b} type B), "
          f"arms: {', '.join(arms)}\n")

    all_results, summaries = [], []
    for arm in arms:
        t0 = time.time()
        results = []
        for scenario in suite:
            # Seeded per scenario, not per arm: every arm faces the identical
            # detection rolls, so differences cannot come from luck.
            rng = np.random.default_rng(abs(hash(scenario.id)) % 2**32)
            results.append(run_scenario(grid, pod, scenario, arm, CFG, rng))
        all_results += results
        summaries.append(summarise(results, arm))
        print(f"  {arm}: {len(results)} runs in {time.time() - t0:.1f}s")

    print(f"\n{'arm':<34} {'type':<5} {'found':>9} {'rate':>7} {'periods*':>10} {'area km2':>10}")
    print("-" * 80)
    for s in summaries:
        for kind in ("A", "B"):
            if kind not in s:
                continue
            d = s[kind]
            print(f"{ARM_LABELS[s['arm']]:<34} {kind:<5} "
                  f"{d['found']:>4}/{d['n']:<4} {100 * d['find_rate']:>6.0f}% "
                  f"{d['mean_periods_censored']:>10.1f} {d['mean_area_km2']:>10.0f}")
    print("\n* mean periods to find, with unfound runs censored at the "
          f"{CFG.max_periods}-period budget.")
    print("  Type A is the null: revision should not help when the premise was right.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "config": CFG.__dict__,
        "summaries": summaries,
        "runs": [r.to_dict() for r in all_results],
    }, indent=2))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
