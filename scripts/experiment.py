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
from searchloop.scenario import generate_suite, stable_seed               # noqa: E402
from searchloop.terrain import load_terrain                  # noqa: E402

ARM_LABELS = {
    "none": "library only (no revision)",
    "heuristic": "blind relocation (no case file)",
    "llm": "nomination from the case file",
}


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval. The normal approximation is badly wrong at these
    sample sizes and near 0 or 1, which is exactly where these rates sit."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    d = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / d
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return (max(centre - half, 0.0), min(centre + half, 1.0))


def _group(runs: list) -> dict:
    out: dict[str, list] = {}
    for r in runs:
        out.setdefault(r.scenario_subkind, []).append(r)
    return dict(sorted(out.items()))


def summarise(results: list, arm: str) -> dict:
    out = {"arm": arm}
    for kind in ("A", "B", "C"):
        sub = [r for r in results if r.scenario_kind == kind]
        if not sub:
            continue
        found = [r for r in sub if r.found]
        periods = [r.periods_to_find for r in found]
        # Unfound runs are censored, not excluded: reporting the median over
        # found runs only would make a worse arm look faster.
        censored = [r.periods_to_find if r.found else CFG.max_periods for r in sub]
        localized = [r for r in sub if r.periods_to_localize is not None]
        out[kind] = {
            "n": len(sub),
            "found": len(found),
            "find_rate": len(found) / len(sub),
            "find_ci": wilson(len(found), len(sub)),
            "median_periods_found_only": float(np.median(periods)) if periods else None,
            "mean_periods_censored": float(np.mean(censored)),
            "mean_area_km2": float(np.mean([r.area_swept_km2 for r in sub])),
            "revisions": float(np.mean([r.revisions for r in sub])),
            # Reasoning metrics: unaffected by detection luck.
            "localize_rate": len(localized) / len(sub),
            "localize_ci": wilson(len(localized), len(sub)),
            "mean_periods_to_localize": (
                float(np.mean([r.periods_to_localize for r in localized]))
                if localized else None),
            "mean_peak_percentile": float(np.mean([r.peak_truth_percentile for r in sub])),
            "by_subkind": {
                k: {
                    "n": len(v),
                    "find_rate": sum(x.found for x in v) / len(v),
                    "localize_rate": sum(x.periods_to_localize is not None for x in v) / len(v),
                }
                for k, v in _group(sub).items()
            },
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-a", type=int, default=30)
    ap.add_argument("--n-b", type=int, default=20)
    ap.add_argument("--n-c", type=int, default=12)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--arms", default="none,heuristic,llm")
    ap.add_argument("--repeats", type=int, default=5,
                    help="independent detection-roll repeats per scenario; "
                         "n=20 scenarios is small, so a single pass is noisy")
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
    suite = generate_suite(grid, args.n_a, args.n_b, args.n_c, seed=args.seed)

    print(f"operating area {grid.shape[0] * grid.cell_m / 1000:.1f} km across, "
          f"{grid.cell_m:.0f} m cells, mean POD {pod.mean():.2f}")
    print(f"{len(suite)} scenarios ({args.n_a} A, {args.n_b} B, {args.n_c} C) "
          f"x {args.repeats} repeats, arms: {', '.join(arms)}\n")

    all_results, summaries = [], []
    for arm in arms:
        t0 = time.time()
        results = []
        for repeat in range(args.repeats):
            for scenario in suite:
                # Seeded per (scenario, repeat) and not per arm, so every arm
                # faces identical detection rolls and nothing separating them
                # can come from luck. stable_seed does not vary across
                # processes, which plain hash() does.
                rng = np.random.default_rng(stable_seed(scenario.id, repeat))
                results.append(run_scenario(grid, pod, scenario, arm, CFG, rng))
        all_results += results
        summaries.append(summarise(results, arm))
        print(f"  {arm}: {len(results)} runs in {time.time() - t0:.1f}s")

    print(f"\n{'arm':<34} {'type':<5} {'find rate (95% CI)':>20} "
          f"{'localised (95% CI)':>20} {'to loc':>10} {'peak':>8}")
    print("-" * 100)
    for s in summaries:
        for kind in ("A", "B", "C"):
            if kind not in s:
                continue
            d = s[kind]
            tl = d["mean_periods_to_localize"]
            fl, fh = d["find_ci"]
            ll, lh = d["localize_ci"]
            print(f"{ARM_LABELS[s['arm']]:<34} {kind:<5} "
                  f"{100 * d['find_rate']:>4.0f}% [{100 * fl:>2.0f}-{100 * fh:<2.0f}] "
                  f"{100 * d['localize_rate']:>4.0f}% [{100 * ll:>2.0f}-{100 * lh:<2.0f}] "
                  f"{(f'{tl:.1f}' if tl else '--'):>10} {d['mean_peak_percentile']:>7.0f}%")

    print(f"\nfind rate   -- subject actually detected within {CFG.max_periods} periods.")
    print("              Capped by sensor POD, so it partly reports detection luck.")
    print("localised   -- true location reached the top decile of belief at any point.")
    print("              This is the reasoning metric: it moves only when the agent")
    print("              reallocates belief correctly, and ignores detection rolls.")
    print("peak pct    -- highest percentile the true location ever reached.")
    print("\nA = premise correct (the null: revision should not help here)")
    print("B = wrong about WHERE the subject started -- no library hypothesis can reach them")
    print("C = wrong about WHO they are; the library already holds the right profile")

    print("\nType B by failure mode:")
    for s in summaries:
        if "B" not in s:
            continue
        print(f"  {ARM_LABELS[s['arm']]}")
        for k, d in s["B"]["by_subkind"].items():
            print(f"    {k:<24} n={d['n']:<3} find {100 * d['find_rate']:>3.0f}%  "
                  f"localise {100 * d['localize_rate']:>3.0f}%")

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
