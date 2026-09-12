"""Render one scenario's search as a filmstrip. Development view, not the demo.

Exists to answer one question before any effort goes into the real frontend:
when the premise is abandoned, does the belief field visibly move?
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, PowerNorm  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from searchloop.config import DEFAULT as CFG        # noqa: E402
from searchloop.grid import build_grid              # noqa: E402
from searchloop.loop import run_scenario            # noqa: E402
from searchloop.pod import pod_field                # noqa: E402
from searchloop.scenario import generate_suite, stable_seed      # noqa: E402
from searchloop.terrain import hillshade, load_terrain  # noqa: E402

BELIEF_CMAP = LinearSegmentedColormap.from_list(
    "belief", ["#00000000", "#2b1f63cc", "#a4442bee", "#ffb020ff", "#fff4d6ff"]
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="heuristic")
    ap.add_argument("--oracle", action="store_true",
                    help="diagnostic: nominate from withheld ground truth, to see "
                         "what a correct revision looks like")
    ap.add_argument("--kind", default="B")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="runs/filmstrip.png")
    args = ap.parse_args()

    terrain = load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom, CFG.radius_tiles)
    grid = build_grid(terrain, CFG.grid_factor, CFG.treeline_m)
    pod = pod_field(grid, CFG.altitude_m)
    shade = hillshade(terrain)[::CFG.grid_factor, ::CFG.grid_factor][: grid.shape[0], : grid.shape[1]]

    suite = generate_suite(grid, 30, 24, 12, seed=args.seed)
    scenario = [s for s in suite if s.kind == args.kind][args.index]

    frames: list[tuple] = []
    swept = np.zeros(grid.shape, dtype=bool)

    def capture(trace, belief, segment):
        for c in segment:
            swept[c] = True
        frames.append((trace.period, belief.joint.copy(), swept.copy(),
                       trace.leader_label, trace.leader_exhaustion,
                       bool(trace.nominations), trace.found))

    if args.oracle:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ceiling_check import oracle_factory
        import searchloop.loop as loop_mod
        loop_mod.nominate_heuristic = oracle_factory(scenario, grid)

    rng = np.random.default_rng(stable_seed(scenario.id))
    result = run_scenario(grid, pod, scenario, args.arm, CFG, rng, on_period=capture)

    flip = next((i for i, f in enumerate(frames) if f[5]), None)
    picks = sorted({0, 1, max((flip or 2) - 1, 0), flip or 3,
                    min((flip or 3) + 1, len(frames) - 1), len(frames) - 1})
    picks = [p for p in picks if p < len(frames)][:6]

    fig, axes = plt.subplots(1, len(picks), figsize=(4.0 * len(picks), 4.6))
    fig.patch.set_facecolor("#0A0C0F")
    if len(picks) == 1:
        axes = [axes]

    for ax, i in zip(axes, picks):
        period, joint, mask, leader, exhaustion, revised, found = frames[i]
        ax.imshow(shade, cmap="gray", vmin=0, vmax=1.4, interpolation="bilinear")
        ax.imshow(np.where(mask, 0.30, np.nan), cmap="Blues_r", vmin=0, vmax=1, alpha=0.55)
        ax.imshow(joint, cmap=BELIEF_CMAP,
                  norm=PowerNorm(0.45, vmin=0, vmax=float(np.percentile(joint, 99.9))),
                  interpolation="bilinear")
        ax.plot(*scenario.ipp_rc[::-1], marker="P", ms=9, mfc="#7ee7ff", mec="#0A0C0F", mew=1.2)
        ax.plot(*scenario.true_rc[::-1], marker="x", ms=11, mec="#ff4d4d", mew=2.4)

        title = f"period {period}"
        if revised:
            title += "  • REVISED"
        if found:
            title += "  • FOUND"
        ax.set_title(title, color="#ff6b6b" if revised else "#e8eaed",
                     fontsize=11, family="monospace", pad=8)
        ax.set_xlabel(f"{leader[:30]}\n{100 * exhaustion:.0f}% ruled out",
                      color="#9aa0a6", fontsize=8, family="monospace")
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#2a2f36")

    fig.suptitle(
        f"{scenario.id}  [{scenario.subkind}]   arm={args.arm}   "
        f"truth {result.true_distance_km:.1f} km from planning point   "
        f"{'FOUND period ' + str(result.periods_to_find) if result.found else 'NOT FOUND'}",
        color="#e8eaed", family="monospace", fontsize=12, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130, facecolor=fig.get_facecolor())
    print(f"{scenario.id} [{scenario.subkind}] arm={args.arm} -> {out}")
    print(f"  revisions={result.revisions} found={result.found} "
          f"period={result.periods_to_find} leader='{result.final_leader}'")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
