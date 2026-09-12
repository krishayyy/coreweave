"""Export one run as frames the frontend can render directly.

Belief fields go out as PNGs rather than JSON arrays: a 192x192 float field per
period is 37k numbers, and deck.gl's BitmapLayer wants an image anyway. The
JSON carries only the timeline -- geography, hypotheses, revisions -- which is
the part the interface actually reasons about.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from searchloop.config import DEFAULT as CFG            # noqa: E402
from searchloop.grid import build_grid                  # noqa: E402
from searchloop.loop import run_scenario                # noqa: E402
from searchloop.pod import pod_field                    # noqa: E402
from searchloop.scenario import generate_suite, stable_seed          # noqa: E402
from searchloop.terrain import hillshade, load_terrain  # noqa: E402


def _to_png(field: np.ndarray, path: Path, gamma: float = 0.45) -> None:
    """Quantise a field to 8-bit with a gamma lift so the tail stays visible."""
    top = float(np.percentile(field, 99.9))
    norm = np.clip(field / top, 0.0, 1.0) ** gamma if top > 0 else np.zeros_like(field)
    Image.fromarray((norm * 255).astype(np.uint8), mode="L").save(path)


def _serpentine(cells: list[tuple[int, int]]) -> list[list[int]]:
    """Order a segment into the mow pattern an aircraft would actually fly."""
    by_row: dict[int, list[int]] = {}
    for r, c in cells:
        by_row.setdefault(r, []).append(c)
    track: list[list[int]] = []
    for i, r in enumerate(sorted(by_row)):
        cols = sorted(by_row[r], reverse=bool(i % 2))
        track.append([r, cols[0]])
        track.append([r, cols[-1]])
    return track


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", default="B")
    ap.add_argument("--index", type=int, default=2)
    ap.add_argument("--arm", default="heuristic")
    ap.add_argument("--oracle", action="store_true")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="web/public/run")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "fields").mkdir(parents=True, exist_ok=True)

    terrain = load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom, CFG.radius_tiles)
    grid = build_grid(terrain, CFG.grid_factor, CFG.treeline_m)
    pod = pod_field(grid, CFG.altitude_m)
    scenario = [s for s in generate_suite(grid, 30, 24, 12, seed=args.seed)
                if s.kind == args.kind][args.index]

    if args.oracle:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from ceiling_check import oracle_factory
        import searchloop.loop as loop_mod
        loop_mod.nominate_heuristic = oracle_factory(scenario, grid)

    frames: list[dict] = []
    swept = np.zeros(grid.shape, dtype=bool)

    def capture(trace, belief, segment):
        for c in segment:
            swept[c] = True
        idx = trace.period
        _to_png(belief.joint, out / "fields" / f"belief_{idx:02d}.png")
        Image.fromarray((swept * 255).astype(np.uint8), mode="L").save(
            out / "fields" / f"swept_{idx:02d}.png")
        frames.append({
            "period": idx,
            "belief": f"fields/belief_{idx:02d}.png",
            "swept": f"fields/swept_{idx:02d}.png",
            "track": _serpentine(segment),
            "found": trace.found,
            "revised": bool(trace.nominations),
            "trigger_fired": trace.trigger_fired,
            "trigger_reason": trace.trigger_reason,
            "nominations": trace.nominations,
            "rejected": trace.rejected_nominations,
            "cumulative_pos": trace.cumulative_pos,
            "entropy": trace.entropy,
            "track_km": trace.track_km,
            "hypotheses": [
                {"label": h.label, "weight": w, "exhaustion": e, "origin": h.origin}
                for h, w, e in belief.ranked()[:6]
            ],
        })

    rng = np.random.default_rng(stable_seed(scenario.id))
    result = run_scenario(grid, pod, scenario, args.arm, CFG, rng, on_period=capture)

    shade = hillshade(terrain)
    Image.fromarray((np.clip(shade, 0, 1) * 255).astype(np.uint8), mode="L").save(
        out / "hillshade.png")

    (out / "run.json").write_text(json.dumps({
        "scenario": {
            "id": scenario.id, "kind": scenario.kind, "subkind": scenario.subkind,
            "case_file": scenario.case_file,
            "late_evidence": [{"period": e.period, "text": e.text}
                              for e in scenario.late_evidence],
            "ipp": list(scenario.ipp_rc),
            # Ground truth is exported for the display only, so the finished
            # frame can show where they actually were. The searcher never saw it.
            "truth": list(scenario.true_rc),
            "truth_distance_km": result.true_distance_km,
        },
        "geo": {
            "north": terrain.lat_north, "south": terrain.lat_south,
            "west": terrain.lon_west, "east": terrain.lon_east,
            "rows": grid.shape[0], "cols": grid.shape[1], "cell_m": grid.cell_m,
        },
        "result": {
            "found": result.found, "periods_to_find": result.periods_to_find,
            "revisions": result.revisions, "arm": result.arm,
            "oracle": bool(args.oracle),
        },
        "frames": frames,
    }, indent=2))

    print(f"{scenario.id} [{scenario.subkind}] -> {out}/run.json")
    print(f"  {len(frames)} frames, found={result.found} period={result.periods_to_find}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
