"""Export several cases, each searched two ways, for the live display.

One case proves the mechanism. Several let someone at a judging table ask the
question they actually want to ask -- show me a different one, show me one where
it fails -- and get an answer rather than a promise.

    python scripts/export_demo.py --cases 5,10,13,18,19

Each case is run twice on identical detection rolls: once by this system, once
by conventional Bayesian search that never revises. Anything that differs
between the two is the method.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from searchloop.config import DEFAULT as CFG              # noqa: E402
from searchloop.grid import build_grid                    # noqa: E402
from searchloop.loop import run_scenario                  # noqa: E402
from searchloop.pod import pod_field                      # noqa: E402
from searchloop.scenario import generate_suite, stable_seed  # noqa: E402
from searchloop.terrain import load_imagery, load_terrain  # noqa: E402


def _to_png(field: np.ndarray, path: Path, gamma: float = 0.62) -> None:
    top = float(np.percentile(field, 99.9))
    norm = np.clip(field / top, 0.0, 1.0) ** gamma if top > 0 else np.zeros_like(field)
    Image.fromarray((norm * 255).astype(np.uint8), mode="L").save(path)


def _serpentine(cells: list[tuple[int, int]]) -> list[list[int]]:
    by_row: dict[int, list[int]] = {}
    for r, c in cells:
        by_row.setdefault(r, []).append(c)
    track: list[list[int]] = []
    for i, r in enumerate(sorted(by_row)):
        cols = sorted(by_row[r], reverse=bool(i % 2))
        step = max(len(cols) // 8, 1)
        track.extend([r, c] for c in cols[::step])
        if cols:
            track.append([r, cols[-1]])
    return track


def run_arm(grid, pod, scenario, arm, out, prefix):
    frames: list[dict] = []
    swept = np.zeros(grid.shape, dtype=bool)

    def capture(trace, belief, segment):
        for c in segment:
            swept[c] = True
        i = trace.period
        _to_png(belief.joint, out / "fields" / f"{prefix}_b{i:02d}.png")
        Image.fromarray((swept * 255).astype(np.uint8), mode="L").save(
            out / "fields" / f"{prefix}_s{i:02d}.png")
        frames.append({
            "period": i,
            "belief": f"fields/{prefix}_b{i:02d}.png",
            "swept": f"fields/{prefix}_s{i:02d}.png",
            "track": _serpentine(segment),
            "found": trace.found,
            "revised": bool(trace.nominations),
            "leader": trace.leader_label,
            "truth_percentile": trace.truth_percentile,
            "trigger_reason": trace.trigger_reason,
            "nominations": trace.nominations,
        })

    result = run_scenario(grid, pod, scenario, arm, CFG,
                          np.random.default_rng(stable_seed(scenario.id)),
                          on_period=capture)
    return frames, result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default="5,10,13,18,19,4")
    ap.add_argument("--out", default=str(ROOT / "web" / "public" / "run"))
    ap.add_argument("--detail", type=int, default=2)
    args = ap.parse_args()

    out = Path(args.out)
    (out / "fields").mkdir(parents=True, exist_ok=True)

    terrain = load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom, CFG.radius_tiles)
    grid = build_grid(terrain, CFG.grid_factor, CFG.treeline_m)
    pod = pod_field(grid, CFG.altitude_m)

    print("fetching aerial imagery...", flush=True)
    load_imagery(terrain, args.detail).resize((1536, 1536), Image.LANCZOS).save(
        out / "imagery.jpg", quality=88)

    suite = [s for s in generate_suite(grid, 30, 24, 12, seed=7) if s.kind == "B"]
    cases = []
    for idx in [int(x) for x in args.cases.split(",")]:
        scenario = suite[idx]
        ours_f, ours_r = run_arm(grid, pod, scenario, "jev", out, f"c{idx}o")
        conv_f, conv_r = run_arm(grid, pod, scenario, "none", out, f"c{idx}c")
        cases.append({
            "index": idx,
            "id": scenario.id,
            "subkind": scenario.subkind,
            "case_file": scenario.case_file,
            "late_evidence": [{"period": e.period, "text": e.text}
                              for e in scenario.late_evidence],
            "ipp": list(scenario.ipp_rc),
            "truth": list(scenario.true_rc),
            "truth_distance_km": ours_r.true_distance_km,
            "ours": {"frames": ours_f, "found": ours_r.found,
                     "periods_to_find": ours_r.periods_to_find,
                     "revisions": ours_r.revisions},
            "conventional": {"frames": conv_f, "found": conv_r.found,
                             "periods_to_find": conv_r.periods_to_find},
        })
        print(f"  {scenario.id} [{scenario.subkind:<21}] "
              f"ours {'p' + str(ours_r.periods_to_find) if ours_r.found else 'not found':<10} "
              f"conventional {'p' + str(conv_r.periods_to_find) if conv_r.found else 'not found'}",
              flush=True)

    (out / "demo.json").write_text(json.dumps({
        "geo": {"north": terrain.lat_north, "south": terrain.lat_south,
                "west": terrain.lon_west, "east": terrain.lon_east,
                "rows": grid.shape[0], "cols": grid.shape[1],
                "cell_m": grid.cell_m},
        # Hours per operational period, so elapsed time can be shown in units a
        # search manager uses rather than in loop iterations.
        "period_hours": 12,
        "cases": cases,
    }, indent=2))
    print(f"\nwrote {out}/demo.json  ({len(cases)} cases)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
