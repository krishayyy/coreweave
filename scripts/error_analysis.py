"""Where does the remaining error actually come from?

Decomposes every nomination the model made against the withheld truth. The
headline number says how often the loop works; this says why it fails, which is
the part that tells you what to fix next.

    python scripts/error_analysis.py [runs/experiment_full.json]
"""
from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from searchloop.config import DEFAULT as CFG          # noqa: E402
from searchloop.grid import build_grid                # noqa: E402
from searchloop.scenario import generate_suite        # noqa: E402
from searchloop.terrain import load_terrain           # noqa: E402

CUES = {
    "transported": "a witness names the direction the vehicle left",
    "wrong_ipp": "a relative names the side of the range",
    "deliberate_deviation": "'the back side' -- names no compass direction, "
                            "must be read against terrain",
}


def main() -> int:
    path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs/experiment_full.json")
    grid = build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor, CFG.treeline_m)
    suite = {s.id: s for s in generate_suite(grid, 30, 24, 12, seed=7)}
    data = json.loads(path.read_text())

    bearing, distance, position = [], [], []
    per_response, by_mode = [], collections.defaultdict(list)
    found_by_mode = collections.defaultdict(list)
    best_per_run = []

    for run in data["runs"]:
        if run["arm"] != "llm" or run["scenario_kind"] != "B":
            continue
        s = suite[run["scenario_id"]]
        dr = s.true_anchor_rc[0] - s.ipp_rc[0]
        dc = s.true_anchor_rc[1] - s.ipp_rc[1]
        true_bearing = math.degrees(math.atan2(dc, -dr)) % 360.0
        true_km = float(np.hypot(dr, dc) * grid.cell_m / 1000.0)
        found_by_mode[s.subkind].append(run["found"])

        run_errors = []
        for trace in run["trace"]:
            response = []
            for nom in trace["nominations"]:
                b_err = abs((nom["bearing_deg"] - true_bearing + 180) % 360 - 180)
                d_err = nom["distance_km"] - true_km
                pos = math.sqrt(true_km**2 + nom["distance_km"]**2
                                - 2 * true_km * nom["distance_km"]
                                * math.cos(math.radians(b_err)))
                bearing.append(b_err)
                distance.append(d_err)
                position.append(pos)
                by_mode[s.subkind].append(b_err)
                response.append((nom["prior"], pos))
                run_errors.append(pos)
            if len(response) >= 2:
                per_response.append(response)
        if run_errors:
            best_per_run.append(min(run_errors))

    b = np.array(bearing); d = np.array(distance); p = np.array(position)
    print(f"{len(b)} nominations across {len(best_per_run)} type B runs\n")

    print("ERROR DECOMPOSITION")
    print(f"  bearing    median {np.median(b):5.0f} deg   within 20 deg: "
          f"{100 * (b <= 20).mean():3.0f}%")
    print(f"  distance   median {np.median(d):+5.1f} km    within 2 km:   "
          f"{100 * (abs(d) <= 2).mean():3.0f}%")
    print(f"  position   median {np.median(p):5.1f} km    within 3 km:   "
          f"{100 * (p <= 3).mean():3.0f}%")
    print("  Distance is essentially solved; the error is angular.\n")

    print("CAN THE MODEL RANK ITS OWN PROPOSALS?")
    top = np.array([min(r, key=lambda x: -x[0])[1] for r in per_response])
    best = np.array([min(x[1] for x in r) for r in per_response])
    conc = disc = 0
    for r in per_response:
        for i in range(len(r)):
            for j in range(i + 1, len(r)):
                a, c = r[i], r[j]
                if a[0] == c[0]:
                    continue
                hi, lo = (a, c) if a[0] > c[0] else (c, a)
                conc += hi[1] < lo[1]
                disc += hi[1] >= lo[1]
    print(f"  its highest-prior pick   median {np.median(top):4.1f} km")
    print(f"  the best one it proposed median {np.median(best):4.1f} km")
    print(f"  concordance of stated prior with accuracy: "
          f"{100 * conc / max(conc + disc, 1):.0f}%  (50% = no signal)")
    print("  The right account is usually present and cannot be picked out in advance.\n")

    print("BEARING ERROR BY WHAT THE CASE FILE SAYS")
    for mode in sorted(by_mode):
        a = np.array(by_mode[mode])
        print(f"  {mode:<21} median {np.median(a):3.0f} deg  within20 "
              f"{100 * (a <= 20).mean():3.0f}%  find {100 * np.mean(found_by_mode[mode]):3.0f}%")
        print(f"    {CUES.get(mode, '')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
