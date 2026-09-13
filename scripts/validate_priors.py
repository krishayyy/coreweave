"""Do the simulated subjects move the way the literature says real ones do?

Real incident data is not obtainable: ISRID is contribution-based rather than
public, and it explicitly excludes media reports, which is the only source that
could be scraped. So the strongest available check is internal but not
circular -- the distance models here are transcribed from published
lost-person-behaviour quantiles, and the scenarios are generated through them,
so the generated population should reproduce those quantiles.

If it does not, the simulation is not implementing the literature it cites,
whatever else it may be doing. That is worth knowing either way.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from searchloop.config import DEFAULT as CFG                   # noqa: E402
from searchloop.grid import build_grid                         # noqa: E402
from searchloop.hypotheses import PROFILES, build_prior_field  # noqa: E402
from searchloop.terrain import load_terrain                    # noqa: E402

# Published distance-from-planning-point quantiles for mountainous / temperate
# terrain, transcribed from the lost-person-behaviour literature (Koester,
# ISRID ring models). These are the numbers the profiles claim to implement.
PUBLISHED = {
    "hiker": (1.9, 11.0),
    "hiker_route": (1.1, 5.0),
    "hunter": (2.6, 14.0),
    "despondent": (1.2, 6.0),
    "dementia": (0.8, 3.5),
    "child": (0.9, 4.0),
    "angler": (1.0, 5.5),
}


def main() -> int:
    grid = build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor, CFG.treeline_m)
    rng = np.random.default_rng(0)
    ipp = (grid.shape[0] // 2, grid.shape[1] // 2)
    rows, cols = grid.shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    dist_km = np.hypot(rr - ipp[0], cc - ipp[1]) * grid.cell_m / 1000.0
    extent = grid.shape[0] * grid.cell_m / 1000.0

    print("Distance from the planning point: published against generated.")
    print("20000 subjects per profile, drawn from the prior the simulation uses.\n")
    print(f"{'profile':<14}{'pub d50':>9}{'gen':>7}{'pub d95':>10}{'gen':>7}   note")
    print("-" * 62)

    worst = 0.0
    for key, (d50, d95) in PUBLISHED.items():
        field = build_prior_field(grid, PROFILES[key], ipp)
        flat = field.ravel() / field.sum()
        draws = rng.choice(flat.size, size=20000, p=flat)
        d = dist_km.ravel()[draws]
        g50, g95 = float(np.percentile(d, 50)), float(np.percentile(d, 95))
        note = "d95 exceeds the map" if d95 > extent / 2 else ""
        print(f"{key:<14}{d50:>9.1f}{g50:>7.1f}{d95:>10.1f}{g95:>7.1f}   {note}")
        worst = max(worst, abs(g50 - d50) / d50)

    print()
    print(f"The operating area is {extent:.0f} km across and the planning point sits")
    print("in it, so a profile whose published d95 exceeds the half-width is")
    print("truncated by the map edge: the generated d95 is censored, not wrong,")
    print("and the censoring is identical for every arm.")
    print(f"\nLargest median deviation: {100 * worst:.0f}%.")
    print("\nThis is not a test against real incidents. It checks that the")
    print("simulation implements the literature it cites -- a precondition for the")
    print("result meaning anything, not a substitute for field data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
