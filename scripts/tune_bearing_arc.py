"""Does spreading a nomination over an arc recover performance under bearing error?

The model reliably identifies WHAT happened and less reliably WHICH DIRECTION.
This asks whether representing that uncertainty honestly -- as an arc rather than
a committed point -- recovers find rate.

Tested with a deliberately noisy nominator rather than with the model, so the
bearing error is a controlled variable. Run on the tuning suite (seed 99); if it
helps here it is adopted and the held-out suite is re-run, and if it does not it
is dropped.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import searchloop.loop as loop_mod                    # noqa: E402
from searchloop.config import DEFAULT as CFG          # noqa: E402
from searchloop.grid import build_grid                # noqa: E402
from searchloop.loop import run_scenario              # noqa: E402
from searchloop.pod import pod_field                  # noqa: E402
from searchloop.revision import Nomination, nominate_heuristic  # noqa: E402
from searchloop.scenario import generate_suite, stable_seed     # noqa: E402
from searchloop.terrain import load_terrain           # noqa: E402


def noisy_factory(scenario, grid, bearing_error_deg: float, sigma: float, seed: int):
    """Nominate the true account with a controlled bearing error."""
    def nominate(belief, g, ipp_rc, rng):
        local = np.random.default_rng(seed)
        dr = scenario.true_anchor_rc[0] - ipp_rc[0]
        dc = scenario.true_anchor_rc[1] - ipp_rc[1]
        true_bearing = math.degrees(math.atan2(dc, -dr)) % 360.0
        distance = float(np.hypot(dr, dc) * grid.cell_m / 1000.0)
        err = local.normal(0.0, bearing_error_deg)
        return [Nomination(
            label="[noisy] account", narrative="", profile_key=scenario.true_profile_key,
            anchor_bearing_deg=(true_bearing + err) % 360.0,
            anchor_distance_km=distance, prior=0.30,
            rationale="diagnostic", anchor_bearing_sigma_deg=sigma,
        )]
    return nominate


def score(grid, pod, suite, bearing_error, sigma, repeats=3):
    found = loc = n = 0
    try:
        for repeat in range(repeats):
            for s in suite:
                loop_mod.nominate_heuristic = noisy_factory(
                    s, grid, bearing_error, sigma, seed=int(stable_seed(s.id, repeat) % 2**31))
                r = run_scenario(grid, pod, s, "heuristic", CFG,
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

    print("tuning suite, 24 type B x 3 repeats. Nominator knows the account and")
    print("the distance; only the bearing is perturbed.\n")
    print(f"{'bearing err':>12} | " + " | ".join(f"sigma={s:<3.0f}" for s in (0, 15, 30, 45)))
    print("-" * 62)
    for err in (0.0, 15.0, 30.0, 45.0):
        cells = []
        for sigma in (0.0, 15.0, 30.0, 45.0):
            f, l = score(grid, pod, suite, err, sigma)
            cells.append(f"{100 * f:3.0f}%/{100 * l:3.0f}%")
        print(f"{err:>10.0f}deg | " + " | ".join(f"{c:<9}" for c in cells))
    print("\ncells are find%/localise%. Rows are the error actually made;")
    print("columns are the uncertainty the nomination declares.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
