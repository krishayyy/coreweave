"""How should proposals be admitted -- all at once, or one at a time?

The measurement that motivates this: across 101 revisions the model's own
highest-prior proposal was 4.7 km from the truth on median, while the *best*
proposal in the same response was 2.2 km. Its stated confidence ranks its own
proposals at 56% concordance, where 50% is no signal. The right account is
usually present and cannot be picked out in advance.

Admitting all three at once therefore splits the prior across one good account
and two bad ones, and the search commits to none of them. The alternative is to
admit one, let the evidence rule it out, and then admit the next -- which is
also how real search is tasked, one hypothesis per operational period.

Tested with a simulated nominator matched to the MEASURED error distribution
rather than with the model, so admission policy is the only variable and the
sweep costs no API calls. Tuning suite only.
"""
from __future__ import annotations

import math
import sys
from dataclasses import replace
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

# Measured on the held-out run: 297 nominations, bearing error median 52 deg
# with 40% inside 20 deg; distance error median +0.2 km, 98% inside 4 km.
BEARING_ERR_QUANTILES = [5.0, 14.0, 30.0, 52.0, 78.0, 110.0, 150.0]
DISTANCE_SD_KM = 1.6


def sample_bearing_error(rng: np.random.Generator) -> float:
    q = BEARING_ERR_QUANTILES
    u = rng.random() * (len(q) - 1)
    i = int(u)
    err = q[i] + (u - i) * (q[min(i + 1, len(q) - 1)] - q[i])
    return err * rng.choice([-1.0, 1.0])


def realistic_factory(scenario, grid, n_proposals: int, seed: int):
    """Three proposals whose error matches what the model actually produces."""
    def nominate(belief, g, ipp_rc, rng):
        local = np.random.default_rng(seed)
        dr = scenario.true_anchor_rc[0] - ipp_rc[0]
        dc = scenario.true_anchor_rc[1] - ipp_rc[1]
        tb = math.degrees(math.atan2(dc, -dr)) % 360.0
        td = float(np.hypot(dr, dc) * grid.cell_m / 1000.0)
        out = []
        for k in range(n_proposals):
            out.append(Nomination(
                label=f"[sim] account {k}", narrative="", profile_key="hiker",
                anchor_bearing_deg=(tb + sample_bearing_error(local)) % 360.0,
                anchor_distance_km=float(np.clip(td + local.normal(0, DISTANCE_SD_KM), 1.0, 14.0)),
                prior=0.20, rationale="simulated",
            ))
        # The model cannot rank its own proposals, so the order carries no
        # information: shuffle, and let admission policy be the only variable.
        local.shuffle(out)
        return out
    return nominate


def score(grid, pod, suite, cfg, n_proposals, repeats=4):
    found = loc = n = 0
    try:
        for repeat in range(repeats):
            for s in suite:
                loop_mod.nominate_heuristic = realistic_factory(
                    s, grid, n_proposals, seed=int(stable_seed(s.id, repeat) % 2**31))
                r = run_scenario(grid, pod, s, "heuristic", cfg,
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

    print("tuning suite, 24 type B x 4 repeats, simulated nominator matched to")
    print("the measured error distribution.\n")
    print(f"{'admitted':>9} {'max revisions':>14} | {'find':>6} {'localise':>9}")
    print("-" * 46)
    best = None
    for admitted, caps in ((3, (2, 3)), (2, (2, 3, 4)), (1, (2, 3, 4, 6))):
        for cap in caps:
            cfg = replace(CFG, max_revisions=cap)
            f, l = score(grid, pod, suite, cfg, admitted)
            flag = ""
            if best is None or f > best[0]:
                best, flag = (f, admitted, cap), "  <-"
            print(f"{admitted:>9} {cap:>14} | {100 * f:>5.0f}% {100 * l:>8.0f}%{flag}")
    print(f"\nbest: admit {best[1]} per revision, cap {best[2]} "
          f"({100 * best[0]:.0f}% find on the tuning suite)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
