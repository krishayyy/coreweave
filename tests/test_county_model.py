"""The county model must learn what is there and, crucially, not what isn't.

The second half is the part that matters. A four-parameter correction fitted on
a handful of cases will happily discover that everyone is found in creeks
because the last three were, and a county that adopted that would search worse
than one that had never learned anything. So the null case -- a county that
genuinely matches the national model -- is tested as strictly as the real ones.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from searchloop.config import DEFAULT as CFG                        # noqa: E402
from searchloop.county import (CountyModel, CountyRegistry,         # noqa: E402
                               ResolvedCase, fit_county)
from searchloop.grid import build_grid                              # noqa: E402
from searchloop.hypotheses import PROFILES, build_prior_field       # noqa: E402
from searchloop.scenario import generate_suite                      # noqa: E402
from searchloop.terrain import load_terrain                         # noqa: E402


@pytest.fixture(scope="module")
def grid():
    return build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor)


def _cases(grid, weights, n, seed=5):
    local = CountyModel("truth", np.asarray(weights, float), 1) if any(weights) else None
    scen = generate_suite(grid, n_a=n, n_b=0, n_c=0, seed=seed, local=local)
    return [ResolvedCase(s.true_anchor_rc, s.true_rc,
                         build_prior_field(grid, PROFILES[s.true_profile_key],
                                           s.true_anchor_rc))
            for s in scen]


def test_a_cold_county_changes_nothing(grid):
    model = CountyModel.cold("Nowhere")
    prior = build_prior_field(grid, PROFILES["hiker"], (60, 60))
    assert np.array_equal(model.adjust(prior, grid, (60, 60)), prior)


def test_it_recovers_a_deviation_that_is_really_there(grid):
    model = fit_county("X", _cases(grid, [0.7, 0.0, 0.0, 0.0], 120), grid)
    assert model.weights[0] > 0.35, model.weights
    # and does not invent a canopy effect that was never planted
    assert abs(model.weights[2]) < model.weights[0]


def test_it_recovers_the_opposite_deviation_too(grid):
    model = fit_county("X", _cases(grid, [-0.7, 0.0, 0.0, 0.0], 120), grid)
    assert model.weights[0] < -0.35, model.weights


def test_a_county_that_is_average_stays_average(grid):
    """The null. Fitting noise here would be worse than not learning at all."""
    model = fit_county("X", _cases(grid, [0.0, 0.0, 0.0, 0.0], 120), grid)
    assert np.abs(model.weights).max() < 0.35, model.weights


def test_few_cases_stay_close_to_the_literature(grid):
    """Partial pooling: two cases must not rewrite the published model."""
    cases = _cases(grid, [0.7, 0.0, 0.0, 0.0], 120)
    small = fit_county("X", cases[:2], grid)
    large = fit_county("X", cases, grid)
    assert np.abs(small.weights).max() <= np.abs(large.weights).max() + 0.5


def test_the_fit_never_lands_on_the_clip_bound(grid):
    """It used to. A fixed step oscillated and stuck at -2.0 on a county whose
    gradient at zero was positive."""
    for n in (2, 4, 8, 32):
        w = fit_county("X", _cases(grid, [0.7, 0.0, 0.0, 0.0], 120)[:n], grid).weights
        assert np.abs(w).max() < 1.99, (n, w)


def test_the_registry_round_trips(tmp_path, grid):
    reg = CountyRegistry(tmp_path / "counties.json")
    reg.put(CountyModel("Clackamas OR", np.array([0.4, -0.1, 0.0, 0.2]), 12))
    reg.save()
    assert np.allclose(CountyRegistry(tmp_path / "counties.json")
                       .get("Clackamas OR").weights, [0.4, -0.1, 0.0, 0.2])
    # a county nobody has flown yet is cold, not missing
    assert CountyRegistry(tmp_path / "counties.json").get("Elsewhere").n_cases == 0
