"""Growing the suite must not rewrite the scenarios already in it.

Every published figure is keyed to a scenario id. If asking for more type B
scenarios changes what B003 or C007 is, then adding cases silently invalidates
every run already measured, and a paired comparison across two files is
comparing different worlds while looking identical. This is the same failure as
seeding from a process-randomised hash, one level up.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest                                     # noqa: E402

from searchloop.config import DEFAULT as CFG      # noqa: E402
from searchloop.config import DEFAULT as CFG     # noqa: E402
from searchloop.grid import build_grid            # noqa: E402
from searchloop.scenario import generate_suite    # noqa: E402
from searchloop.terrain import load_terrain       # noqa: E402


@pytest.fixture(scope="module")
def grid():
    return build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor)


def _index(suite):
    return {s.id: (s.kind, s.subkind, s.ipp_rc, s.true_rc,
                   s.true_anchor_rc, s.true_profile_key, s.case_file)
            for s in suite}


def test_growing_one_family_leaves_the_others_untouched(grid):
    small = _index(generate_suite(grid, n_a=4, n_b=3, n_c=2, seed=0))
    grown = _index(generate_suite(grid, n_a=4, n_b=9, n_c=2, seed=0))

    for sid, value in small.items():
        assert grown[sid] == value, f"{sid} changed when the suite grew"
    assert len(grown) == len(small) + 6


def test_the_same_request_is_reproducible(grid):
    assert _index(generate_suite(grid, n_a=3, n_b=3, n_c=2, seed=0)) == \
        _index(generate_suite(grid, n_a=3, n_b=3, n_c=2, seed=0))


def test_a_different_seed_is_a_different_suite(grid):
    a = _index(generate_suite(grid, n_a=3, n_b=3, n_c=2, seed=0))
    b = _index(generate_suite(grid, n_a=3, n_b=3, n_c=2, seed=1))
    assert a != b
