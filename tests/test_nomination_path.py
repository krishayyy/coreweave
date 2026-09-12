"""The nomination path, exercised with a test double instead of a live model.

Verifies the property the whole design rests on: a nominated hypothesis gets no
special treatment. It is converted into a prior by the same function the
published library uses, and the same Bayesian update can kill it.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from searchloop import llm, revision                                    # noqa: E402
from searchloop.belief import Belief, SearchRecord                      # noqa: E402
from searchloop.config import DEFAULT as CFG                            # noqa: E402
from searchloop.grid import build_grid                                  # noqa: E402
from searchloop.hypotheses import library_hypotheses                    # noqa: E402
from searchloop.pod import pod_field                                    # noqa: E402
from searchloop.revision import Nomination, should_revise, to_hypothesis  # noqa: E402
from searchloop.terrain import load_terrain                             # noqa: E402

VALID = """Here is my assessment.
```json
[{"label": "Transported to the east approach",
  "narrative": "Accepted a lift and started from a different access point.",
  "profile_key": "hiker", "anchor_bearing_deg": 90, "anchor_distance_km": 7.5,
  "prior": 0.3, "rationale": "A witness saw them accept a lift.",
  "evidence_cited": ["accepting a lift from another vehicle"]}]
```"""

MALFORMED = '[{"label": "Nonsense", "profile_key": "wizard", ' \
            '"anchor_bearing_deg": 45, "anchor_distance_km": 3, "prior": 0.9}]'


@pytest.fixture(scope="module")
def world():
    grid = build_grid(load_terrain(CFG.center_lat, CFG.center_lon, CFG.zoom,
                                   CFG.radius_tiles), CFG.grid_factor)
    return grid, pod_field(grid)


def test_parses_fenced_json_into_nominations(monkeypatch, world):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: VALID)
    grid, _ = world
    belief = Belief.from_hypotheses(library_hypotheses(grid, (96, 96)))
    noms = revision.nominate_llm(belief, grid, "case file", revision.RevisionTrigger(True, "r"))
    assert len(noms) == 1
    assert noms[0].profile_key == "hiker"
    assert noms[0].anchor_distance_km == 7.5


def test_unknown_profile_is_rejected_not_coerced(monkeypatch, world):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: MALFORMED)
    grid, _ = world
    belief = Belief.from_hypotheses(library_hypotheses(grid, (96, 96)))
    noms = revision.nominate_llm(belief, grid, "case file", revision.RevisionTrigger(True, "r"))
    assert to_hypothesis(noms[0], grid, (96, 96), 0) is None, "unknown profile must be refused"


def test_nomination_prior_is_capped(monkeypatch, world):
    monkeypatch.setattr(llm, "complete", lambda *a, **k: MALFORMED)
    grid, _ = world
    belief = Belief.from_hypotheses(library_hypotheses(grid, (96, 96)))
    noms = revision.nominate_llm(belief, grid, "case file", revision.RevisionTrigger(True, "r"))
    assert noms[0].prior <= revision.MAX_NOMINATION_PRIOR


def test_bearing_maps_to_the_right_quadrant(world):
    grid, _ = world
    ipp = (96, 96)
    east = to_hypothesis(Nomination("e", "", "hiker", 90, 7.0, 0.2), grid, ipp, 0)
    north = to_hypothesis(Nomination("n", "", "hiker", 0, 7.0, 0.2), grid, ipp, 0)
    assert east.anchor_rc[1] > ipp[1] and abs(east.anchor_rc[0] - ipp[0]) < 3
    assert north.anchor_rc[0] < ipp[0] and abs(north.anchor_rc[1] - ipp[1]) < 3


def test_a_nominated_hypothesis_can_be_killed_by_evidence(world):
    """The property the design depends on: nomination confers no protection."""
    grid, pod = world
    ipp = (96, 96)
    belief = Belief.from_hypotheses(library_hypotheses(grid, ipp))
    hyp = to_hypothesis(Nomination("wrong", "", "hiker", 90, 8.0, 0.2), grid, ipp, 0)
    belief.add_hypothesis(hyp, 0.35)
    idx = belief.hypotheses.index(hyp)
    assert belief.weights[idx] > 0.2

    # Search exactly where this hypothesis says to look, and find nothing.
    # Enough cells to cover the bulk of its mass -- re-sweeping a small set has
    # sharply diminishing returns, since mass there is already driven toward zero.
    field = belief.fields[idx]
    top = np.argsort(field, axis=None)[::-1][:14000]
    cells = [tuple(map(int, np.unravel_index(i, field.shape))) for i in top]
    for period in range(6):
        belief.observe(SearchRecord(period, cells, [float(pod[c]) for c in cells],
                                    False, 0.0, 0.0))

    assert belief.exhaustion[idx] > 0.75, "repeated non-detection must rule it out"
    assert belief.weights[idx] < 0.2, "and must cost it leadership"


def test_trigger_never_fires_on_a_fresh_search(world):
    grid, _ = world
    belief = Belief.from_hypotheses(library_hypotheses(grid, (96, 96)))
    assert not should_revise(belief, CFG, 0, None).fired
