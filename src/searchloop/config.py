"""Operating parameters, in one place so every experiment arm shares them."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Operating area. Mt Hood, Oregon Cascades: strong relief, real drainages,
    # genuine SAR terrain. Zoom 12 over 3x3 tiles is a ~20 km box at 107 m cells
    # -- wide enough that the published distance profiles (hiker d95 = 11 km)
    # actually discriminate rather than blanketing the whole map.
    center_lat: float = 45.3736
    center_lon: float = -121.6960
    zoom: int = 12
    radius_tiles: int = 1
    grid_factor: int = 4
    treeline_m: float = 1800.0

    # One iteration is an operational period, not a single sortie: real searches
    # plan in ~12 hour periods flying several aircraft alongside ground teams,
    # and the revision step belongs at that planning boundary. 400 km of total
    # track per period sweeps roughly 31 km2 of a 424 km2 operating area.
    period_track_km: float = 400.0
    altitude_m: float = 90.0

    # Revision fires when the leading hypothesis has been substantially ruled out
    # and nothing has replaced it. Never on a fixed iteration count.
    # The threshold was selected on a separate tuning suite (seed 99) against the
    # oracle nominator, and applied unchanged to the reported held-out suite --
    # see scripts/tune_trigger.py. Choosing it on the reported scenarios would be
    # fitting the test set.
    exhaustion_trigger: float = 0.50
    min_periods_between_revisions: int = 2
    # Each revision admits 2-3 hypotheses. Left uncapped the mixture dilutes:
    # the joint spreads across a dozen competing accounts and the search stops
    # committing to any of them. Swept on the tuning suite.
    max_revisions: int = 2
    max_periods: int = 16

    prune_floor: float = 1e-4
    seed: int = 0


DEFAULT = Config()
