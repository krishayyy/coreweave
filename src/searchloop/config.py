"""Operating parameters, in one place so every experiment arm shares them."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Operating area. Mt Hood, Oregon Cascades: strong relief, real drainages,
    # genuine SAR terrain. Zoom 13 over 3x3 tiles is a ~10 km box at 54 m cells.
    center_lat: float = 45.3736
    center_lon: float = -121.6960
    zoom: int = 13
    radius_tiles: int = 1
    grid_factor: int = 4
    treeline_m: float = 1800.0

    # One iteration is an operational period, not a single sortie: real searches
    # plan in ~12 hour periods flying several aircraft, and the revision step
    # belongs at that planning boundary.
    period_track_km: float = 240.0
    altitude_m: float = 90.0

    # Revision fires when the leading hypothesis has been substantially ruled out
    # and nothing has replaced it. Never on a fixed iteration count.
    exhaustion_trigger: float = 0.55
    min_periods_between_revisions: int = 2
    max_periods: int = 16

    prune_floor: float = 1e-4
    seed: int = 0


DEFAULT = Config()
