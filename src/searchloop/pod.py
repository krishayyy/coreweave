"""Probability of detection.

Standard SAR formulation: POD = 1 - exp(-coverage), where coverage is the ratio
of area effectively swept to area searched. Effective sweep width shrinks under
canopy and on steep ground, and grows with a lower, slower pass.

This is deterministic and is the same for every arm of the experiment -- the
comparison between arms is never confounded by the sensor model.
"""
from __future__ import annotations

import numpy as np

from .grid import SearchGrid

# Nominal sweep width for a thermal/RGB pass over open ground, in metres.
BASE_SWEEP_WIDTH_M = 180.0


def sweep_width(grid: SearchGrid, altitude_m: float = 90.0) -> np.ndarray:
    """Per-cell effective sweep width in metres."""
    # Canopy is the dominant term: a subject under closed conifer is often simply
    # not visible from above, at any altitude.
    canopy_factor = 1.0 - 0.80 * grid.canopy
    # Steep ground foreshortens the view and casts occluding shadow.
    slope_factor = np.clip(1.0 - (grid.slope - 20.0) / 70.0, 0.35, 1.0)
    # Higher passes cover more ground per unit track but resolve less detail.
    alt_factor = float(np.clip(1.0 + (altitude_m - 90.0) / 300.0, 0.6, 1.6))
    return BASE_SWEEP_WIDTH_M * canopy_factor * slope_factor * alt_factor


def pod_field(grid: SearchGrid, altitude_m: float = 90.0, passes: int = 1) -> np.ndarray:
    """POD in [0, 1) for a cell given `passes` sweeps across it."""
    # One pass across a cell lays down track length ~= cell width.
    coverage = sweep_width(grid, altitude_m) * grid.cell_m * passes / (grid.cell_m**2)
    return 1.0 - np.exp(-coverage)
