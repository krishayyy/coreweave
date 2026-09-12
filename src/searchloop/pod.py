"""Probability of detection, and the altitude trade-off that governs it.

Standard SAR formulation: POD = 1 - exp(-coverage), where coverage is the ratio
of area effectively swept to area searched.

The altitude trade-off is the part that makes altitude a decision rather than a
constant. Flying higher widens the field of view, so a single pass covers more
ground -- but a person subtends fewer pixels, so a smaller fraction of what
passes through the frame is actually seen. Effective sweep width is the product
of the two, and it peaks somewhere in the middle.

Where it peaks depends on the ground. Under closed canopy the subject is visible
only through gaps, and gaps are harder to exploit from height, so dense cover
pushes the optimum down. Open scree does not.

None of this involves a model. It is geometry and a sensor curve, and it is what
makes "what altitude should this sortie fly" a question the planner can answer
by arithmetic.
"""
from __future__ import annotations

import numpy as np

from .grid import SearchGrid

# Field of view at the reference altitude, in metres across track.
BASE_FOV_M = 180.0
REFERENCE_ALT_M = 90.0

# Altitudes the aircraft can actually be tasked to fly.
ALTITUDE_CHOICES = (50.0, 75.0, 100.0, 130.0, 170.0)


def sweep_width(grid: SearchGrid, altitude_m: float = REFERENCE_ALT_M) -> np.ndarray:
    """Per-cell effective sweep width in metres.

    Field of view grows linearly with altitude; the fraction of the frame in
    which a person is actually resolvable falls with it. The product has an
    interior maximum, and canopy moves that maximum down.
    """
    fov_m = BASE_FOV_M * (altitude_m / REFERENCE_ALT_M)

    # Pixels on target: a person is a fixed size, the frame is not.
    resolution = float(np.clip(1.0 - (altitude_m - 50.0) / 260.0, 0.18, 1.0))

    # Canopy costs more from height -- seeing through a gap needs a steeper look.
    canopy_penalty = 0.62 + 0.30 * np.clip((altitude_m - 50.0) / 160.0, 0.0, 1.0)
    canopy_factor = 1.0 - canopy_penalty * grid.canopy

    # Steep ground foreshortens the view and casts occluding shadow.
    slope_factor = np.clip(1.0 - (grid.slope - 20.0) / 70.0, 0.35, 1.0)

    return np.clip(fov_m * resolution * canopy_factor * slope_factor, 1.0, None)


def pod_field(grid: SearchGrid, altitude_m: float = REFERENCE_ALT_M,
              passes: int = 1) -> np.ndarray:
    """POD in [0, 1) for a cell given `passes` sweeps across it."""
    coverage = sweep_width(grid, altitude_m) * grid.cell_m * passes / (grid.cell_m**2)
    return 1.0 - np.exp(-coverage)
