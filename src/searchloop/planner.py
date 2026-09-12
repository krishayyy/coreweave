"""Sortie planning: which segment to sweep next.

Aerial SAR does not fly a line, it is assigned a *segment* and mows it in a
serpentine pattern. So a sortie here selects a contiguous region of high
probability-of-success and sweeps it, with the region's size set by how much
track the aircraft can fly given its endurance and sweep width.

Deliberately simple, and identical across every arm of the experiment: the
contribution of this project is which hypotheses exist, not how cleverly the
drone routes inside one. Holding the planner fixed means differences between
arms cannot be attributed to it.
"""
from __future__ import annotations

import heapq
import math

import numpy as np

from .grid import SearchGrid
from .pod import sweep_width

_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def segment_capacity(grid: SearchGrid, endurance_km: float, altitude_m: float = 100.0) -> int:
    """How many cells a sortie can actually mow.

    Serpentine coverage lays down `endurance_km` of track at a lane spacing of
    one sweep width, so the area covered is track x sweep width.
    """
    sw_m = float(np.median(sweep_width(grid, altitude_m)))
    area_m2 = endurance_km * 1000.0 * sw_m
    return max(int(area_m2 / grid.cell_m**2), 1)


def plan_sortie(
    grid: SearchGrid,
    joint: np.ndarray,
    pod: np.ndarray,
    start_rc: tuple[int, int],
    budget_cells: int,
    transit_penalty: float = 0.35,
) -> list[tuple[int, int]]:
    """Grow the highest-value contiguous segment the aircraft can sweep.

    Seeds on the best cell after a transit discount, then grows outward,
    always absorbing the highest-value cell on the frontier. The result is a
    connected blob -- a real assignable search segment.
    """
    value = joint * pod
    rows, cols = grid.shape

    rr, cc = np.mgrid[0:rows, 0:cols]
    transit_km = np.hypot(rr - start_rc[0], cc - start_rc[1]) * grid.cell_m / 1000.0
    discounted = value * np.exp(-transit_penalty * transit_km)
    seed = np.unravel_index(int(np.argmax(discounted)), value.shape)

    segment: list[tuple[int, int]] = []
    seen = np.zeros((rows, cols), dtype=bool)
    frontier: list[tuple[float, int, int]] = [(-float(value[seed]), int(seed[0]), int(seed[1]))]
    seen[seed] = True

    while frontier and len(segment) < budget_cells:
        _, r, c = heapq.heappop(frontier)
        segment.append((r, c))
        for dr, dc in _NEIGHBOURS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc]:
                seen[nr, nc] = True
                heapq.heappush(frontier, (-float(value[nr, nc]), nr, nc))

    return segment


def sortie_pos(joint: np.ndarray, pod: np.ndarray, cells: list[tuple[int, int]]) -> float:
    """Probability of success: chance this sortie makes contact."""
    rows = np.array([c[0] for c in cells])
    cols = np.array([c[1] for c in cells])
    return float(np.sum(joint[rows, cols] * pod[rows, cols]))


def plan_sortie_adaptive(
    grid: SearchGrid,
    joint: np.ndarray,
    start_rc: tuple[int, int],
    endurance_km: float,
    altitude_m: float | None = None,
    transit_penalty: float = 0.35,
) -> tuple[list[tuple[int, int]], np.ndarray, float, float]:
    """Allocate effort where it pays, instead of sweeping everything equally.

    Mowing a segment at uniform spacing spends the same effort on a cell holding
    a tenth of the belief as on one holding a thousandth. Koopman showed the
    optimal allocation for exponential detection is not uniform: with a fixed
    amount of search effort the marginal return must be equal everywhere, which
    gives

        effort(cell) = max(0, ln(P(cell) / lambda))

    with lambda set so the total meets the budget. Ground below the threshold
    gets nothing -- it is not worth the lane -- and the effort saved is spent
    thickening coverage where belief actually is.

    This is where the loop searches better rather than believing better, and it
    is arithmetic: no model is consulted, and the same allocation runs for every
    experimental arm.

    Returns the segment, the per-cell detection field the allocation produced,
    the mean effort actually applied, and the expected probability of success.
    """
    from .pod import sweep_width
    REFERENCE_ALT_M = 90.0

    altitude = altitude_m if altitude_m is not None else REFERENCE_ALT_M
    width_m = sweep_width(grid, altitude)

    # Budget expressed as total effort: one unit is coverage 1.0 over one cell.
    swept_area_m2 = endurance_km * 1000.0 * float(np.median(width_m))
    budget = swept_area_m2 / grid.cell_m**2

    # Transit discount, so the aircraft does not chase isolated specks.
    rows, cols = grid.shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    transit_km = np.hypot(rr - start_rc[0], cc - start_rc[1]) * grid.cell_m / 1000.0
    value = joint * np.exp(-transit_penalty * transit_km)
    # Terrain that resists detection needs more effort for the same return.
    value = value * np.clip(width_m / float(np.median(width_m)), 0.25, 1.4)

    # Solve for lambda by bisection: effort is monotone decreasing in lambda.
    positive = value[value > 0]
    if positive.size == 0:
        return [], np.zeros(grid.shape), 0.0, 0.0
    lo, hi = float(positive.min()) * 1e-6, float(positive.max())
    for _ in range(60):
        mid = math.sqrt(lo * hi)
        effort = np.maximum(0.0, np.log(np.maximum(value, 1e-300) / mid))
        if effort.sum() > budget:
            lo = mid
        else:
            hi = mid
    effort = np.maximum(0.0, np.log(np.maximum(value, 1e-300) / math.sqrt(lo * hi)))

    pod = np.clip(1.0 - np.exp(-effort), 0.0, 0.98)
    segment = [(int(r), int(c)) for r, c in zip(*np.nonzero(effort > 0.01))]
    if not segment:
        return [], pod, 0.0, 0.0
    pos = sortie_pos(joint, pod, segment)
    return segment, pod, float(effort[effort > 0].mean()), pos
