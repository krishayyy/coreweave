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

import numpy as np

from .grid import SearchGrid
from .pod import sweep_width

_NEIGHBOURS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def segment_capacity(grid: SearchGrid, endurance_km: float, altitude_m: float = 90.0) -> int:
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
