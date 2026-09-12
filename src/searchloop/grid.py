"""The search grid: a coarsened terrain raster plus the features behaviour depends on.

Lost-person movement is not radial. People walk downhill and they follow
drainages. So the grid carries a D8 flow-accumulation network, which is both the
behavioural substrate for the hypothesis priors and the thing that makes the map
read as real terrain.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .terrain import Terrain, slope_degrees

# D8 neighbour offsets and their travel distance multiplier.
_D8 = [(-1, 0, 1.0), (1, 0, 1.0), (0, -1, 1.0), (0, 1, 1.0),
       (-1, -1, 1.4142), (-1, 1, 1.4142), (1, -1, 1.4142), (1, 1, 1.4142)]


@dataclass(frozen=True)
class SearchGrid:
    """Coarse grid over the operating area.

    All fields are (rows, cols) arrays aligned with `elevation`.
    """

    terrain: Terrain
    factor: int
    elevation: np.ndarray
    slope: np.ndarray
    flow_accum: np.ndarray      # cells draining through each cell
    drainage: np.ndarray        # [0,1] normalised drainage strength
    canopy: np.ndarray          # [0,1] vegetation density proxy
    cell_m: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    def rc_to_latlon(self, r: float, c: float) -> tuple[float, float]:
        return self.terrain.rc_to_latlon((r + 0.5) * self.factor, (c + 0.5) * self.factor)

    def latlon_to_rc(self, lat: float, lon: float) -> tuple[int, int]:
        r, c = self.terrain.latlon_to_rc(lat, lon)
        rows, cols = self.shape
        return int(np.clip(r // self.factor, 0, rows - 1)), int(np.clip(c // self.factor, 0, cols - 1))

    def distance_m(self, a: tuple[int, int], b: tuple[int, int]) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1]) * self.cell_m)


def _block_mean(a: np.ndarray, f: int) -> np.ndarray:
    rows = (a.shape[0] // f) * f
    cols = (a.shape[1] // f) * f
    return a[:rows, :cols].reshape(rows // f, f, cols // f, f).mean(axis=(1, 3))


def flow_accumulation(elevation: np.ndarray) -> np.ndarray:
    """D8 flow accumulation: how many cells drain through each cell.

    Processing cells from high to low means every contributor is resolved before
    the cell it drains into, so one pass suffices.
    """
    rows, cols = elevation.shape
    accum = np.ones((rows, cols), dtype=np.float64)
    order = np.argsort(elevation, axis=None)[::-1]

    for flat in order:
        r, c = divmod(int(flat), cols)
        z = elevation[r, c]
        best_slope = 0.0
        best = None
        for dr, dc, dist in _D8:
            nr, nc = r + dr, c + dc
            if not (0 <= nr < rows and 0 <= nc < cols):
                continue
            drop = (z - elevation[nr, nc]) / dist
            if drop > best_slope:
                best_slope = drop
                best = (nr, nc)
        if best is not None:
            accum[best] += accum[r, c]
    return accum


def build_grid(terrain: Terrain, factor: int = 4, treeline_m: float = 1800.0) -> SearchGrid:
    """Coarsen terrain and derive the behavioural feature layers."""
    elev = _block_mean(terrain.elevation, factor)
    slope = _block_mean(slope_degrees(terrain), factor)

    accum = flow_accumulation(elev)
    # Log-compress: accumulation is heavy-tailed, and what matters behaviourally
    # is "is this a channel" not "how many cells exactly".
    drain = np.log1p(accum)
    drain = (drain - drain.min()) / max(float(np.ptp(drain)), 1e-9)

    # Canopy proxy. Real vegetation rasters exist but need authentication, so
    # this is derived from terrain -- honest about being a proxy, but shaped by
    # the things that actually break up forest cover in this range:
    #   * elevation relative to treeline
    #   * steep ground, which goes to talus and cliff rather than timber
    #   * locally rugged ground, which is rockier and more broken
    # A uniform "everything below treeline is closed canopy" is the unrealistic
    # option: it erases meadows, talus, burn scars and rock.
    canopy = np.clip(1.0 - (elev - treeline_m + 250.0) / 750.0, 0.0, 1.0)
    canopy *= np.clip(1.0 - (slope - 28.0) / 22.0, 0.15, 1.0)

    # Local relief over a 5-cell window: rugged ground carries less timber.
    pad = np.pad(elev, 2, mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(pad, (5, 5))
    roughness = windows.std(axis=(-2, -1))
    rough_n = roughness / max(float(np.percentile(roughness, 95)), 1e-9)
    canopy *= np.clip(1.0 - 0.55 * rough_n, 0.30, 1.0)

    ns, ew = terrain.cell_size_m
    return SearchGrid(
        terrain=terrain,
        factor=factor,
        elevation=elev,
        slope=slope,
        flow_accum=accum,
        drainage=drain,
        canopy=np.clip(canopy, 0.0, 1.0),
        cell_m=float((ns + ew) / 2.0 * factor),
    )
