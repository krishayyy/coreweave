"""Terrain loading from AWS Terrarium elevation tiles.

Terrarium tiles are public, key-free, global, and are the same tiles deck.gl's
TerrainLayer consumes -- so the simulation and the frontend render identical
ground truth.

Encoding: elevation_meters = (R * 256 + G + B / 256) - 32768
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from PIL import Image

TILE_URL = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png"
TILE_PX = 256
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "tiles"


def deg2tile(lat: float, lon: float, z: int) -> tuple[int, int]:
    """Slippy-map tile containing a lat/lon at zoom z."""
    lat_rad = math.radians(lat)
    n = 2.0**z
    x = int((lon + 180.0) / 360.0 * n)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def tile2deg(x: int, y: int, z: int) -> tuple[float, float]:
    """NW corner of a tile, as (lat, lon)."""
    n = 2.0**z
    lon = x / n * 360.0 - 180.0
    lat = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lat, lon


def _fetch_tile(z: int, x: int, y: int, session: requests.Session) -> np.ndarray:
    """One tile's elevation in meters, shape (256, 256). Cached on disk."""
    cache = CACHE_DIR / str(z) / str(x) / f"{y}.png"
    if cache.exists():
        img = Image.open(cache).convert("RGB")
    else:
        resp = session.get(TILE_URL.format(z=z, x=x, y=y), timeout=30)
        resp.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(resp.content)
        img = Image.open(io.BytesIO(resp.content)).convert("RGB")

    rgb = np.asarray(img, dtype=np.float64)
    return (rgb[:, :, 0] * 256.0 + rgb[:, :, 1] + rgb[:, :, 2] / 256.0) - 32768.0


@dataclass(frozen=True)
class Terrain:
    """An elevation raster with its geographic bounds.

    `elevation` is north-up: row 0 is the northern edge.
    """

    elevation: np.ndarray
    lat_north: float
    lat_south: float
    lon_west: float
    lon_east: float
    zoom: int

    @property
    def shape(self) -> tuple[int, int]:
        return self.elevation.shape

    @property
    def cell_size_m(self) -> tuple[float, float]:
        """Approximate (north-south, east-west) metres per pixel."""
        rows, cols = self.elevation.shape
        lat_mid = math.radians((self.lat_north + self.lat_south) / 2.0)
        ns = (self.lat_north - self.lat_south) * 111_320.0 / rows
        ew = (self.lon_east - self.lon_west) * 111_320.0 * math.cos(lat_mid) / cols
        return ns, ew

    def latlon_to_rc(self, lat: float, lon: float) -> tuple[int, int]:
        rows, cols = self.elevation.shape
        fr = (self.lat_north - lat) / (self.lat_north - self.lat_south) * rows
        fc = (lon - self.lon_west) / (self.lon_east - self.lon_west) * cols
        r = int(np.clip(fr, 0, rows - 1))
        c = int(np.clip(fc, 0, cols - 1))
        return r, c

    def rc_to_latlon(self, r: float, c: float) -> tuple[float, float]:
        rows, cols = self.elevation.shape
        lat = self.lat_north - (r + 0.5) / rows * (self.lat_north - self.lat_south)
        lon = self.lon_west + (c + 0.5) / cols * (self.lon_east - self.lon_west)
        return lat, lon


def load_terrain(lat: float, lon: float, zoom: int = 12, radius_tiles: int = 1) -> Terrain:
    """Mosaic the tiles around a centre point into one Terrain."""
    cx, cy = deg2tile(lat, lon, zoom)
    xs = range(cx - radius_tiles, cx + radius_tiles + 1)
    ys = range(cy - radius_tiles, cy + radius_tiles + 1)

    session = requests.Session()
    rows = [np.hstack([_fetch_tile(zoom, x, y, session) for x in xs]) for y in ys]
    mosaic = np.vstack(rows)

    lat_n, lon_w = tile2deg(min(xs), min(ys), zoom)
    lat_s, lon_e = tile2deg(max(xs) + 1, max(ys) + 1, zoom)
    return Terrain(mosaic, lat_n, lat_s, lon_w, lon_e, zoom)


def slope_degrees(t: Terrain) -> np.ndarray:
    """Per-cell slope in degrees, from the elevation gradient."""
    ns, ew = t.cell_size_m
    dz_dy, dz_dx = np.gradient(t.elevation, ns, ew)
    return np.degrees(np.arctan(np.hypot(dz_dx, dz_dy)))


def hillshade(t: Terrain, azimuth: float = 315.0, altitude: float = 45.0) -> np.ndarray:
    """Shaded relief in [0, 1]. Used for the matplotlib view during development."""
    ns, ew = t.cell_size_m
    dz_dy, dz_dx = np.gradient(t.elevation, ns, ew)
    slope = np.arctan(np.hypot(dz_dx, dz_dy))
    aspect = np.arctan2(-dz_dx, dz_dy)
    az = np.radians(360.0 - azimuth + 90.0)
    alt = np.radians(altitude)
    shaded = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
    return np.clip(shaded, 0.0, 1.0)


# Aerial imagery for the display. USGS National Map: public domain, no key, and
# the same slippy-tile scheme as the elevation tiles, so a mosaic over the same
# tile range lines up with the terrain exactly.
IMAGERY_URL = ("https://basemap.nationalmap.gov/arcgis/rest/services/"
               "USGSImageryOnly/MapServer/tile/{z}/{y}/{x}")
IMAGERY_CACHE = Path(__file__).resolve().parents[2] / "data" / "imagery"


def load_imagery(terrain: Terrain, detail: int = 2) -> Image.Image:
    """Aerial imagery covering exactly the same ground as `terrain`.

    `detail` is how many zoom levels finer than the elevation tiles to fetch;
    each level doubles the resolution and quadruples the number of requests.

    This is a basemap, not a sensor feed. It shows the real ground the search is
    happening over -- which is the point, since the terrain, the drainages and
    the behaviour models are all real too. Nothing here is a simulated camera.
    """
    zoom = terrain.zoom + detail
    step = 2**detail
    x0, y0 = deg2tile(terrain.lat_north - 1e-9, terrain.lon_west + 1e-9, zoom)
    x1, y1 = deg2tile(terrain.lat_south + 1e-9, terrain.lon_east - 1e-9, zoom)
    # Snap to the elevation mosaic's footprint so the two align exactly.
    x0 = (x0 // step) * step
    y0 = (y0 // step) * step
    cols = terrain.shape[1] // TILE_PX * step
    rows = terrain.shape[0] // TILE_PX * step

    session = requests.Session()
    session.headers["User-Agent"] = "searchloop/1.0"
    canvas = Image.new("RGB", (cols * TILE_PX, rows * TILE_PX))

    for dy in range(rows):
        for dx in range(cols):
            x, y = x0 + dx, y0 + dy
            cache = IMAGERY_CACHE / str(zoom) / str(x) / f"{y}.jpg"
            if cache.exists():
                tile = Image.open(cache).convert("RGB")
            else:
                try:
                    resp = session.get(IMAGERY_URL.format(z=zoom, x=x, y=y), timeout=30)
                    resp.raise_for_status()
                    cache.parent.mkdir(parents=True, exist_ok=True)
                    cache.write_bytes(resp.content)
                    tile = Image.open(io.BytesIO(resp.content)).convert("RGB")
                except Exception:
                    tile = Image.new("RGB", (TILE_PX, TILE_PX), (18, 20, 24))
            canvas.paste(tile, (dx * TILE_PX, dy * TILE_PX))
    return canvas
