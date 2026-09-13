"""The hypothesis library: competing accounts of what happened.

Each hypothesis is a story that makes a spatial prediction. The distance-from-IPP
distributions are approximations of the published lost-person behaviour
literature (Koester's ISRID ring models), which report distance quantiles by
subject category and terrain. They are approximations and are labelled as such;
the point is that the priors come from a real empirical taxonomy rather than
from intuition.

Terrain affinities encode documented movement tendencies: hikers travel downhill
and follow drainages; despondent subjects seek significant high ground; anglers
stay on water.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .grid import SearchGrid


@dataclass(frozen=True)
class SubjectProfile:
    """An empirical behaviour category."""

    key: str
    label: str
    # Distance from the initial planning point, in km, at the 50th and 95th
    # percentile. Approximated from ISRID mountainous-terrain ring models.
    d50_km: float
    d95_km: float
    drainage_affinity: float   # >0 follows watercourses
    downhill_affinity: float   # >0 travels to lower ground
    highpoint_affinity: float  # >0 seeks elevation / views
    slope_aversion: float      # >0 avoids steep ground
    base_rate: float           # prior weight before any evidence
    narrative: str


PROFILES: dict[str, SubjectProfile] = {
    p.key: p
    for p in [
        SubjectProfile(
            "hiker", "Lost hiker, down-drainage", 1.9, 11.0,
            drainage_affinity=1.6, downhill_affinity=1.4, highpoint_affinity=0.0,
            slope_aversion=0.8, base_rate=0.40,
            narrative="Became disoriented off-trail and travelled downhill, "
                      "following a drainage in the expectation it would lead out.",
        ),
        SubjectProfile(
            "hiker_route", "Lost hiker, holding route", 1.1, 5.0,
            drainage_affinity=0.3, downhill_affinity=0.4, highpoint_affinity=0.3,
            slope_aversion=1.2, base_rate=0.18,
            narrative="Stayed near the intended route, moving slowly or sheltering "
                      "after injury or nightfall.",
        ),
        SubjectProfile(
            "hunter", "Hunter / off-trail traveller", 2.6, 14.0,
            drainage_affinity=1.1, downhill_affinity=0.7, highpoint_affinity=0.5,
            slope_aversion=0.4, base_rate=0.08,
            narrative="Deliberately off-trail and comfortable in terrain; "
                      "ranged far before being overtaken by weather or injury.",
        ),
        SubjectProfile(
            "despondent", "Despondent subject", 1.2, 6.0,
            drainage_affinity=0.1, downhill_affinity=-0.6, highpoint_affinity=1.8,
            slope_aversion=0.2, base_rate=0.10,
            narrative="Sought a significant or scenic location, typically high "
                      "ground with a view, away from other people.",
        ),
        SubjectProfile(
            "dementia", "Dementia / cognitive impairment", 0.8, 3.5,
            drainage_affinity=0.2, downhill_affinity=0.9, highpoint_affinity=0.0,
            slope_aversion=1.6, base_rate=0.06,
            narrative="Travelled until reaching an obstacle, following the path of "
                      "least resistance and rarely deviating from linear features.",
        ),
        SubjectProfile(
            "child", "Child, 7-12", 0.9, 4.0,
            drainage_affinity=0.9, downhill_affinity=1.1, highpoint_affinity=0.0,
            slope_aversion=1.3, base_rate=0.05,
            narrative="Moved downhill and toward water, then sheltered when tired "
                      "or frightened.",
        ),
        SubjectProfile(
            "angler", "Angler / water-bound", 1.0, 5.5,
            drainage_affinity=2.6, downhill_affinity=0.8, highpoint_affinity=0.0,
            slope_aversion=1.1, base_rate=0.04,
            narrative="Stayed on the watercourse, moving along it rather than "
                      "away from it.",
        ),
    ]
}


@dataclass
class Hypothesis:
    """A scored account of what happened, with its spatial prediction."""

    id: str
    label: str
    narrative: str
    profile: SubjectProfile
    anchor_rc: tuple[int, int]
    prior_field: np.ndarray
    origin: str = "library"          # "library" | "nominated"
    born_iteration: int = 0
    rationale: str = ""
    _tags: list[str] = field(default_factory=list)


def _radial_weight(dist_km: np.ndarray, d50: float, d95: float) -> np.ndarray:
    """Per-cell weight whose radial marginal matches the published quantiles.

    The published figures are quantiles of distance from the planning point, so
    the lognormal below is a density in distance. Laying it on a grid does not
    preserve that: the number of cells at radius r grows with r, so assigning
    each cell the density gives a radial marginal of pdf(r) * r, biased
    outward. Measured before the correction, generated medians ran 2.7x the
    published values -- a simulated hiker sat 5.2 km out where the literature
    says 1.9.

    Dividing by r once for the density and once for the area element makes the
    marginal come out as intended.
    """
    mu = np.log(max(d50, 1e-3))
    # z(0.95) = 1.645 for the standard normal.
    sigma = max((np.log(max(d95, d50 * 1.05)) - mu) / 1.645, 1e-3)
    d = np.clip(dist_km, 0.02, None)
    pdf = np.exp(-((np.log(d) - mu) ** 2) / (2 * sigma**2)) / d
    return pdf / d


def build_prior_field(
    grid: SearchGrid, profile: SubjectProfile, anchor_rc: tuple[int, int]
) -> np.ndarray:
    """P(location | hypothesis): radial distance model shaped by terrain affinity."""
    rows, cols = grid.shape
    rr, cc = np.mgrid[0:rows, 0:cols]
    dist_km = np.hypot(rr - anchor_rc[0], cc - anchor_rc[1]) * grid.cell_m / 1000.0

    weight = _radial_weight(dist_km, profile.d50_km, profile.d95_km)

    # Terrain shaping, in log space so affinities compose multiplicatively.
    elev = grid.elevation
    elev_n = (elev - elev.min()) / max(float(np.ptp(elev)), 1e-9)
    anchor_elev_n = float(elev_n[anchor_rc])
    downhill = np.clip(anchor_elev_n - elev_n, -1.0, 1.0)

    log_shape = (
        profile.drainage_affinity * grid.drainage
        + profile.downhill_affinity * downhill
        + profile.highpoint_affinity * elev_n
        - profile.slope_aversion * np.clip((grid.slope - 25.0) / 30.0, 0.0, 1.5)
    )
    field_ = weight * np.exp(log_shape)

    total = field_.sum()
    return field_ / total if total > 0 else np.full(field_.shape, 1.0 / field_.size)


def library_hypotheses(grid: SearchGrid, ipp_rc: tuple[int, int],
                       county=None) -> list[Hypothesis]:
    """The deterministic arm: every published profile, anchored at the IPP.

    `county` is the local correction this county has learned from its own
    resolved cases. It shapes where each published profile expects to find the
    subject, and it does not add, remove or reweight hypotheses -- the same
    seven profiles compete on the same terms. A county that has never resolved
    a case passes None and gets the published model exactly.
    """
    return [
        Hypothesis(
            id=p.key,
            label=p.label,
            narrative=p.narrative,
            profile=p,
            anchor_rc=ipp_rc,
            prior_field=(build_prior_field(grid, p, ipp_rc) if county is None
                         else county.adjust(build_prior_field(grid, p, ipp_rc),
                                            grid, ipp_rc)),
            origin="library",
        )
        for p in PROFILES.values()
    ]
