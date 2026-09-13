"""What a county learns that its neighbours do not.

The published lost-person models are national. They say a hiker's terrain
affinities once, for every hiker everywhere. But the reason a search unit gets
good at its own ground is that its ground is not the national average: in one
county the drainages are walkable and people follow them out, in the next they
are choked with alder and people stay on the ridges. A drone flying the same
county for a year should end up knowing that. A drone flying the next county
over should not inherit it, because it is not true there.

So the learned object here is not a replacement for the published prior. It is
a correction to it, local to one county, shared by every drone in that county:

    p_county(cell) proportional to  p_published(cell) * exp(w . f(cell))

where f is the terrain description of a cell -- drainage, downhill, elevation,
slope excess -- and w is fitted from that county's own resolved cases. With
w = 0 the county believes exactly what the literature says, which is where
every county starts.

Fitting is a softmax regression over cells and is convex, so there is one
optimum and no initialisation to tune. The gradient is the difference between
the terrain where subjects were actually found and the terrain the current
belief expects them to be found in:

    grad = mean_i f(truth_i) - mean_i E_p[f]

which is worth stating plainly because it is the whole learning rule: move
toward the terrain that keeps being right, in proportion to how surprised the
current model was to find them there.

The regulariser is the part that makes this safe on real caseloads. A county
resolves a handful of searches a year, and four free parameters fitted on three
cases will happily conclude that everyone is found in creeks because the last
three were. The L2 penalty is scaled so it dominates when n is small and
relaxes as evidence accumulates -- partial pooling toward the national model,
with the county's own data earning its way in rather than being trusted on
arrival. A county with two cases is still, correctly, almost entirely the
literature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .grid import SearchGrid

# Names are fixed here because a stored county model is read back by index.
#
# Absolute elevation is deliberately NOT a feature. `downhill` is defined as
# the anchor's normalised elevation minus the cell's, so the two differ only by
# a constant and only their difference is identifiable: fitting both produced
# exactly mirrored weights that cancelled, which is a defect in the design and
# not a finding about terrain. `canopy` replaces it, and is the more useful
# feature anyway because the published shaping does not use it at all -- it is
# genuinely new information a county can discover about its own ground.
FEATURES = ("drainage", "downhill", "canopy", "slope_excess")

# Prior strength in units of observations: the penalty is equivalent to having
# seen PRIOR_WEIGHT cases that agreed exactly with the published model. Set by
# recovery testing -- with a known deviation planted and 200 cases, 8.0 still
# shrank a true +1.0 down to +0.68, which is over-shrinkage at a sample size
# where the county's own evidence should have won. At 2.0 a county still needs
# most of a season before it departs meaningfully, without being held back once
# it has the cases.
PRIOR_WEIGHT = 2.0

# A correction larger than this is almost certainly a small-sample artefact
# rather than a real local effect, so the fit is not allowed to go there.
MAX_ABS_WEIGHT = 2.0


def terrain_features(grid: SearchGrid, anchor_rc: tuple[int, int]) -> np.ndarray:
    """(features, rows, cols) description of every cell, relative to an anchor.

    `downhill` and `elevation` are measured against the anchor rather than in
    absolute metres so a model learned in one county describes a *relationship*
    -- 'subjects here end up below where they started' -- which is the only
    kind of statement that could transfer at all. Absolute elevation would not
    even be defined in the next county.
    """
    elev = grid.elevation
    elev_n = (elev - elev.min()) / max(float(np.ptp(elev)), 1e-9)
    anchor_elev_n = float(elev_n[anchor_rc])
    raw = np.stack([
        grid.drainage,
        np.clip(anchor_elev_n - elev_n, -1.0, 1.0),
        grid.canopy,
        np.clip((grid.slope - 25.0) / 30.0, 0.0, 1.5),
    ])
    # Standardise each feature over the county. Without this the coefficients
    # are on wildly different scales -- drainage sits near 0.13 with very
    # little spread, so its weight has to be large to say anything, and a
    # penalty that is reasonable for canopy barely touches it. A county with no
    # real deviation at all was fitting +0.5 on drainage from sampling noise
    # alone. In standardised units a weight means "per standard deviation of
    # this terrain feature in this county", which is also the only form in
    # which a weight could sensibly transfer to different ground.
    mean = raw.mean(axis=(1, 2), keepdims=True)
    sd = raw.std(axis=(1, 2), keepdims=True)
    return (raw - mean) / np.clip(sd, 1e-6, None)


@dataclass
class CountyModel:
    """The shared local knowledge of one county. Every drone in it reads this."""

    county: str
    weights: np.ndarray = field(
        default_factory=lambda: np.zeros(len(FEATURES)))
    n_cases: int = 0

    @classmethod
    def cold(cls, county: str) -> "CountyModel":
        """A county that has never resolved a case believes the literature."""
        return cls(county=county)

    def adjust(self, prior_field: np.ndarray, grid: SearchGrid,
               anchor_rc: tuple[int, int]) -> np.ndarray:
        """Apply the county's correction to a published prior field."""
        if self.n_cases == 0 or not np.any(self.weights):
            return prior_field
        feats = terrain_features(grid, anchor_rc)
        log_adj = np.tensordot(self.weights, feats, axes=(0, 0))
        # Subtract the max before exponentiating: the correction is only
        # meaningful up to a constant, and without this a large positive
        # weight overflows before normalisation can rescue it.
        adjusted = prior_field * np.exp(log_adj - log_adj.max())
        total = adjusted.sum()
        return adjusted / total if total > 0 else prior_field

    def describe(self) -> str:
        if self.n_cases == 0:
            return f"{self.county}: no resolved cases, using the published model"
        parts = [f"{name} {w:+.2f}" for name, w in zip(FEATURES, self.weights)
                 if abs(w) >= 0.05]
        body = ", ".join(parts) if parts else "no departure from the literature"
        return f"{self.county}: {self.n_cases} cases -> {body}"

    def to_dict(self) -> dict:
        return {"county": self.county, "weights": self.weights.tolist(),
                "n_cases": self.n_cases, "features": list(FEATURES)}

    @classmethod
    def from_dict(cls, d: dict) -> "CountyModel":
        if list(d.get("features", FEATURES)) != list(FEATURES):
            raise ValueError(
                f"stored model for {d.get('county')} uses a different feature "
                "set and cannot be read by this version")
        return cls(county=d["county"], weights=np.asarray(d["weights"], float),
                   n_cases=int(d["n_cases"]))


@dataclass
class ResolvedCase:
    """One closed search: where we looked from, and where the subject was."""

    anchor_rc: tuple[int, int]
    found_rc: tuple[int, int]
    prior_field: np.ndarray     # what the published model believed at the time


def fit_county(county: str, cases: list[ResolvedCase], grid: SearchGrid,
               steps: int = 800, lr: float = 0.5) -> CountyModel:
    """Fit the county correction by penalised maximum likelihood.

    Convex in `weights`, so there is one optimum and no initialisation that
    changes the answer -- but convexity does not make a fixed step size safe.
    At lr=2.0 the iteration overshot, the expected-feature term overcorrected,
    and the weights oscillated until they stuck against the clip bound, landing
    at -2.0 on a county whose gradient at zero was +0.18. The step is therefore
    halved whenever it fails to improve the objective, which makes convergence
    a property of the routine rather than of a constant that happened to work.
    """
    w = np.zeros(len(FEATURES))
    if not cases:
        return CountyModel(county=county)

    prepared = []
    for case in cases:
        feats = terrain_features(grid, case.anchor_rc)
        flat = feats.reshape(len(FEATURES), -1)
        base = np.log(np.clip(case.prior_field.ravel(), 1e-12, None))
        idx = case.found_rc[0] * grid.shape[1] + case.found_rc[1]
        prepared.append((flat, base, flat[:, idx], float(base[idx])))

    n = len(cases)
    penalty = PRIOR_WEIGHT / n     # relaxes as the county accumulates evidence

    def objective(weights: np.ndarray) -> float:
        total = 0.0
        for flat, base, truth, base_truth in prepared:
            logits = base + weights @ flat
            m = float(logits.max())
            log_z = m + float(np.log(np.exp(logits - m).sum()))
            total += base_truth + float(weights @ truth) - log_z
        return total / n - 0.5 * penalty * float(weights @ weights)

    def gradient(weights: np.ndarray) -> np.ndarray:
        grad = np.zeros(len(FEATURES))
        for flat, base, truth, _ in prepared:
            logits = base + weights @ flat
            logits -= logits.max()
            p = np.exp(logits)
            p /= p.sum()
            grad += truth - flat @ p
        return grad / n - penalty * weights

    current = objective(w)
    step = lr
    for _ in range(steps):
        candidate = np.clip(w + step * gradient(w), -MAX_ABS_WEIGHT, MAX_ABS_WEIGHT)
        value = objective(candidate)
        if value < current:
            step *= 0.5
            if step < 1e-6:
                break
            continue
        w, current = candidate, value

    return CountyModel(county=county, weights=w, n_cases=n)


class CountyRegistry:
    """Every county's model, on disk. This is the shared layer between drones.

    A drone does not own what it learns. It contributes a resolved case to its
    county and reads back the county's model, so the tenth drone to fly a
    county starts where the ninth left off, and a drone moved to a neighbouring
    county correctly starts over.
    """

    def __init__(self, path: Path):
        self.path = path
        self.models: dict[str, CountyModel] = {}
        if path.exists():
            raw = json.loads(path.read_text())
            self.models = {k: CountyModel.from_dict(v) for k, v in raw.items()}

    def get(self, county: str) -> CountyModel:
        return self.models.get(county) or CountyModel.cold(county)

    def put(self, model: CountyModel) -> None:
        self.models[model.county] = model

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(
            {k: v.to_dict() for k, v in self.models.items()}, indent=2))
