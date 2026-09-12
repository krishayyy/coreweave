"""The belief state: a Bayesian mixture over competing hypotheses.

This is the deterministic heart of the system. Every hypothesis -- whether it
came from the published library or was nominated by a language model -- is
scored here by the same update on the same evidence. Nomination confers no
authority: a hypothesis that does not explain the evidence loses weight and
dies, regardless of its origin.

The fields are held UNNORMALISED. A hypothesis' remaining mass is then exactly
P(no contact so far | H_i), which is both the Bayesian likelihood that drives the
mixture weights and the directly interpretable quantity the revision trigger
needs: how much of what this story predicted have we now ruled out.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .hypotheses import Hypothesis


@dataclass
class SearchRecord:
    """One sortie's worth of evidence."""

    iteration: int
    cells: list[tuple[int, int]]
    pod: list[float]
    found: bool
    pos: float               # probability of success this sortie carried
    track_km: float


@dataclass
class Belief:
    """Mixture over hypotheses, with unnormalised per-hypothesis location fields."""

    hypotheses: list[Hypothesis]
    priors: np.ndarray                   # P(H_i) before any search evidence
    fields: list[np.ndarray]             # unnormalised: sum == P(no contact | H_i)
    history: list[SearchRecord] = field(default_factory=list)

    @classmethod
    def from_hypotheses(cls, hypotheses: list[Hypothesis]) -> "Belief":
        p = np.array([h.profile.base_rate for h in hypotheses], dtype=np.float64)
        return cls(
            hypotheses=list(hypotheses),
            priors=p / p.sum(),
            fields=[h.prior_field.copy() for h in hypotheses],
        )

    # -- derived quantities -------------------------------------------------

    @property
    def survival(self) -> np.ndarray:
        """P(no contact so far | H_i) for each hypothesis."""
        return np.array([float(f.sum()) for f in self.fields])

    @property
    def cumulative_pos(self) -> float:
        """P(we would have made contact by now), over the whole mixture.

        Derived, never accumulated: summing per-sortie POS double-counts, because
        each sortie's POS is measured against a joint that has already been
        renormalised. The honest quantity is simply the mass consumed.
        """
        return float(1.0 - np.sum(self.priors * self.survival))

    @property
    def exhaustion(self) -> np.ndarray:
        """Fraction of each hypothesis' predicted mass now ruled out."""
        return 1.0 - self.survival

    @property
    def weights(self) -> np.ndarray:
        """P(H_i | evidence), by Bayes: prior x likelihood of the non-detections."""
        w = self.priors * self.survival
        total = w.sum()
        return w / total if total > 0 else np.full_like(w, 1.0 / len(w))

    @property
    def joint(self) -> np.ndarray:
        """P(location | evidence), marginalised over hypotheses. What gets rendered."""
        out = np.zeros_like(self.fields[0])
        for p, f in zip(self.priors, self.fields):
            out += p * f
        total = out.sum()
        return out / total if total > 0 else out

    @property
    def leader(self) -> tuple[Hypothesis, float]:
        w = self.weights
        i = int(np.argmax(w))
        return self.hypotheses[i], float(w[i])

    def ranked(self) -> list[tuple[Hypothesis, float, float]]:
        """(hypothesis, posterior weight, exhaustion), best first."""
        w, e = self.weights, self.exhaustion
        return [(self.hypotheses[i], float(w[i]), float(e[i])) for i in np.argsort(w)[::-1]]

    def entropy(self) -> float:
        """Shannon entropy of the posterior over hypotheses, in nats."""
        w = self.weights
        w = w[w > 0]
        return float(-np.sum(w * np.log(w)))

    # -- evidence -----------------------------------------------------------

    def observe(self, record: SearchRecord) -> None:
        """Apply a sortie's non-detection across the searched cells.

        For every hypothesis, each searched cell's mass is scaled by (1 - POD):
        the chance the subject was there AND we still missed them. No
        renormalisation -- the mass that disappears is precisely the probability
        this sortie would have found them.
        """
        self.history.append(record)
        if record.found:
            return

        rows = np.array([c[0] for c in record.cells])
        cols = np.array([c[1] for c in record.cells])
        keep = 1.0 - np.asarray(record.pod)

        for f in self.fields:
            f[rows, cols] *= keep

    def add_hypothesis(self, hypothesis: Hypothesis, prior: float) -> None:
        """Admit a nominated hypothesis into the mixture.

        `prior` is P(H_new) -- a prior, not a verdict. The existing priors are
        scaled down to make room, and the next non-detection scores the newcomer
        on exactly the same footing as everything already present.
        """
        prior = float(np.clip(prior, 1e-4, 0.9))
        self.hypotheses.append(hypothesis)
        self.fields.append(hypothesis.prior_field.copy())
        self.priors = np.append(self.priors * (1.0 - prior), prior)
        self.priors /= self.priors.sum()

    def prune(self, floor: float = 1e-4) -> int:
        """Drop hypotheses the evidence has effectively killed."""
        keep = self.weights >= floor
        if keep.all() or keep.sum() == 0:
            return 0
        dropped = int((~keep).sum())
        self.hypotheses = [h for h, k in zip(self.hypotheses, keep) if k]
        self.fields = [f for f, k in zip(self.fields, keep) if k]
        self.priors = self.priors[keep]
        self.priors /= self.priors.sum()
        return dropped
