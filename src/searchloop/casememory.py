"""Experience: what the agent remembers from searches that were resolved.

Rewriting your own instructions is one way to get better, and it is the weaker
one. An abstract rule has to be right about every future case at once, it is
hard to validate, and -- as this project found the hard way, twice -- a
confident diagnosis of your own failures is often simply wrong.

Precedent is the other way. When a search resolves, the truth becomes known:
this is what the evidence said, and this is where the subject had actually
started. That pairing is worth keeping. Facing a new case, the agent retrieves
the resolved cases whose evidence most resembles the one in front of it and is
shown what turned out to be true.

It is how an experienced incident commander works, and it has a property the
instruction-rewriting approach does not: it improves monotonically with
experience without anyone having to decide what the lesson was. Case 40 has
thirty-nine precedents. Case 1 has none.
"""
from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

CASE_MEMORY = Path(__file__).resolve().parents[2] / "data" / "cases.json"

# Words that carry no signal about what kind of case this is.
_STOP = {
    "the", "a", "an", "and", "or", "of", "to", "at", "in", "on", "for", "from",
    "was", "were", "is", "are", "be", "been", "had", "has", "have", "that",
    "this", "it", "its", "their", "they", "them", "he", "she", "his", "her",
    "with", "by", "not", "no", "as", "but", "who", "which", "when", "after",
    "before", "she", "said", "says", "reports", "recalls", "confirms",
}


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z]+", text.lower())
            if len(w) > 2 and w not in _STOP]


@dataclass
class CaseRecord:
    """One resolved search. Everything here was unknown while it was running."""

    id: str
    evidence: str              # the decisive item from the case file
    opening: str               # what was known at hour zero
    true_bearing_deg: float    # from the planning point to where they began
    true_distance_km: float
    true_profile: str
    found: bool
    premise_error_km: float    # how far the original premise was from the truth

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        """One precedent, as the agent reads it."""
        return (
            f"  Evidence was: \"{self.evidence.strip()}\"\n"
            f"    It turned out the subject had started {self.true_distance_km:.1f} km "
            f"from the planning point on bearing {self.true_bearing_deg:.0f} degrees "
            f"({_compass(self.true_bearing_deg)}), behaving as a {self.true_profile}."
        )


def _compass(deg: float) -> str:
    points = ["north", "north-east", "east", "south-east",
              "south", "south-west", "west", "north-west"]
    return points[int((deg + 22.5) % 360 // 45)]


@dataclass
class CaseMemory:
    """Resolved searches, retrievable by how much their evidence resembles a new one."""

    # Tuned by the self-improvement loop; instance calls may still override.
    DEFAULT_K: int = 3

    cases: list[CaseRecord] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path = CASE_MEMORY) -> "CaseMemory":
        if not path.exists():
            return cls()
        raw = json.loads(path.read_text())
        return cls([CaseRecord(**c) for c in raw.get("cases", [])])

    def save(self, path: Path = CASE_MEMORY) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {"cases": [c.to_dict() for c in self.cases]}, indent=2))

    def add(self, record: CaseRecord) -> None:
        self.cases.append(record)

    def __len__(self) -> int:
        return len(self.cases)

    # -- retrieval ---------------------------------------------------------

    def _idf(self) -> dict[str, float]:
        n = max(len(self.cases), 1)
        df: Counter[str] = Counter()
        for case in self.cases:
            df.update(set(_tokens(case.evidence + " " + case.opening)))
        return {w: math.log((n + 1) / (c + 1)) + 1.0 for w, c in df.items()}

    def retrieve(self, evidence: str, opening: str, k: int = 3) -> list[CaseRecord]:
        """The k resolved cases whose evidence most resembles this one.

        Cosine similarity over tf-idf weighted tokens. Deliberately not an
        embedding model: it adds a dependency and a failure mode, and the thing
        being matched is short, concrete, and lexically distinctive -- a witness
        naming a direction reads much like another witness naming a direction.
        """
        if not self.cases:
            return []
        idf = self._idf()
        query = Counter(_tokens(evidence + " " + opening))
        qv = {w: c * idf.get(w, 1.0) for w, c in query.items()}
        qn = math.sqrt(sum(v * v for v in qv.values())) or 1.0

        scored: list[tuple[float, CaseRecord]] = []
        for case in self.cases:
            cv_counts = Counter(_tokens(case.evidence + " " + case.opening))
            cv = {w: c * idf.get(w, 1.0) for w, c in cv_counts.items()}
            cn = math.sqrt(sum(v * v for v in cv.values())) or 1.0
            dot = sum(qv.get(w, 0.0) * v for w, v in cv.items())
            scored.append((dot / (qn * cn), case))

        scored.sort(key=lambda x: -x[0])
        return [case for score, case in scored[:k] if score > 0.05]

    def prompt_section(self, evidence: str, opening: str, k: int | None = None) -> str:
        """Retrieved precedent, rendered for the nomination prompt.

        Empty when nothing has been resolved yet, so the first search runs on
        exactly the prompt it always ran on -- which is what makes a learning
        curve measurable rather than asserted.
        """
        hits = self.retrieve(evidence, opening, k if k is not None else self.DEFAULT_K)
        if not hits:
            return ""
        lines = [
            f"PRECEDENT. You have resolved {len(self.cases)} searches before this "
            f"one. These are the most similar, and in each the truth is now known:",
            "",
        ]
        lines += [case.render() for case in hits]
        lines += [
            "",
            "These are not rules, they are outcomes. Weigh them against the case "
            "in front of you.",
        ]
        return "\n".join(lines)


def record_from(scenario, grid, found: bool) -> CaseRecord:
    """Write up a resolved search. Only called once the truth is known."""
    dr = scenario.true_anchor_rc[0] - scenario.ipp_rc[0]
    dc = scenario.true_anchor_rc[1] - scenario.ipp_rc[1]
    bearing = math.degrees(math.atan2(dc, -dr)) % 360.0
    distance = float(np.hypot(dr, dc) * grid.cell_m / 1000.0)
    evidence = (scenario.late_evidence[0].text if scenario.late_evidence
                else scenario.case_file)
    return CaseRecord(
        id=scenario.id,
        evidence=evidence,
        opening=scenario.case_file,
        true_bearing_deg=bearing,
        true_distance_km=distance,
        true_profile=scenario.true_profile_key,
        found=found,
        premise_error_km=distance,
    )
