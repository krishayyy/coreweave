"""Blind scenario generation.

The searcher never sees the true location or the true story -- only a case file
and an initial planning point, which is all a real incident commander gets at
hour zero.

Generation is procedural and seeded rather than model-authored. That is
deliberate: a model that writes the scenario and a model that solves it share a
prior, and any apparent skill could be collusion rather than inference. Procedural
generation makes leakage structurally impossible.

Two families:

  Type A -- the obvious reading is correct. The subject behaves like their
            apparent category, from the apparent starting point.

  Type B -- the obvious reading is wrong. These are the documented ways real
            searches fail: the subject was transported elsewhere, deviated
            deliberately, was mis-categorised, or the planning point itself is
            wrong.

Every Type B case file contains evidence sufficient to infer the true account.
Without that the nomination task would be guessing, and the experiment would
measure luck instead of reasoning.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .grid import SearchGrid
from .hypotheses import PROFILES, build_prior_field

TYPE_A_PROFILES = ["hiker", "hiker_route", "hunter", "despondent", "dementia", "child", "angler"]

# Displacement failures: the subject started somewhere other than the planning
# point. No profile in the library can reach them, because every library
# hypothesis is anchored at the planning point.
B_KINDS = ["transported", "deliberate_deviation", "wrong_ipp"]

# Category failure: the planning point is right but the behaviour profile was
# misjudged. The library already contains the correct profile at a low base
# rate, so ordinary Bayesian search can recover this on its own -- which is
# exactly why it is reported separately rather than blended into type B.
C_KINDS = ["miscategorised"]


@dataclass
class Evidence:
    """One item in the case file. `period` is when it becomes available."""

    period: int
    text: str
    kind: str = "witness"


@dataclass
class Scenario:
    id: str
    kind: str                      # "A", "B" (displacement) or "C" (category)
    subkind: str
    ipp_rc: tuple[int, int]
    true_rc: tuple[int, int]
    # Where the subject actually began. For Type A this is the planning point;
    # for Type B it is displaced, and recovering it is what the outer loop is
    # really doing -- a hypothesis names a starting point, not a destination.
    true_anchor_rc: tuple[int, int]
    true_profile_key: str
    true_account: str              # withheld from the searcher; for scoring only
    case_file: str                 # what the searcher sees at hour zero
    late_evidence: list[Evidence] = field(default_factory=list)

    def briefing(self, period: int) -> str:
        """Case file plus whatever evidence has arrived by this period."""
        parts = [self.case_file]
        arrived = [e for e in self.late_evidence if e.period <= period]
        if arrived:
            parts.append("\nSubsequent developments:")
            parts += [f"  [period {e.period}] {e.text}" for e in arrived]
        return "\n".join(parts)


def _sample_truth(grid: SearchGrid, profile_key: str, anchor: tuple[int, int],
                  rng: np.random.Generator, local=None) -> tuple[int, int]:
    """Sample where the subject actually ended up.

    `local` is the county's REAL behaviour, which is not what the searcher
    believes. Passing None means this county is exactly the national average,
    and is the null case the experiment needs. The searcher's library is built
    from the published profile either way -- it never sees this.
    """
    field_ = build_prior_field(grid, PROFILES[profile_key], anchor)
    if local is not None:
        field_ = local.adjust(field_, grid, anchor)
    return _sample_from_field(field_, rng)


def _sample_from_field(field_: np.ndarray, rng: np.random.Generator) -> tuple[int, int]:
    flat = field_.ravel().astype(np.float64)
    flat = np.clip(flat, 0, None)
    idx = int(rng.choice(flat.size, p=flat / flat.sum()))
    return divmod(idx, field_.shape[1])


def _displace(grid: SearchGrid, rc: tuple[int, int], km: float,
              rng: np.random.Generator,
              bearing_deg: float | None = None) -> tuple[int, int]:
    """Move an anchor a given distance, clipped on-grid.

    A bearing may be supplied so the geometry agrees with the story the case
    file tells. Placing the subject in a random direction while the evidence
    says "the back side" makes the directional cue meaningless, and the
    experiment would then be measuring luck on bearing rather than inference.
    """
    rows, cols = grid.shape
    cells = km * 1000.0 / grid.cell_m
    for attempt in range(24):
        if bearing_deg is None:
            theta = rng.uniform(0, 2 * math.pi)
            dr, dc = cells * math.sin(theta), cells * math.cos(theta)
        else:
            # Compass bearing: 0 = north = decreasing row.
            jitter = rng.normal(0.0, 12.0) if attempt < 12 else rng.uniform(-45, 45)
            theta = math.radians(bearing_deg + jitter)
            dr, dc = -cells * math.cos(theta), cells * math.sin(theta)
        r, c = int(round(rc[0] + dr)), int(round(rc[1] + dc))
        if 2 <= r < rows - 2 and 2 <= c < cols - 2:
            return r, c
    return int(np.clip(rc[0], 2, rows - 3)), int(np.clip(rc[1], 2, cols - 3))


def far_side_bearing(grid: SearchGrid, ipp_rc: tuple[int, int]) -> float:
    """Compass bearing from the planning point toward the far side of the massif.

    This is what a witness means by "the back side" -- the direction through the
    high ground and out the other side.
    """
    summit = np.unravel_index(int(np.argmax(grid.elevation)), grid.elevation.shape)
    return math.degrees(math.atan2(summit[1] - ipp_rc[1],
                                   -(summit[0] - ipp_rc[0]))) % 360.0


def _compass(deg: float) -> str:
    points = ["north", "north-east", "east", "south-east",
              "south", "south-west", "west", "north-west"]
    return points[int((deg + 22.5) % 360 // 45)]


_A_TEMPLATES = {
    "hiker": "{name}, {age}, set out alone from the {ipp_name} trailhead at {t0} on a day hike "
             "and failed to return by dark. Experienced but unfamiliar with this area. Day pack, "
             "no shelter. Weather turned to low cloud and rain by mid-afternoon. Vehicle still at "
             "the trailhead.",
    "hiker_route": "{name}, {age}, was hiking the loop from {ipp_name} with a partner who turned "
                   "back early. {name} intended to complete the loop and meet them at the car. "
                   "Partner reports {they} had a sore ankle at the split. Vehicle at the trailhead.",
    "hunter": "{name}, {age}, an experienced hunter, left camp near {ipp_name} before first light "
              "and has not returned. Comfortable off-trail and known to range widely. Rifle, "
              "day pack, no radio contact since departure.",
    "despondent": "{name}, {age}, left home without saying where {they} {were} going. Family "
                  "reports a difficult period and a recent loss. Vehicle located at {ipp_name}. "
                  "Family states {they} had spoken about this area meaning something to {them}.",
    "dementia": "{name}, {age}, has moderate dementia and walked away from a family gathering at "
                "{ipp_name}. Known to walk long distances without apparent objective. Dressed "
                "lightly for the conditions.",
    "child": "{name}, {age}, became separated from {their} family group near {ipp_name} while "
             "the group was packing up. Last seen heading toward the treeline. No pack, no "
             "additional clothing.",
    "angler": "{name}, {age}, went to fish the creek below {ipp_name} and did not return. "
              "Vehicle at the pullout, waders and gear missing from it.",
}

_B_TEMPLATES = {
    "transported": (
        "{name}, {age}, was dropped at the {ipp_name} trailhead intending a day hike and did not "
        "return. Vehicle is at {ipp_name}.",
        "A second party at the trailhead recalls someone matching {name}'s description "
        "accepting a lift mid-morning from a vehicle that left heading {direction}.",
        "The subject was driven to a different access point and began from there; the planning "
        "point is several kilometres from where they actually entered the terrain.",
    ),
    "deliberate_deviation": (
        "{name}, {age}, told family {they} {were} hiking the standard route from {ipp_name}. "
        "Vehicle at the trailhead. Overdue since {t0}.",
        "A message on {their} phone, sent at 06:40, reads: \"heading over the back side first, "
        "will loop round after\" -- recipient unidentified.",
        "The subject never intended the stated route and crossed to the far side of the divide "
        "before beginning the described hike.",
    ),
    "miscategorised": (
        "{name}, {age}, is overdue from a day hike departing {ipp_name}. Family describe {them} "
        "as a keen and regular hiker. Vehicle at the trailhead.",
        "On a follow-up call, a sibling discloses that {name} had stopped taking prescribed "
        "medication some weeks ago and had been talking about wanting to \"get above everything\".",
        "The subject is a despondent-category subject, not a lost hiker; movement is toward "
        "significant high ground rather than downhill.",
    ),
    "wrong_ipp": (
        "A vehicle registered to {name}, {age}, was found at {ipp_name} after {they} {were} "
        "reported overdue from a hike in the area.",
        "The registered owner's brother confirms the vehicle was lent out the previous week and "
        "that {name} was driven to an access point on the {direction} side of the range.",
        "The planning point is derived from a vehicle that does not indicate where the subject "
        "actually started.",
    ),
}

_NAMES = ["Dana Whitlock", "Marco Reyes", "Priya Raman", "Tom Aldridge", "Kate Nkemelu",
          "Sam Oyelaran", "Elena Vasquez", "Ben Hartley", "Noor Haddad", "Jesse Lamarr"]
_IPP_NAMES = ["Cloud Cap", "Tilly Jane", "Cooper Spur", "Vista Ridge", "Elk Cove",
              "Polallie Creek", "Mount Hood Meadows", "Timberline"]
_PRONOUNS = [("they", "were", "them", "their"), ("they", "were", "them", "their")]


def _fill(template: str, rng: np.random.Generator) -> tuple[str, dict]:
    they, were, them, their = _PRONOUNS[0]
    ctx = {
        "direction": "",
        "name": str(rng.choice(_NAMES)), "age": int(rng.integers(19, 74)),
        "ipp_name": str(rng.choice(_IPP_NAMES)),
        "t0": f"{int(rng.integers(6, 10)):02d}:{int(rng.choice([0, 15, 30, 45])):02d}",
        "they": they, "were": were, "them": them, "their": their,
    }
    return template.format(**ctx), ctx


def generate(grid: SearchGrid, scenario_id: str, kind: str, rng: np.random.Generator,
             local=None) -> Scenario:
    """One scenario. `kind` is "A" (premise correct), "B" (wrong about where the
    subject started) or "C" (right place, wrong behaviour category).

    `local` is this county's real deviation from the published behaviour model.
    It shapes where the subject is actually found and is never shown to the
    searcher. None means the county matches the literature exactly.
    """
    rows, cols = grid.shape
    ipp = (int(rng.integers(rows // 4, 3 * rows // 4)), int(rng.integers(cols // 4, 3 * cols // 4)))

    if kind == "A":
        key = str(rng.choice(TYPE_A_PROFILES))
        text, ctx = _fill(_A_TEMPLATES[key], rng)
        true_rc = _sample_truth(grid, key, ipp, rng, local)
        return Scenario(scenario_id, "A", key, ipp, true_rc, ipp, key,
                        f"Subject behaved as a {PROFILES[key].label}.", text)

    subkind = str(rng.choice(B_KINDS if kind == "B" else C_KINDS))
    opening, late_text, account = _B_TEMPLATES[subkind]
    text, ctx = _fill(opening, rng)

    if subkind == "miscategorised":
        true_key = "despondent"
        true_anchor = ipp
        bearing = None
    else:
        true_key = "hiker"
        # The geometry must agree with the story. "The back side" means through
        # the high ground; a named direction means that direction.
        bearing = (far_side_bearing(grid, ipp) if subkind == "deliberate_deviation"
                   else float(rng.uniform(0, 360)))
        true_anchor = _displace(grid, ipp, float(rng.uniform(6.5, 10.0)), rng, bearing)
        # Recover the bearing actually achieved after clipping, so the witness
        # statement describes where the subject really went.
        bearing = math.degrees(math.atan2(true_anchor[1] - ipp[1],
                                          -(true_anchor[0] - ipp[0]))) % 360.0

    # The witness statement is written only once the geometry is settled, so it
    # describes where the subject actually went. The same context is reused so
    # names stay consistent across the case file.
    ctx["direction"] = _compass(bearing) if bearing is not None else "unknown"
    late = late_text.format(**ctx)

    true_rc = _sample_truth(grid, true_key, true_anchor, rng, local)
    return Scenario(
        scenario_id, kind, subkind, ipp, true_rc, true_anchor, true_key, account, text,
        late_evidence=[Evidence(period=int(rng.integers(2, 5)), text=late)],
    )


def generate_suite(
    grid: SearchGrid, n_a: int = 30, n_b: int = 20, n_c: int = 12, seed: int = 0,
    local=None
) -> list[Scenario]:
    """A: premise correct. B: wrong about WHERE. C: wrong about WHO.

    Each scenario gets its own generator seeded from its own id, rather than
    drawing in turn from one shared stream. With a shared stream, asking for
    more type B scenarios shifts every draw after them, so growing the suite
    silently rewrites the type C scenarios and quietly invalidates any run
    already measured against them. Deriving per scenario makes the suite
    append-only: B024 is the same scenario whether you asked for 25 or 500.
    """
    def make(prefix: str, kind: str, count: int) -> list[Scenario]:
        out = []
        for i in range(count):
            sid = f"{prefix}{i:03d}"
            rng = np.random.default_rng(stable_seed(seed, sid))
            out.append(generate(grid, sid, kind, rng, local))
        return out

    return make("A", "A", n_a) + make("B", "B", n_b) + make("C", "C", n_c)


def stable_seed(*parts: object) -> int:
    """A seed that does not change between processes.

    Python randomises str.__hash__ per process unless PYTHONHASHSEED is fixed,
    so seeding from hash(scenario.id) silently reseeds every run and makes
    results irreproducible. Hash explicitly instead.
    """
    import hashlib

    digest = hashlib.sha256("|".join(str(p) for p in parts).encode()).digest()
    return int.from_bytes(digest[:8], "big")
