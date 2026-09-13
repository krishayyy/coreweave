"""Nomination via TypeSafe's System One model.

The language model is asked to write accounts and name a bearing. It is good at
the first and poor at the second: measured over 297 nominations, distance error
was +0.2 km median while bearing error was 52 degrees, and its own stated
confidence ranked its proposals at 56% concordance -- no better than chance.

A System One model is a different instrument. It does not write; it answers
questions whose shape you define, and returns a probability for every option.
That removes the ranking problem entirely, because nothing needs ranking: asked
which direction the subject started in, it returns a distribution over compass
points, and a distribution is what the belief mixture consumes anyway.

So the division of labour follows the measurements. Direction comes from the
System One model as a calibrated distribution. Distance comes from the same
call but is treated as weak evidence, because its confidence there is low and
the language model was already accurate at it.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from . import tracing

ENDPOINT = "https://api.typesafe.ai/v1/systemone"

# Cached by request digest, for the same reason the language model calls are:
# every other arm reproduces exactly, and an arm that does not is the one whose
# numbers cannot be checked. Repeats of a scenario also recur identically.
CACHE_DIR = Path(__file__).resolve().parents[2] / "data" / "jev_cache"

# Sixteen points, not eight. Eight-point bins are 45 degrees wide, so asking
# for one of eight puts a floor of about 22 degrees on the answer -- and the
# measured error was 19 degrees, which is the model answering as precisely as
# the question allowed rather than as precisely as it could. Halving the bin
# halves that floor.
COMPASS = {
    "north": 0.0, "north-north-east": 22.5, "north-east": 45.0,
    "east-north-east": 67.5, "east": 90.0, "east-south-east": 112.5,
    "south-east": 135.0, "south-south-east": 157.5, "south": 180.0,
    "south-south-west": 202.5, "south-west": 225.0, "west-south-west": 247.5,
    "west": 270.0, "west-north-west": 292.5, "north-west": 315.0,
    "north-north-west": 337.5,
}

# Likewise finer: the expected distance is a probability-weighted average over
# these midpoints, so coarse bands quantise the answer before it is used.
DISTANCE_BANDS = [
    ("under 1.5 km from the planning point", 0.8),
    ("1.5 to 3 km from the planning point", 2.2),
    ("3 to 4.5 km from the planning point", 3.8),
    ("4.5 to 6 km from the planning point", 5.2),
    ("6 to 7.5 km from the planning point", 6.8),
    ("7.5 to 9 km from the planning point", 8.2),
    ("9 to 11 km from the planning point", 10.0),
    ("more than 11 km from the planning point", 12.5),
]


class JevUnavailable(RuntimeError):
    """No TypeSafe credentials configured."""


def available() -> bool:
    return bool(os.getenv("TYPESAFE_API_KEY"))


@dataclass
class Reading:
    """One System One evaluation of a case."""

    # P(the subject began somewhere other than the planning point). This gates
    # and scales every relocation: a calibrated model that is unconvinced by a
    # tip produces a small number here, and the nomination it is obliged to
    # give for direction and distance is admitted with correspondingly little
    # weight, or not at all.
    premise_wrong: float
    premise_confidence: float
    direction_probs: dict[str, float]
    direction_confidence: float
    distance_km: float
    distance_probs: dict[str, float]
    distance_confidence: float
    profile_probs: dict[str, float]
    profile_confidence: float
    usage: dict[str, int]

    def expected_distance_km(self, searched_km: float = 0.0) -> float:
        """Probability-weighted distance, conditioned on what was ruled out.

        The premise failed because the ground near the planning point was swept
        without contact, so the subject is not there -- and a plain weighted
        average over every band, including the ones already eliminated, drags
        the estimate inward toward ground we have proven empty.

        Conditioning on the search removes those bands and renormalises, which
        is just Bayes on information already in hand. Measured on a case where
        direction was correct and the answer was still missed: the estimate was
        7.6 km against a true 10.6, short by a quarter, with the near bands
        carrying weight that the search had already disproved.
        """
        weights = []
        for i, (_, midpoint) in enumerate(DISTANCE_BANDS):
            probability = self.distance_probs.get(str(i), 0.0)
            if midpoint < searched_km:
                probability = 0.0       # swept, and empty
            weights.append((midpoint, probability))

        total = sum(p for _, p in weights)
        if total <= 1e-6:
            # Everything the model believed has been ruled out; take the first
            # band beyond the searched radius rather than falling back inward.
            beyond = [m for m, _ in weights if m >= searched_km]
            return beyond[0] if beyond else self.distance_km
        return sum(m * p for m, p in weights) / total


@tracing.op
def ask(state: str, profiles: dict[str, str], model: str | None = None,
        timeout: int = 45) -> Reading:
    """One evaluation: which direction, how far, and what kind of subject."""
    key = os.getenv("TYPESAFE_API_KEY")
    if not key:
        raise JevUnavailable("TYPESAFE_API_KEY is not set")

    payload: dict[str, Any] = {
        "model": model or os.getenv("TYPESAFE_MODEL", "jev-latest"),
        "state": state,
        "questions": {
            # Asked first and deliberately: every other question here
            # presupposes that the subject began somewhere other than the
            # planning point, so without this one the model has no way to say
            # that the original premise still holds. It was structurally
            # required to nominate a relocation, which is exactly what a false
            # lead exploits.
            "premise_wrong": {
                "type": "choice",
                "instructions": (
                    "Does the case file, including anything that arrived after "
                    "the search began, indicate that the subject began their "
                    "journey somewhere other than the planning point? Reports "
                    "that were checked and traced to someone else, vehicles "
                    "belonging to unrelated people, and alerts that did not "
                    "develop are not evidence that the planning point is wrong."
                ),
                "criteria": {
                    "displaced": "The subject began somewhere other than the "
                                 "planning point, and the evidence says so.",
                    "as_planned": "The planning point is still the best account "
                                  "of where the subject began. Nothing in the "
                                  "file contradicts it.",
                },
            },
            "start_direction": {
                "type": "choice",
                "instructions": (
                    "In which compass direction from the planning point did the "
                    "subject actually begin their journey? If the case file states "
                    "a direction, that statement is the strongest evidence available."
                ),
                "criteria": {
                    name: f"The subject actually started to the {name} "
                          f"(bearing {int(deg)} degrees) of the planning point."
                    for name, deg in COMPASS.items()
                },
            },
            "start_distance": {
                "type": "score",
                "instructions": (
                    "How far from the planning point did the subject actually "
                    "begin? The ground within a few kilometres has already been "
                    "searched without contact."
                ),
                "criteria": [label for label, _ in DISTANCE_BANDS],
            },
            "subject_profile": {
                "type": "choice",
                "instructions": "Which behaviour pattern best describes this subject?",
                "criteria": profiles,
            },
        },
    }

    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()).hexdigest()
    cached = CACHE_DIR / f"{digest}.json"
    if cached.exists():
        body = json.loads(cached.read_text())
    else:
        resp = requests.post(
            ENDPOINT,
            headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
            json=payload, timeout=timeout,
        )
        resp.raise_for_status()
        body = resp.json()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cached.write_text(json.dumps(body))
    answers = body["answers"]

    premise = answers["premise_wrong"]
    direction = answers["start_direction"]
    distance = answers["start_distance"]
    profile = answers["subject_profile"]

    return Reading(
        premise_wrong=float(premise["probabilities"].get("displaced", 1.0)),
        premise_confidence=float(premise["confidence"]),
        direction_probs={k: float(v) for k, v in direction["probabilities"].items()},
        direction_confidence=float(direction["confidence"]),
        distance_km=float(distance["score"]),
        distance_probs={k: float(v) for k, v in distance["probabilities"].items()},
        distance_confidence=float(distance["confidence"]),
        profile_probs={k: float(v) for k, v in profile["probabilities"].items()},
        profile_confidence=float(profile["confidence"]),
        usage=body.get("usage", {}),
    )
