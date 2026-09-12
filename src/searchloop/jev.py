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

import os
from dataclasses import dataclass
from typing import Any

import requests

from . import tracing

ENDPOINT = "https://api.typesafe.ai/v1/systemone"

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

    direction_probs: dict[str, float]
    direction_confidence: float
    distance_km: float
    distance_probs: dict[str, float]
    distance_confidence: float
    profile_probs: dict[str, float]
    profile_confidence: float
    usage: dict[str, int]

    def expected_distance_km(self) -> float:
        """Probability-weighted distance across the bands."""
        total = 0.0
        for i, (_, midpoint) in enumerate(DISTANCE_BANDS):
            total += self.distance_probs.get(str(i), 0.0) * midpoint
        return total or self.distance_km


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

    resp = requests.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
        json=payload, timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    answers = body["answers"]

    direction = answers["start_direction"]
    distance = answers["start_distance"]
    profile = answers["subject_profile"]

    return Reading(
        direction_probs={k: float(v) for k, v in direction["probabilities"].items()},
        direction_confidence=float(direction["confidence"]),
        distance_km=float(distance["score"]),
        distance_probs={k: float(v) for k, v in distance["probabilities"].items()},
        distance_confidence=float(distance["confidence"]),
        profile_probs={k: float(v) for k, v in profile["probabilities"].items()},
        profile_confidence=float(profile["confidence"]),
        usage=body.get("usage", {}),
    )
