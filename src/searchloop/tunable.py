"""The knobs the agent is allowed to turn on itself.

Three changes measurably improved this system: resolving direction to sixteen
compass points instead of eight, committing to one account when the model was
confident instead of always hedging across three, and conditioning the distance
estimate on ground already swept. All three were found by a human reading the
system's own error statistics.

None of them were sentences. Every attempt to improve this system by writing it
a better instruction failed -- twice by a human, twice by the agent itself,
arriving at the same wrong rule from the same data. The lever that works is
configuration, so that is the space the self-improvement loop should search.

Each parameter here is real, bounded, and has a defensible effect on behaviour.
A proposal outside the bounds is rejected before it is ever measured.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Knob:
    name: str
    lo: float
    hi: float
    default: float
    integer: bool
    what: str

    def clamp(self, value: float) -> float:
        v = max(self.lo, min(self.hi, float(value)))
        return round(v) if self.integer else round(v, 3)


KNOBS: dict[str, Knob] = {k.name: k for k in [
    Knob("exhaustion_trigger", 0.20, 0.70, 0.50, False,
         "how much of the leading account must be ruled out before the premise "
         "is reconsidered. Lower reconsiders sooner and risks abandoning a good "
         "hypothesis; higher wastes periods on one that has already failed."),
    Knob("max_revisions", 1, 4, 2, True,
         "how many times one search may change its mind. Each revision admits "
         "new accounts, and too many dilutes the mixture until the search "
         "commits to nothing."),
    Knob("min_periods_between_revisions", 1, 4, 2, True,
         "cooling-off between revisions, so a single bad period cannot trigger "
         "a cascade."),
    Knob("evidence_trigger_floor", 0.05, 0.40, 0.15, False,
         "how far the current account must already be ruled out before newly "
         "arriving evidence is allowed to reopen the premise immediately."),
    Knob("direction_mass_cutoff", 0.60, 0.98, 0.85, False,
         "how much of the direction distribution to act on. Lower commits to "
         "the most likely direction; higher hedges across more of them, which "
         "covers more ground but splits the sweep."),
    Knob("direction_min_probability", 0.03, 0.25, 0.08, False,
         "the smallest direction probability worth admitting as an account at "
         "all."),
    Knob("max_nomination_prior", 0.20, 0.50, 0.35, False,
         "the most belief any single proposed account may take from the "
         "existing mixture."),
    Knob("precedent_count", 0, 5, 3, True,
         "how many resolved cases to retrieve as precedent when proposing."),
]}

DEFAULTS: dict[str, float] = {k.name: k.default for k in KNOBS.values()}

# A plausible first guess by someone who has not measured anything: wait a long
# time before doubting the premise, change your mind at most once, ignore
# arriving evidence until the account is nearly dead, hedge across every
# direction the model will admit, and keep no memory of past searches.
#
# Every one of these is defensible a priori. All of them are wrong, and the
# tuning loop has to discover that from its own results -- which is a far
# better test of the loop than asking it to improve a configuration a human
# already optimised by hand.
NAIVE: dict[str, float] = {
    "exhaustion_trigger": 0.70,
    "max_revisions": 1,
    "min_periods_between_revisions": 4,
    "evidence_trigger_floor": 0.40,
    "direction_mass_cutoff": 0.98,
    "direction_min_probability": 0.03,
    "max_nomination_prior": 0.20,
    "precedent_count": 0,
}


def describe(settings: dict[str, float] | None = None) -> str:
    """The knobs, as the agent reads them.

    This used to print `currently {k.default}` -- the hand-tuned default --
    regardless of what the system was actually running. Under `--from-naive`
    that handed the model the eight values it was supposed to rediscover from
    its own measurements, and quietly turned the experiment into a copying
    exercise. The current value now comes from the live configuration or is
    omitted entirely.
    """
    def value(k: Knob) -> str:
        if settings is None:
            return ""
        return f"currently {settings.get(k.name, k.default)}, "

    return "\n".join(
        f"  {k.name}: {value(k)}allowed {k.lo} to {k.hi}"
        f"{' (whole numbers)' if k.integer else ''}\n      {k.what}"
        for k in KNOBS.values()
    )


def validate(name: str, value: Any) -> tuple[str, float] | None:
    """Clamp a proposal into range, or reject an unknown parameter outright."""
    knob = KNOBS.get(str(name))
    if knob is None:
        return None
    try:
        return knob.name, knob.clamp(float(value))
    except (TypeError, ValueError):
        return None


def apply(settings: dict[str, float]) -> None:
    """Install a configuration across the modules that read these values."""
    from . import revision
    from .casememory import CaseMemory

    revision.MAX_NOMINATION_PRIOR = settings["max_nomination_prior"]
    revision.DIRECTION_MASS_CUTOFF = settings["direction_mass_cutoff"]
    revision.DIRECTION_MIN_PROBABILITY = settings["direction_min_probability"]
    CaseMemory.DEFAULT_K = int(settings["precedent_count"])


def as_config(base, settings: dict[str, float]):
    """A Config carrying the loop-level settings from this configuration."""
    from dataclasses import replace
    return replace(
        base,
        exhaustion_trigger=settings["exhaustion_trigger"],
        max_revisions=int(settings["max_revisions"]),
        min_periods_between_revisions=int(settings["min_periods_between_revisions"]),
        evidence_trigger_floor=settings["evidence_trigger_floor"],
    )
