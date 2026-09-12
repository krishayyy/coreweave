# Search loop

An autonomous search agent that changes its mind about what happened.

Every deployed search-planning system optimises **inside an assumption**: a human
builds a prior from the case file, and the mathematics takes over. The failure
mode that kills people is not bad arithmetic, it is a wrong premise. When the
subject never went down the drainage — when they took a ride out, or were never
on the mountain at all — a conventional system updates its posterior forever and
only grows more confident about the wrong valley.

This system runs two nested loops:

- **Inner loop** (per operational period) — plan a segment, sweep it, apply the
  non-detection, replan. Fully deterministic Bayesian search.
- **Outer loop** (on disconfirmation) — when every hypothesis has been
  substantially ruled out, propose *new accounts of what happened* and admit them
  into the mixture. The inner loop's objective is rewritten, not renormalised.

**The language model can only nominate. The mathematics decides.** A nominated
hypothesis is converted into a prior field and scored by the identical Bayesian
update as every hypothesis from the published library. It cannot move the
aircraft, cannot discard a search area, and cannot weight itself. A hallucinated
hypothesis simply fails to explain the evidence and dies.

## Grounding

- **Terrain** — real elevation from AWS Terrarium tiles (public, key-free); the
  same tiles deck.gl renders, so the simulation and the display share one source
  of ground truth. Default area is Mt Hood, Oregon Cascades.
- **Behaviour** — hypothesis priors approximate the published lost-person
  behaviour literature (Koester's ISRID ring models), which reports
  distance-from-IPP quantiles by subject category and terrain.
- **Detection** — standard SAR formulation, `POD = 1 - exp(-coverage)`, with
  effective sweep width reduced under canopy and on steep ground.

## Layout

    src/searchloop/
      terrain.py      elevation tiles -> raster, slope, hillshade
      grid.py         coarsened grid + D8 flow accumulation (drainage network)
      pod.py          probability of detection
      hypotheses.py   subject profiles and their prior fields
      belief.py       Bayesian mixture over hypotheses
      planner.py      segment selection
      config.py       shared operating parameters

## The experiment

Three arms, run against identical scenarios with identical detection rolls:

| arm | revises? | reads the case file? |
|---|---|---|
| `none` | no | -- |
| `heuristic` | yes | **no** -- relocates blindly toward unsearched mass |
| `llm` | yes | yes |

The `heuristic` arm is what makes the result meaningful. Without it, beating the
baseline would only show that *trying somewhere else* helps. Beating `heuristic`
is what shows the model is reading the evidence.

Scenarios come in three families:

- **A** -- the premise is correct. This is the null: revision should not help,
  and if it never hurts here something is wrong with the experiment.
- **B** -- wrong about **where** the subject started. They were transported,
  deviated deliberately, or the planning point rests on a false premise. No
  hypothesis in the library can reach them, because every library hypothesis is
  anchored at the planning point.
- **C** -- wrong about **who** they are. The planning point is right but the
  behaviour category was misjudged. The library already contains the correct
  profile at a low base rate.

        python scripts/experiment.py --n-a 30 --n-b 24 --n-c 12 --repeats 5

## Results

Held-out suite (seed 7), five independent detection-roll repeats per scenario,
Wilson 95% intervals.

    arm                           family   find rate        localised       to loc
    library only (no revision)    A        78% [71-84]      81% [74-87]        2.1
    library only (no revision)    B        31% [23-40]      46% [37-55]        6.7
    library only (no revision)    C       100% [94-100]     97% [89-99]        1.0
    blind relocation              A        66% [58-73]      78% [71-84]        1.6
    blind relocation              B        22% [16-31]      27% [20-35]        3.2
    blind relocation              C       100% [94-100]    100% [94-100]       1.1

Two metrics, because they measure different things:

**find rate** -- the subject was actually detected. Capped by sensor POD, so it
partly reports detection luck rather than reasoning.

**localised** -- the true location reached the top decile of belief at any point.
This moves only when the agent reallocates belief correctly and is unaffected by
detection rolls. It is the reasoning metric.

### What the baseline already does, and what it cannot do

**Type C is solved at 100%.** Ordinary Bayesian search recovers a misjudged
behaviour category on its own, because the correct profile was in the mixture the
whole time and the evidence promotes it. The library is not a strawman.

**Type B is where it fails, at 31%.** Every library hypothesis is anchored at the
planning point, so when the subject started somewhere else there is no amount of
evidence that can move the mixture to them. The premise is outside the
hypothesis space.

So the claim is narrow and specific: *conventional search already handles being
wrong about **who** someone is. It cannot handle being wrong about **where** they
started.*

**Revision is not free.** On type A it costs 12 points -- abandoning a correct
hypothesis to chase a bad one is a real and expensive mistake.

**Moving the search is not understanding why.** Blind relocation drops type B
from 31% to 22% and localisation from 46% to 27%.

### The reachable ceiling

A diagnostic nominator with access to withheld ground truth reaches **79%** on
type B. That bounds what any reasoning quality could deliver, and a reported
result above it would indicate a leak rather than a finding.

The revision threshold was selected on a separate tuning suite (seed 99) and
applied unchanged here, so it is not fitted to these numbers. Seeding is
explicit rather than derived from `hash()`, which Python randomises per process
-- results reproduce exactly across runs.

## The demo scenario

`B005` -- *deliberate deviation*. What the incident commander has at hour zero:

> Tom Aldridge, 50, told family they were hiking the standard route from
> Timberline. Vehicle at the trailhead. Overdue since 09:30.

Three operational periods in, with nothing found, one more item arrives:

> A message on their phone, sent at 06:40, reads: "heading over the back side
> first, will loop round after" -- recipient unidentified.

The subject crossed the divide before starting the hike he described. He is
6.9 km from where everyone is looking, and no amount of searching the stated
route will reach him.

Run it:

    export GROQ_API_KEY=...        # or ANTHROPIC_ / WANDB_ / TYPESAFE_
    python scripts/demo.py         # the model in the loop
    python scripts/demo.py --oracle  # diagnostic: what a correct account looks like

    python3 -m http.server 5173 --directory web   # the display

`demo.py` streams the reasoning to the terminal as it happens -- evidence
arriving, the premise failing, each proposed account with the fragment of the
case file it cites -- and refreshes the display when it finishes.

## Reproducing

    python scripts/experiment.py --n-a 30 --n-b 24 --n-c 12 --repeats 3
    python scripts/ceiling_check.py      # the diagnostic upper bound
    python scripts/tune_trigger.py       # threshold sweep, tuning suite only
    python scripts/tune_revisions.py     # revision cap sweep, tuning suite only
    python -m pytest tests/

Every arm is seeded from an explicit digest rather than `hash()`, and model
responses are cached by prompt, so all three arms reproduce exactly.

## Observability

Weave instrumentation is optional by construction -- absent credentials every
decorator is a pass-through and the loop is unchanged. What is traced is the
revision rather than the search: what the evidence had ruled out, what was
proposed in response, and what was refused. The rejected proposals are the
informative rows, because they show the mixture declining a bad account instead
of absorbing it.
