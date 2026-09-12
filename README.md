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

Three arms, run against the identical scenario suite with identical detection
rolls, so nothing separating them can come from luck:

| arm | revises? | reads the case file? |
|---|---|---|
| `none` | no | -- |
| `heuristic` | yes | **no** -- relocates blindly toward unsearched mass |
| `llm` | yes | yes |

The `heuristic` arm is the one that makes the result meaningful. Without it,
beating the baseline would only show that *trying somewhere else* helps. Beating
`heuristic` is what shows the model is reading the evidence.

Scenarios come in two families. Type A: the obvious reading is correct. Type B:
it is wrong, in one of the four documented ways real searches fail. **Type A is
a null** -- revision should not help when the premise was right, and reporting
that plainly is what makes the Type B result credible.

    python scripts/experiment.py --n-a 30 --n-b 20

## Results on the held-out suite

30 type A and 20 type B scenarios, seed 7. Every arm sees identical scenarios and
identical detection rolls.

    arm                                type    found   rate   periods*
    library only (no revision)         A       24/30    80%       6.4
    library only (no revision)         B        8/20    40%      12.7
    blind relocation (no case file)    A       20/30    67%       7.2
    blind relocation (no case file)    B        4/20    20%      13.1

    * mean periods to find, unfound runs censored at the 16-period budget

Two things worth stating plainly, because they are what make any later result
credible:

**Revision is not free.** On type A, where the original premise was correct,
revising *costs* 13 points. Abandoning a good hypothesis to chase a bad one is a
real and expensive mistake, not a hypothetical one.

**Moving the search is not the same as understanding why.** Blind relocation
halves the type B find rate, from 40% to 20%. Anything that beats the baseline
has to be doing more than deciding to look elsewhere.

### The reachable ceiling

A diagnostic nominator with access to the withheld ground truth reaches **55%**
on the same type B suite. That is the upper bound on what any reasoning quality
could deliver here: the language model arm is competing for the band between
40% and 55%, and a claim above 55% would indicate a leak, not a result.

The revision threshold was selected on a separate tuning suite (seed 99) and
applied unchanged to the suite reported above, so it is not fitted to these
numbers.

## Status

Deterministic core, scenario suite, revision trigger and experiment harness
complete and tested. The LLM arm is implemented and unit-tested against a test
double; it needs credentials to run for real.
