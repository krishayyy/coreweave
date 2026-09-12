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

## Install

    python3 -m pip install -r requirements.txt          # to run
    python3 -m pip install -r requirements-dev.txt      # to run the tests and Weave

No API key is needed for the deterministic arms, and terrain comes from a
public, key-free tile source -- nothing in the baseline can fail for want of a
credential.

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

Held-out suite (seed 7), three independent detection-roll repeats per scenario.
Every arm sees identical scenarios with identical detection rolls.

    arm                           family   find rate        localised
    library only (no revision)    A        79% [69-86]      82% [73-89]
    library only (no revision)    B        43% [32-55]      54% [43-65]
    library only (no revision)    C        89% [75-96]      75% [59-86]
    blind relocation              A        77% [67-84]      82% [73-89]
    blind relocation              B        35% [25-46]      54% [43-65]
    blind relocation              C        81% [65-90]      75% [59-86]
    case-file nomination          A        77% [67-84]      86% [77-91]
    case-file nomination          B        60% [48-70]      75% [64-84]
    case-file nomination          C        86% [71-94]      67% [50-80]

### Significance

The arms run on identical scenarios, so they are paired and an unpaired interval
discards exactly the information that makes the comparison sharp. Repeats of a
scenario are not independent either -- they share the scenario -- so repeats are
averaged within a scenario first and the bootstrap resamples **scenarios**.
Treating 72 runs as 72 independent trials would overstate significance
considerably.

    Type B, paired, 24 scenarios          delta      95% CI            p
    find rate   vs library only          +16.7pp   [ +4.2, +30.6]   0.010   significant
    find rate   vs blind relocation      +25.0pp   [ +9.7, +40.3]   0.002   significant
    localised   vs library only          +20.8pp   [ -1.4, +43.1]   0.079   NOT significant
    localised   vs blind relocation      +20.8pp   [ -1.4, +43.1]   0.080   NOT significant

    Type A, paired, 30 scenarios (null)
    find rate   vs library only           -2.2pp   [-11.1,  +6.7]   0.596   no effect

        python scripts/significance.py

### What this does and does not show

**It works on the family it was built for.** On type B, reading the case file
raises the find rate by 17 points over ordinary Bayesian search and by 25 points
over relocating without reading. Both are significant under a paired test that
resamples scenarios.

**Localisation is directionally positive but not significant.** +20.8 points with
an interval that just crosses zero at 24 scenarios. The effect looks real and
this suite is too small to establish it. It is reported as a null, not rounded
into the headline.

**It does no harm where the premise was right.** Type A is flat (-2.2pp, p=0.60).
An earlier configuration cost 12 points there; capping revisions removed that,
and the cost of being wrong about being wrong is now close to zero.

**It captures a little under half the available headroom.** Baseline 43%,
nomination 60%, and a diagnostic nominator with access to withheld ground truth
reaches 83%. The remaining 23 points are cases where the model proposed a
plausible account that was not the right one -- most of the error is in bearing,
not in the story.

**Type C is unaffected**, as it should be: the library already solves it and
there is nothing for revision to add.

### Things that did not work

**Spreading a nomination over an arc of bearings.** Most of the remaining error
is angular: the model identifies *what* happened more reliably than *which
direction*. Representing that honestly -- as an arc rather than a committed
bearing -- looked like the obvious fix. Tested against a nominator with a
controlled bearing error, so the error was a variable rather than a confound:

    bearing err | sigma=0   | sigma=15  | sigma=30  | sigma=45
           0deg |  72%/ 62% |  69%/ 69% |  60%/ 65% |  53%/ 39%
          15deg |  58%/ 53% |  62%/ 56% |  57%/ 50% |  53%/ 46%
          30deg |  47%/ 42% |  54%/ 43% |  46%/ 46% |  51%/ 43%
          45deg |  43%/ 40% |  43%/ 42% |  47%/ 39% |  46%/ 43%

Matching the declared uncertainty to the real error gains 4-7 points, which is
inside the noise at 24 scenarios, and it costs 3 points when the bearing was
good. Not adopted. The capability is in the code and defaults to off; taking a
change on evidence this thin is the same overfitting the rest of the method
avoids.

        python scripts/tune_bearing_arc.py

## The loop that persists

The inner loop searches. The outer loop notices its premise is wrong and
replaces it. Neither gets better over time -- the hundredth search is no wiser
than the first, because nothing survives the end of a run.

So there is a third loop, running at the timescale of many searches:

1. **Search.** Run a batch, and fail at some of them.
2. **Review.** Score every proposal the agent made against the truth it never
   saw. Summarise into statistics and concrete worked misses.
3. **Propose.** Show the agent its own record and ask what instruction would
   have prevented those errors.
4. **Validate.** Measure the candidate on a fold it was not derived from.
5. **Gate.** Keep it only if it clears a margin. Record it either way.

        python scripts/self_improve.py --rounds 2

Accepted lessons are injected into the nomination prompt for every future
search. Rejected ones stay in the book with their measured effect, so the agent
does not re-propose them and the record shows what was tried.

### Why there is a gate

Because I got this wrong myself, in exactly the way the gate exists to catch.

Reviewing 297 of the agent's own nominations showed that bearing error was
6 degrees where the cue required terrain reasoning and 55-60 degrees where the
case file simply *named* a direction. That is a clean, legible diagnosis: the
model is good at the hard inference and bad at the lookup. I wrote a compass
table into the prompt and predicted bearing error would fall.

On the held-out suite it **rose**, 55 to 68 and 60 to 70 degrees, and the effect
on find rate was unchanged. The mechanism was a correlation that looked good.
The instruction has been reverted, because it earned no place on the evidence.

Then the self-improvement loop, shown the same record, proposed **almost exactly
the same instruction** -- give a stated direction priority over terrain cues --
with the same rationale and the same predicted effect. Two independent
reviewers, one human and one model, reached the same plausible and wrong
conclusion from the same data.

The gate rejected it.

That is the whole argument for building one. A model asked to improve its own
prompt will always produce something that sounds like an improvement, and so
will a person. Whether it is one is an empirical question, and the answer is
often no. Self-improvement without a validation gate is not self-improvement;
it is a prompt slowly filling with plausible noise.

## Reproducing

    python scripts/experiment.py --n-a 30 --n-b 24 --n-c 12 --repeats 3
    python scripts/self_improve.py --rounds 2    # the loop that persists
    python scripts/error_analysis.py     # where the remaining error comes from
    python scripts/ceiling_check.py      # the diagnostic upper bound
    python scripts/tune_trigger.py       # threshold sweep, tuning suite only
    python scripts/tune_revisions.py     # revision cap sweep, tuning suite only
    python -m pytest tests/

Every arm is seeded from an explicit digest rather than `hash()`, and model
responses are cached by prompt, so all three arms reproduce exactly.

## Observability

Weave instrumentation is optional by construction -- absent credentials every
decorator is a pass-through and the loop is unchanged. Credentials are checked
*before* `weave.init` is called, because it prompts on stdin and blocks when it
finds none, and a frozen terminal during a live search is a worse failure than
no tracing at all.

    wandb login          # then tracing starts automatically What is traced is the
revision rather than the search: what the evidence had ruled out, what was
proposed in response, and what was refused. The rejected proposals are the
informative rows, because they show the mixture declining a bad account instead
of absorbing it.
