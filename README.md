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

### A System One model does better than a language model here

The language model arm's measured failure was never that its best proposal was
bad -- the best proposal in each response sat 2.2 km from the truth. It was that
it could not tell which of its own proposals was the best one: stated confidence
ranked them at 56% concordance, where 50% is no signal.

TypeSafe's System One model removes that problem rather than mitigating it. It
does not write. You define the shape of the answer and it returns a probability
for every option, so asked which direction the subject started in it returns a
distribution over compass points -- and a distribution is exactly what the
belief mixture already consumes. There is no selection step because nothing
needs selecting.

    arm                                family   find rate        localised
    library only (no revision)         B        43% [32-55]      54% [43-65]
    blind relocation                   B        31% [21-42]      47% [36-59]
    System One calibrated distribution B        69% [58-79]      78% [67-86]
    oracle -- told the true answer     B        83%              78%

    paired, 24 scenarios               delta      95% CI            p
    find      vs library only         +26.4pp   [+13.9, +38.9]   0.000  significant
    find      vs blind relocation     +38.9pp   [+23.6, +54.2]   0.000  significant
    localised vs library only         +23.6pp   [ +1.4, +45.8]   0.049  significant
    localised vs blind relocation     +30.6pp   [+15.3, +47.2]   0.000  significant

    paired, 30 scenarios, type A       delta      95% CI            p
    find      vs library only          -2.2pp   [ -8.9,  +3.3]   0.613  no harm

43% to 69% against a ceiling of 83%: about two thirds of the available headroom,
with both metrics significant and no measurable cost on cases where the premise
was already right.

### Two ceilings that were mine, not the model's

The gap to the oracle closed in two steps, and neither was a better model.

**The question was too coarse.** Direction was asked as a choice among eight
compass points -- 45 degree bins -- so the answer could not be better than about
22 degrees, and the measured error was 19. That is a model answering as
precisely as the question allows, not as precisely as it can. Sixteen points
took bearing error from 19 degrees to 11, and the share within 20 degrees from
51% to 88%.

**The confidence was being discarded.** Up to three directions were admitted
whenever they cleared a fixed threshold, regardless of how sure the model was.
The tell was that the oracle localises the subject *less* often than this arm
does -- 78% against 82% -- and still finds more of them, because it commits to
one account and the search concentrates while three accounts split the sweep
three ways. Hedging in proportion to actual doubt, rather than always, gained
five points on type B and recovered four on type A.

Both were ceilings imposed by how the system asked, and the second one is the
sharper lesson: the whole reason to use a calibrated model is to act on the
calibration. Reading the probability and then ignoring it gives up most of what
it is for.

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

### Searching better, as distinct from believing better

The planner is held fixed across arms so differences cannot be attributed to
routing. That invites a fair question: does the loop ever get better at
*searching*, or only at *believing*? Three attempts, and the answer is that
there is very little there to get.

**Adaptive altitude.** Height widens the field of view and costs resolution, so
effective sweep width has an interior maximum -- and under canopy the optimum
moves down, since seeing through a gap needs a steeper look. Real trade-off,
but no adaptive signal: the optimum sits at the same altitude whether belief is
concentrated or diffuse, because coverage and detection both scale with sweep
width. There is nothing to choose.

**Choosing how thoroughly to sweep.** A fixed track budget buys a small area
searched well or a large area searched poorly: swept area is
(track x sweep width) / c and detection within it is 1 - exp(-c). Plausibly this
should depend on concentration. It does not -- c = 1 won in every regime tested,
from the full library mixture down to a profile with a 0.8 km median radius.

**Koopman's optimal effort allocation.** The classical result: for exponential
detection the optimum is not uniform but `effort = max(0, ln(P / lambda))`, with
lambda set to the budget. Implemented and measured against uniform mowing:
-6%, -8%, +1% on diffuse, moderate and tight beliefs respectively.

    belief                     POS uniform   POS optimal    change
    diffuse (whole library)        0.310        0.291         -6%
    moderate (hiker profile)       0.191        0.176         -8%
    tight (dementia profile)       0.480        0.484         +1%

Break-even at best, because greedily taking the highest-value cells at uniform
coverage is already a close approximation to the optimum. So the fixed planner
was not a limitation being papered over -- the search tactics were already near
their ceiling, and all the available headroom is in the premise.

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

### Learning from precedent instead of rules

Rewriting instructions is the weaker way to improve, and in this project it kept
failing. An abstract rule has to be right about every future case at once, it is
hard to validate, and a confident diagnosis of your own failures is often wrong
-- which happened twice here, once to a human and once to the agent, arriving at
the same wrong rule from the same data.

So the agent also keeps **resolved cases**. When a search ends the truth becomes
known: this is what the evidence said, and this is where the subject had
actually started. Facing a new case, it retrieves the resolved ones whose
evidence most resembles this one and is shown what turned out to be true.

    PRECEDENT. You have resolved 12 searches before this one. These are the most
    similar, and in each the truth is now known:

      Evidence was: "a vehicle that left heading north-east."
        It turned out the subject had started 8.6 km from the planning point on
        bearing 58 degrees (north-east), behaving as a hiker.

That teaches a convention by example rather than by instruction, which is
precisely what the instruction-rewriting attempts could not do. It also adapts
per case: different evidence retrieves different precedent.

Retrieval is tf-idf cosine over the evidence text. Deliberately not an embedding
model -- that is a dependency and a failure mode, and the text being matched is
short and lexically distinctive.

        python scripts/learning_curve.py --n 16

#### Result

16 cases run in sequence, twice: once with memory accumulating, once with
nothing remembered. Identical scenarios, identical detection rolls. Case 1 is
identical in both arms by construction -- with no precedent there is nothing to
retrieve -- which is a useful check that the only difference between the arms is
memory.

    paired over the same cases        bearing error of the best account proposed
      with memory                       15.9 deg
      without                           23.8 deg
      difference                        -7.9 deg   95% CI [-17.1, 0.0]  p = 0.051
      better on 7 cases, worse on 3, tied on 5
      find rate                         60% vs 53%

A 33% reduction in angular error, sitting exactly on the significance line at
n=15. Not rounded down: p = 0.051 is not significant at 0.05.

**Where it wins is the interesting part.** The gains are concentrated in
`wrong_ipp` -- 62 to 38 degrees, 46 to 1, 56 to 11. That is the same failure mode
that two attempts at writing an abstract rule could not fix, once by a human and
once by the agent. Showing it three concrete resolved cases did what telling it
a rule could not.

**What the shape does not show.** The gap is roughly constant rather than
widening, and the two lines converge by the end. So this is evidence that
precedent helps *from the first retrieved case onward*, not that the benefit
compounds with experience. A compounding effect would need far more than 16
cases to detect, and claiming one from this chart would be reading it wrong.

**A design error worth recording.** This was first measured as a difference in
differences -- first half against second half, memory against control. That
metric reported +11 degrees, pointing the wrong way, because the control started
worse and had more room to regress toward the mean. The paired comparison is the
correct analysis and is the one used everywhere else in this project; it simply
was not applied here at first.

## Exploring it

    marimo run notebooks/explorer.py

A judging table is a conversation, not a screening. People walk up mid-sentence
and want to poke at the thing: show me one where it fails; what was it running
on at period six; what happens if the witness had not named a direction. A
recorded demo answers none of that.

The explorer is reactive -- change the experiment, the arm, the case or the
operational period and everything below recomputes. Every run is already
serialised with per-period leaders, disconfirmation and the nominations that
fired, so this is wiring rather than new machinery.

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
