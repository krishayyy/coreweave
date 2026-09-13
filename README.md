# Search loop

An autonomous search agent that changes its mind about what happened.

**On the cases where the initial premise is wrong, standard search doctrine
finds the subject 35% of the time. This finds them 83% of the time.**

    +48.1 points, paired on the same 72 scenarios, p < 0.001
    against a real but small cost of 2.8 points on the cases where the
    premise was right, which each carry a false lead designed to bait it

That baseline is not a strawman and it is worth being precise about what it is.
The control arm implements what a trained incident commander actually does:
take the subject category, apply the published distance-from-planning-point
model for that category, weight it by terrain, allocate effort where the
probability of success is highest, and update on every negative sweep. It is
the ISRID/Koester method, run without fatigue, without ego, and without
anchoring on the first theory of the case. It is a *generous* representation of
human practice, not a weak one.

It still finds three in ten, because the method has no way to doubt its own
premise. Every deployed search-planning system optimises **inside an
assumption**: a human builds a prior from the case file, and the mathematics
takes over. The failure mode that kills people is not bad arithmetic, it is a
wrong premise. When the subject never went down the drainage — when they took a
ride out, or were never on the mountain at all — a conventional system updates
its posterior forever and only grows more confident about the wrong valley.

The cases in the type B suite are the documented ways this happens: the subject
was transported out of the area, deviated deliberately, or the planning point
itself was wrong. On the cases where the premise *was* right, this system costs
**-2.8pp (p=0.030)**, and those cases each carry a false lead, so the suite is
actively trying to bait it into moving.

That cost is worth stating precisely, because at half this sample size it did
not reach significance and an earlier version of this README called it "no
measurable cost". Doubling the correct-premise cases from 30 to 60 made a small
real effect detectable. It is small, it is not zero, and it is the price of
being willing to doubt.

Because the gain is conditional on the premise being wrong, the honest question
is how often that has to happen for the system to be worth running:

    break-even premise-error rate    5.5%   95% CI [1.0, 11.4]

Above that the system is net positive across a whole caseload; below it, the
cost of doubting outweighs the wins. It was 10.0% before the doubt gate. Net
benefit is demonstrable from about a 15% error rate upward.

Every decision takes **4 ms** on a 21 km box and 10 ms on a 48 km county, which
is about five orders of magnitude faster than the flight it is planning.

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

## What this is, and what it is not

This is a **planning system evaluated in simulation**. No aircraft was flown and
none is claimed. The sensor is a parameter -- a nominal 180 m effective sweep
width for a thermal and RGB pass, degraded under canopy and on steep ground --
not a particular product.

That is the correct standard for this problem rather than a compromise. You
cannot run a controlled experiment on real missing people: you would need the
same case searched twice by two different methods, with the answer withheld
from both. A simulator is what makes the ground truth withheld, the arms
comparable and the result measurable. One real flight would be an anecdote;
what follows is 594 runs with a control.

What is real: the terrain, the drainage networks derived from it, the
distance-from-planning-point distributions taken from published lost-person
behaviour literature, and the detection formulation, which is the standard one
used in search and rescue. What is synthetic: the incidents, generated
procedurally so no model can have seen the answers.

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

        python scripts/experiment.py --n-a 30 --n-b 24 --n-c 12 --repeats 3

## Results

### Why a System One model rather than a language model

The first working version of this had a language model write accounts in prose.
Its measured failure was never that its best proposal was bad -- the best
proposal in each response sat 2.2 km from the truth. It was that the model
could not tell which of its own proposals was the best one: across 297
nominations, stated confidence ranked them at **56% concordance**, where 50% is
no signal. Its top pick averaged 4.7 km from the truth; the best pick in the
same response averaged 2.2 km.

That measurement is the reason for everything that follows. It has since been
re-run under the current configuration -- the evidence trigger, the
sixteen-point compass and the distance conditioning -- against the identical 66
scenarios and the same seed, so the two arms are paired run for run and appear
side by side below.

TypeSafe's System One model removes that problem rather than mitigating it. It
does not write. You define the shape of the answer and it returns a probability
for every option, so asked which direction the subject started in it returns a
distribution over compass points -- and a distribution is exactly what the
belief mixture already consumes. There is no selection step because nothing
needs selecting.

### Knowing when *not* to change your mind

A system willing to revise its premise is a system that can be talked out of a
correct search. Real case files are mostly false leads -- a sighting traced to
another party, a vehicle belonging to someone unrelated, a dog alert that did
not develop -- and every correct-premise case in this suite now carries one,
so the suite can fail a system for being too willing to move.

The first version failed it. The reason was the question, not the model. Every
question in the System One payload presupposed displacement -- *"in which
compass direction did the subject actually begin?"* -- so no answer was
available meaning "where you already think". It was structurally obliged to
nominate a relocation every time it was consulted.

It is now asked the prior question first, and every relocation is gated and
scaled by the answer: P(the subject began somewhere other than the planning
point). A half-convinced model moves half as much belief; below a floor it
moves none.

Measured on a tuning fold held separate from everything above:

    premise RIGHT, with a false lead    median P(wrong)  0.34
    premise WRONG                       median P(wrong)  0.69
    separation                          AUC 0.813

The comparison that matters is against the trigger this replaces. **Exhaustion
-- the fraction of the predicted area covered without contact, which is how
Bayesian search systems conventionally decide to doubt themselves -- fired at a
median of 0.67 on both families. That is an AUC of 0.5: chance.** Coverage
cannot answer "am I wrong", because coming up empty happens just as often when
you are looking in exactly the right place. Asking the question directly
carries information; inferring it from how much ground you have covered does
not.

#### Three attempts to drive the correct-premise cost to zero, none of which worked

The -2.8pp is the price of being willing to doubt, and it would be better at
zero. It resists.

First, a look at where it comes from. Splitting the correct-premise cases by
whether the system revised:

    scenarios where it revised (n=25)     doctrine 94.7%   System One 88.0%
    scenarios where it did not (n=35)     doctrine 100.0%  System One 100.0%

Most of the apparent gap is selection rather than damage. Those 25 are the
cases where sweeping the right ground came up empty, so doctrine does worse on
them too -- the search is unlucky, which is also what makes the system doubt
itself. The causal part is the remaining -6.7pp on the cases where revision
fires, which averages to -2.8pp overall. An earlier reading of this that
attributed the whole gap to revision was confounded.

**Attempt one: stop a losing nomination from steering the aircraft.** Of the
runs that revised and then failed, seven of nine still had a library hypothesis
leading at the end -- the nomination lost the argument and cost the search
anyway, which suggested it was taking just enough mass to move the sortie. So
the planner was restricted to accounts holding at least half the leader's
posterior, leaving the rest in the mixture to grow or die on evidence. It
recovered almost nothing on correct-premise cases (-8.3pp to -8.3pp at a half
share, -6.2pp at three quarters) and cost type B six points. Reverted.

**Attempt two: raise the doubt floor.** The frontier is flat:

    floor   correct-premise   premise-wrong   break-even
     0.40        -8.3pp          +56.2pp        12.9%
     0.50        -8.3pp          +52.1pp        13.8%
     0.60        -6.2pp          +41.7pp        13.0%
     0.70        -6.2pp          +41.7pp        13.0%

Every point recovered on the correct-premise cases costs roughly its own worth
on the cases the system exists for. There is no setting that buys a better
trade, which is what a flat break-even column means.

**Attempt three: cap how much belief a nomination can take.** Already capped,
at 0.35 of the mixture and scaled by the model's own doubt, so the original
account always retains the majority. Lowering it further had no effect on the
heuristic arm and the knob does not reach the calibrated arm's damage.

The conclusion is that the residual cost is not a threshold problem. It is the
40% of false leads on which the premise judgement is genuinely wrong -- the
model puts P(premise wrong) above 0.40 on a report that was already traced to
somebody else. Reducing it further needs a better-calibrated answer to that
question, not a different cutoff on the same answer.

The floor is 0.40, chosen on the tuning fold: it blocks 60% of false leads
while keeping 91% of the real displacements. A floor of 0.60 blocks 70% of
false leads but discards a third of the cases the system exists to solve.

This is also what reduced the cost on correct-premise cases, from -5.6pp to
-2.8pp, while *raising* the gain on the displaced ones -- a confident reading
now earns a stronger nomination than a hesitant one, where before every reading
was admitted at full strength. The remaining cost is small but, at 60
correct-premise cases, statistically real (p=0.030); at half that sample it did
not reach significance, which was a fact about the sample rather than about the
system.

    arm                                family   find rate        localised
    library only (no revision)         A        98% [94-99]      96% [92-98]
    library only (no revision)         B        35% [29-41]      39% [33-46]
    library only (no revision)         C       100% [95-100]    100% [95-100]
    blind relocation (reads nothing)   A        97% [93-98]      94% [90-97]
    blind relocation (reads nothing)   B        28% [22-34]      31% [26-38]
    blind relocation (reads nothing)   C        99% [93-100]    100% [95-100]
    System One calibrated distribution A        95% [91-97]      97% [94-99]
    System One calibrated distribution B        83% [77-87]      87% [82-91]
    System One calibrated distribution C       100% [95-100]    100% [95-100]

    paired, 72 scenarios               delta      95% CI            p
    find      vs library only         +48.1pp   [+36.6, +59.3]   0.000  significant
    find      vs blind relocation     +55.1pp   [+44.0, +65.7]   0.000  significant
    localised vs library only         +48.1pp   [+36.1, +59.7]   0.000  significant
    localised vs blind relocation     +55.6pp   [+44.4, +66.7]   0.000  significant

    paired, 60 scenarios, type A       delta      95% CI            p
    find      vs library only          -2.8pp   [ -6.1,  -0.6]   0.030  a real cost
    find      vs blind relocation      -1.7pp   [ -5.0,  +1.1]   0.286  not significant

The type A row is the one worth pausing on. Prose revision does help on type B
-- 57% against the library's 31% -- so the language model is not useless at
this. But it pays for that help by damaging the cases where the premise was
correct all along, dropping type A from 90% to 74%, because it argues itself
into relocating a search that was already pointed at the subject. The
calibrated arm does not, and the difference on the null is larger and better
separated than the difference on type B. Being unable to rank your own
hypotheses does not only cost you the wins; it costs you the cases you had
already won.

Every figure comes from `runs/experiment_large.json`: 60 type A, 72 type B and
24 type C scenarios, three detection-roll repeats, Wilson 95% intervals. The
suite was grown from 66 scenarios to 156; because scenario generation is
append-only, the original 66 are bit-identical and the earlier run remains a
valid subset rather than a superseded one.

The prose language-model arm is not in this table. It was measured before the
false leads and the doubt gate existed (`runs/experiment_llm_current.json`,
57% on type B against the calibrated arm's 81%, and 74% on type A against 90%)
and running it against a suite it has never seen would be comparing two
different experiments. Its result is reported in the section above rather than
placed in a table that would imply it was run under these conditions.

    python scripts/experiment.py --n-a 60 --n-b 72 --n-c 24 --repeats 3
    python scripts/significance.py runs/experiment_large.json jev

### What a county knows that its neighbour does not

The published behaviour models are national. They state a hiker's terrain
affinities once, for everywhere. But a search unit gets good at its own ground
precisely because its ground is not the national average: in one county the
drainages are the walkable way out and people follow them, in the next they are
dry, choked and go nowhere, and people stay off them. A drone flying the same
county for a year should end up knowing that. A drone flying the next county
should not inherit it, because it is not true there.

So the county layer learns a correction to the published prior rather than a
replacement for it:

    p_county(cell)  proportional to  p_published(cell) * exp(w . f(cell))

`f` describes a cell's terrain -- drainage, downhill from the anchor, canopy,
excess slope -- and `w` is fitted from that county's own resolved cases. Every
drone in a county reads the same `w` and contributes its closed cases back to
it, so the tenth drone starts where the ninth left off. `w = 0` is the
published model, and is where every county begins.

The learning rule is worth stating plainly because it is the entire mechanism:

    gradient  =  terrain where subjects were actually found
               - terrain the current belief expected to find them in

Move toward the terrain that keeps being right, in proportion to how surprised
you were. The problem is convex, so there is one optimum and nothing to seed.

The regulariser is what makes it safe on a real caseload. A county resolves a
handful of searches a year, and four free parameters fitted on three cases will
cheerfully conclude that everybody is found in creeks because the last three
were. The penalty is scaled by 1/n, so it dominates early and relaxes as
evidence accumulates -- the county's own data earns its way in rather than
being trusted on arrival.

Measured on three real jurisdictions, chosen for genuinely different terrain: a
glaciated volcano, barren high granite, and a forested Appalachian ridge. Two
were given a local deviation from the literature; the third was given none.
That third county is the important one.

    county          published   home   transplanted from next door
    Clackamas OR       4.2%     3.3%        7.9%
    Inyo CA            2.4%     2.4%        3.6%
    Buncombe NC        2.0%     2.0%        3.0%

Fraction of the county swept before reaching the subject, searching the
highest-probability cell first, on held-out cases. Lower is better.

Clackamas recovers **+0.72 against a planted +0.70** and sweeps a fifth less
ground. Buncombe is the null -- genuinely the national average -- and correctly
learns nothing, landing at `[+0.01, -0.07, -0.06, -0.09]`, neither helped nor
harmed. That is the result that makes the other two worth believing, because a
method that "learns" a correction for an average county is fitting noise and
its wins elsewhere would mean nothing.

Inyo recovers its deviation correctly, **-0.88 against a planted -0.70**, and
gains nothing operationally. Learning the right thing and gaining nothing is a
real outcome and it is reported rather than dropped: there is less leverage in
learning to avoid terrain the published prior already weights low than in
learning to favour terrain it ignores.

**Transplanting a neighbouring county's model is worse than having learned
nothing, in all three counties.** That is the actual claim. What a county
learns is true of its ground and not of ground in general, which is why this
has to be a per-county ecosystem rather than one national model that keeps
getting better.

    python scripts/county_learning.py 96

#### It improves the prior. It does not improve the outcome.

Wired into the operational loop -- the same drone, the same code, flying
held-out cases -- the county model does not find more people:

    search window      cold   experienced    delta
      1 period        54.2%      54.7%      +0.5pp
      2 periods       72.4%      69.8%      -2.6pp
      3 periods       78.6%      79.2%      +0.5pp
      4 periods       83.3%      83.9%      +0.5pp
      6 periods       88.0%      88.5%      +0.5pp

That is noise, at every window length. A better prior did not convert into a
better search.

My first explanation was granularity: one sortie sweeps a contiguous 7.4% of
this county, while ideal cell-by-cell ordering reaches the subject after
3.3-4.2%, so the subject sits inside the first blob either way and a ranking
gain smaller than the smallest action available cannot be acted on. That
predicts the gain should appear once a sortie is small relative to the county.
It does not:

    county        sortie      cold   experienced    delta
    21x21 km       7.4%      95.8%      92.4%      -3.5pp
    34x34 km       2.5%      95.1%      95.1%      +0.0pp
    48x48 km       1.2%      97.2%      97.9%      +0.7pp

The prediction failed, so the explanation was wrong. The likelier reason is
plainer: the find rate sits at 95-97% across every scale tested, because the
radial term dominates and the subject is near the anchor. There is no headroom
for a better prior to claim. Whether local knowledge pays off under conditions
that actually bite -- a displaced anchor, a genuinely large search area, a
subject who did not stay near where they started -- is not settled here.

So the honest scope of this section is narrower than it first appears. What is
demonstrated is that a county learns a correction that is real when there is
one, absent when there is not, and **not transferable to its neighbours**. What
is *not* demonstrated is that any of this finds more people. Those are
different claims and only the first is supported.

#### What this does not yet show

The local deviations were planted by me, and their size is my choice. The
experiment tests whether the method recovers a deviation that exists, ignores
one that does not, and refuses to transfer -- it does not establish how large
real county-to-county deviations are. That number would have to come from real
resolved-case records, which is exactly the data a county already holds and
this system would consume.

### The simulation was not implementing the literature it cites

Real incident data is not obtainable. ISRID is contribution-based rather than
downloadable, and it explicitly excludes media reports -- the only source that
could be scraped -- so reconstructing cases from news would use exactly the
evidence its own maintainers reject.

What could be checked is whether the simulated subjects move the way the
published quantiles say real ones do, since those quantiles are what the
profiles claim to implement. They did not:

    profile        published d50    generated (before)    generated (after)
    hiker               1.9 km            5.2 km                2.5 km
    dementia            0.8 km            2.2 km                0.8 km
    child               0.9 km            2.6 km                1.0 km
    angler              1.0 km            3.3 km                1.1 km

Simulated subjects sat about 2.7 times too far from the planning point. The
published figures are quantiles of *distance*, so the lognormal is a density in
distance -- but the number of grid cells at radius r grows with r, so assigning
each cell the density gives a radial marginal of pdf(r) x r, biased outward. A
missing Jacobian.

Corrected, the largest median deviation falls from 233% to 31%, and the
remaining gap is accounted for: profiles whose published d95 exceeds the map
half-width are truncated by the edge, and terrain affinity deliberately moves
mass off a pure radial.

**Every number on this page was re-measured afterwards.** The effect got larger
rather than smaller: with subjects correctly placed near their starting point,
a correct account is actually worth something, where before a right answer
still left the subject scattered across the map.

        python scripts/validate_priors.py

This is not a test against real incidents. It checks that the simulation
implements the literature it cites, which is a precondition for the result
meaning anything, not a substitute for field data.

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
The tell was that the oracle localised the subject *less* often than this arm
did -- 78% against 82%, both measured before the doubt gate and the false leads
existed -- and still found more of them, because it commits to
one account and the search concentrates while three accounts split the sweep
three ways. Hedging in proportion to actual doubt, rather than always, gained
five points on type B and recovered four on type A.

Both were ceilings imposed by how the system asked, and the second one is the
sharper lesson: the whole reason to use a calibrated model is to act on the
calibration. Reading the probability and then ignoring it gives up most of what
it is for.

### What "days" means, and what it depends on

The display reports days, and that number is assembled from one real convention
and one chosen parameter.

Real: one loop iteration is one **operational period of twelve hours**, which is
standard incident-command practice. A search is planned, flown and reviewed on
that cadence, and the premise is reconsidered at that boundary.

Chosen: **400 km of total track per period**, which sweeps about 32 km² of a
425 km² operating area. That is a plausible figure for a multi-aircraft day over
forested terrain, but it was picked to pace the simulation rather than taken
from an operations manual. Every "days" figure inherits it.

So absolute durations here are indicative, not predictive. The obvious next
question is whether the advantage is an artifact of that choice. It is not, but
it is not independent of it either:

    track/period   swept/period   conventional   this system   ratio
       200 km          16 km2          23%           50%       2.18x
       300 km          24 km2          35%           60%       1.71x
       400 km          32 km2          44%           69%       1.57x
       600 km          47 km2          62%           77%       1.23x

**The method helps most when search capacity is scarce relative to the area**,
which is when it matters. Given enough aircraft, everything gets covered
eventually and being wrong about the premise costs less. The reported figure
sits mid-range; a better-resourced operation would see a smaller gap and a
worse-resourced one a larger.

### What the 50 points is actually measuring

Correcting the Jacobian made the effect larger, which is the direction that
should invite suspicion rather than celebration. It holds up, but it changes
what the number means.

With subjects correctly placed, a type B subject sits **1.7 km from where they
started and 8.5 km from the planning point**, and the library prior ranks their
true location at the 52nd percentile -- no better than a coin. So type B has
become close to binary: identify the displaced starting point and the subject
is within a couple of kilometres of it; miss it and no amount of searching near
the planning point will reach them.

That is a property of the problem rather than of the suite. Real subjects do
stay near where they began -- a median of 1.9 km for hikers is the published
figure -- which is precisely why a search anchored on a wrong planning point
fails. The earlier version, with subjects scattered five kilometres from their
own starting point, was the unrealistic one and it compressed the difference
between a right and a wrong account.

But it means the honest reading of +50 points is **"how often does it identify
where the subject actually started"**, not "how much better does it sweep
ground". Those are different claims and only the first is supported here.

### The strongest objection to all of this

The evaluation is synthetic and I designed both sides of it. That is the
attack a reviewer should lead with, so it belongs here rather than in a
footnote.

**The scenario families encode a theory.** Type B is built from four documented
ways a search goes wrong -- the subject was transported, deviated deliberately,
was mis-categorised, or the planning point rests on a false premise. The system
is built to recover from exactly those. A result on a distribution I authored
is weaker evidence than a result on one I did not, and no amount of statistical
care inside the suite repairs that.

What keeps it honest rather than circular:

- The *behaviour* is not mine. Distance-from-planning-point distributions come
  from the published lost-person literature, not from intuition, and the
  detection model is the standard SAR formulation. I chose which failure modes
  to simulate; I did not choose how subjects move once displaced.
- The generator is procedural and seeded, and the true location is withheld
  from every arm. Whatever advantage exists is not information leakage.
- The baseline gets the same theory. Conventional Bayesian search is anchored
  at the planning point *because that is what deployed systems do*, not because
  it was handicapped. It solves type C outright at 89%, which is what a
  non-strawman looks like.

What would actually answer it: replaying documented historical searches, where
the incident was authored by the world. That is the right next experiment and
it is not in this repo.

**One terrain.** Every scenario is Mt Hood. Drainage structure, treeline and
canopy all differ elsewhere, and the drainage-following behaviour that the
priors encode matters more in dissected terrain than on open tundra. The
loading on terrain is unmeasured.

**n = 24 carries the thesis.** Type B is the family the argument rests on, and
twenty-four scenarios give intervals wide enough that the find-rate result
(+26.4pp, p = 0.001) is solid while localisation (+23.6pp, p = 0.018) would be
more comfortable at twice the size. Every interval on this page is reported
rather than summarised for that reason.

**The most robust finding is not in the arm tables.** Across 297 nominations
the language model could not rank its own proposals -- 56% concordance where
50% is chance, top pick 4.7 km from truth against 2.2 km for the best pick in
the same response. That measurement survived every configuration change made
since, and it is the reason the system asks for a distribution instead of
prose. If one number here generalises beyond this suite, it is that one.

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

### It recovered a setting a human found by hand

Started from a deliberately naive configuration -- every value a plausible
first guess, all of them wrong -- and asked to improve itself from its own
measurements:

    exhaustion_trigger   0.7 -> 0.5
      objective  +0.023 +/- 0.011 on twelve held-out cases
      find rate  +0 pp
      ACCEPTED

    direction_mass_cutoff  0.98 -> 0.9
      objective  +0.001 +/- 0.003
      REVERTED, below the bar

**0.5 is the value a hand sweep independently landed on**, and the prompt did
not contain it: the model was shown `exhaustion_trigger: currently 0.7, allowed
0.2 to 0.7` and nothing else. An earlier version of this experiment *did* leak
the answer -- the knob description printed the hand-tuned default as "currently"
regardless of what was running, so every naive run was handed the eight values
it existed to rediscover. Those runs were void and were rerun.

**The honest caveat:** 0.5 is a round number inside a 0.2-0.7 range, so
"proposed a round value that happened to be right" is a live alternative to
"reasoned its way to the optimum". What is not in doubt is the acceptance:
the change was measured on cases it was not tuned on and cleared the bar by
about two standard errors.

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

### The one accepted lesson did not survive confirmation

The self-improvement loop accepted exactly one instruction, `lesson-003`, on a
fold that showed +9.9 degrees of bearing improvement. Re-measured on a
confirmation fold it had not been selected on:

    before        70.0 deg
    after         54.7 deg
    gain         +15.3 deg   95% CI [-0.6, +39.2]   p = 0.070
                 4 better, 2 worse, 6 bit-identical   (n = 12)

The point estimate is larger than the selection fold's, and it still does not
hold. **The median gain is zero** -- half the fold was unchanged to the degree,
meaning the instruction did not alter the model's proposal at all. And one
scenario, B011 at 134 degrees to 1 degree, carries the entire mean: remove it
and +15.3 becomes +4.6, under the loop's own 5 degree bar.

That is precisely the "carried by one case" failure the gate's
improved-versus-worsened tie-break exists to catch, and it slipped through
because 4 > 2. The lesson has been withdrawn -- `accepted` is now false, with
the confirmation recorded against it so it is not re-proposed.

So the honest standing of the self-improvement loop is: **the gate provably
distinguishes candidates, and it has not yet accepted an instruction that
survived confirmation.** Those are separate claims and only the first is
established. The configuration change it found (`exhaustion_trigger` 0.7 to
0.5) is a different mechanism and is unaffected -- it was measured on held-out
cases and cleared by about two standard errors.

Lessons are injected only into the prose language-model nomination path. The
calibrated System One arm, which produces every headline figure, never reads
them, so nothing in the results table depends on this.

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
