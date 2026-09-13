# Data sources

Everything the experiment reads, and where it comes from. Nothing here is
proprietary and nothing requires a key.

## 1. Terrain — real

Elevation comes from the AWS Terrarium public dataset, a global DEM served as
slippy tiles:

    https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png

The headline suite uses zoom 12 around **45.3736, -121.6960** — Mount Hood,
Oregon — which gives a 21 km box at roughly 107 m per cell. The county
experiment additionally loads Mount Whitney (36.5785, -118.2923) and Mount
Mitchell (35.7649, -82.2651).

Drainage networks are **not** downloaded. They are computed from that elevation
with a D8 flow-accumulation pass (`src/searchloop/grid.py`), so the watercourses
in the display are derived from the real surface rather than drawn.

Aerial imagery for the display is the USGS National Map, aligned to the same
tile scheme:

    https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}

Both are public domain and key-free. Tiles are cached under `data/tiles` and
`data/imagery`.

## 2. Lost-person behaviour — published statistics, approximated

Search planning uses distance-from-planning-point quantiles by subject
category, from Koester's lost-person behaviour work and the ISRID incident
database (~145,000 recorded searches). The seven profiles used here carry a
median (d50) and 95th percentile (d95) distance in kilometres:

    key            label                               d50    d95   base rate
    hiker          Lost hiker, down-drainage           1.9   11.0     0.40
    hiker_route    Lost hiker, holding route           1.1    5.0     0.18
    hunter         Hunter / off-trail traveller        2.6   14.0     0.08
    despondent     Despondent subject                  1.2    6.0     0.10
    dementia       Dementia / cognitive impairment     0.8    3.5     0.06
    child          Child, 7-12                         0.9    4.0     0.05
    angler         Angler / water-bound                1.0    5.5     0.04

**These are approximations of the published mountainous-terrain ring models,
not a transcription of a specific ISRID table.** ISRID itself is not publicly
downloadable; it is held by its maintainer and contains records of real missing
people. The terrain affinities alongside them (drainage, downhill, high ground,
slope aversion) are our encoding of the qualitative behaviour each category is
documented to show, and are not published constants.

Source: Robert J. Koester, *Lost Person Behavior: A Search and Rescue Guide on
Where to Look — for Land, Air and Water*.

## 3. Scenarios — generated, and checked against the above

The 156 cases are generated procedurally from those distributions
(`src/searchloop/scenario.py`), not written by a model. A model that authors
the scenario and a model that solves it share a prior, and any apparent skill
could then be collusion rather than inference. Procedural generation makes that
leakage structurally impossible.

Generation is verified against the published quantiles:

    python scripts/validate_priors.py

It prints generated d50/d95 per profile against the published values. This
check is what caught a missing Jacobian that was placing subjects 2.7x too far
from their planning point; the largest median deviation fell from 233% to 31%.

## 4. Why there is no dataset of real searches

The experiment needs the *same case searched twice*, once by each method, with
the answer withheld from both. Real incidents give one outcome and no
counterfactual: you learn what happened, never what would have happened. No
such dataset exists and none can, which is why this runs in simulation against
a control arm rather than as a demonstration reel.

## 5. What is NOT used, and why

Drone imagery datasets such as SARD teach a model to spot a person in a frame.
That is detection. This project is about where to point the camera, and the
sensor here is a parameter — probability of detection from effective sweep
width, degraded under canopy and on slope — held identical across every
experimental arm so the comparison is never confounded by it. A real detector
would improve every arm equally and would not change the reported difference.

## 6. Outputs

    runs/experiment_large.json     156 scenarios x 3 arms x 3 repeats, the headline
    runs/county_learning.json      the per-county learning experiment (null result)
    runs/confirm_lesson003.log     the self-improvement lesson that failed confirmation
    web/public/run/evidence.json   what the Evidence view renders, generated from the above

Weave traces: https://wandb.ai/krishaysuresh/searchloop
