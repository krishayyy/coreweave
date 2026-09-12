# Display

Canvas2D, no dependencies. A live demo should not need a CDN to be reachable,
and every transition here is hand-controlled.

The rule the renderer follows is that **nothing ever pops**. Belief fields
cross-fade between periods, hypothesis weights ease toward their targets rather
than snapping, and the aircraft is interpolated between samples. Discrete
updates are the single thing that most makes software read as cheap.

    python3 -m http.server 5173 --directory web

Frames come from `scripts/export_frames.py`, which writes belief and coverage
masks as 8-bit PNGs plus a `run.json` timeline.

## Notes on the visual choices

- **Terrain is re-tinted at load** into a cool dark ramp. Raw greyscale hillshade
  competes with the belief field and makes the page read as a hiking map rather
  than an instrument.
- **The belief palette is transparent across its bottom third.** Probability mass
  has a very long tail; painting the tail turns the whole map into a uniform glow
  and the field stops reading as a concentration.
- **Only the aircraft's recent trail is drawn.** Stroking the full serpentine at
  once renders as scan lines across the map instead of something flying.
- **Hypothesis bars are scaled against the leader**, not against 1.0 -- posteriors
  sit well below 0.5 and an absolute scale makes every bar a stub.
