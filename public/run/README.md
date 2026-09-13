# Generated display assets

Everything here is produced by `scripts/export_demo.py` and committed on purpose:
the demo then runs from a clean clone with no API key, which is what a judging
table needs.

- `imagery.jpg` — USGS National Map aerial imagery (public domain), covering
  exactly the same ground as the elevation tiles. A basemap, not a sensor feed.
- `fields/*.png` — belief and coverage masks, one pair per operational period
  per arm, 8-bit greyscale and coloured in the browser.
- `demo.json` — the timeline: cases, evidence, nominations, and how highly each
  arm ranked the subject's actual location at every period.

To regenerate:

    python scripts/export_demo.py --cases 5,10,13,1,18,20,19,4
