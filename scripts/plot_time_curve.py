"""Plot find rate against search budget."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
GROUND, TEXT, DIM, RULE = "#0a0c0f", "#e8eaed", "#8b93a1", "#1c2128"
WARM, COLD, FAINT = "#ffb020", "#7ee7ff", "#565e6b"

SERIES = [("none", "conventional search", DIM, "-"),
          ("jev", "this system", WARM, "-"),
          ("heuristic", "oracle — told the true answer", COLD, "--")]


def main() -> int:
    curve = json.loads((ROOT / (sys.argv[1] if len(sys.argv) > 1
                                else "runs/time_curve.json")).read_text())
    fig, ax = plt.subplots(figsize=(10, 5.4))
    fig.patch.set_facecolor(GROUND); ax.set_facecolor(GROUND)

    for key, label, colour, style in SERIES:
        pts = curve[key]
        x = [p["days"] for p in pts]
        y = [100 * p["rate"] for p in pts]
        ax.plot(x, y, style, lw=2.4 if key == "jev" else 1.8, color=colour,
                label=label, marker="o", ms=4)

    ours8 = 100 * next(p["rate"] for p in curve["jev"] if p["periods"] == 16)
    conv = curve["none"]
    matched = next((p for p in conv if 100 * p["rate"] >= ours8), None)
    if matched:
        ax.plot([8, matched["days"]], [ours8, ours8], ":", lw=1.4, color="#ff5a52")
        ax.annotate(f"{matched['days'] - 8:.0f} days sooner",
                    xy=((8 + matched["days"]) / 2, ours8), xytext=(0, 9),
                    textcoords="offset points", ha="center",
                    color="#ff5a52", fontsize=10, family="monospace")

    ax.set_xlabel("search budget  (days, at 12-hour operational periods)",
                  color=DIM, fontsize=10, family="monospace")
    ax.set_ylabel("subject located  (%)", color=DIM, fontsize=10, family="monospace")
    ax.tick_params(colors=DIM, labelsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("bottom", "left"):
        ax.spines[s].set_color(RULE)
    ax.grid(True, color=RULE, lw=0.6, alpha=0.7); ax.set_axisbelow(True)
    ax.set_ylim(0, 100)

    lg = ax.legend(frameon=False, fontsize=10, loc="lower right")
    for t in lg.get_texts():
        t.set_color(TEXT); t.set_family("monospace")
    ax.set_title("Everyone finds them eventually. The question is how long.",
                 color=TEXT, fontsize=13, family="monospace", loc="left", pad=14)

    out = ROOT / "runs" / "time_curve.png"
    fig.tight_layout(); fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
