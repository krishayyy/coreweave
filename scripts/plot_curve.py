"""Render the learning curve.

Two lines on one axis: bearing error as experience accumulates, with memory and
without. The control is the point of the chart -- a single falling line proves
nothing, because the tail of any sequence can happen to be easier.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

GROUND = "#0a0c0f"
TEXT = "#e8eaed"
DIM = "#8b93a1"
RULE = "#1c2128"
WARM = "#ffb020"
COLD = "#7ee7ff"


def running_mean(xs: list[float], window: int = 5) -> np.ndarray:
    """Trailing mean: each point is that case and the few before it."""
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        out.append(float(np.mean(xs[lo:i + 1])))
    return np.array(out)


def series(rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    pts = [(r["i"], r["bearing_error"]) for r in rows if r["bearing_error"] is not None]
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def main() -> int:
    path = ROOT / (sys.argv[1] if len(sys.argv) > 1 else "runs/learning_curve.json")
    data = json.loads(path.read_text())

    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.patch.set_facecolor(GROUND)
    ax.set_facecolor(GROUND)

    for rows, colour, label in (
        (data["control"], DIM, "without memory  (control)"),
        (data["with_memory"], WARM, "with memory  (precedent retrieved)"),
    ):
        x, y = series(rows)
        ax.plot(x, y, "o", ms=4, color=colour, alpha=0.35)
        ax.plot(x, running_mean(list(y)), "-", lw=2.2, color=colour, label=label)

    ax.set_xlabel("case number  (experience accumulates left to right)",
                  color=DIM, fontsize=10, family="monospace")
    ax.set_ylabel("bearing error of the best account proposed  (degrees)",
                  color=DIM, fontsize=10, family="monospace")
    ax.tick_params(colors=DIM, labelsize=9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color(RULE)
    ax.grid(True, color=RULE, lw=0.6, alpha=0.7)
    ax.set_axisbelow(True)

    legend = ax.legend(frameon=False, fontsize=10, loc="upper right")
    for text in legend.get_texts():
        text.set_color(TEXT)
        text.set_family("monospace")

    ax.set_title("Does it get better with experience?",
                 color=TEXT, fontsize=13, family="monospace", loc="left", pad=14)

    out = ROOT / "runs" / "learning_curve.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
