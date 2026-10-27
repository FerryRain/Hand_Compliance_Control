"""Shared, headless-safe plotting helpers for the module demos."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path


COLORS = {
  "blue": "#2878B5",
  "cyan": "#45B8AC",
  "green": "#43AA8B",
  "orange": "#F8961E",
  "pink": "#F15A7A",
  "red": "#D1495B",
  "purple": "#7B61A8",
  "navy": "#17324D",
  "gray": "#6B7280",
  "light": "#EEF3F7",
}


def get_pyplot():
  """Return pyplot configured for reproducible rendering without a display."""

  cache = Path(tempfile.gettempdir()) / "handcomp-matplotlib"
  cache.mkdir(parents=True, exist_ok=True)
  os.environ.setdefault("MPLCONFIGDIR", str(cache))
  import matplotlib

  matplotlib.use("Agg")
  import matplotlib.pyplot as plt

  plt.rcParams.update(
    {
      "axes.facecolor": "#F8FAFC",
      "axes.edgecolor": "#AAB6C2",
      "axes.grid": True,
      "axes.labelcolor": COLORS["navy"],
      "axes.titlecolor": COLORS["navy"],
      "figure.facecolor": "white",
      "font.size": 10,
      "grid.alpha": 0.25,
      "grid.linestyle": "--",
      "legend.frameon": False,
      "savefig.facecolor": "white",
      "text.color": COLORS["navy"],
      "xtick.color": COLORS["gray"],
      "ytick.color": COLORS["gray"],
    }
  )
  return plt


def save_figure(figure, output_path: Path) -> Path:
  output = Path(output_path)
  output.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output, dpi=160, bbox_inches="tight")
  return output
