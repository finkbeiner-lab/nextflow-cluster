"""Shared figure style for the evidence document (static PNG, light surface).

Colours come from the validated reference palette (dataviz skill):
  roles   - live #2a78d6 (blue), dead #eb6834 (orange), S.A. #4a3aa7 (violet),
            untreated = muted ink, ambiguous band = gridline grey
  lines   - fixed order 14C, 13I, 09B, 63D -> slots 1-4 (aqua/yellow are < 3:1 on
            the surface, so cross-line figures always carry direct labels or a table)
"""

from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

LIVE = "#2a78d6"
DEAD = "#eb6834"
SA = "#4a3aa7"
UT = "#898781"
AMBIG = "#e1e0d9"
LOWTAIL = "#c3c2b7"

LINE_ORDER = ["CTR-33114.C", "CTR-33113.2I", "XDP-33109.2B", "XDP-33363.D"]
LINE_SHORT = {"CTR-33114.C": "14C", "CTR-33113.2I": "13I", "XDP-33109.2B": "09B", "XDP-33363.D": "63D"}
LINE_COLOR = {"CTR-33114.C": "#2a78d6", "CTR-33113.2I": "#eb6834", "XDP-33109.2B": "#1baf7a",
              "XDP-33363.D": "#eda100"}
EXP_ORDER = ["XDP1", "XDP3", "XDP5", "XDP8", "XDP10", "XDP11"]

RATIO_TICKS = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]


def setup() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.family": ["DejaVu Sans"], "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold",
        "axes.labelsize": 9, "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "legend.fontsize": 8, "lines.linewidth": 2.0, "text.color": INK,
    })


def log_ratio_axis(ax, axis: str = "x", ticks: Iterable[float] = RATIO_TICKS) -> None:
    """Data are log10(ratio); label the ticks in linear ratio (what Julia reads)."""
    locs = [np.log10(t) for t in ticks]
    fmt = FuncFormatter(lambda v, _: f"{10 ** v:g}")
    target = ax.xaxis if axis == "x" else ax.yaxis
    target.set_major_locator(FixedLocator(locs))
    target.set_major_formatter(fmt)
    target.set_minor_formatter(NullFormatter())


def save(fig, path: Path, dpi: int = 110) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    try:  # palette-quantise to keep the self-contained HTML small
        from PIL import Image
        im = Image.open(path).convert("RGB").quantize(colors=128, method=Image.MEDIANCUT)
        im.save(path, optimize=True)
    except Exception:  # noqa: BLE001 - quantisation is an optimisation only
        pass
    return path


setup()


OR_TICKS = [0.125, 0.25, 0.5, 1, 2, 4, 8]


def or_axis(ax, axis: str = "y") -> None:
    """Log-scaled odds-ratio axis labelled in plain values (0.25, 0.5, 1, 2, 4)."""
    target = ax.yaxis if axis == "y" else ax.xaxis
    target.set_major_locator(FixedLocator(OR_TICKS))
    target.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    target.set_minor_formatter(NullFormatter())
