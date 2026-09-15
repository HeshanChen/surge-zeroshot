"""Publication figure style (nature-figure skill python backend rules + Okabe-Ito palette).
Usage: from pubstyle import apply, save_pub, PAL, MM"""
import os as _os
_ROOT = _os.environ.get('SURGE_ROOT') or _os.path.abspath(_os.path.join(_os.path.dirname(_os.path.abspath(__file__)), '..', '..'))
import matplotlib as mpl

def apply():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none", "pdf.fonttype": 42,
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.spines.right": False, "axes.spines.top": False,
        "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "legend.frameon": False, "figure.dpi": 150,
    })

MM = 1/25.4                       # mm -> inch
W1, W2 = 89*MM, 183*MM            # Nature single / double column widths

PAL = {                           # Okabe-Ito (colorblind-safe)
    "blue": "#0072B2", "sky": "#56B4E9", "orange": "#E69F00",
    "verm": "#D55E00", "green": "#009E73", "purple": "#CC79A7",
    "yellow": "#F0E442", "grey": "#999999", "black": "#000000",
}

def panel_label(ax, letter, dx=-0.12, dy=1.06):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

def save_pub(fig, path_noext, dpi=600):
    fig.savefig(f"{path_noext}.pdf", bbox_inches="tight")
    fig.savefig(f"{path_noext}.png", dpi=dpi, bbox_inches="tight")
