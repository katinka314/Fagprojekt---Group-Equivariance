"""
NOter:
Generer resultat-plots for channel/parameter x datasize testen (param_datasize_2).

Grid (Fashion-MNIST, n_rings=6 fast, padding, img_size=40):
  datasize (fraction): 0.005, 0.01, 0.03, 0.05, 0.1   (0.5%, 1%, 3%, 5%, 10%)
  channels:  2, 4, 8, 16, 32
  modeller:  CNN, GE_CNN

To grupper af plots:

  Gruppe 1 - fast datasize, accuracy vs ANTAL PARAMETRE (5 plots):
    For hver datasize plottes begge modeller mod deres EGET parametertal
    (ikke channels), saa sammenligningen er pr. parameter. Hver model faar
    sine 5 (eller faerre) channel-punkter placeret ved deres n_params.

  Gruppe 2 - parameter-matchede par, accuracy vs datasize (4 plots):
    Samme channels giver IKKE samme parametertal (GE-CNN har ~3x flere params
    pr. kanal), saa vi sammenligner i stedet par der matcher parametertallet:
      CNN 2ch  vs GE-CNN 2ch
      CNN 4ch  vs GE-CNN 4ch
      CNN 16ch vs GE-CNN 8ch
      CNN 32ch vs GE-CNN 16ch
    Legenden viser channel-tallet for hver model.

Usikkerhed. linje = mean over de 15 seeds, skygge = 95% konfidensinterval for
mean (t-fordeling)
"""

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "reports" / "param_datasize_2"
OUT_DIR = DATA_DIR / "plots"

DATASIZES = [0.005, 0.01, 0.03, 0.05, 0.1]
CHANNELS = [2, 4, 8, 16, 32]
MODELS = ["CNN", "GE_CNN"]

# Parameter-matchede par (CNN channels, GE-CNN channels)
PAIRINGS = [(2, 2), (4, 4), (16, 8), (32, 16)]

# GE-CNN ch32 udelades i param-plottene
PARAM_PLOT_CHANNELS = {"CNN": [2, 4, 8, 16, 32], "GE_CNN": [2, 4, 8, 16]}

COLORS = {"CNN": "tab:orange", "GE_CNN": "tab:blue"}
MARKERS = {"CNN": "s", "GE_CNN": "o"}
LABELS = {"CNN": "CNN", "GE_CNN": "GE-CNN"}


GE_START, GE_END = (0.0, 0.6, 0.1), (0.1, 0.3, 0.85)      # groen -> blaa
CNN_START, CNN_END = (0.95, 0.75, 0.0), (0.75, 0.0, 0.0)  # gul -> roed


def lerp(c1, c2, t):
    return tuple(a + (b - a) * t for a, b in zip(c1, c2))

# t_{0.975} (to-sidet 95%) pr. frihedsgrad; df>30 ~ normalfordeling (1.96).
T_TABLE = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447,
           7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179,
           13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110, 18: 2.101,
           19: 2.093, 20: 2.086, 21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064,
           25: 2.060, 26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042}


def t_crit(df):
    return T_TABLE.get(df, 1.96) if df > 0 else 0.0


def mean_and_ci(values):
    """Returner (mean, halv-bredde af 95% CI for mean)."""
    n = len(values)
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in values) / (n - 1)
    half = t_crit(n - 1) * math.sqrt(var) / math.sqrt(n)
    return m, half


def load_cell(model, ds, ch):
    """Returner (accs, n_params) for en celle, eller None hvis den mangler."""
    path = DATA_DIR / model / f"datasize_{ds}" / f"channels_{ch}" / "metrics.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)
    accs = [s["final_test_acc"] for s in data["per_seed"]]
    return (accs, data["n_params"]) if accs else None


# stats[(model, ds, ch)] = (mean, ci_half, n_seeds, n_params)
stats = {}
for model in MODELS:
    for ds in DATASIZES:
        for ch in CHANNELS:
            cell = load_cell(model, ds, ch)
            if cell is None:
                continue
            accs, n_params = cell
            m, half = mean_and_ci(accs)
            stats[(model, ds, ch)] = (m, half, len(accs), n_params)

OUT_DIR.mkdir(parents=True, exist_ok=True)


def draw(ax, points, color, marker, label, alpha=0.2, band=True, linestyle="-"):
    """points: liste af (x, mean, ci_half), tegnes som linje + (valgfrit) 95% CI-baand."""
    points = sorted(points, key=lambda p: p[0])
    if not points:
        return
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    ax.plot(xs, ys, marker=marker, color=color, label=label, linestyle=linestyle)
    if band:
        lo = [p[1] - p[2] for p in points]
        hi = [p[1] + p[2] for p in points]
        ax.fill_between(xs, lo, hi, color=color, alpha=alpha)


# gruppe 1: fast datasize, accuracy vs antal parametre 
for ds in DATASIZES:
    fig, ax = plt.subplots(figsize=(7, 5))
    for model in MODELS:
        pts = []
        for ch in PARAM_PLOT_CHANNELS[model]:
            key = (model, ds, ch)
            if key in stats:
                m, half, _, n_params = stats[key]
                pts.append((n_params, m, half))
        draw(ax, pts, COLORS[model], MARKERS[model], LABELS[model])
    ax.set_xscale("log")
    ax.set_xlabel("Number of parameters (log scale)")
    ax.set_ylabel("Test accuracy (mean ± 95% CI)")
    ax.set_title(f"Data size = {ds*100:g}%: accuracy vs parameter count")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / f"fixed_datasize_{ds}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"gemt {out}")

#Gruppe 2: parameter-matchede par, accuracy vs datasize
ds_pct = [d * 100 for d in DATASIZES]
for cnn_ch, ge_ch in PAIRINGS:
    fig, ax = plt.subplots(figsize=(7, 5))
    for model, ch in [("CNN", cnn_ch), ("GE_CNN", ge_ch)]:
        pts, n_params = [], None
        for ds, x in zip(DATASIZES, ds_pct):
            key = (model, ds, ch)
            if key in stats:
                m, half, _, n_params = stats[key]
                pts.append((x, m, half))
        label = f"{LABELS[model]} ({ch} ch"
        label += f", {n_params/1000:.0f}k params)" if n_params else " ch)"
        draw(ax, pts, COLORS[model], MARKERS[model], label)
    ax.set_xscale("log")
    ax.set_xticks(ds_pct)
    ax.set_xticklabels([f"{p:g}%" for p in ds_pct])
    ax.set_xlabel("Training set size (% of full set, log scale)")
    ax.set_ylabel("Test accuracy (mean ± 95% CI)")
    ax.set_title(f"Parameter-matched: CNN {cnn_ch}ch vs GE-CNN {ge_ch}ch")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    out = OUT_DIR / f"param_matched_cnn{cnn_ch}_ge{ge_ch}.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"gemt {out}")

# Samlet plot: alle 4 par (8 kurver) i ét, med farver
n_pair = len(PAIRINGS)
fig, ax = plt.subplots(figsize=(9, 6))


def params_of(model, ch):
    for ds in DATASIZES:
        if (model, ds, ch) in stats:
            return stats[(model, ds, ch)][3]
    return None


for i, (cnn_ch, ge_ch) in enumerate(PAIRINGS):
    t = i / (n_pair - 1)
    pts = [(x, stats[("GE_CNN", ds, ge_ch)][0], stats[("GE_CNN", ds, ge_ch)][1])
           for ds, x in zip(DATASIZES, ds_pct) if ("GE_CNN", ds, ge_ch) in stats]
    p = params_of("GE_CNN", ge_ch)
    label = f"GE-CNN ({ge_ch} ch, {p/1000:.0f}k params)"
    draw(ax, pts, lerp(GE_START, GE_END, t), "o", label, band=False)

for i, (cnn_ch, ge_ch) in enumerate(PAIRINGS):
    t = i / (n_pair - 1)
    pts = [(x, stats[("CNN", ds, cnn_ch)][0], stats[("CNN", ds, cnn_ch)][1])
           for ds, x in zip(DATASIZES, ds_pct) if ("CNN", ds, cnn_ch) in stats]
    p = params_of("CNN", cnn_ch)
    label = f"CNN ({cnn_ch} ch, {p/1000:.0f}k params)"
    draw(ax, pts, lerp(CNN_START, CNN_END, t), "s", label, band=False, linestyle="--")

ax.set_xscale("log")
ax.set_xticks(ds_pct)
ax.set_xticklabels([f"{p:g}%" for p in ds_pct])
ax.set_xlabel("Training set size (% of full set, log scale)")
ax.set_ylabel("Test accuracy (mean ± 95% CI)")
ax.set_title("Parameter-matched comparisons: GE-CNN (green→blue) vs CNN (yellow→red)")
ax.grid(True, alpha=0.3)
ax.legend(ncol=2, fontsize=8)
fig.tight_layout()
out = OUT_DIR / "param_matched_combined.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"gemt {out}")

# samlet plot: alle 5 datasizes (10 kurver), accuracy vs params
# Én distinkt farve pr. datasize (delt af begge modeller); model skelnes via en
# linjestil hvor: GE-CNN = fuld linje, CNN = stiplet.
DATASIZE_COLORS = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
fig, ax = plt.subplots(figsize=(9, 6))

for i, ds in enumerate(DATASIZES):
    color = DATASIZE_COLORS[i % len(DATASIZE_COLORS)]
    for model, marker, ls in [("GE_CNN", "o", "-"), ("CNN", "s", "--")]:
        pts = []
        for ch in PARAM_PLOT_CHANNELS[model]:
            key = (model, ds, ch)
            if key in stats:
                m, half, _, n_params = stats[key]
                pts.append((n_params, m, half))
        draw(ax, pts, color, marker, "_nolegend_", band=False, linestyle=ls)

ax.set_xscale("log")
ax.set_xlabel("Number of parameters (log scale)")
ax.set_ylabel("Test accuracy (mean ± 95% CI)")
ax.set_title("Accuracy vs parameter count, by data size (GE-CNN solid, CNN dashed)")
ax.grid(True, alpha=0.3)


order = list(reversed(range(len(DATASIZES))))
color_handles = [Line2D([0], [0], color=DATASIZE_COLORS[i], lw=2) for i in order]
color_labels = [f"{DATASIZES[i]*100:g}%" for i in order]
style_handles = [Line2D([0], [0], color="black", lw=2, linestyle="-", marker="o"),
                 Line2D([0], [0], color="black", lw=2, linestyle="--", marker="s")]
leg1 = ax.legend(color_handles, color_labels, title="Data size", loc="lower right", fontsize=8)
ax.add_artist(leg1)
ax.legend(style_handles, ["GE-CNN", "CNN"], title="Model", loc="upper left", fontsize=8)

fig.tight_layout()
out = OUT_DIR / "fixed_datasize_combined.png"
fig.savefig(out, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"gemt {out}")

print(f"\nFærdig - {len(DATASIZES)} datasize-plots + {len(PAIRINGS)} par-plots + 2 samlede i {OUT_DIR}")
