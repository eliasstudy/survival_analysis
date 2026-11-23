"""Creates figures/dataset_overview.pdf (dataset analysis figure of the paper) and prints the
cohort statistics used in Table 2.

The cohorts and folds are read from the dataset_splits.csv files written by the baseline notebooks
(identical for all models), so the figure describes exactly the patients that were evaluated.

Usage (from self_paced_research/Paper/figures):
    python make_dataset_figure.py
Requires: numpy, pandas, matplotlib, h5py
"""
import glob
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]                                   # .../self_paced_research
DATA = PROJECT.parent / "Datensatz" / "data-transfer-student"
COHORTS = ["TCGA-GBMLGG", "TCGA-KIRC", "TCGA-KIRP"]
LABEL = {c: c.replace("TCGA-", "") for c in COHORTS}

# validated categorical palette (first three slots, all-pairs CVD-safe) + secondary encoding
COLOR = {"TCGA-GBMLGG": "#2a78d6", "TCGA-KIRC": "#eb6834", "TCGA-KIRP": "#1baf7a"}
LSTYLE = {"TCGA-GBMLGG": "-", "TCGA-KIRC": "--", "TCGA-KIRP": "-."}
MARKER = {"TCGA-GBMLGG": "o", "TCGA-KIRC": "s", "TCGA-KIRP": "^"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
EVENT_C, CENS_C = "#1c5cab", "#86b6ef"                      # ordinal blue ramp (validated)

SUBTYPE_NAME = {"GBM": "GBM", "AASTR": "AASTR", "AOAST": "AOAST", "OAST": "OAST",
                "ODG": "ODG", "ASTR": "ASTR"}


def load_cohort(cohort: str) -> pd.DataFrame:
    """Cohort + folds as used for training (latest MCAT run), merged with clinical variables."""
    runs = glob.glob(str(PROJECT / "MCAT" / "runs" / f"{cohort}_mcat_*"))
    run = max(runs, key=lambda d: re.search(r"(\d{8}_\d{6})$", d).group(1))
    df = pd.read_csv(Path(run) / "dataset_splits.csv")
    clin = pd.read_csv(DATA / cohort / "clinical" / "clinical_survival_per_slide.csv")
    clin["slide_id"] = clin["slide_id"].str.replace(".svs", "", regex=False)
    return df.merge(clin[["slide_id", "age", "is_female", "oncotree_code"]], on="slide_id", how="left")


def kaplan_meier(time: np.ndarray, event: np.ndarray):
    """Product-limit estimator; returns step-function knots (t, S(t))."""
    t_out, s_out, s = [0.0], [1.0], 1.0
    for u in np.unique(time[event == 1]):
        at_risk = (time >= u).sum()
        deaths = ((time == u) & (event == 1)).sum()
        s *= 1.0 - deaths / at_risk
        t_out.append(u)
        s_out.append(s)
    return np.array(t_out), np.array(s_out)


def surv_at(t_knots, s_knots, t):
    return s_knots[np.searchsorted(t_knots, t, side="right") - 1]


def reverse_km_median(time, event):
    t, s = kaplan_meier(time, 1 - event)                     # censoring treated as the "event"
    below = np.where(s <= 0.5)[0]
    return t[below[0]] if len(below) else np.nan


def main():
    data = {c: load_cohort(c) for c in COHORTS}

    # ---------------- statistics for Table 2 ----------------
    for c, df in data.items():
        n, ev = len(df), int(df.event.sum())
        site = df.groupby("site").agg(n=("event", "size"), ev=("event", "sum"))
        site = site[site.n >= 10]
        folds = df.groupby("fold").event.agg(["size", "sum"])
        wsi = DATA / c / "wsi_features" / "conch_v15"
        n_patch = np.array([h5py.File(wsi / f"{s}.h5", "r")["features"].shape[0] for s in df.slide_id])
        print(f"{c}: n={n}, events={ev}, censored={1 - ev / n:.1%}, "
              f"median follow-up (reverse KM)={reverse_km_median(df.survival_months.values, df.event.values):.1f} m, "
              f"sites={df.site.nunique()}, site event rate (n>=10)={(site.ev / site.n).min():.0%}-{(site.ev / site.n).max():.0%}, "
              f"events/fold={int(folds['sum'].min())}-{int(folds['sum'].max())}, "
              f"patches/WSI median={np.median(n_patch):.0f} [{n_patch.min()}, {n_patch.max()}]")

    # ---------------- figure ----------------
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral"],
        "mathtext.fontset": "stix", "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.2,
        "axes.edgecolor": "#898781", "axes.linewidth": 0.5, "xtick.color": INK2, "ytick.color": INK2,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "xtick.major.size": 2, "ytick.major.size": 2,
        "axes.labelcolor": INK, "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    # Column-width figure (3.25 in) with the two panels the dataset critique refers to.
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.72), gridspec_kw={"width_ratios": [1.0, 1.05]})
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=GRID, linewidth=0.4)
        ax.set_axisbelow(True)

    # (a) event rate per tissue source site (sites with >= 10 patients)
    ax = axes[0]
    rng = np.random.default_rng(0)
    for i, (c, df) in enumerate(data.items()):
        site = df.groupby("site").agg(n=("event", "size"), ev=("event", "sum"))
        site = site[site.n >= 10]
        rate = site.ev / site.n
        x = i + rng.uniform(-0.18, 0.18, len(site))
        ax.scatter(x, 100 * rate, s=2.2 * site.n / 4, marker=MARKER[c], color=COLOR[c],
                   alpha=0.85, edgecolors="white", linewidths=0.4, zorder=3)
        ax.hlines(100 * df.event.mean(), i - 0.32, i + 0.32, color=INK2, linewidth=0.8, zorder=2)
        if c == "TCGA-KIRC":
            b0 = site.index.get_loc("B0")
            ax.annotate("B0 = fold 0", xy=(x[b0], 100 * rate.iloc[b0]), xytext=(i + 0.05, 74),
                        fontsize=5.6, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.4))
    ax.set_xticks(range(len(COHORTS)), [LABEL[c] for c in COHORTS])
    ax.tick_params(axis="x", labelsize=6.0, pad=1.5)
    ax.set_xlim(-0.55, 2.55)
    ax.set_ylim(-4, 104)
    ax.set_ylabel("Event rate per site [%]")
    ax.set_title("(a) Event rate per site", loc="left")

    # (b) GBMLGG: patients and events per histological subtype
    ax = axes[1]
    g = data["TCGA-GBMLGG"].groupby("oncotree_code").agg(n=("event", "size"), ev=("event", "sum"))
    g["rate"] = g.ev / g.n
    g = g.sort_values("rate")
    xs = np.arange(len(g))
    ax.bar(xs, g.ev, width=0.62, color=EVENT_C, edgecolor="white", linewidth=0.6, label="Events")
    ax.bar(xs, g.n - g.ev, bottom=g.ev, width=0.62, color=CENS_C, edgecolor="white", linewidth=0.6,
           label="Censored")
    for xi, (_, r) in zip(xs, g.iterrows()):
        ax.text(xi, r.n + 2, f"{r.rate:.0%}", ha="center", va="bottom", fontsize=5.4, color=INK2)
    ax.set_xticks(xs, [SUBTYPE_NAME[k] for k in g.index], rotation=35, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", labelsize=5.6, pad=1.2)
    ax.set_ylim(0, 178)
    ax.set_yticks([0, 50, 100, 150])
    ax.set_ylabel("Patients")
    ax.set_title("(b) GBMLGG subtypes", loc="left")
    ax.legend(loc="upper left", ncol=2, frameon=False, borderaxespad=0.1, columnspacing=0.8,
              handlelength=1.0, handletextpad=0.3, fontsize=5.8)

    fig.tight_layout(pad=0.25, w_pad=0.8)
    out = HERE / "dataset_overview.pdf"
    fig.savefig(out)
    print("saved", out)


if __name__ == "__main__":
    main()
