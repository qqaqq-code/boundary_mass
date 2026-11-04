"Plot normal/tangent geometry and backbone-selection results. See docs/EXPERIMENTS.md for usage."

from utils.paths import RESULTS_DIR, FIGURES_DIR, result_file
from pathlib import Path
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ART = RESULTS_DIR
FIG = FIGURES_DIR
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.6,
        "figure.dpi": 180,
        "savefig.dpi": 260,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.04,
    }
)

SCALE_COLORS = {
    0.5: "#8C510A",
    1.0: "#01665E",
    2.0: "#4393C3",
    4.0: "#5E3C99",
}
BACKBONE_COLORS = {
    "R50": "#E69F00",
    "ViT-B": "#CC79A7",
    "DINO-MS": "#0072B2",
    "SigLIP-MS": "#009E73",
}


def prettify(ax):
    ax.grid(True, axis="both")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def controlled_real_feature():
    curves = pd.read_csv(
        result_file(
            "normal_tangent/controlled_real_feature_learning_curves.csv"
        )
    )
    diff = pd.read_csv(
        result_file(
            "normal_tangent/controlled_real_feature_sample_difficulty.csv"
        )
    )

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0), constrained_layout=True)

    ax = axes[0]
    grouped = curves.groupby(["normal_scale", "spc"], as_index=False)["acc"].mean()
    for scale, g in grouped.groupby("normal_scale"):
        ax.plot(
            g["spc"],
            g["acc"] * 100,
            marker="o",
            linewidth=1.8,
            markersize=4,
            color=SCALE_COLORS.get(float(scale), None),
            label=f"s={scale:g}",
        )
    ax.set_xscale("log")
    ax.set_xlabel("labels per class")
    ax.set_ylabel("test accuracy (%)")
    ax.set_title("Low-shot learning curves")
    ax.legend(frameon=False, ncol=2, handlelength=1.4, columnspacing=0.8)
    prettify(ax)

    ax = axes[1]
    for scale, g in diff.groupby("normal_scale"):
        ax.scatter(
            g["log1p_psi_eff"],
            g["aulc"],
            s=34,
            alpha=0.84,
            edgecolor="white",
            linewidth=0.35,
            color=SCALE_COLORS.get(float(scale), None),
            label=f"s={scale:g}",
        )
    ax.set_xlabel(r"$\log(1+\hat\psi_{\rm eff})$")
    ax.set_ylabel("AULC")
    ax.set_title("Boundary mass vs. AULC")
    prettify(ax)

    ax = axes[2]
    for scale, g in diff.groupby("normal_scale"):
        ax.scatter(
            g["log1p_psi_eff"],
            g["label_budget_to_90pct_full"],
            s=34,
            alpha=0.84,
            edgecolor="white",
            linewidth=0.35,
            color=SCALE_COLORS.get(float(scale), None),
        )
    ax.set_xlabel(r"$\log(1+\hat\psi_{\rm eff})$")
    ax.set_ylabel("labels to target")
    ax.set_title("Boundary mass vs. label budget")
    prettify(ax)

    fig.savefig(FIG / "fig04a_normal_scaling.png")


def direction_controls():
    df = pd.read_csv(
        result_file("normal_tangent/controlled_direction_sample_difficulty.csv")
    )
    mean = (
        df.groupby(["direction", "scale"], as_index=False)
        .agg(
            logpsi=("log1p_psi_eff", "mean"),
            budget=("label_budget_to_90pct_full", "mean"),
            aulc=("aulc", "mean"),
        )
        .sort_values(["direction", "scale"])
    )
    colors = {"normal": "#0072B2", "tangent": "#D55E00"}

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 2.65), constrained_layout=True)
    for ax, y, title, ylabel in [
        (axes[0], "logpsi", "Tube coefficient", r"$\log(1+\hat\psi_{\rm eff})$"),
        (axes[1], "aulc", "Low-shot AULC", "AULC"),
        (axes[2], "budget", "Target label budget", "labels to target"),
    ]:
        for direction, g in mean.groupby("direction"):
            ax.plot(
                g["scale"],
                g[y],
                marker="o",
                linewidth=2,
                markersize=4.5,
                color=colors[direction],
                label=direction,
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks([0.5, 1, 2, 4])
        ax.set_xticklabels(["0.5", "1", "2", "4"])
        ax.set_xlabel("scale")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        prettify(ax)
    axes[0].legend(frameon=False, loc="best")
    fig.savefig(FIG / "fig04b_tangent_control.png")


def backbone_experiment():
    df = pd.read_csv(
        result_file(
            "backbone_selection/strict_label_budget_table.csv"
        )
    )
    stats = json.load(
        open(
            result_file(
                "transferability/transferability_baseline_stats.json"
            )
        )
    )
    inc = pd.DataFrame(stats["incremental"])
    # These archived rows also appear in table11_transferability_controlled_regression.tex.
    labels = {
        "LogME + H-score": "LogME + H-score",
        "LogME + H-score + TransRate": "+ TransRate",
        "feature transferability set": "feature transferability set",
        "all non-$\\psi$ + transferability": "full diagnostic set",
    }
    inc = inc.set_index("base").loc[list(labels)].reset_index()
    fig, axes = plt.subplots(
        1, 2, figsize=(7.25, 2.65), width_ratios=[1, 1.02], constrained_layout=True
    )

    ax = axes[0]
    for backbone, g in df.groupby("backbone"):
        ax.scatter(
            g["log1p_psi"],
            g["label_budget"],
            s=28,
            alpha=0.78,
            edgecolor="white",
            linewidth=0.35,
            color=BACKBONE_COLORS.get(backbone, None),
            label=backbone,
        )
    ax.set_xlabel(r"held-out train $\log(1+\hat\psi_{\rm eff})$")
    ax.set_ylabel("test label budget (spc)")
    ax.set_title("Boundary mass and label demand")
    ax.legend(
        frameon=False,
        ncol=2,
        handlelength=1.0,
        columnspacing=0.6,
        loc="upper left",
        fontsize=7,
    )
    prettify(ax)

    ax = axes[1]
    inc = inc.assign(label=inc["base"].map(labels))
    y = np.arange(len(inc))
    x = inc["delta_r2"].to_numpy()
    lo = x - inc["delta_r2_ci_low"].to_numpy()
    hi = inc["delta_r2_ci_high"].to_numpy() - x
    ax.barh(y, x, color=["#4C78A8"] * 3 + ["#C65D26"], alpha=0.92, height=0.6)
    ax.errorbar(
        x, y, xerr=[lo, hi], fmt="none", ecolor="#222222", elinewidth=1, capsize=2
    )
    ax.set_yticks(y)
    ax.set_yticklabels(inc["label"])
    ax.invert_yaxis()
    ax.set_xlim(0, 0.165)
    ax.set_xticks([0, 0.05, 0.10, 0.15])
    for row, value in zip(y, x):
        ax.text(value + 0.003, row + 0.26, f"{value:.3f}", va="center", fontsize=8)
    ax.set_xlabel(r"incremental $\Delta R^2$ from $\hat\psi$ (95% CI)")
    ax.set_title("Transferability controls")
    prettify(ax)

    fig.savefig(FIG / "fig05_backbone_selection.png")
    fig.savefig(FIG / "fig05_backbone_selection.pdf")
    plt.close(fig)


if __name__ == "__main__":
    controlled_real_feature()
    direction_controls()
    backbone_experiment()
