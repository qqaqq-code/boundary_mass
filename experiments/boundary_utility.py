"Plot the relationship between boundary mass and downstream utility. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, FIGURES_DIR, result_file

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

STUDY_ROOT = PROJECT_ROOT
df = pd.read_csv(
    result_file("boundary_utility/boundary_utility_epochs.csv")
)
df["dataset"] = np.where(
    df["run_path"].str.lower().str.contains("tiny"), "Tiny-ImageNet", "CIFAR-100"
)
df["spc"] = pd.to_numeric(df["train_samples_per_class"], errors="coerce")
df["mode"] = df["mode"].fillna("unknown")
df = df.dropna(subset=["utility", "log1p_psieff", "heldout_psieff", "spc"])
final = df.sort_values("epoch").groupby("run_path", as_index=False).tail(1).copy()
final["boundary_efficiency"] = final["utility"] / final["log1p_psieff"].clip(lower=1e-9)


def corr(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


within = []
for (dataset, spc), g in final.groupby(["dataset", "spc"]):
    within.append(
        {
            "dataset": dataset,
            "spc": int(spc),
            "n": int(len(g)),
            "corr_logpsi_utility": corr(g["log1p_psieff"], g["utility"]),
        }
    )

slopes = []
for run, g in df.groupby("run_path"):
    g = g.sort_values("epoch")
    if len(g) < 5 or g["log1p_psieff"].std() == 0:
        continue
    slope = float(np.polyfit(g["log1p_psieff"], g["utility"], 1)[0])
    slopes.append(
        {
            "run_path": run,
            "mode": g["mode"].iloc[0],
            "spc": int(g["spc"].iloc[0]),
            "dataset": g["dataset"].iloc[0],
            "slope": slope,
        }
    )
slopes_df = pd.DataFrame(slopes)

mode_eff = (
    final.groupby("mode")["boundary_efficiency"]
    .agg(["mean", "std", "count"])
    .reset_index()
)
mode_slope = (
    slopes_df.groupby("mode")["slope"].agg(["mean", "std", "count"]).reset_index()
    if len(slopes_df)
    else pd.DataFrame()
)

fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.7))
colors = {
    "standard": "#1f77b4",
    "weight_decay": "#ff7f0e",
    "psieff": "#2ca02c",
    "unknown": "#7f7f7f",
}
for mode, g in df.groupby("mode"):
    for _, run_g in list(g.groupby("run_path"))[:35]:
        axes[0].plot(
            run_g["log1p_psieff"],
            run_g["utility"],
            color=colors.get(mode, "#777777"),
            alpha=0.10,
            lw=0.8,
        )
for mode, g in final.groupby("mode"):
    axes[1].scatter(
        g["log1p_psieff"],
        g["utility"],
        s=18,
        alpha=0.55,
        label=mode,
        color=colors.get(mode, "#777777"),
    )
if not mode_eff.empty:
    axes[2].bar(
        mode_eff["mode"],
        mode_eff["mean"],
        color=[colors.get(m, "#777777") for m in mode_eff["mode"]],
    )
axes[0].set_title("training trajectories")
axes[1].set_title("final checkpoints by intervention")
axes[2].set_title("BoundaryEfficiency")
for ax in axes[:2]:
    ax.set_xlabel("log(1 + psi_eff)")
    ax.set_ylabel("utility")
axes[2].set_ylabel("utility / log(1 + psi_eff)")
axes[1].legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "fig10_boundary_utility.png", dpi=220)
plt.close(fig)

summary = {
    "num_epoch_rows": int(len(df)),
    "num_runs": int(final["run_path"].nunique()),
    "mixed_final_corr_logpsi_utility": corr(final["log1p_psieff"], final["utility"]),
    "within_spc_correlations": within,
    "mode_boundary_efficiency": mode_eff.to_dict(orient="records"),
    "mode_trajectory_slope": mode_slope.to_dict(orient="records")
    if len(mode_slope)
    else [],
}
(result_file("boundary_utility/fig10_grouped_analysis.json")).write_text(
    json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
)

valid_corrs = [
    r["corr_logpsi_utility"] for r in within if r["corr_logpsi_utility"] is not None
]
median_corr = float(np.median(valid_corrs)) if valid_corrs else None
best_eff = (
    mode_eff.sort_values("mean", ascending=False).iloc[0]["mode"]
    if not mode_eff.empty
    else "unavailable"
)
text = (
    f"Grouped analysis covers {len(df):,} epoch records and {final['run_path'].nunique()} runs. "
    f"The mixed final correlation is {summary['mixed_final_corr_logpsi_utility']:.3f}; "
    f"within-regime median correlation is {median_corr:.3f} when computable. "
    f"The highest mean BoundaryEfficiency among recovered interventions is {best_eff}. "
    "Thus Figure 10 is used as a state-plane diagnostic rather than as an optimality theorem."
)
(result_file("boundary_utility/fig10_interpretation.tex")).write_text(
    text, encoding="utf-8"
)
print(json.dumps(summary, indent=2, sort_keys=True))
