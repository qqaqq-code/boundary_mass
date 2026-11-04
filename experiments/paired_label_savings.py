"Summarize paired label savings from pilot-label selection. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, RESULTS_DIR, result_file

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
ART = RESULTS_DIR
N_BOOT = 5000
RNG = np.random.default_rng(20260503)


def ci(vals):
    arr = np.asarray(vals, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> None:
    R = pd.read_csv(result_file("pilot_selection/pilot_policy_selection.csv"))
    rows = []
    for spc in sorted(R.pilot_spc.unique()):
        base = R[(R.pilot_spc == spc) & (R.policy == "combined non-$\\psi$")].set_index(
            "pair_id"
        )
        plus = R[
            (R.pilot_spc == spc) & (R.policy == "combined non-$\\psi$ + $\\psi$")
        ].set_index("pair_id")
        idx = np.array(sorted(set(base.index) & set(plus.index)))
        diff = (
            base.loc[idx, "chosen_budget"].to_numpy()
            - plus.loc[idx, "chosen_budget"].to_numpy()
        )
        rel = diff / np.maximum(base.loc[idx, "chosen_budget"].to_numpy(), 1e-9)
        win = (diff > 1e-9).astype(float)
        boot_win, boot_diff, boot_rel = [], [], []
        for _ in range(N_BOOT):
            sample = RNG.choice(np.arange(len(idx)), size=len(idx), replace=True)
            boot_win.append(float(win[sample].mean()))
            boot_diff.append(float(diff[sample].mean()))
            boot_rel.append(float(rel[sample].mean()))
        win_ci = ci(boot_win)
        diff_ci = ci(boot_diff)
        rel_ci = ci(boot_rel)
        rows.append(
            {
                "pilot_spc": int(spc),
                "n": int(len(idx)),
                "psi_reduces_budget_rate": float(win.mean()),
                "psi_reduces_budget_rate_ci": win_ci,
                "mean_budget_reduction": float(diff.mean()),
                "mean_budget_reduction_ci": diff_ci,
                "mean_relative_reduction": float(rel.mean()),
                "mean_relative_reduction_ci": rel_ci,
            }
        )
    stats_path = result_file("pilot_selection/pilot_policy_stats.json")
    stats = json.loads(stats_path.read_text())
    stats["paired_nonpsi_to_nonpsi_plus_psi_ci"] = rows
    stats_path.write_text(json.dumps(stats, indent=2))
    tex = []
    for r in rows:
        tex.append(
            f"{r['pilot_spc']} & ${100 * r['psi_reduces_budget_rate']:.1f}[{100 * r['psi_reduces_budget_rate_ci'][0]:.1f},{100 * r['psi_reduces_budget_rate_ci'][1]:.1f}]$ & "
            f"${r['mean_budget_reduction']:.2f}[{r['mean_budget_reduction_ci'][0]:.2f},{r['mean_budget_reduction_ci'][1]:.2f}]$ & "
            f"${100 * r['mean_relative_reduction']:.1f}[{100 * r['mean_relative_reduction_ci'][0]:.1f},{100 * r['mean_relative_reduction_ci'][1]:.1f}]$\\\\"
        )
    (
        result_file("pilot_selection/pilot_policy_pair_savings_rows.tex")
    ).write_text("\n".join(tex) + "\n")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
