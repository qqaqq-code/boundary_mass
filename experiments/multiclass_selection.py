"Evaluate multiclass backbone-selection policies. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, RESULTS_DIR, result_file

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
ART = RESULTS_DIR

NONPSI_FEATURES = [
    "diag_nll",
    "diag_ece",
    "diag_acc",
    "median_margin",
    "effective_rank",
    "logme",
    "h_score",
    "transrate",
    "nce_cluster",
]


def ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def summarize(
    rows: pd.DataFrame, task_col: str = "task_id"
) -> dict[str, float | list[float]]:
    rng = np.random.default_rng(20260507)
    task_ids = np.array(sorted(rows[task_col].unique()))
    boot = {
        key: []
        for key in [
            "base_regret",
            "policy_regret",
            "budget_reduction",
            "base_aulc",
            "policy_aulc",
        ]
    }
    for _ in range(5000):
        sampled = rng.choice(task_ids, size=len(task_ids), replace=True)
        sample = pd.concat(
            [rows[rows[task_col] == task_id] for task_id in sampled], ignore_index=True
        )
        for key in boot:
            boot[key].append(float(sample[key].mean()))
    return {
        "n": int(len(task_ids)),
        "base_regret": float(rows.base_regret.mean()),
        "base_regret_ci": ci(boot["base_regret"]),
        "policy_regret": float(rows.policy_regret.mean()),
        "policy_regret_ci": ci(boot["policy_regret"]),
        "budget_reduction": float(rows.budget_reduction.mean()),
        "budget_reduction_ci": ci(boot["budget_reduction"]),
        "base_aulc": float(rows.base_aulc.mean()),
        "policy_aulc": float(rows.policy_aulc.mean()),
        "aulc_delta": float((rows.policy_aulc - rows.base_aulc).mean()),
        "aulc_delta_ci": ci(
            np.asarray(boot["policy_aulc"]) - np.asarray(boot["base_aulc"])
        ),
        "win_rate": float((rows.policy_regret < rows.base_regret).mean()),
        "same_budget_rate": float((rows.policy_regret == rows.base_regret).mean()),
    }


def policy_rows(
    data: pd.DataFrame,
    mode: str,
    psi_col: str | None = None,
    abstain_quantile: float | None = None,
) -> pd.DataFrame:
    rows = []
    for task_id, subset in data.groupby("task_id"):
        train = data[data.task_id != task_id]
        base_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        base_model.fit(train[NONPSI_FEATURES], train["label_budget"])
        test = subset.copy()
        test["pred_budget"] = base_model.predict(test[NONPSI_FEATURES])
        ranked = test.sort_values("pred_budget")
        base = ranked.iloc[0]
        top2 = ranked.head(2)
        if mode == "base":
            chosen = base
        elif mode == "top2":
            assert psi_col is not None
            if abstain_quantile is not None:
                train_gaps = []
                for _, train_subset in train.groupby("task_id"):
                    train_ranked = train_subset.assign(
                        pred_budget=base_model.predict(train_subset[NONPSI_FEATURES])
                    ).sort_values("pred_budget")
                    train_gaps.append(
                        float(
                            train_ranked.pred_budget.iloc[1]
                            - train_ranked.pred_budget.iloc[0]
                        )
                    )
                threshold = float(np.quantile(train_gaps, abstain_quantile))
                gap = float(top2.pred_budget.iloc[1] - top2.pred_budget.iloc[0])
                chosen = base if gap > threshold else top2.sort_values(psi_col).iloc[0]
            else:
                chosen = top2.sort_values(psi_col).iloc[0]
        elif mode == "plus_regression":
            assert psi_col is not None
            plus_model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
            plus_model.fit(train[NONPSI_FEATURES + [psi_col]], train["label_budget"])
            test["plus_pred_budget"] = plus_model.predict(
                test[NONPSI_FEATURES + [psi_col]]
            )
            chosen = test.sort_values("plus_pred_budget").iloc[0]
        else:
            raise ValueError(mode)
        best_budget = float(test.label_budget.min())
        rows.append(
            {
                "task_id": int(task_id),
                "base_chosen": str(base.backbone),
                "policy_chosen": str(chosen.backbone),
                "best_budget": best_budget,
                "base_regret": float(base.label_budget - best_budget),
                "policy_regret": float(chosen.label_budget - best_budget),
                "budget_reduction": float(base.label_budget - chosen.label_budget),
                "base_aulc": float(base.aulc),
                "policy_aulc": float(chosen.aulc),
                "pred_gap": float(top2.pred_budget.iloc[1] - top2.pred_budget.iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    table = pd.read_csv(
        result_file("multiclass_selection/tiny_multiclass_label_budget_table.csv")
    )
    summaries: dict[str, dict[str, float | list[float]]] = {}
    configs = {
        "nonpsi_base": ("base", None, None),
        "ovr_top2": ("top2", "log1p_psi", None),
        "ovr_top2_abstain_median_gap": ("top2", "log1p_psi", 0.50),
        "interface_top2": ("top2", "log1p_psi_interface", None),
        "interface_top2_abstain_median_gap": ("top2", "log1p_psi_interface", 0.50),
        "nonpsi_plus_ovr_regression": ("plus_regression", "log1p_psi", None),
        "nonpsi_plus_interface_regression": (
            "plus_regression",
            "log1p_psi_interface",
            None,
        ),
    }
    for name, (mode, psi_col, abstain_q) in configs.items():
        rows = policy_rows(table, mode, psi_col, abstain_q)
        rows.to_csv(result_file(f"tiny_multiclass_{name}_policy.csv"), index=False)
        summaries[name] = summarize(rows)

    task_stats = []
    for task_id, subset in table.groupby("task_id"):
        best = float(subset.label_budget.min())
        task_stats.append(
            {
                "task_id": int(task_id),
                "best_budget": best,
                "best_at_min_grid": float(best <= 2.05),
                "top2_contains_best": float(
                    subset.sort_values("label_budget").iloc[0].backbone
                    in set(
                        policy_rows(table, "base")
                        .query("task_id == @task_id")
                        .base_chosen
                    )
                ),
                "num_zero_regret_backbones": int(
                    (np.isclose(subset.label_budget, best)).sum()
                ),
            }
        )
    stats = {
        "summaries": summaries,
        "headroom": {
            "tasks": int(table.task_id.nunique()),
            "mean_best_budget": float(pd.DataFrame(task_stats).best_budget.mean()),
            "min_grid_best_rate": float(
                pd.DataFrame(task_stats).best_at_min_grid.mean()
            ),
            "mean_zero_regret_backbones": float(
                pd.DataFrame(task_stats).num_zero_regret_backbones.mean()
            ),
        },
    }
    (
        result_file("multiclass_selection/tiny_multiclass_policy_supplement.json")
    ).write_text(json.dumps(stats, indent=2))
    rows = []
    for label, key in [
        ("Non-$\\psi$ base", "nonpsi_base"),
        ("OvR top-2", "ovr_top2"),
        ("OvR abstain", "ovr_top2_abstain_median_gap"),
        ("Interface top-2", "interface_top2"),
        ("Interface abstain", "interface_top2_abstain_median_gap"),
        ("Non-$\\psi$ + OvR reg.", "nonpsi_plus_ovr_regression"),
        ("Non-$\\psi$ + interface reg.", "nonpsi_plus_interface_regression"),
    ]:
        summary = summaries[key]
        rows.append(
            f"{label} & ${summary['policy_regret']:.3f}$ & ${summary['budget_reduction']:.3f}$ & "
            f"${summary['aulc_delta']:.3f}$ & ${summary['win_rate']:.2f}$\\\\"
        )
    (
        result_file("multiclass_selection/tiny_multiclass_policy_supplement_rows.tex")
    ).write_text("\n".join(rows) + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
