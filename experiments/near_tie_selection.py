"Evaluate shortlist selection among closely matched backbones. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, RESULTS_DIR, FIGURES_DIR, result_file

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
FIG = FIGURES_DIR
ART.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

RNG = np.random.default_rng(20260504)
N_BOOT = 5000
BASE_FEATURES = [
    "diag_nll",
    "diag_ece",
    "diag_acc",
    "median_margin",
    "effective_rank",
    "gsa_proxy",
    "noise_flip_010",
    "logme",
    "h_score",
    "transrate",
    "nce_cluster",
]


def ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def paired_sign_p(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan")
    observed = float(values.mean())
    centered = values - observed
    null = []
    for _ in range(N_BOOT):
        signs = RNG.choice([-1.0, 1.0], size=len(centered))
        null.append(float((centered * signs).mean()))
    return float((1.0 + sum(v >= observed for v in null)) / (len(null) + 1.0))


def load_table() -> pd.DataFrame:
    path = result_file(
        "transferability/transferability_label_budget_table.csv"
    )
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}; run python run.py transferability first"
        )
    data = (
        pd.read_csv(path)
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=BASE_FEATURES + ["log1p_psi", "label_budget", "aulc"])
    )
    data = data.copy()
    data["pred_budget"] = np.nan
    for pair_id in sorted(data.pair_id.unique()):
        train = data[data.pair_id != pair_id]
        test = data[data.pair_id == pair_id]
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train[BASE_FEATURES], train["label_budget"])
        data.loc[test.index, "pred_budget"] = model.predict(test[BASE_FEATURES])
    data.to_csv(
        result_file("near_tie_selection/near_tie_lopo_predictions.csv"), index=False
    )
    return data


def shortlist_policy(data: pd.DataFrame, k: int) -> pd.DataFrame:
    rows = []
    for pair_id, sub in data.groupby("pair_id"):
        best_budget = float(sub.label_budget.min())
        base = sub.sort_values("pred_budget").iloc[0]
        chosen = sub.sort_values("pred_budget").head(k).sort_values("log1p_psi").iloc[0]
        rows.append(
            {
                "pair_id": int(pair_id),
                "k": int(k),
                "base_chosen": base.backbone,
                "psi_chosen": chosen.backbone,
                "base_regret": float(base.label_budget - best_budget),
                "psi_regret": float(chosen.label_budget - best_budget),
                "budget_reduction": float(base.label_budget - chosen.label_budget),
                "base_rel_regret": float(
                    (base.label_budget - best_budget) / (best_budget + 1e-9)
                ),
                "psi_rel_regret": float(
                    (chosen.label_budget - best_budget) / (best_budget + 1e-9)
                ),
                "base_aulc": float(base.aulc),
                "psi_aulc": float(chosen.aulc),
                "psi_win": float(chosen.label_budget < base.label_budget),
                "same_budget": float(chosen.label_budget == base.label_budget),
            }
        )
    return pd.DataFrame(rows)


def summarize_shortlist(rows: pd.DataFrame) -> dict[str, float]:
    pair_ids = np.array(sorted(rows.pair_id.unique()))
    boot = {
        "base_regret": [],
        "psi_regret": [],
        "budget_reduction": [],
        "psi_rel_regret": [],
        "psi_aulc": [],
    }
    for _ in range(N_BOOT):
        sampled = RNG.choice(pair_ids, size=len(pair_ids), replace=True)
        sample = pd.concat(
            [rows[rows.pair_id == pair_id] for pair_id in sampled], ignore_index=True
        )
        for key in boot:
            boot[key].append(float(sample[key].mean()))
    return {
        "k": int(rows.k.iloc[0]),
        "n": int(rows.pair_id.nunique()),
        "base_regret": float(rows.base_regret.mean()),
        "base_regret_ci": ci(boot["base_regret"]),
        "psi_regret": float(rows.psi_regret.mean()),
        "psi_regret_ci": ci(boot["psi_regret"]),
        "budget_reduction": float(rows.budget_reduction.mean()),
        "budget_reduction_ci": ci(boot["budget_reduction"]),
        "paired_sign_p": paired_sign_p(rows.budget_reduction.to_numpy()),
        "psi_win_rate": float(rows.psi_win.mean()),
        "same_budget_rate": float(rows.same_budget.mean()),
        "psi_rel_regret": float(rows.psi_rel_regret.mean()),
        "psi_rel_regret_ci": ci(boot["psi_rel_regret"]),
        "psi_aulc": float(rows.psi_aulc.mean()),
        "psi_aulc_ci": ci(boot["psi_aulc"]),
    }


def pairwise_near_tie(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pair_id, sub in data.groupby("pair_id"):
        records = list(sub.to_dict("records"))
        for i in range(len(records)):
            for j in range(i + 1, len(records)):
                left, right = records[i], records[j]
                base = left if left["pred_budget"] <= right["pred_budget"] else right
                low_psi = left if left["log1p_psi"] <= right["log1p_psi"] else right
                high_psi = right if low_psi is left else left
                if low_psi["label_budget"] < high_psi["label_budget"]:
                    win_score = 1.0
                elif low_psi["label_budget"] == high_psi["label_budget"]:
                    win_score = 0.5
                else:
                    win_score = 0.0
                rows.append(
                    {
                        "pair_id": int(pair_id),
                        "baseline_gap": float(
                            abs(left["pred_budget"] - right["pred_budget"])
                        ),
                        "baseline_budget": float(base["label_budget"]),
                        "low_psi_budget": float(low_psi["label_budget"]),
                        "high_psi_budget": float(high_psi["label_budget"]),
                        "budget_reduction_vs_baseline": float(
                            base["label_budget"] - low_psi["label_budget"]
                        ),
                        "budget_saving_vs_high_psi": float(
                            high_psi["label_budget"] - low_psi["label_budget"]
                        ),
                        "lower_psi_win_score": win_score,
                        "lower_psi_strict_win": float(
                            low_psi["label_budget"] < high_psi["label_budget"]
                        ),
                        "lower_psi_tie": float(
                            low_psi["label_budget"] == high_psi["label_budget"]
                        ),
                    }
                )
    pairs = pd.DataFrame(rows)
    pairs["gap_bin"] = pd.qcut(
        pairs.baseline_gap,
        q=[0.0, 0.33, 0.66, 1.0],
        labels=["near tie", "medium gap", "large gap"],
        duplicates="drop",
    )
    pairs.to_csv(result_file("near_tie_selection/near_tie_pairwise_experiment.csv"), index=False)
    return pairs


def summarize_pairwise(pairs: pd.DataFrame) -> list[dict[str, float]]:
    out = []
    pair_ids = np.array(sorted(pairs.pair_id.unique()))
    for gap_bin, sub in pairs.groupby("gap_bin", observed=True):
        boot_reduction, boot_saving, boot_score = [], [], []
        for _ in range(N_BOOT):
            sampled = RNG.choice(pair_ids, size=len(pair_ids), replace=True)
            sample = pd.concat(
                [sub[sub.pair_id == pair_id] for pair_id in sampled], ignore_index=True
            )
            if len(sample) == 0:
                continue
            boot_reduction.append(float(sample.budget_reduction_vs_baseline.mean()))
            boot_saving.append(float(sample.budget_saving_vs_high_psi.mean()))
            boot_score.append(float(sample.lower_psi_win_score.mean()))
        values = sub.budget_reduction_vs_baseline.to_numpy()
        out.append(
            {
                "gap_bin": str(gap_bin),
                "n": int(len(sub)),
                "mean_baseline_gap": float(sub.baseline_gap.mean()),
                "budget_reduction_vs_baseline": float(
                    sub.budget_reduction_vs_baseline.mean()
                ),
                "budget_reduction_vs_baseline_ci": ci(boot_reduction),
                "saving_vs_high_psi": float(sub.budget_saving_vs_high_psi.mean()),
                "saving_vs_high_psi_ci": ci(boot_saving),
                "lower_psi_win_score": float(sub.lower_psi_win_score.mean()),
                "lower_psi_win_score_ci": ci(boot_score),
                "strict_win_rate": float(sub.lower_psi_strict_win.mean()),
                "tie_rate": float(sub.lower_psi_tie.mean()),
                "paired_sign_p": paired_sign_p(values),
            }
        )
    return out


def write_tex(
    short_rows: list[dict[str, float]], pair_rows: list[dict[str, float]]
) -> None:
    shortlist_lines = []
    for row in short_rows:
        shortlist_lines.append(
            f"top-{row['k']} then lowest $\\psi$ & ${row['base_regret']:.2f}$ & "
            f"${row['psi_regret']:.2f}[{row['psi_regret_ci'][0]:.2f},{row['psi_regret_ci'][1]:.2f}]$ & "
            f"${row['budget_reduction']:.2f}[{row['budget_reduction_ci'][0]:.2f},{row['budget_reduction_ci'][1]:.2f}]$ & "
            f"${100 * row['psi_win_rate']:.1f}$ & ${row['psi_aulc']:.3f}$\\\\"
        )
    (result_file("near_tie_selection/shortlist_policy_rows.tex")).write_text(
        "\n".join(shortlist_lines) + "\n"
    )

    pair_lines = []
    for row in pair_rows:
        pair_lines.append(
            f"{row['gap_bin']} & {row['n']} & ${row['mean_baseline_gap']:.2f}$ & "
            f"${row['budget_reduction_vs_baseline']:.2f}[{row['budget_reduction_vs_baseline_ci'][0]:.2f},{row['budget_reduction_vs_baseline_ci'][1]:.2f}]$ & "
            f"${100 * row['lower_psi_win_score']:.1f}[{100 * row['lower_psi_win_score_ci'][0]:.1f},{100 * row['lower_psi_win_score_ci'][1]:.1f}]$\\\\"
        )
    (result_file("near_tie_selection/near_tie_pairwise_rows.tex")).write_text(
        "\n".join(pair_lines) + "\n"
    )


def plot_pairwise(pair_rows: list[dict[str, float]]) -> None:
    try:
        import matplotlib.pyplot as plt

        labels = [row["gap_bin"] for row in pair_rows]
        x = np.arange(len(labels))
        reductions = np.array(
            [row["budget_reduction_vs_baseline"] for row in pair_rows]
        )
        yerr = np.array(
            [
                [
                    row["budget_reduction_vs_baseline"]
                    - row["budget_reduction_vs_baseline_ci"][0]
                    for row in pair_rows
                ],
                [
                    row["budget_reduction_vs_baseline_ci"][1]
                    - row["budget_reduction_vs_baseline"]
                    for row in pair_rows
                ],
            ]
        )
        scores = 100 * np.array([row["lower_psi_win_score"] for row in pair_rows])
        score_yerr = np.array(
            [
                [
                    100
                    * (row["lower_psi_win_score"] - row["lower_psi_win_score_ci"][0])
                    for row in pair_rows
                ],
                [
                    100
                    * (row["lower_psi_win_score_ci"][1] - row["lower_psi_win_score"])
                    for row in pair_rows
                ],
            ]
        )
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
        axes[0].bar(x, reductions, color=["#4C78A8", "#72B7B2", "#F58518"], width=0.65)
        axes[0].errorbar(
            x, reductions, yerr=yerr, fmt="none", color="black", capsize=3, linewidth=1
        )
        axes[0].axhline(0, color="#333333", linewidth=0.8)
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(labels, rotation=12)
        axes[0].set_ylabel("budget change vs. non-psi pick")
        axes[0].set_title("Use psi only as tie-breaker")
        axes[1].bar(x, scores, color=["#4C78A8", "#72B7B2", "#F58518"], width=0.65)
        axes[1].errorbar(
            x,
            scores,
            yerr=score_yerr,
            fmt="none",
            color="black",
            capsize=3,
            linewidth=1,
        )
        axes[1].axhline(50, color="#333333", linestyle="--", linewidth=0.8)
        axes[1].set_ylim(45, max(70, float(np.nanmax(scores + score_yerr[1]) + 5)))
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=12)
        axes[1].set_ylabel("lower-psi win/tie score (%)")
        axes[1].set_title("Pairwise boundary burden")
        fig.tight_layout()
        fig.savefig(FIG / "near_tie_selection_selection.png", dpi=220)
        fig.savefig(FIG / "near_tie_selection_curve.pdf")
        plt.close(fig)
    except Exception as exc:
        print(f"plot failed: {exc}", flush=True)


def write_external_table() -> None:
    strict = json.loads(
        (
            result_file(
                "transferability/transferability_baseline_stats.json"
            )
        ).read_text()
    )
    tiny = json.loads(
        (
            result_file("tiny_imagenet/tiny_strict_label_budget_stats.json")
        ).read_text()
    )
    strict_row = next(
        row
        for row in strict["incremental"]
        if row["base"] == "all non-$\\psi$ + transferability"
    )
    tiny_row = next(
        row
        for row in tiny["incremental"]
        if row["base"] == "all non-$\\psi$ + transferability"
    )
    lines = [
        f"CIFAR-100 binary & 40 & 4 & ${strict_row['r2_base']:.3f}$ & ${strict_row['r2_plus_psi']:.3f}$ & ${strict_row['delta_r2']:.3f}$ & ${strict_row['paired_signflip_p']:.3f}$\\\\",
        f"Tiny-ImageNet binary & 40 & 4 & ${tiny_row['r2_base']:.3f}$ & ${tiny_row['r2_plus_psi']:.3f}$ & ${tiny_row['delta_r2']:.3f}$ & ${tiny_row['paired_signflip_p']:.3f}$\\\\",
    ]
    (result_file("multiclass_selection/external_validation_rows.tex")).write_text(
        "\n".join(lines) + "\n"
    )


def main() -> None:
    data = load_table()
    short_tables = []
    short_summaries = []
    for k in [2, 3]:
        rows = shortlist_policy(data, k)
        rows.to_csv(result_file(f"shortlist_top{k}_selection.csv"), index=False)
        short_tables.append(rows)
        short_summaries.append(summarize_shortlist(rows))
    pairwise = pairwise_near_tie(data)
    pair_summaries = summarize_pairwise(pairwise)
    write_tex(short_summaries, pair_summaries)
    plot_pairwise(pair_summaries)
    write_external_table()
    stats = {
        "protocol": {
            "base_predictor": "leave-one-class-pair-out Ridge from non-psi probe-quality and transferability metrics to label budget",
            "shortlist": "choose lowest predicted label budget, or choose lowest psi inside the top-k predicted shortlist",
            "pairwise_near_tie": "for each task/backbone pair, compare lower-psi choice against the lower predicted-budget non-psi choice within baseline-gap bins",
            "features": BASE_FEATURES,
        },
        "shortlist": short_summaries,
        "pairwise": pair_summaries,
    }
    (result_file("near_tie_selection/near_tie_shortlist_stats.json")).write_text(
        json.dumps(stats, indent=2)
    )
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
