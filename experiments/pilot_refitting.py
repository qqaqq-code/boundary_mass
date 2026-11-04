"""Measure pilot-head refitting and selection stability across random seeds."""
from __future__ import annotations
import argparse
import itertools
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import log_loss
from utils import binary_protocol as protocol
from utils.paths import PROJECT_ROOT, result_file

CACHES = protocol.CACHES
PILOT_SPCS = (2, 5, 10)
DIAG_FRAC = 0.30
SINGLE_POLICIES = {
    "choose by $\\psi$": ("log1p_psi", "low"),
    "choose by margin": ("median_margin", "high"),
    "choose by LogME": ("logme", "high"),
    "choose by H-score": ("h_score", "high"),
}


REG_POLICIES = {
    "NLL/ECE": ["diag_nll", "diag_ece"],
    "LogME/H-score": ["logme", "h_score"],
    "transferability set": ["logme", "h_score", "transrate", "nce_cluster"],
    "combined non-$\\psi$": [
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
    ],
    "combined non-$\\psi$ + $\\psi$": [
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
        "log1p_psi",
    ],
}


SEED_METRICS = (
    "psi_eff",
    "log1p_psi",
    "median_margin",
    "diag_nll",
    "diag_ece",
    "diag_acc",
    "effective_rank",
    "gsa_proxy",
    "noise_flip_010",
)


def make_pairs() -> list[tuple[int, int]]:
    rng = np.random.default_rng(20260429)
    pairs: list[tuple[int, int]] = [
        (class_id, class_id + 1) for class_id in range(0, 40, 2)
    ]
    seen = set(pairs)
    while len(pairs) < 40:
        left, right = rng.choice(100, 2, replace=False)
        pair = tuple(sorted((int(left), int(right))))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = singular * singular / max(1, features.shape[0] - 1)
    total = eigenvalues.sum()
    return float(total * total / (np.square(eigenvalues).sum() + 1e-12))


def ece_binary(probability: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    confidence = np.maximum(probability, 1.0 - probability)
    prediction = (probability >= 0.5).astype(int)
    correctness = (prediction == labels).astype(float)
    result = 0.0
    edges = np.linspace(0.0, 1.0, bins + 1)
    for low, high in zip(edges[:-1], edges[1:]):
        selected = (confidence >= low) & (
            confidence < (high if high < 1.0 else high + 1e-8)
        )
        if selected.any():
            result += float(
                selected.mean()
                * abs(
                    confidence[selected].mean()
                    - correctness[selected].mean()
                )
            )
    return result


def gsa_proxy(model: object) -> float:
    weight = model.coef_.reshape(-1)
    offset = float(model.intercept_[0]) / (np.linalg.norm(weight) + 1e-12)
    return float(2.0 * math.exp(-0.5 * offset * offset) / math.sqrt(2.0 * math.pi))


def noise_flip(
    model: object, features: np.ndarray, alpha: float = 0.10, repeats: int = 3
) -> float:
    baseline = model.predict(features)
    rng = np.random.default_rng(123)
    scale = np.std(features, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    values = []
    for _ in range(repeats):
        noise = rng.normal(0, alpha, size=features.shape).astype("float32")
        values.append(float(np.mean(model.predict(features + noise * scale) != baseline)))
    return float(np.mean(values))


def compute_seed_metrics(seeds: tuple[int, ...]) -> pd.DataFrame:
    pairs = make_pairs()
    rows: list[dict[str, object]] = []
    total_units = len(CACHES) * len(pairs)
    complete = 0

    for backbone_id, (backbone, cache_dir) in enumerate(CACHES.items()):
        cache_path = cache_dir / "train_features.pt"
        if not cache_path.is_file():
            raise FileNotFoundError(f"missing frozen feature cache: {cache_path}")
        train = protocol.load_features(cache_path)
        features_all = train["features"]
        labels_all = train["labels"]

        for pair_id, (class_a, class_b) in enumerate(pairs):
            pair_mask = (labels_all == class_a) | (labels_all == class_b)
            pair_features = np.asarray(features_all[pair_mask], dtype="float32")
            pair_labels = np.asarray(labels_all[pair_mask] == class_b, dtype="int64")
            probe_idx, diag_idx = protocol.split_probe_diag(
                pair_labels, pair_id, backbone_id
            )
            probe_features, diag_features = protocol.standardize_from_probe(
                pair_features[probe_idx], pair_features[diag_idx]
            )
            probe_labels = pair_labels[probe_idx]
            diag_labels = pair_labels[diag_idx]
            class0 = np.flatnonzero(probe_labels == 0)
            class1 = np.flatnonzero(probe_labels == 1)
            rank = effective_rank(diag_features)
            diag_indices = protocol.sample_indices(np.sort(diag_idx))

            for spc in PILOT_SPCS:
                for seed in seeds:
                    rng = np.random.default_rng(
                        5000
                        + pair_id * 37
                        + backbone_id * 11
                        + spc * 3
                        + seed
                    )
                    selected = np.r_[
                        rng.choice(class0, min(spc, len(class0)), replace=False),
                        rng.choice(class1, min(spc, len(class1)), replace=False),
                    ]
                    model = protocol.fit_linear_head(
                        probe_features[selected], probe_labels[selected]
                    )
                    probability = model.predict_proba(diag_features)[:, 1]
                    prediction = (probability >= 0.5).astype(int)
                    distances = protocol.signed_distance(model, diag_features)
                    psi, _, eps = protocol.psi_metrics(distances, diag_labels)
                    selected_pair_idx = probe_idx[selected]
                    rows.append(
                        {
                            "pair_id": pair_id,
                            "class_a": class_a,
                            "class_b": class_b,
                            "backbone": backbone,
                            "backbone_id": backbone_id,
                            "pilot_spc": spc,
                            "seed": seed,
                            "pilot_n": int(len(selected)),
                            "pilot_class0_n": int(
                                np.sum(probe_labels[selected] == 0)
                            ),
                            "pilot_class1_n": int(
                                np.sum(probe_labels[selected] == 1)
                            ),
                            "pilot_indices": protocol.sample_indices(
                                np.sort(selected_pair_idx)
                            ),
                            "diag_indices": diag_indices,
                            "leakage_count": int(
                                np.intersect1d(selected_pair_idx, diag_idx).size
                            ),
                            "diag_n": int(len(diag_labels)),
                            "psi_eff": psi,
                            "log1p_psi": math.log1p(psi),
                            "psi_eps": eps,
                            "median_margin": float(
                                np.median(np.abs(distances))
                            ),
                            "diag_nll": float(
                                log_loss(
                                    diag_labels,
                                    np.vstack(
                                        [1.0 - probability, probability]
                                    ).T,
                                    labels=[0, 1],
                                )
                            ),
                            "diag_ece": ece_binary(probability, diag_labels),
                            "diag_acc": float(
                                np.mean(prediction == diag_labels)
                            ),
                            "effective_rank": rank,
                            "gsa_proxy": gsa_proxy(model),
                            "noise_flip_010": noise_flip(model, diag_features),
                            "fit_iterations": int(model.n_iter_[0]),
                        }
                    )

            complete += 1
            if complete % 20 == 0:
                print(
                    f"completed {complete}/{total_units} task-backbone units",
                    flush=True,
                )

    raw = pd.DataFrame(rows)
    backbone_order = {name: index for index, name in enumerate(CACHES)}
    raw["_backbone_order"] = raw.backbone.map(backbone_order)
    raw = raw.sort_values(
        ["pair_id", "_backbone_order", "pilot_spc", "seed"]
    ).drop(columns="_backbone_order")
    return raw.reset_index(drop=True)


def choose_single(
    frame: pd.DataFrame, policy: str, metric: str, direction: str
) -> pd.DataFrame:
    rows = []
    for (spc, pair_id), subset in frame.groupby(["pilot_spc", "pair_id"]):
        best = subset.loc[subset.label_budget.idxmin()]
        chosen = subset.sort_values(
            metric, ascending=(direction == "low")
        ).iloc[0]
        rows.append(
            {
                "pilot_spc": int(spc),
                "pair_id": int(pair_id),
                "policy": policy,
                "chosen": chosen.backbone,
                "best": best.backbone,
                "correct": bool(chosen.backbone == best.backbone),
                "chosen_budget": float(chosen.label_budget),
                "best_budget": float(best.label_budget),
                "regret": float(chosen.label_budget - best.label_budget),
                "relative_regret": float(
                    (chosen.label_budget - best.label_budget)
                    / (best.label_budget + 1e-9)
                ),
                "chosen_aulc": float(chosen.aulc),
            }
        )
    return pd.DataFrame(rows)


def choose_lopo(
    frame: pd.DataFrame, policy: str, predictors: list[str]
) -> pd.DataFrame:
    rows = []
    for (spc, pair_id), subset in frame.groupby(["pilot_spc", "pair_id"]):
        train = frame[
            (frame.pilot_spc == spc) & (frame.pair_id != pair_id)
        ].copy()
        test = subset.copy()
        columns = predictors + ["label_budget"]
        train = train[
            ["pair_id", "backbone", "aulc"] + columns
        ].replace([np.inf, -np.inf], np.nan).dropna()
        test = test[
            ["pair_id", "backbone", "aulc"] + columns
        ].replace([np.inf, -np.inf], np.nan).dropna()
        mean = train[predictors].mean(axis=0)
        std = train[predictors].std(axis=0).replace(0, 1.0)
        prediction = LinearRegression().fit(
            ((train[predictors] - mean) / std).to_numpy(),
            train.label_budget.to_numpy(),
        ).predict(((test[predictors] - mean) / std).to_numpy())
        test = test.copy()
        test["pred_budget"] = prediction
        chosen = test.sort_values("pred_budget").iloc[0]
        best = subset.loc[subset.label_budget.idxmin()]
        rows.append(
            {
                "pilot_spc": int(spc),
                "pair_id": int(pair_id),
                "policy": policy,
                "chosen": chosen.backbone,
                "best": best.backbone,
                "correct": bool(chosen.backbone == best.backbone),
                "chosen_budget": float(chosen.label_budget),
                "best_budget": float(best.label_budget),
                "regret": float(chosen.label_budget - best.label_budget),
                "relative_regret": float(
                    (chosen.label_budget - best.label_budget)
                    / (best.label_budget + 1e-9)
                ),
                "chosen_aulc": float(chosen.aulc),
            }
        )
    return pd.DataFrame(rows)


def select_all_policies(frame: pd.DataFrame) -> pd.DataFrame:
    frames = [
        choose_single(frame, name, metric, direction)
        for name, (metric, direction) in SINGLE_POLICIES.items()
    ]
    frames.extend(
        choose_lopo(frame, name, predictors)
        for name, predictors in REG_POLICIES.items()
    )
    return pd.concat(frames, ignore_index=True)


def seed_policy_selections(
    raw: pd.DataFrame, truth: pd.DataFrame, transfer: pd.DataFrame
) -> pd.DataFrame:
    metric_columns = list(SEED_METRICS)
    static_columns = ["logme", "h_score", "transrate", "nce_cluster"]
    transfer_subset = transfer[
        ["pair_id", "class_a", "class_b", "backbone"] + static_columns
    ]
    base = raw[
        [
            "pair_id",
            "class_a",
            "class_b",
            "backbone",
            "pilot_spc",
            "seed",
        ]
        + metric_columns
    ].merge(
        transfer_subset,
        on=["pair_id", "class_a", "class_b", "backbone"],
        how="left",
        validate="many_to_one",
    )
    base = base.merge(
        truth[["pair_id", "backbone", "label_budget", "aulc"]],
        on=["pair_id", "backbone"],
        how="left",
        validate="many_to_one",
    )
    outputs = []
    for seed, current in base.groupby("seed", sort=True):
        selected = select_all_policies(current)
        selected.insert(0, "seed", int(seed))
        outputs.append(selected)
    return pd.concat(outputs, ignore_index=True)


def bootstrap_interval(
    values: pd.Series, rng: np.random.Generator, repeats: int
) -> tuple[float, float]:
    array = values.to_numpy(dtype=float)
    sampled = rng.choice(
        array, size=(repeats, len(array)), replace=True
    ).mean(axis=1)
    return float(np.percentile(sampled, 2.5)), float(
        np.percentile(sampled, 97.5)
    )


def summarize_policy_agreement(
    selections: pd.DataFrame, n_boot: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_rows = []
    for (spc, policy, pair_id), subset in selections.groupby(
        ["pilot_spc", "policy", "pair_id"]
    ):
        indexed = subset.set_index("seed")
        seeds = sorted(indexed.index)
        agreements = []
        budget_agreements = []
        absolute_budget_differences = []
        for left, right in itertools.combinations(seeds, 2):
            agreements.append(
                indexed.loc[left, "chosen"] == indexed.loc[right, "chosen"]
            )
            left_budget = float(indexed.loc[left, "chosen_budget"])
            right_budget = float(indexed.loc[right, "chosen_budget"])
            budget_agreements.append(
                bool(np.isclose(left_budget, right_budget, rtol=0.0, atol=1e-12))
            )
            absolute_budget_differences.append(abs(left_budget - right_budget))
        counts = indexed.chosen.value_counts()
        task_rows.append(
            {
                "pilot_spc": int(spc),
                "policy": policy,
                "pair_id": int(pair_id),
                "pairwise_selection_agreement": float(np.mean(agreements)),
                "pairwise_budget_agreement": float(
                    np.mean(budget_agreements)
                ),
                "mean_abs_budget_difference": float(
                    np.mean(absolute_budget_differences)
                ),
                "unanimous_selection": bool(counts.max() == len(seeds)),
                "modal_selection_share": float(counts.max() / len(seeds)),
            }
        )
    task = pd.DataFrame(task_rows)
    summary_rows = []
    rng = np.random.default_rng(20260804)
    for (spc, policy), subset in task.groupby(["pilot_spc", "policy"]):
        row: dict[str, object] = {
            "pilot_spc": int(spc),
            "policy": policy,
            "seeds": int(
                selections[
                    (selections.pilot_spc == spc)
                    & (selections.policy == policy)
                ].seed.nunique()
            ),
            "tasks": int(subset.pair_id.nunique()),
        }
        for column in (
            "pairwise_selection_agreement",
            "pairwise_budget_agreement",
            "mean_abs_budget_difference",
            "unanimous_selection",
            "modal_selection_share",
        ):
            values = subset[column].astype(float)
            low, high = bootstrap_interval(values, rng, n_boot)
            row[column] = float(values.mean())
            row[column + "_ci_low"] = low
            row[column + "_ci_high"] = high
        summary_rows.append(row)
    return task, pd.DataFrame(summary_rows)


def summarize_psi_stability(
    raw: pd.DataFrame,
    policy_summary: pd.DataFrame,
    n_boot: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    task_rows = []
    backbones = list(CACHES)
    for (spc, pair_id), subset in raw.groupby(["pilot_spc", "pair_id"]):
        unit = subset.groupby("backbone").agg(
            psi_mean=("psi_eff", "mean"),
            psi_sd=("psi_eff", "std"),
            logpsi_sd=("log1p_psi", "std"),
        )
        unit["psi_cv"] = unit.psi_sd / unit.psi_mean.abs()
        pivot = subset.pivot(
            index="backbone", columns="seed", values="psi_eff"
        ).reindex(backbones)
        seeds = sorted(pivot.columns)
        rank_correlations = []
        order_agreements = []
        for left, right in itertools.combinations(seeds, 2):
            rank_correlations.append(
                float(spearmanr(pivot[left], pivot[right]).statistic)
            )
            agreements = []
            for backbone_a, backbone_b in itertools.combinations(backbones, 2):
                left_delta = (
                    pivot.loc[backbone_a, left]
                    - pivot.loc[backbone_b, left]
                )
                right_delta = (
                    pivot.loc[backbone_a, right]
                    - pivot.loc[backbone_b, right]
                )
                agreements.append(
                    bool(np.sign(left_delta) == np.sign(right_delta))
                )
            order_agreements.append(float(np.mean(agreements)))
        task_rows.append(
            {
                "pilot_spc": int(spc),
                "pair_id": int(pair_id),
                "psi_refit_sd": float(unit.psi_sd.mean()),
                "log1p_psi_refit_sd": float(unit.logpsi_sd.mean()),
                "psi_cv": float(unit.psi_cv.mean()),
                "within_pair_seed_spearman": float(
                    np.mean(rank_correlations)
                ),
                "pairwise_rank_agreement": float(
                    np.mean(order_agreements)
                ),
            }
        )
    task = pd.DataFrame(task_rows)
    summary_rows = []
    rng = np.random.default_rng(20260805)
    for spc, subset in task.groupby("pilot_spc"):
        row: dict[str, object] = {
            "pilot_spc": int(spc),
            "seeds": int(raw[raw.pilot_spc == spc].seed.nunique()),
            "tasks": int(subset.pair_id.nunique()),
            "backbones": len(CACHES),
        }
        for column in (
            "psi_refit_sd",
            "log1p_psi_refit_sd",
            "psi_cv",
            "within_pair_seed_spearman",
            "pairwise_rank_agreement",
        ):
            low, high = bootstrap_interval(subset[column], rng, n_boot)
            row[column] = float(subset[column].mean())
            row[column + "_ci_low"] = low
            row[column + "_ci_high"] = high

        for policy, output_name in (
            ("choose by $\\psi$", "psi_selection_agreement"),
            (
                "combined non-$\\psi$ + $\\psi$",
                "combined_plus_psi_selection_agreement",
            ),
        ):
            match = policy_summary[
                (policy_summary.pilot_spc == spc)
                & (policy_summary.policy == policy)
            ].iloc[0]
            row[output_name] = float(match.pairwise_selection_agreement)
            row[output_name + "_ci_low"] = float(
                match.pairwise_selection_agreement_ci_low
            )
            row[output_name + "_ci_high"] = float(
                match.pairwise_selection_agreement_ci_high
            )
            budget_name = output_name.replace(
                "selection_agreement", "budget_agreement"
            )
            row[budget_name] = float(match.pairwise_budget_agreement)
            row[budget_name + "_ci_low"] = float(
                match.pairwise_budget_agreement_ci_low
            )
            row[budget_name + "_ci_high"] = float(
                match.pairwise_budget_agreement_ci_high
            )
        summary_rows.append(row)
    return task, pd.DataFrame(summary_rows)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-boot", type=int, default=10000)
    parser.add_argument("--raw-metrics", type=Path)
    parser.add_argument("--selections", type=Path)
    parser.add_argument("--reference-budgets", type=Path, default=result_file("backbone_selection/strict_label_budget_table.csv"))
    parser.add_argument("--transfer-metrics", type=Path, default=result_file("transferability/transferability_baseline_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/pilot_refitting")
    args = parser.parse_args()
    if args.n_seeds < 2 or args.n_boot < 100:
        parser.error("Use at least two seeds and 100 bootstrap samples.")
    raw = pd.read_csv(args.raw_metrics) if args.raw_metrics else compute_seed_metrics(tuple(range(args.n_seeds)))
    if args.selections:
        selections = pd.read_csv(args.selections)
    else:
        selections = seed_policy_selections(raw, pd.read_csv(args.reference_budgets), pd.read_csv(args.transfer_metrics))
    policy_task, policy_summary = summarize_policy_agreement(selections, args.n_boot)
    psi_task, summary = summarize_psi_stability(raw, policy_summary, args.n_boot)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in [("seed_metrics", raw), ("selections", selections), ("policy_by_task", policy_task),
                        ("policy_summary", policy_summary), ("stability_by_task", psi_task), ("stability", summary)]:
        frame.to_csv(args.output_dir / f"{name}.csv", index=False)
    print(summary[["pilot_spc", "log1p_psi_refit_sd", "pairwise_rank_agreement", "combined_plus_psi_budget_agreement"]].to_string(index=False))

if __name__ == "__main__":
    main()
