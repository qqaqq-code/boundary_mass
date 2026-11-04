"""Compare Random, Uncertainty and Hybrid label acquisition."""
from __future__ import annotations
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from utils import binary_protocol as protocol
from utils.paths import PROJECT_ROOT, result_file

CACHES = protocol.CACHES
TRUTH_PATH = result_file("backbone_selection/strict_label_budget_table.csv")
POLICIES = ("random", "uncertainty", "hybrid")
DISPLAY_NAMES = {"random": "Random", "uncertainty": "Uncertainty", "hybrid": "Hybrid"}
def make_pairs(n_pairs: int) -> list[tuple[int, int]]:
    if not 1 <= n_pairs <= 40:
        raise ValueError("n_pairs must be between 1 and 40")
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
    return pairs[:n_pairs]


def standardize_from_probe(
    probe: np.ndarray,
    diagnostic: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the fixed-split experiment: normalization is fit on the full probe pool."""
    mean = probe.mean(axis=0, keepdims=True)
    std = probe.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (
        ((probe - mean) / std).astype("float32"),
        ((diagnostic - mean) / std).astype("float32"),
        ((test - mean) / std).astype("float32"),
    )


def initial_rng(
    pair_id: int, backbone_id: int, seed: int
) -> np.random.Generator:
    return np.random.default_rng(
        710_000 + pair_id * 1009 + backbone_id * 101 + seed
    )


def policy_rng(
    policy: str, pair_id: int, backbone_id: int, seed: int
) -> np.random.Generator:
    return np.random.default_rng(
        910_000
        + POLICIES.index(policy) * 100_003
        + pair_id * 1009
        + backbone_id * 101
        + seed
    )


def balanced_pilot(
    labels: np.ndarray,
    samples_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    selected = np.concatenate(
        [
            rng.choice(
                np.flatnonzero(labels == class_value),
                samples_per_class,
                replace=False,
            )
            for class_value in (0, 1)
        ]
    )
    return rng.permutation(selected).astype("int64")


def balanced_random_additions(
    labels: np.ndarray,
    selected: np.ndarray,
    additions_per_class: int,
    rng: np.random.Generator,
) -> np.ndarray:
    additions = []
    selected_mask = np.zeros(len(labels), dtype=bool)
    selected_mask[selected] = True
    for class_value in (0, 1):
        candidates = np.flatnonzero(
            (labels == class_value) & ~selected_mask
        )
        additions.append(
            rng.choice(candidates, additions_per_class, replace=False)
        )
    return rng.permutation(np.concatenate(additions)).astype("int64")


def approximately_balanced_random_additions(
    labels: np.ndarray,
    candidates: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Balanced random half of Hybrid, with deterministic odd-count handling."""
    desired = {
        0: count // 2,
        1: count - count // 2,
    }
    additions: list[np.ndarray] = []
    for class_value in (0, 1):
        class_candidates = candidates[labels[candidates] == class_value]
        take = min(desired[class_value], len(class_candidates))
        additions.append(
            rng.choice(class_candidates, take, replace=False)
            if take
            else np.empty(0, dtype="int64")
        )
    current = np.concatenate(additions).astype("int64")
    if len(current) < count:
        remaining = candidates[~np.isin(candidates, current)]
        fill = rng.choice(
            remaining, count - len(current), replace=False
        ).astype("int64")
        current = np.concatenate([current, fill])
    return rng.permutation(current).astype("int64")


def uncertainty_additions(
    model: object,
    features: np.ndarray,
    candidates: np.ndarray,
    count: int,
) -> tuple[np.ndarray, float, float]:
    margins = np.abs(protocol.signed_distance(model, features[candidates]))
    order = np.lexsort((candidates, margins))
    chosen_offsets = order[:count]
    chosen = candidates[chosen_offsets].astype("int64")
    cutoff = float(margins[chosen_offsets].max())
    next_margin = (
        float(margins[order[count]])
        if count < len(order)
        else float("inf")
    )
    return chosen, cutoff, next_margin


def fit_and_score(
    features: np.ndarray,
    labels: np.ndarray,
    selected: np.ndarray,
    test_features: np.ndarray,
    test_labels: np.ndarray,
) -> tuple[object, float, int]:
    model = protocol.fit_linear_head(features[selected], labels[selected])
    accuracy = float(model.score(test_features, test_labels))
    iterations = int(model.n_iter_[0])
    return model, accuracy, iterations


def interp_budget(
    budgets: np.ndarray, accuracies: np.ndarray, target: float
) -> tuple[float, bool]:
    """Use the log-grid interpolation."""
    budgets = np.asarray(budgets, dtype=float)
    accuracies = np.asarray(accuracies, dtype=float)
    order = np.argsort(budgets)
    budgets = budgets[order]
    accuracies = accuracies[order]
    if np.nanmax(accuracies) < target:
        return float(budgets.max() * 1.5), False
    if accuracies[0] >= target:
        return float(budgets[0]), True
    for index in range(1, len(budgets)):
        if accuracies[index] >= target:
            lower_x, upper_x = (
                np.log(budgets[index - 1]),
                np.log(budgets[index]),
            )
            lower_y, upper_y = (
                accuracies[index - 1],
                accuracies[index],
            )
            fraction = (
                0.0
                if abs(upper_y - lower_y) < 1e-9
                else (target - lower_y) / (upper_y - lower_y)
            )
            return (
                float(
                    np.exp(
                        lower_x + fraction * (upper_x - lower_x)
                    )
                ),
                True,
            )
    return float(budgets.max() * 1.5), False


def run_experiment(
    n_pairs: int,
    budgets: tuple[int, ...],
    seeds: tuple[int, ...],
    target_ratio: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pairs = make_pairs(n_pairs)
    truth = pd.read_csv(TRUTH_PATH)
    truth = truth[
        truth["pair_id"].isin(range(n_pairs))
        & truth["backbone"].isin(CACHES)
    ].copy()
    truth_lookup = {
        (int(row.pair_id), str(row.backbone)): row
        for row in truth.itertuples()
    }
    best_full_by_pair = (
        truth.groupby("pair_id")["target_full_acc"].max().to_dict()
    )

    curve_rows: list[dict[str, object]] = []
    pilot_rows: list[dict[str, object]] = []
    event_rows: list[dict[str, object]] = []
    full_rows: list[dict[str, object]] = []
    started = time.perf_counter()
    total_units = len(pairs) * len(CACHES)
    completed_units = 0

    for backbone_id, (backbone, cache_dir) in enumerate(CACHES.items()):
        train = protocol.load_features(
            cache_dir / "train_features.pt"
        )
        test = protocol.load_features(
            cache_dir / "test_features.pt"
        )
        train_features_all = train["features"]
        train_labels_all = train["labels"]
        test_features_all = test["features"]
        test_labels_all = test["labels"]

        for pair_id, (class_a, class_b) in enumerate(pairs):
            train_mask = (train_labels_all == class_a) | (
                train_labels_all == class_b
            )
            test_mask = (test_labels_all == class_a) | (
                test_labels_all == class_b
            )
            pair_features = np.asarray(
                train_features_all[train_mask], dtype="float32"
            )
            pair_labels = np.asarray(
                train_labels_all[train_mask] == class_b, dtype="int64"
            )
            pair_test_features = np.asarray(
                test_features_all[test_mask], dtype="float32"
            )
            pair_test_labels = np.asarray(
                test_labels_all[test_mask] == class_b, dtype="int64"
            )
            probe_idx, diag_idx = protocol.split_probe_diag(
                pair_labels, pair_id, backbone_id
            )
            probe_features, diagnostic_features, test_features = (
                standardize_from_probe(
                    pair_features[probe_idx],
                    pair_features[diag_idx],
                    pair_test_features,
                )
            )
            probe_labels = pair_labels[probe_idx]
            diagnostic_labels = pair_labels[diag_idx]
            target_row = truth_lookup[(pair_id, backbone)]
            target = target_ratio * float(best_full_by_pair[pair_id])

            full_model = protocol.fit_linear_head(
                probe_features, probe_labels
            )
            fitted_full_accuracy = float(
                full_model.score(test_features, pair_test_labels)
            )
            full_rows.append(
                {
                    "pair_id": pair_id,
                    "backbone": backbone,
                    "target_accuracy": target,
                    "stored_full_accuracy": float(
                        target_row.target_full_acc
                    ),
                    "fitted_full_accuracy": fitted_full_accuracy,
                    "full_accuracy_abs_diff": abs(
                        fitted_full_accuracy
                        - float(target_row.target_full_acc)
                    ),
                    "full_fit_iterations": int(full_model.n_iter_[0]),
                }
            )

            for seed in seeds:
                pilot = balanced_pilot(
                    probe_labels,
                    budgets[0],
                    initial_rng(pair_id, backbone_id, seed),
                )
                pilot_pair_indices = probe_idx[pilot]
                leakage = int(
                    np.intersect1d(pilot_pair_indices, diag_idx).size
                )
                pilot_model, pilot_accuracy, pilot_iterations = fit_and_score(
                    probe_features,
                    probe_labels,
                    pilot,
                    test_features,
                    pair_test_labels,
                )
                distances = protocol.signed_distance(
                    pilot_model, diagnostic_features
                )
                psi, _, eps = protocol.psi_metrics(
                    distances, diagnostic_labels
                )
                pilot_indices = protocol.sample_indices(
                    np.sort(pilot_pair_indices)
                )
                diag_indices = protocol.sample_indices(np.sort(diag_idx))
                pilot_rows.append(
                    {
                        "pair_id": pair_id,
                        "class_a": class_a,
                        "class_b": class_b,
                        "backbone": backbone,
                        "backbone_id": backbone_id,
                        "seed": seed,
                        "pilot_spc": budgets[0],
                        "pilot_n": int(len(pilot)),
                        "pilot_class0_n": int(
                            np.sum(probe_labels[pilot] == 0)
                        ),
                        "pilot_class1_n": int(
                            np.sum(probe_labels[pilot] == 1)
                        ),
                        "pilot_indices": pilot_indices,
                        "diag_indices": diag_indices,
                        "leakage_count": leakage,
                        "psi_eff": psi,
                        "log1p_psi": math.log1p(psi),
                        "psi_eps": eps,
                        "pilot_accuracy": pilot_accuracy,
                        "pilot_fit_iterations": pilot_iterations,
                        "target_accuracy": target,
                    }
                )

                for policy in POLICIES:
                    rng = policy_rng(
                        policy, pair_id, backbone_id, seed
                    )
                    selected = pilot.copy()
                    model = pilot_model
                    curve_rows.append(
                        {
                            "pair_id": pair_id,
                            "class_a": class_a,
                            "class_b": class_b,
                            "backbone": backbone,
                            "backbone_id": backbone_id,
                            "seed": seed,
                            "policy": policy,
                            "spc": budgets[0],
                            "selected_n": int(len(selected)),
                            "selected_class0_n": int(
                                np.sum(probe_labels[selected] == 0)
                            ),
                            "selected_class1_n": int(
                                np.sum(probe_labels[selected] == 1)
                            ),
                            "selection_indices": protocol.sample_indices(
                                np.sort(probe_idx[selected])
                            ),
                            "pilot_indices": pilot_indices,
                            "diag_indices": diag_indices,
                            "leakage_count": leakage,
                            "accuracy": pilot_accuracy,
                            "target_accuracy": target,
                            "psi_eff": psi,
                            "log1p_psi": math.log1p(psi),
                            "fit_iterations": pilot_iterations,
                        }
                    )

                    current_spc = budgets[0]
                    for next_spc in budgets[1:]:
                        total_additions = 2 * (
                            next_spc - current_spc
                        )
                        selected_mask = np.zeros(
                            len(probe_labels), dtype=bool
                        )
                        selected_mask[selected] = True
                        candidates = np.flatnonzero(~selected_mask)
                        uncertainty_count = 0
                        random_count = 0
                        cutoff = float("nan")
                        next_margin = float("nan")

                        if policy == "random":
                            additions = balanced_random_additions(
                                probe_labels,
                                selected,
                                next_spc - current_spc,
                                rng,
                            )
                            random_count = len(additions)
                        elif policy == "uncertainty":
                            (
                                additions,
                                cutoff,
                                next_margin,
                            ) = uncertainty_additions(
                                model,
                                probe_features,
                                candidates,
                                total_additions,
                            )
                            uncertainty_count = len(additions)
                        elif policy == "hybrid":
                            uncertainty_count = total_additions // 2
                            random_count = (
                                total_additions - uncertainty_count
                            )
                            (
                                uncertainty_part,
                                cutoff,
                                next_margin,
                            ) = uncertainty_additions(
                                model,
                                probe_features,
                                candidates,
                                uncertainty_count,
                            )
                            random_candidates = candidates[
                                ~np.isin(candidates, uncertainty_part)
                            ]
                            random_part = (
                                approximately_balanced_random_additions(
                                    probe_labels,
                                    random_candidates,
                                    random_count,
                                    rng,
                                )
                            )
                            additions = np.concatenate(
                                [uncertainty_part, random_part]
                            )
                        else:
                            raise ValueError(policy)

                        selected = np.concatenate(
                            [selected, additions]
                        ).astype("int64")
                        model, accuracy, iterations = fit_and_score(
                            probe_features,
                            probe_labels,
                            selected,
                            test_features,
                            pair_test_labels,
                        )
                        selected_pair_indices = probe_idx[selected]
                        leakage = int(
                            np.intersect1d(
                                selected_pair_indices, diag_idx
                            ).size
                        )
                        event_rows.append(
                            {
                                "pair_id": pair_id,
                                "backbone": backbone,
                                "seed": seed,
                                "policy": policy,
                                "from_spc": current_spc,
                                "to_spc": next_spc,
                                "added_n": int(len(additions)),
                                "uncertainty_added_n": uncertainty_count,
                                "random_added_n": random_count,
                                "uncertainty_cutoff": cutoff,
                                "next_unselected_margin": next_margin,
                                "uncertainty_order_valid": bool(
                                    np.isnan(cutoff)
                                    or cutoff <= next_margin + 1e-12
                                ),
                            }
                        )
                        curve_rows.append(
                            {
                                "pair_id": pair_id,
                                "class_a": class_a,
                                "class_b": class_b,
                                "backbone": backbone,
                                "backbone_id": backbone_id,
                                "seed": seed,
                                "policy": policy,
                                "spc": next_spc,
                                "selected_n": int(len(selected)),
                                "selected_class0_n": int(
                                    np.sum(
                                        probe_labels[selected] == 0
                                    )
                                ),
                                "selected_class1_n": int(
                                    np.sum(
                                        probe_labels[selected] == 1
                                    )
                                ),
                                "selection_indices": protocol.sample_indices(
                                    np.sort(selected_pair_indices)
                                ),
                                "pilot_indices": pilot_indices,
                                "diag_indices": diag_indices,
                                "leakage_count": leakage,
                                "accuracy": accuracy,
                                "target_accuracy": target,
                                "psi_eff": psi,
                                "log1p_psi": math.log1p(psi),
                                "fit_iterations": iterations,
                            }
                        )
                        current_spc = next_spc

            completed_units += 1
            print(
                f"[{completed_units:03d}/{total_units}] "
                f"backbone={backbone} pair={pair_id + 1}/{n_pairs} "
                f"elapsed={time.perf_counter() - started:.1f}s",
                flush=True,
            )

        del (
            train,
            test,
            train_features_all,
            train_labels_all,
            test_features_all,
            test_labels_all,
        )

    return (
        pd.DataFrame(curve_rows),
        pd.DataFrame(pilot_rows),
        pd.DataFrame(event_rows),
        pd.DataFrame(full_rows),
    )


def compute_budgets(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_columns = [
        "pair_id",
        "class_a",
        "class_b",
        "backbone",
        "seed",
        "policy",
    ]
    for keys, trajectory in curves.groupby(group_columns, sort=True):
        trajectory = trajectory.sort_values("spc")
        target = float(trajectory["target_accuracy"].iloc[0])
        budget, reached = interp_budget(
            trajectory["spc"].to_numpy(),
            trajectory["accuracy"].to_numpy(),
            target,
        )
        rows.append(
            {
                **dict(zip(group_columns, keys)),
                "psi_eff": float(trajectory["psi_eff"].iloc[0]),
                "log1p_psi": float(
                    trajectory["log1p_psi"].iloc[0]
                ),
                "target_accuracy": target,
                "realized_budget": budget,
                "target_reached": reached,
                "first_grid_budget": (
                    int(
                        trajectory.loc[
                            trajectory["accuracy"] >= target, "spc"
                        ].iloc[0]
                    )
                    if reached
                    else 150
                ),
            }
        )
    return pd.DataFrame(rows)


def task_seed_residual_spearman(frame: pd.DataFrame) -> float:
    current = frame[["pair_id", "seed", "log1p_psi", "realized_budget"]]
    psi_residual = current["log1p_psi"] - current.groupby(
        ["pair_id", "seed"]
    )["log1p_psi"].transform("mean")
    budget_residual = current["realized_budget"] - current.groupby(
        ["pair_id", "seed"]
    )["realized_budget"].transform("mean")
    return float(spearmanr(psi_residual, budget_residual).statistic)


def selection_regret(frame: pd.DataFrame) -> tuple[float, int, int]:
    regrets = []
    zero_regret = 0
    for _, unit in frame.groupby(["pair_id", "seed"], sort=True):
        chosen = unit.sort_values(
            ["psi_eff", "backbone"], ascending=[True, True]
        ).iloc[0]
        best = float(unit["realized_budget"].min())
        regret = float(chosen.realized_budget - best)
        regrets.append(regret)
        zero_regret += int(np.isclose(regret, 0.0))
    return float(np.mean(regrets)), zero_regret, len(regrets)

def summarize(budgets):
    rows = []
    for policy in POLICIES:
        current = budgets[budgets.policy == policy]
        averaged = current.groupby(["pair_id", "backbone"])[["log1p_psi", "realized_budget"]].mean()
        regret, _, _ = selection_regret(current)
        rows.append({
            "policy": DISPLAY_NAMES[policy],
            "mean_budget": current.realized_budget.mean(),
            "seed_averaged_spearman": spearmanr(averaged.log1p_psi, averaged.realized_budget).statistic,
            "selection_regret": regret,
            "task_seed_residual_spearman": task_seed_residual_spearman(current),
        })
    return pd.DataFrame(rows)


def main():
    global TRUTH_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-pairs", type=int, default=40)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--pilot-spc", type=int, choices=(2, 5), default=2)
    parser.add_argument("--target-ratio", type=float, default=0.95)
    parser.add_argument("--reference-budgets", type=Path, default=TRUTH_PATH)
    parser.add_argument("--curves", type=Path)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "outputs/acquisition")
    args = parser.parse_args()
    if args.n_seeds < 1 or not 0 < args.target_ratio <= 1:
        parser.error("Use a positive seed count and a target ratio in (0, 1].")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    TRUTH_PATH = args.reference_budgets
    if args.curves:
        curves = pd.read_csv(args.curves)
    else:
        grid = (2, 5, 10, 20, 50, 100) if args.pilot_spc == 2 else (5, 10, 20, 50, 100)
        curves, pilots, events, fitted = run_experiment(args.n_pairs, grid, tuple(range(args.n_seeds)), args.target_ratio)
        pilots.to_csv(args.output_dir / "pilot_scores.csv", index=False)
        events.to_csv(args.output_dir / "acquisition_events.csv", index=False)
    budgets = compute_budgets(curves)
    summary = summarize(budgets)
    curves.to_csv(args.output_dir / "learning_curves.csv", index=False)
    budgets.to_csv(args.output_dir / "budgets.csv", index=False)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
