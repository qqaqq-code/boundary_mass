"""Compare four boundary estimators across 40 five-way tasks and five splits."""
from __future__ import annotations
import argparse
import itertools
import json
import math
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata, spearmanr
from utils import multiclass_protocol as protocol
from utils.paths import PROJECT_ROOT

DEFAULT_OUTPUT = PROJECT_ROOT / "outputs/multiclass_comparison"
PRIMARY_QUANTILE = 0.20
SMOOTH_TAU = 0.05
AVAILABLE_CLASSES = 80
DEFAULT_TASKS = 40
DEFAULT_SEEDS = 5
COMPARISONS = (
    (
        "Hard OvR vs. Hard Interface",
        "Multiclass aggregation",
        "hard_ovr",
        "hard_interface",
    ),
    (
        "Soft OvR vs. Hard OvR",
        "OvR smoothing",
        "soft_ovr",
        "hard_ovr",
    ),
    (
        "Soft Interface vs. Hard Interface",
        "Interface smoothing",
        "soft_interface",
        "hard_interface",
    ),
    (
        "Soft OvR vs. Soft Interface",
        "End-to-end implementation",
        "soft_ovr",
        "soft_interface",
    ),
)


def make_expanded_tasks(n_tasks: int) -> list[list[int]]:
    """Fixed balanced task sample; each class occurs nearly equally often."""
    if not 1 <= n_tasks <= 80:
        raise ValueError("n_tasks must be between 1 and 80")
    rng = np.random.default_rng(20260727)
    tasks: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    while len(tasks) < n_tasks:
        permutation = rng.permutation(AVAILABLE_CLASSES)
        for offset in range(0, AVAILABLE_CLASSES, protocol.WAYS):
            task = tuple(
                sorted(
                    int(value)
                    for value in permutation[
                        offset : offset + protocol.WAYS
                    ]
                )
            )
            if task in seen:
                continue
            seen.add(task)
            tasks.append(list(task))
            if len(tasks) == n_tasks:
                break
    return tasks


def split_probe_diag_seeded(
    labels: np.ndarray,
    task_id: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Shared across backbones for a task/seed; disjoint and stratified."""
    rng = np.random.default_rng(
        77000 + 1009 * task_id + 7919 * seed
    )
    probe_parts, diag_parts = [], []
    for class_value in sorted(np.unique(labels)):
        class_indices = np.flatnonzero(labels == class_value)
        class_indices = rng.permutation(class_indices)
        diag_count = max(
            25,
            int(round(protocol.DIAG_FRAC * len(class_indices))),
        )
        diag_count = min(
            diag_count, max(1, len(class_indices) - 60)
        )
        diag_parts.append(class_indices[:diag_count])
        probe_parts.append(class_indices[diag_count:])
    return np.concatenate(probe_parts), np.concatenate(diag_parts)


def gaussian_kernel(distance: np.ndarray, eps: float) -> np.ndarray:
    return np.exp(-0.5 * (distance / eps) ** 2) / (
        eps * math.sqrt(2.0 * math.pi)
    )


def gaussian_ovr(
    scores: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    eps: float,
) -> float:
    total = 0.0
    sample_count, class_count = scores.shape
    for class_offset in range(class_count):
        other_scores = scores.copy()
        other_scores[:, class_offset] = -np.inf
        rivals = np.argmax(other_scores, axis=1)
        gaps = scores[:, class_offset] - scores[
            np.arange(sample_count), rivals
        ]
        normals = weights[class_offset] - weights[rivals]
        distances = gaps / np.maximum(
            np.linalg.norm(normals, axis=1), 1e-12
        )
        class_probability = float(np.mean(labels == class_offset))
        beta = max(
            min(class_probability, 1.0 - class_probability), 1e-12
        )
        total += float(gaussian_kernel(distances, eps).mean() / beta)
    return total


def gaussian_interface(
    scores: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    eps: float,
    kappa: float,
) -> float:
    total = 0.0
    class_count = scores.shape[1]
    for class_a, class_b in itertools.combinations(
        range(class_count), 2
    ):
        normal = weights[class_a] - weights[class_b]
        distances = (
            scores[:, class_a] - scores[:, class_b]
        ) / max(np.linalg.norm(normal), 1e-12)
        pair_score = np.minimum(
            scores[:, class_a], scores[:, class_b]
        )
        other_scores = np.delete(scores, [class_a, class_b], axis=1)
        gate_margin = pair_score - other_scores.max(axis=1)
        gate = expit(gate_margin / kappa)
        probability_a = float(np.mean(labels == class_a))
        probability_b = float(np.mean(labels == class_b))
        beta = max(min(probability_a, probability_b), 1e-12)
        total += float(
            np.mean(gate * gaussian_kernel(distances, eps)) / beta
        )
    return total


def soft_tube_membership(
    distance: np.ndarray,
    eps: float,
    tau: float = SMOOTH_TAU,
) -> np.ndarray:
    """Paper surrogate for the hard indicator 1{|distance| <= eps}."""
    return expit((eps - np.abs(distance)) / tau)


def soft_ovr(
    scores: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    eps: float,
) -> float:
    """OvR score with the paper's sigmoid-smoothed tube membership."""
    total = 0.0
    sample_count, class_count = scores.shape
    for class_offset in range(class_count):
        other_scores = scores.copy()
        other_scores[:, class_offset] = -np.inf
        rivals = np.argmax(other_scores, axis=1)
        gaps = scores[:, class_offset] - scores[
            np.arange(sample_count), rivals
        ]
        normals = weights[class_offset] - weights[rivals]
        distances = gaps / np.maximum(
            np.linalg.norm(normals, axis=1), 1e-12
        )
        class_probability = float(np.mean(labels == class_offset))
        beta = max(
            min(class_probability, 1.0 - class_probability), 1e-12
        )
        total += float(
            soft_tube_membership(distances, eps).mean()
            / (eps * beta)
        )
    return total


def soft_interface(
    scores: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    top1: np.ndarray,
    top2: np.ndarray,
    eps: float,
) -> float:
    """Interface score with only the hard tube edge sigmoid-smoothed."""
    total = 0.0
    class_count = scores.shape[1]
    low = np.minimum(top1, top2)
    high = np.maximum(top1, top2)
    for class_a, class_b in itertools.combinations(
        range(class_count), 2
    ):
        active = (low == class_a) & (high == class_b)
        normal = weights[class_a] - weights[class_b]
        distances = (
            scores[:, class_a] - scores[:, class_b]
        ) / max(np.linalg.norm(normal), 1e-12)
        probability_a = float(np.mean(labels == class_a))
        probability_b = float(np.mean(labels == class_b))
        beta = max(min(probability_a, probability_b), 1e-12)
        total += float(
            np.mean(active * soft_tube_membership(distances, eps))
            / (eps * beta)
        )
    return total


def soft_interface_with_active_gate(
    scores: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    top1: np.ndarray,
    top2: np.ndarray,
    eps: float,
) -> float:
    """Gaussian tube smoothing with the hard argmax-interface gate fixed."""
    total = 0.0
    class_count = scores.shape[1]
    low = np.minimum(top1, top2)
    high = np.maximum(top1, top2)
    for class_a, class_b in itertools.combinations(
        range(class_count), 2
    ):
        active = (low == class_a) & (high == class_b)
        normal = weights[class_a] - weights[class_b]
        distances = (
            scores[:, class_a] - scores[:, class_b]
        ) / max(np.linalg.norm(normal), 1e-12)
        probability_a = float(np.mean(labels == class_a))
        probability_b = float(np.mean(labels == class_b))
        beta = max(min(probability_a, probability_b), 1e-12)
        total += float(
            np.mean(active * gaussian_kernel(distances, eps)) / beta
        )
    return total


def sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def comparison_summary(
    raw: pd.DataFrame,
    quantile: float,
    comparison: str,
    factor: str,
    left_name: str,
    right_name: str,
) -> dict[str, float | int | str]:
    suffix = f"q{int(round(quantile * 100)):02d}"
    left_column = f"{left_name}_{suffix}"
    right_column = f"{right_name}_{suffix}"
    task_spearman: list[float] = []
    rank_displacements: list[float] = []
    pairwise_matches = 0
    pairwise_total = 0
    selected_matches = 0

    for _, task in raw.groupby(["task_id", "seed"], sort=True):
        task = task.sort_values("backbone").reset_index(drop=True)
        left = task[left_column].to_numpy()
        right = task[right_column].to_numpy()
        task_spearman.append(
            float(spearmanr(left, right).statistic)
        )
        left_ranks = rankdata(left, method="average")
        right_ranks = rankdata(right, method="average")
        rank_displacements.extend(
            np.abs(left_ranks - right_ranks).tolist()
        )
        for left_index in range(len(task)):
            for right_index in range(left_index + 1, len(task)):
                left_sign = sign(
                    float(left[left_index] - left[right_index])
                )
                right_sign = sign(
                    float(right[left_index] - right[right_index])
                )
                pairwise_matches += int(left_sign == right_sign)
                pairwise_total += 1
        left_selected = str(
            task.loc[int(np.argmin(left)), "backbone"]
        )
        right_selected = str(
            task.loc[int(np.argmin(right)), "backbone"]
        )
        selected_matches += int(left_selected == right_selected)

    task_seed_units = int(
        raw[["task_id", "seed"]].drop_duplicates().shape[0]
    )
    return {
        "comparison": comparison,
        "factor": factor,
        "quantile": quantile,
        "tasks": int(raw["task_id"].nunique()),
        "seeds": int(raw["seed"].nunique()),
        "task_seed_units": task_seed_units,
        "backbones": int(raw["backbone"].nunique()),
        "within_task_spearman": float(np.mean(task_spearman)),
        "within_task_spearman_median": float(
            np.median(task_spearman)
        ),
        "pairwise_matches": pairwise_matches,
        "pairwise_total": pairwise_total,
        "pairwise_concordance": pairwise_matches / pairwise_total,
        "selected_matches": selected_matches,
        "selected_total": task_seed_units,
        "selected_backbone_agreement": (
            selected_matches / task_seed_units
        ),
        "median_rank_displacement": float(
            np.median(rank_displacements)
        ),
        "mean_rank_displacement": float(
            np.mean(rank_displacements)
        ),
    }

def compute_scores(n_tasks, n_seeds):
    tasks = make_expanded_tasks(n_tasks)
    seeds = list(range(n_seeds))
    started = time.monotonic()

    missing = [
        str(path)
        for path in protocol.CACHES.values()
        if not (path / "train_features.pt").exists()
    ]
    if missing:
        raise FileNotFoundError(f"Missing feature caches: {missing}")

    rows: list[dict[str, float | int | str]] = []
    balanced_diagnostic_classes = True
    common_label_sets: dict[str, set[int]] = {}
    for backbone_id, (backbone, cache_path) in enumerate(
        protocol.CACHES.items()
    ):
        features_all, labels_all = protocol.load_cache(cache_path)
        common_label_sets[backbone] = set(
            int(value) for value in np.unique(labels_all)
        )
        for task_id, classes in enumerate(tasks):
            task_mask = np.isin(labels_all, classes)
            task_features = features_all[task_mask]
            task_labels = protocol.relabel(
                labels_all[task_mask], classes
            )
            for seed in seeds:
                probe_idx, diag_idx = split_probe_diag_seeded(
                    task_labels, task_id, seed
                )
                probe_raw = task_features[probe_idx]
                diag_raw = task_features[diag_idx]
                probe_labels = task_labels[probe_idx]
                diag_labels = task_labels[diag_idx]
                counts = np.bincount(
                    diag_labels, minlength=protocol.WAYS
                )
                balanced_diagnostic_classes &= bool(
                    np.all(counts == counts[0])
                )
                probe_features, diag_features = (
                    protocol.standardize_from_probe(
                        probe_raw, diag_raw
                    )
                )
                model = protocol.fit_multiclass(
                    probe_features, probe_labels
                )
                scores = model.decision_function(diag_features)
                weights = model.coef_
                top1, top2, active_distances = (
                    protocol.active_interface_distances(scores, weights)
                )
                top_gap = (
                    scores[np.arange(len(scores)), top1]
                    - scores[np.arange(len(scores)), top2]
                )
                record: dict[str, float | int | str] = {
                    "task_id": task_id,
                    "classes": ",".join(map(str, classes)),
                    "seed": seed,
                    "backbone": backbone,
                    "probe_n": len(probe_idx),
                    "diag_n": len(diag_idx),
                    "diag_accuracy": float(
                        model.score(diag_features, diag_labels)
                    ),
                    "fit_iterations": int(model.n_iter_[0]),
                }
                for quantile in protocol.QUANTILES:
                    suffix = f"q{int(round(quantile * 100)):02d}"
                    eps = max(
                        0.05,
                        float(
                            np.quantile(
                                np.abs(active_distances), quantile
                            )
                        ),
                    )
                    kappa = max(
                        0.05,
                        float(np.quantile(top_gap, quantile)),
                    )
                    record[f"eps_{suffix}"] = eps
                    record[f"kappa_{suffix}"] = kappa
                    record[f"hard_ovr_{suffix}"] = (
                        protocol.hard_ovr(
                            scores, weights, diag_labels, eps
                        )
                    )
                    record[f"soft_ovr_{suffix}"] = soft_ovr(
                        scores, weights, diag_labels, eps
                    )
                    record[f"gaussian_ovr_{suffix}"] = gaussian_ovr(
                        scores, weights, diag_labels, eps
                    )
                    record[f"gaussian_ovr_varmatch_{suffix}"] = (
                        gaussian_ovr(
                            scores,
                            weights,
                            diag_labels,
                            eps / math.sqrt(3.0),
                        )
                    )
                    record[f"hard_interface_{suffix}"] = (
                        protocol.hard_interface(
                            scores,
                            weights,
                            diag_labels,
                            top1,
                            top2,
                            eps,
                        )
                    )
                    record[f"soft_interface_{suffix}"] = (
                        soft_interface(
                            scores,
                            weights,
                            diag_labels,
                            top1,
                            top2,
                            eps,
                        )
                    )
                    record[f"gaussian_interface_{suffix}"] = (
                        gaussian_interface(
                            scores,
                            weights,
                            diag_labels,
                            eps,
                            kappa,
                        )
                    )
                    record[
                        f"gaussian_interface_active_{suffix}"
                    ] = soft_interface_with_active_gate(
                        scores,
                        weights,
                        diag_labels,
                        top1,
                        top2,
                        eps,
                    )
                    record[
                        f"gaussian_interface_active_varmatch_{suffix}"
                    ] = soft_interface_with_active_gate(
                        scores,
                        weights,
                        diag_labels,
                        top1,
                        top2,
                        eps / math.sqrt(3.0),
                    )
                    record[
                        f"gaussian_interface_varmatch_{suffix}"
                    ] = gaussian_interface(
                        scores,
                        weights,
                        diag_labels,
                        eps / math.sqrt(3.0),
                        kappa,
                    )
                rows.append(record)
            if (task_id + 1) % 5 == 0:
                print(
                    f"processed {backbone}: "
                    f"{task_id + 1}/{len(tasks)} tasks",
                    flush=True,
                )
        del features_all, labels_all
        print(f"completed {backbone}", flush=True)

    raw = pd.DataFrame(rows).sort_values(
        ["task_id", "seed", "backbone"]
    ).reset_index(drop=True)
    return raw

def summarize_scores(raw):
    summaries = []
    for quantile in protocol.QUANTILES:
        for comparison, factor, left_name, right_name in COMPARISONS:
            summaries.append(
                comparison_summary(
                    raw,
                    quantile,
                    comparison,
                    factor,
                    left_name,
                    right_name,
                )
            )
    sensitivity = pd.DataFrame(summaries)
    primary = sensitivity[
        np.isclose(sensitivity["quantile"], PRIMARY_QUANTILE)
    ].copy()

    formatted = pd.DataFrame(
        {
            "Comparison": primary["comparison"],
            "Factor being tested": primary["factor"],
            "Within-task Spearman ↑": primary[
                "within_task_spearman"
            ].map(lambda value: f"{value:.3f}"),
            "Pairwise concordance ↑": primary[
                "pairwise_concordance"
            ].map(lambda value: f"{100 * value:.1f}%"),
            "Selected-backbone agreement ↑": primary[
                "selected_backbone_agreement"
            ].map(lambda value: f"{100 * value:.1f}%"),
            "Mean rank displacement ↓": primary[
                "mean_rank_displacement"
            ].map(lambda value: f"{value:.4f}"),
        }
    )

    return sensitivity, formatted

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-tasks", type=int, default=40)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--scores", type=Path)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not 1 <= args.n_seeds <= 20:
        parser.error("--n-seeds must be between 1 and 20")
    raw = pd.read_csv(args.scores) if args.scores else compute_scores(args.n_tasks, args.n_seeds)
    sensitivity, summary = summarize_scores(raw)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(args.output_dir / "scores.csv", index=False)
    sensitivity.to_csv(args.output_dir / "bandwidth_sensitivity.csv", index=False)
    summary.to_csv(args.output_dir / "comparison.csv", index=False)
    print(summary.to_string(index=False))

if __name__ == "__main__":
    main()
