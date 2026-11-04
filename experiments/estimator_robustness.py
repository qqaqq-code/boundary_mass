"Measure sensitivity to tube width and feature normalization. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, RUNS_DIR, FEATURES_DIR, RESULTS_DIR, result_file

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

ROOT = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
ART = RESULTS_DIR
ART.mkdir(parents=True, exist_ok=True)

CACHES = {
    "R50": RUNS_DIR / "cifar100_resnet50_pretrained/feature_cache_resnet50_imagenetv2",
    "ViT-B": RUNS_DIR / "cifar100_vitb16_targeted/feature_cache_vitb16_imagenetv1",
    "DINO-MS": FEATURES_DIR / "cifar100/dino_ms",
    "SigLIP-MS": FEATURES_DIR / "cifar100/siglip_ms",
}

N_PAIRS = 40
DIAG_FRAC = 0.30
RNG = np.random.default_rng(20260505)


def make_pairs() -> list[tuple[int, int]]:
    rng = np.random.default_rng(20260429)
    pairs: list[tuple[int, int]] = []
    for class_id in range(0, 40, 2):
        pairs.append((class_id, class_id + 1))
    seen = set(tuple(sorted(pair)) for pair in pairs)
    while len(pairs) < N_PAIRS:
        left, right = rng.choice(100, 2, replace=False)
        pair = tuple(sorted((int(left), int(right))))
        if pair not in seen:
            seen.add(pair)
            pairs.append(pair)
    return pairs


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    train = torch.load(path / "train_features.pt", map_location="cpu")
    return train["features"].float().numpy().astype("float32"), train[
        "labels"
    ].numpy().astype("int64")


def split_probe_diag(
    labels: np.ndarray, pair_id: int, backbone_id: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(88000 + 101 * pair_id + backbone_id)
    probe_parts, diag_parts = [], []
    for class_value in [0, 1]:
        class_indices = np.flatnonzero(labels == class_value)
        class_indices = rng.permutation(class_indices)
        diag_count = max(20, int(round(DIAG_FRAC * len(class_indices))))
        diag_count = min(diag_count, max(1, len(class_indices) - 60))
        diag_parts.append(class_indices[:diag_count])
        probe_parts.append(class_indices[diag_count:])
    return np.concatenate(probe_parts), np.concatenate(diag_parts)


def standardize(probe: np.ndarray, diag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = probe.mean(axis=0, keepdims=True)
    std = probe.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((probe - mean) / std).astype("float32"), ((diag - mean) / std).astype(
        "float32"
    )


def l2_normalize(probe: np.ndarray, diag: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    probe_std, diag_std = standardize(probe, diag)
    probe_norm = np.maximum(np.linalg.norm(probe_std, axis=1, keepdims=True), 1e-12)
    diag_norm = np.maximum(np.linalg.norm(diag_std, axis=1, keepdims=True), 1e-12)
    return (probe_std / probe_norm).astype("float32"), (diag_std / diag_norm).astype(
        "float32"
    )


def pca_whiten(
    probe: np.ndarray, diag: np.ndarray, components: int = 128
) -> tuple[np.ndarray, np.ndarray]:
    probe_std, diag_std = standardize(probe, diag)
    component_count = min(components, probe_std.shape[0] - 2, probe_std.shape[1])
    pca = PCA(
        n_components=component_count,
        whiten=True,
        svd_solver="randomized",
        random_state=123,
    )
    return pca.fit_transform(probe_std).astype("float32"), pca.transform(
        diag_std
    ).astype("float32")


def fit_binary_head(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        C=1.0, penalty="l2", solver="liblinear", max_iter=500, tol=1e-4, random_state=0
    )
    model.fit(features, labels)
    return model


def signed_distance(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    weights = model.coef_.reshape(-1)
    return model.decision_function(features) / (np.linalg.norm(weights) + 1e-12)


def psi_kernel(distances: np.ndarray, labels: np.ndarray, eps: float) -> float:
    beta = max(min(float(np.mean(labels == 1)), float(np.mean(labels == 0))), 1e-6)
    kernel = np.exp(-0.5 * (distances / eps) ** 2) / (eps * math.sqrt(2.0 * math.pi))
    return float(kernel.mean() / beta)


def hard_tube(distances: np.ndarray, labels: np.ndarray, eps: float) -> float:
    beta = max(min(float(np.mean(labels == 1)), float(np.mean(labels == 0))), 1e-6)
    return float(np.mean(np.abs(distances) <= eps) / (eps * beta))


def eps_grid(distances: np.ndarray) -> dict[str, float]:
    abs_distances = np.abs(distances)
    median = max(float(np.median(abs_distances)), 1e-6)
    return {
        "fixed0.05": 0.05,
        "q10": max(0.05, float(np.quantile(abs_distances, 0.10))),
        "q20": max(0.05, float(np.quantile(abs_distances, 0.20))),
        "q30": max(0.05, float(np.quantile(abs_distances, 0.30))),
        "margin0.25": max(0.05, 0.25 * median),
    }


def summarize_selection(records: pd.DataFrame, metric: str) -> dict[str, float]:
    truth = pd.read_csv(
        result_file(
            "backbone_selection/strict_label_budget_table.csv"
        )
    )
    merged = records[["pair_id", "backbone", metric]].merge(
        truth[["pair_id", "backbone", "label_budget", "aulc"]],
        on=["pair_id", "backbone"],
        how="inner",
    )
    rows = []
    for pair_id, subset in merged.groupby("pair_id"):
        best_budget = float(subset.label_budget.min())
        chosen = subset.sort_values(metric).iloc[0]
        rows.append(
            {
                "pair_id": int(pair_id),
                "regret": float(chosen.label_budget - best_budget),
                "rel_regret": float(
                    (chosen.label_budget - best_budget) / (best_budget + 1e-9)
                ),
                "aulc": float(chosen.aulc),
            }
        )
    summary = pd.DataFrame(rows)
    return {
        "mean_regret": float(summary.regret.mean()),
        "mean_rel_regret": float(summary.rel_regret.mean()),
        "mean_aulc": float(summary.aulc.mean()),
    }


def main() -> None:
    caches = {name: load_cache(path) for name, path in CACHES.items()}
    pairs = make_pairs()
    rows = []
    transforms = {
        "raw": lambda probe, diag: (probe.astype("float32"), diag.astype("float32")),
        "standardized": standardize,
        "l2_standardized": l2_normalize,
        "pca128_whitened": pca_whiten,
    }
    for pair_id, (class_a, class_b) in enumerate(pairs):
        for backbone_id, (backbone, (features_all, labels_all)) in enumerate(
            caches.items()
        ):
            mask = (labels_all == class_a) | (labels_all == class_b)
            features_pair = features_all[mask]
            labels_pair = (labels_all[mask] == class_b).astype(int)
            probe_idx, diag_idx = split_probe_diag(labels_pair, pair_id, backbone_id)
            probe_raw = features_pair[probe_idx]
            diag_raw = features_pair[diag_idx]
            probe_labels = labels_pair[probe_idx]
            diag_labels = labels_pair[diag_idx]
            for transform_name, transform_fn in transforms.items():
                probe_features, diag_features = transform_fn(probe_raw, diag_raw)
                model = fit_binary_head(probe_features, probe_labels)
                distances = signed_distance(model, diag_features)
                eps_values = eps_grid(distances)
                record = {
                    "pair_id": pair_id,
                    "class_a": class_a,
                    "class_b": class_b,
                    "backbone": backbone,
                    "transform": transform_name,
                }
                for eps_name, eps_value in eps_values.items():
                    record[f"psi_{eps_name}"] = psi_kernel(
                        distances, diag_labels, eps_value
                    )
                    record[f"hard_{eps_name}"] = hard_tube(
                        distances, diag_labels, eps_value
                    )
                rows.append(record)
        if (pair_id + 1) % 5 == 0:
            print(f"processed robustness pair {pair_id + 1}/{len(pairs)}", flush=True)

    records = pd.DataFrame(rows)
    records.to_csv(
        result_file(
            "estimator_robustness/estimator_tube_width_normalization_experiment.csv"
        ),
        index=False,
    )
    default = records[records["transform"] == "standardized"].copy()
    default["log_default"] = np.log1p(default["psi_q20"])

    stability_rows = []
    for column in [
        "psi_fixed0.05",
        "psi_q10",
        "psi_q20",
        "psi_q30",
        "psi_margin0.25",
        "hard_q20",
    ]:
        corr = spearmanr(
            np.log1p(default["psi_q20"]), np.log1p(default[column])
        ).correlation
        selection = summarize_selection(
            default.assign(metric=np.log1p(default[column])), "metric"
        )
        stability_rows.append(
            {
                "setting": column,
                "n": int(len(default)),
                "spearman_vs_default": float(corr),
                **selection,
            }
        )

    transform_rows = []
    base = records[records["transform"] == "standardized"][
        ["pair_id", "backbone", "psi_q20"]
    ].rename(columns={"psi_q20": "base"})
    for transform_name, subset in records.groupby("transform"):
        current = subset[["pair_id", "backbone", "psi_q20"]].merge(
            base, on=["pair_id", "backbone"], how="inner"
        )
        corr = spearmanr(
            np.log1p(current["base"]), np.log1p(current["psi_q20"])
        ).correlation
        selection = summarize_selection(
            subset.assign(metric=np.log1p(subset["psi_q20"])), "metric"
        )
        transform_rows.append(
            {
                "transform": transform_name,
                "n": int(len(current)),
                "spearman_vs_standardized": float(corr),
                **selection,
            }
        )

    stats = {
        "protocol": {
            "dataset": "CIFAR-100 strict 40-pair experiment",
            "backbones": list(CACHES),
            "tube_widths": ["fixed0.05", "q10", "q20", "q30", "margin0.25"],
            "normalizations": list(transforms),
            "target": "rank stability and label-budget regret under single-metric psi selection",
        },
        "tube_width": stability_rows,
        "normalization": transform_rows,
    }
    (
        result_file(
            "estimator_robustness/estimator_tube_width_normalization_stats.json"
        )
    ).write_text(json.dumps(stats, indent=2))

    width_tex = []
    label_map = {
        "psi_fixed0.05": "fixed $\\epsilon=0.05$",
        "psi_q10": "$q_{10}$ tube",
        "psi_q20": "$q_{20}$ tube",
        "psi_q30": "$q_{30}$ tube",
        "psi_margin0.25": "margin-scaled tube",
        "hard_q20": "hard $q_{20}$ tube",
    }
    for row in stability_rows:
        width_tex.append(
            f"{label_map[row['setting']]} & {row['n']} & ${row['spearman_vs_default']:.3f}$ & "
            f"${row['mean_regret']:.2f}$ & ${100 * row['mean_rel_regret']:.1f}$ & ${row['mean_aulc']:.3f}$\\\\"
        )
    (result_file("estimator_robustness/estimator_tube_width_rows.tex")).write_text(
        "\n".join(width_tex) + "\n"
    )

    transform_tex = []
    transform_map = {
        "raw": "raw features",
        "standardized": "standardized",
        "l2_standardized": "L2-normalized",
        "pca128_whitened": "PCA-whitened",
    }
    for row in transform_rows:
        transform_tex.append(
            f"{transform_map[row['transform']]} & {row['n']} & ${row['spearman_vs_standardized']:.3f}$ & "
            f"${row['mean_regret']:.2f}$ & ${100 * row['mean_rel_regret']:.1f}$ & ${row['mean_aulc']:.3f}$\\\\"
        )
    (result_file("estimator_robustness/estimator_normalization_rows.tex")).write_text(
        "\n".join(transform_tex) + "\n"
    )
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
