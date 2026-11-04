"Measure diagnostic-sampling reliability with a fixed fitted head. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, RUNS_DIR, FEATURES_DIR, RESULTS_DIR, result_file

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

ROOT = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
ART = RESULTS_DIR
ART.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "R50": RUNS_DIR / "cifar100_resnet50_pretrained/feature_cache_resnet50_imagenetv2",
    "ViT-B": RUNS_DIR / "cifar100_vitb16_targeted/feature_cache_vitb16_imagenetv1",
    "DINO-MS": FEATURES_DIR / "cifar100/dino_ms",
    "SigLIP-MS": FEATURES_DIR / "cifar100/siglip_ms",
}

N_PAIRS = 40
DIAG_FRAC = 0.30
N_SPLITS = 100
RNG = np.random.default_rng(20260502)


def has_cache(d: Path) -> bool:
    return (d / "train_features.pt").exists() and (d / "test_features.pt").exists()


def load_cache(d: Path) -> tuple[np.ndarray, np.ndarray]:
    tr = torch.load(d / "train_features.pt", map_location="cpu")
    return tr["features"].float().numpy().astype("float32"), tr[
        "labels"
    ].numpy().astype("int64")


def make_pairs() -> list[tuple[int, int]]:
    rng = np.random.default_rng(20260429)
    pairs: list[tuple[int, int]] = []
    for a in range(0, 40, 2):
        pairs.append((a, a + 1))
    seen = set(tuple(sorted(p)) for p in pairs)
    while len(pairs) < N_PAIRS:
        a, b = rng.choice(100, 2, replace=False)
        p = tuple(sorted((int(a), int(b))))
        if p not in seen:
            seen.add(p)
            pairs.append(p)
    return pairs


def split_probe_diag(
    y: np.ndarray, pair_id: int, backbone_id: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(88000 + 101 * pair_id + backbone_id)
    probe_parts, diag_parts = [], []
    for cls in [0, 1]:
        ids = np.flatnonzero(y == cls)
        ids = rng.permutation(ids)
        n_diag = max(20, int(round(DIAG_FRAC * len(ids))))
        n_diag = min(n_diag, max(1, len(ids) - 60))
        diag_parts.append(ids[:n_diag])
        probe_parts.append(ids[n_diag:])
    return np.concatenate(probe_parts), np.concatenate(diag_parts)


def standardize_from_probe(
    X_probe: np.ndarray, X_diag: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mu = X_probe.mean(axis=0, keepdims=True)
    sd = X_probe.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return ((X_probe - mu) / sd).astype("float32"), ((X_diag - mu) / sd).astype(
        "float32"
    )


def fit_lr(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        C=1.0, penalty="l2", solver="liblinear", max_iter=500, tol=1e-4, random_state=0
    )
    clf.fit(X, y)
    return clf


def signed_dist(clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    w = clf.coef_.reshape(-1)
    return clf.decision_function(X) / (np.linalg.norm(w) + 1e-12)


def psi_kernel(dist: np.ndarray, y: np.ndarray, eps: float) -> float:
    beta = max(min(float(np.mean(y == 1)), float(np.mean(y == 0))), 1e-6)
    kern = np.exp(-0.5 * (dist / eps) ** 2) / (eps * math.sqrt(2.0 * math.pi))
    return float(kern.mean() / beta)


def hard_tube(dist: np.ndarray, y: np.ndarray, eps: float) -> float:
    beta = max(min(float(np.mean(y == 1)), float(np.mean(y == 0))), 1e-6)
    return float(np.mean(np.abs(dist) <= eps) / (2.0 * eps * beta + 1e-12))


def stratified_halves(y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    a_parts, b_parts = [], []
    for cls in [0, 1]:
        ids = np.flatnonzero(y == cls)
        ids = rng.permutation(ids)
        mid = len(ids) // 2
        a_parts.append(ids[:mid])
        b_parts.append(ids[mid:])
    return np.concatenate(a_parts), np.concatenate(b_parts)


def ci(vals: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def main() -> None:
    pairs = make_pairs()
    backbones = {k: load_cache(v) for k, v in CANDIDATES.items() if has_cache(v)}
    records = []
    head_data = []
    for pair_id, (a, b) in enumerate(pairs):
        for backbone_id, (name, (Xtr_all, ytr_all)) in enumerate(backbones.items()):
            mtr = (ytr_all == a) | (ytr_all == b)
            X_pair = Xtr_all[mtr]
            y_pair = (ytr_all[mtr] == b).astype(int)
            probe_idx, diag_idx = split_probe_diag(y_pair, pair_id, backbone_id)
            X_probe_raw, y_probe = X_pair[probe_idx], y_pair[probe_idx]
            X_diag_raw, y_diag = X_pair[diag_idx], y_pair[diag_idx]
            X_probe, X_diag = standardize_from_probe(X_probe_raw, X_diag_raw)
            clf = fit_lr(X_probe, y_probe)
            dist = signed_dist(clf, X_diag)
            eps = max(0.05, float(np.quantile(np.abs(dist), 0.20)))
            psi = psi_kernel(dist, y_diag, eps)
            hard = hard_tube(dist, y_diag, eps)
            head_data.append(
                {
                    "pair_id": pair_id,
                    "backbone_id": backbone_id,
                    "backbone": name,
                    "dist": dist.astype("float64"),
                    "y": y_diag.astype("int64"),
                    "eps": eps,
                }
            )
            records.append(
                {
                    "pair_id": pair_id,
                    "class_a": a,
                    "class_b": b,
                    "backbone": name,
                    "diag_n": int(len(y_diag)),
                    "eps": eps,
                    "psi_eff": psi,
                    "log1p_psi": math.log1p(psi),
                    "hard_tube_psi": hard,
                    "log1p_hard_tube_psi": math.log1p(hard),
                }
            )
        if (pair_id + 1) % 10 == 0:
            print("prepared heads", pair_id + 1, flush=True)

    head_df = pd.DataFrame(records)
    head_df.to_csv(
        result_file("estimator_reliability/estimator_reliability_heads.csv"), index=False
    )

    # Recompute split-half estimates from features so each half is a real held-out sample.
    split_rows = []
    for split_id in range(N_SPLITS):
        a_vals, b_vals = [], []
        rel_errs = []
        for h in head_data:
            dist = h["dist"]
            y_diag = h["y"]
            eps = h["eps"]
            idx_a, idx_b = stratified_halves(
                y_diag,
                seed=20260502
                + 1009 * split_id
                + 17 * int(h["pair_id"])
                + int(h["backbone_id"]),
            )
            psi_a = math.log1p(psi_kernel(dist[idx_a], y_diag[idx_a], eps))
            psi_b = math.log1p(psi_kernel(dist[idx_b], y_diag[idx_b], eps))
            a_vals.append(psi_a)
            b_vals.append(psi_b)
            rel_errs.append(
                abs(psi_a - psi_b) / (0.5 * (abs(psi_a) + abs(psi_b)) + 1e-12)
            )
        split_rows.append(
            {
                "split_id": split_id,
                "pearson": float(pearsonr(a_vals, b_vals)[0]),
                "spearman": float(spearmanr(a_vals, b_vals)[0]),
                "median_relative_error": float(np.median(rel_errs)),
            }
        )
        if (split_id + 1) % 20 == 0:
            print("split", split_id + 1, flush=True)

    split_df = pd.DataFrame(split_rows)
    split_df.to_csv(
        result_file("estimator_reliability/estimator_split_half_reliability.csv"),
        index=False,
    )

    smooth = head_df["log1p_psi"].to_numpy()
    hard = head_df["log1p_hard_tube_psi"].to_numpy()
    hard_agreement = {
        "n": int(len(head_df)),
        "pearson": float(pearsonr(smooth, hard)[0]),
        "spearman": float(spearmanr(smooth, hard)[0]),
        "median_abs_log_diff": float(np.median(np.abs(smooth - hard))),
    }
    split_summary = {
        "n_splits": N_SPLITS,
        "n_heads": int(len(head_df)),
        "pearson_median": float(split_df.pearson.median()),
        "pearson_ci_low": ci(split_df.pearson)[0],
        "pearson_ci_high": ci(split_df.pearson)[1],
        "spearman_median": float(split_df.spearman.median()),
        "spearman_ci_low": ci(split_df.spearman)[0],
        "spearman_ci_high": ci(split_df.spearman)[1],
        "median_relative_error": float(split_df.median_relative_error.median()),
        "median_relative_error_ci_low": ci(split_df.median_relative_error)[0],
        "median_relative_error_ci_high": ci(split_df.median_relative_error)[1],
    }
    summary = {
        "protocol": {
            "diagnostic_split": "train/diagnostic-heldout split from the strict label-budget experiment",
            "eps": "fixed per head as max(0.05, q20(abs(signed distance))) on the full diagnostic-heldout split",
            "split_half_repetitions": N_SPLITS,
        },
        "split_half": split_summary,
        "hard_tube_vs_kernel": hard_agreement,
    }
    (result_file("estimator_reliability/estimator_reliability_summary.json")).write_text(
        json.dumps(summary, indent=2)
    )

    rows = [
        f"split-half $\\log(1+\\hat\\psi)$ & {split_summary['n_heads']} & "
        f"${split_summary['pearson_median']:.3f}[{split_summary['pearson_ci_low']:.3f},{split_summary['pearson_ci_high']:.3f}]$ & "
        f"${split_summary['spearman_median']:.3f}[{split_summary['spearman_ci_low']:.3f},{split_summary['spearman_ci_high']:.3f}]$ & "
        f"${100 * split_summary['median_relative_error']:.1f}[{100 * split_summary['median_relative_error_ci_low']:.1f},{100 * split_summary['median_relative_error_ci_high']:.1f}]\\%$\\\\",
        f"hard-tube vs. kernel & {hard_agreement['n']} & ${hard_agreement['pearson']:.3f}$ & "
        f"${hard_agreement['spearman']:.3f}$ & ${hard_agreement['median_abs_log_diff']:.3f}$\\\\",
    ]
    (result_file("estimator_reliability/estimator_reliability_rows.tex")).write_text(
        "\n".join(rows) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
