"Compare OvR and interface aggregation on 16 five-way tasks. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.tiny_labels import align_labels
from utils.paths import PROJECT_ROOT, RUNS_DIR, FEATURES_DIR, RESULTS_DIR, result_file

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
CACHE_ROOT = PROJECT_ROOT
ART = RESULTS_DIR
ART.mkdir(parents=True, exist_ok=True)

R50_CACHE = (
    RUNS_DIR / "tiny_imagenet_resnet50_targeted/feature_cache_resnet50_imagenetv2"
)
TINY_CACHE = FEATURES_DIR / "tiny_imagenet"
CACHES = {
    "R50": R50_CACHE,
    "ViT-B": TINY_CACHE / "ViT-B",
    "DINO-MS": TINY_CACHE / "DINO-MS",
    "SigLIP-MS": TINY_CACHE / "SigLIP-MS",
}

N_TASKS = 16
WAYS = 5
DIAG_FRAC = 0.30
SPCS = [2, 5, 10, 20, 50]
SEEDS = [0, 1]
N_BOOT = 5000
RNG = np.random.default_rng(20260505)
MAX_DIM = 128

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


def make_tasks() -> list[list[int]]:
    return [
        list(range(task_id * WAYS, (task_id + 1) * WAYS)) for task_id in range(N_TASKS)
    ]


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = torch.load(path / "train_features.pt", map_location="cpu")
    val_name = (
        "val_features.pt" if (path / "val_features.pt").exists() else "test_features.pt"
    )
    val = torch.load(path / val_name, map_location="cpu")
    return (
        train["features"].float().numpy().astype("float32"),
        align_labels(train, path),
        val["features"].float().numpy().astype("float32"),
        align_labels(val, path),
    )


def relabel(labels: np.ndarray, classes: list[int]) -> np.ndarray:
    mapping = {class_id: offset for offset, class_id in enumerate(classes)}
    return np.asarray([mapping[int(label)] for label in labels], dtype="int64")


def split_probe_diag(
    labels: np.ndarray, task_id: int, backbone_id: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(66000 + 101 * task_id + backbone_id)
    probe_parts, diag_parts = [], []
    for class_value in sorted(np.unique(labels)):
        class_indices = np.flatnonzero(labels == class_value)
        class_indices = rng.permutation(class_indices)
        diag_count = max(25, int(round(DIAG_FRAC * len(class_indices))))
        diag_count = min(diag_count, max(1, len(class_indices) - 60))
        diag_parts.append(class_indices[:diag_count])
        probe_parts.append(class_indices[diag_count:])
    return np.concatenate(probe_parts), np.concatenate(diag_parts)


def standardize_from_probe(
    probe: np.ndarray, diag: np.ndarray, val: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = probe.mean(axis=0, keepdims=True)
    std = probe.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (
        ((probe - mean) / std).astype("float32"),
        ((diag - mean) / std).astype("float32"),
        ((val - mean) / std).astype("float32"),
    )


def fit_multiclass(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=700, tol=1e-4, random_state=0
    )
    model.fit(features, labels)
    return model


def ece_multiclass(probs: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == labels).astype(float)
    value = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lower, upper in zip(edges[:-1], edges[1:]):
        mask = (conf >= lower) & (conf < (upper if upper < 1 else upper + 1e-8))
        if mask.any():
            value += mask.mean() * abs(conf[mask].mean() - correct[mask].mean())
    return float(value)


def effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    eigenvalues = singular_values * singular_values / max(1, features.shape[0] - 1)
    total = eigenvalues.sum()
    return float(total * total / (np.square(eigenvalues).sum() + 1e-12))


def compact_features(features: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    centered = features.astype("float64", copy=False) - features.mean(
        axis=0, keepdims=True
    )
    if centered.shape[1] <= max_dim:
        return centered
    pca = PCA(
        n_components=min(max_dim, centered.shape[0] - 2, centered.shape[1]),
        svd_solver="randomized",
        random_state=123,
    )
    return pca.fit_transform(centered)


def h_score(features: np.ndarray, labels: np.ndarray) -> float:
    compact = compact_features(features)
    sample_count, dim = compact.shape
    covariance = (compact.T @ compact) / max(sample_count, 1)
    lam = 1e-3 * float(np.trace(covariance) / max(dim, 1))
    covariance = covariance + (lam + 1e-8) * np.eye(dim)
    global_mean = compact.mean(axis=0)
    between = np.zeros((dim, dim), dtype="float64")
    for class_value in np.unique(labels):
        class_features = compact[labels == class_value]
        diff = (class_features.mean(axis=0) - global_mean).reshape(-1, 1)
        between += (len(class_features) / sample_count) * (diff @ diff.T)
    return float(np.trace(np.linalg.solve(covariance, between)))


def coding_rate(features: np.ndarray, eps: float = 0.10) -> float:
    normalized = features.astype("float64", copy=False)
    sample_count, dim = normalized.shape
    if sample_count <= 1 or dim == 0:
        return 0.0
    normalized = normalized / np.maximum(
        np.linalg.norm(normalized, axis=1, keepdims=True), 1e-12
    )
    matrix = np.eye(dim) + (dim / (eps * eps)) * (
        (normalized.T @ normalized) / max(sample_count, 1)
    )
    sign, logdet = np.linalg.slogdet(matrix)
    return float(0.5 * logdet) if sign > 0 else float("nan")


def transrate_score(features: np.ndarray, labels: np.ndarray) -> float:
    compact = compact_features(features)
    total = coding_rate(compact)
    conditional = 0.0
    for class_value in np.unique(labels):
        class_features = compact[labels == class_value]
        conditional += (len(class_features) / len(labels)) * coding_rate(class_features)
    return float(total - conditional)


def logme_evidence_single(features: np.ndarray, target: np.ndarray) -> float:
    compact = compact_features(features)
    centered_target = target.astype("float64", copy=False) - target.mean()
    sample_count, dim = compact.shape
    left_vectors, singular_values, _ = np.linalg.svd(compact, full_matrices=False)
    projected_target = left_vectors.T @ centered_target
    target_norm = float(centered_target @ centered_target)
    alpha, beta = 1.0, 1.0
    for _ in range(50):
        denom = alpha + beta * singular_values * singular_values
        gamma = float(np.sum((beta * singular_values * singular_values) / denom))
        mean_coef = beta * singular_values * projected_target / denom
        mean_norm = float(np.sum(mean_coef * mean_coef))
        pred_coef = singular_values * mean_coef
        residual = max(
            float(
                target_norm
                - 2 * np.sum(pred_coef * projected_target)
                + np.sum(pred_coef * pred_coef)
            ),
            1e-12,
        )
        new_alpha = float(np.clip(gamma / max(mean_norm, 1e-12), 1e-8, 1e8))
        new_beta = float(np.clip((sample_count - gamma) / residual, 1e-8, 1e8))
        if abs(new_alpha - alpha) / alpha < 1e-4 and abs(new_beta - beta) / beta < 1e-4:
            alpha, beta = new_alpha, new_beta
            break
        alpha, beta = new_alpha, new_beta
    denom = alpha + beta * singular_values * singular_values
    mean_coef = beta * singular_values * projected_target / denom
    mean_norm = float(np.sum(mean_coef * mean_coef))
    pred_coef = singular_values * mean_coef
    residual = max(
        float(
            target_norm
            - 2 * np.sum(pred_coef * projected_target)
            + np.sum(pred_coef * pred_coef)
        ),
        1e-12,
    )
    evidence = (
        0.5 * dim * math.log(alpha)
        + 0.5 * sample_count * math.log(beta)
        - 0.5 * np.sum(np.log(denom))
        - 0.5 * beta * residual
        - 0.5 * alpha * mean_norm
        - 0.5 * sample_count * math.log(2 * math.pi)
    )
    return float(evidence / sample_count)


def logme_score(features: np.ndarray, labels: np.ndarray) -> float:
    return float(
        np.mean(
            [
                logme_evidence_single(
                    features, (labels == class_value).astype("float64")
                )
                for class_value in np.unique(labels)
            ]
        )
    )


def nce_cluster_score(features: np.ndarray, labels: np.ndarray) -> float:
    compact = compact_features(features)
    cluster_count = int(min(30, max(2, len(labels) // 15)))
    clusters = KMeans(n_clusters=cluster_count, n_init=5, random_state=123).fit_predict(
        compact
    )
    entropy = 0.0
    class_values = sorted(np.unique(labels))
    for cluster_id in range(cluster_count):
        mask = clusters == cluster_id
        if not mask.any():
            continue
        cluster_prob = float(mask.mean())
        for class_value in class_values:
            class_prob = float(np.mean(labels[mask] == class_value))
            if class_prob > 1e-12:
                entropy -= cluster_prob * class_prob * math.log(class_prob)
    return float(-entropy)


def ovr_boundary_mass(
    model: LogisticRegression, features: np.ndarray, labels: np.ndarray
) -> float:
    scores = model.decision_function(features)
    weights = model.coef_
    class_values = list(model.classes_)
    total = 0.0
    for class_offset, class_value in enumerate(class_values):
        other_scores = scores.copy()
        other_scores[:, class_offset] = -np.inf
        other_offsets = np.argmax(other_scores, axis=1)
        score_gap = (
            scores[:, class_offset] - scores[np.arange(len(scores)), other_offsets]
        )
        normal_norms = np.linalg.norm(
            weights[class_offset] - weights[other_offsets], axis=1
        )
        distances = score_gap / np.maximum(normal_norms, 1e-12)
        eps = max(0.05, float(np.quantile(np.abs(distances), 0.20)))
        class_mask = labels == class_value
        beta = max(min(float(class_mask.mean()), 1.0 - float(class_mask.mean())), 1e-6)
        kernel = np.exp(-0.5 * (distances / eps) ** 2) / (
            eps * math.sqrt(2.0 * math.pi)
        )
        total += float(kernel.mean() / beta)
    return total


def interface_boundary_mass(
    model: LogisticRegression, features: np.ndarray, labels: np.ndarray
) -> float:
    scores = model.decision_function(features)
    weights = model.coef_
    class_values = list(model.classes_)
    top_gap = np.sort(scores, axis=1)[:, -1] - np.sort(scores, axis=1)[:, -2]
    kappa = max(0.05, float(np.quantile(top_gap, 0.20)))
    total = 0.0
    for offset_a, class_a in enumerate(class_values):
        for offset_b in range(offset_a + 1, len(class_values)):
            class_b = class_values[offset_b]
            pair_scores = np.minimum(scores[:, offset_a], scores[:, offset_b])
            other_scores = np.delete(scores, [offset_a, offset_b], axis=1)
            gate_margin = pair_scores - other_scores.max(axis=1)
            gate = 1.0 / (1.0 + np.exp(-gate_margin / kappa))
            normal = weights[offset_a] - weights[offset_b]
            distances = (scores[:, offset_a] - scores[:, offset_b]) / max(
                np.linalg.norm(normal), 1e-12
            )
            eps = max(0.05, float(np.quantile(np.abs(distances), 0.20)))
            kernel = (
                gate
                * np.exp(-0.5 * (distances / eps) ** 2)
                / (eps * math.sqrt(2.0 * math.pi))
            )
            prob_a = float(np.mean(labels == class_a))
            prob_b = float(np.mean(labels == class_b))
            beta = max(min(prob_a, prob_b), 1e-6)
            total += float(kernel.mean() / beta)
    return total


def interp_budget(
    spcs: np.ndarray, accs: np.ndarray, target: float
) -> tuple[float, bool]:
    order = np.argsort(spcs)
    spcs = np.asarray(spcs, dtype=float)[order]
    accs = np.asarray(accs, dtype=float)[order]
    if np.nanmax(accs) < target:
        return float(spcs.max() * 1.5), False
    if accs[0] >= target:
        return float(spcs[0]), True
    for index in range(1, len(spcs)):
        if accs[index] >= target:
            lower_x, upper_x = np.log(spcs[index - 1]), np.log(spcs[index])
            lower_y, upper_y = accs[index - 1], accs[index]
            fraction = (
                0.0
                if abs(upper_y - lower_y) < 1e-9
                else (target - lower_y) / (upper_y - lower_y)
            )
            return float(np.exp(lower_x + fraction * (upper_x - lower_x))), True
    return float(spcs.max() * 1.5), False


def compute_metrics() -> pd.DataFrame:
    caches = {name: load_cache(path) for name, path in CACHES.items()}
    metric_rows, curve_rows = [], []
    for task_id, classes in enumerate(make_tasks()):
        for backbone_id, (
            backbone,
            (train_features_all, train_labels_all, val_features_all, val_labels_all),
        ) in enumerate(caches.items()):
            train_mask = np.isin(train_labels_all, classes)
            val_mask = np.isin(val_labels_all, classes)
            train_features = train_features_all[train_mask]
            train_labels = relabel(train_labels_all[train_mask], classes)
            val_features = val_features_all[val_mask]
            val_labels = relabel(val_labels_all[val_mask], classes)
            probe_idx, diag_idx = split_probe_diag(train_labels, task_id, backbone_id)
            probe_raw, diag_raw = train_features[probe_idx], train_features[diag_idx]
            probe_labels, diag_labels = train_labels[probe_idx], train_labels[diag_idx]
            probe_features, diag_features, val_features_std = standardize_from_probe(
                probe_raw, diag_raw, val_features
            )
            full_model = fit_multiclass(probe_features, probe_labels)
            diag_probs = full_model.predict_proba(diag_features)
            diag_scores = full_model.decision_function(diag_features)
            sorted_scores = np.sort(diag_scores, axis=1)
            full_acc = float(full_model.score(val_features_std, val_labels))
            psi_eff = ovr_boundary_mass(full_model, diag_features, diag_labels)
            psi_interface = interface_boundary_mass(
                full_model, diag_features, diag_labels
            )
            metric_rows.append(
                {
                    "task_id": task_id,
                    "classes": ",".join(map(str, classes)),
                    "backbone": backbone,
                    "ways": WAYS,
                    "probe_n": int(len(probe_labels)),
                    "diag_n": int(len(diag_labels)),
                    "target_full_acc": full_acc,
                    "psi_eff": psi_eff,
                    "log1p_psi": math.log1p(psi_eff),
                    "psi_interface": psi_interface,
                    "log1p_psi_interface": math.log1p(psi_interface),
                    "median_margin": float(
                        np.median(sorted_scores[:, -1] - sorted_scores[:, -2])
                    ),
                    "diag_acc": float(
                        (diag_probs.argmax(axis=1) == diag_labels).mean()
                    ),
                    "diag_nll": float(
                        log_loss(diag_labels, diag_probs, labels=list(range(WAYS)))
                    ),
                    "diag_ece": ece_multiclass(diag_probs, diag_labels),
                    "effective_rank": effective_rank(diag_features),
                    "logme": logme_score(diag_features, diag_labels),
                    "h_score": h_score(diag_features, diag_labels),
                    "transrate": transrate_score(diag_features, diag_labels),
                    "nce_cluster": nce_cluster_score(diag_features, diag_labels),
                }
            )
            class_indices = {
                class_value: np.flatnonzero(probe_labels == class_value)
                for class_value in range(WAYS)
            }
            for spc in SPCS:
                for seed in SEEDS:
                    rng = np.random.default_rng(
                        91000 + task_id * 97 + backbone_id * 13 + spc * 5 + seed
                    )
                    selected = []
                    for class_value, indices in class_indices.items():
                        selected.extend(
                            rng.choice(
                                indices, min(spc, len(indices)), replace=False
                            ).tolist()
                        )
                    selected = np.asarray(selected, dtype=int)
                    small_model = fit_multiclass(
                        probe_features[selected], probe_labels[selected]
                    )
                    curve_rows.append(
                        {
                            "task_id": task_id,
                            "backbone": backbone,
                            "spc": spc,
                            "seed": seed,
                            "acc": float(
                                small_model.score(val_features_std, val_labels)
                            ),
                        }
                    )
        print(f"processed multiclass task {task_id + 1}/{N_TASKS}", flush=True)

    metrics = pd.DataFrame(metric_rows)
    curves = (
        pd.DataFrame(curve_rows)
        .groupby(["task_id", "backbone", "spc"], as_index=False)
        .acc.mean()
    )
    rows = []
    best_full = metrics.groupby("task_id").target_full_acc.max().to_dict()
    for _, metric_row in metrics.iterrows():
        subset = curves[
            (curves.task_id == metric_row.task_id)
            & (curves.backbone == metric_row.backbone)
        ].sort_values("spc")
        target = 0.90 * best_full[int(metric_row.task_id)]
        budget, reached = interp_budget(subset.spc.values, subset.acc.values, target)
        log_spc = np.log(subset.spc.values.astype(float))
        acc = subset.acc.values.astype(float)
        rows.append(
            {
                **metric_row.to_dict(),
                "target_acc": target,
                "label_budget": budget,
                "target_reached": reached,
                "aulc": float(np.trapz(acc, log_spc) / (log_spc[-1] - log_spc[0])),
                "acc_5spc": float(subset[subset.spc == 5].acc.iloc[0]),
            }
        )
    table = pd.DataFrame(rows)
    metrics.to_csv(
        result_file("multiclass_selection/tiny_multiclass_metrics.csv"), index=False
    )
    curves.to_csv(
        result_file("multiclass_selection/tiny_multiclass_learning_curves.csv"),
        index=False,
    )
    table.to_csv(
        result_file("multiclass_selection/tiny_multiclass_label_budget_table.csv"),
        index=False,
    )
    return table


def ci(values: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def policy_rows(
    data: pd.DataFrame, task_col: str = "task_id", psi_column: str = "log1p_psi"
) -> pd.DataFrame:
    rows = []
    for task_id, subset in data.groupby(task_col):
        train = data[data[task_col] != task_id]
        model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        model.fit(train[NONPSI_FEATURES], train["label_budget"])
        test = subset.copy()
        test["pred_budget"] = model.predict(test[NONPSI_FEATURES])
        best_budget = float(test.label_budget.min())
        base = test.sort_values("pred_budget").iloc[0]
        top2 = test.sort_values("pred_budget").head(2).sort_values(psi_column).iloc[0]
        rows.append(
            {
                task_col: int(task_id),
                "base_chosen": base.backbone,
                "psi_chosen": top2.backbone,
                "base_regret": float(base.label_budget - best_budget),
                "psi_regret": float(top2.label_budget - best_budget),
                "budget_reduction": float(base.label_budget - top2.label_budget),
                "base_aulc": float(base.aulc),
                "psi_aulc": float(top2.aulc),
                "psi_win": float(top2.label_budget < base.label_budget),
                "same_budget": float(top2.label_budget == base.label_budget),
            }
        )
    return pd.DataFrame(rows)


def summarize_policy(rows: pd.DataFrame, task_col: str) -> dict[str, float]:
    task_ids = np.array(sorted(rows[task_col].unique()))
    boot = {
        key: [] for key in ["base_regret", "psi_regret", "budget_reduction", "psi_aulc"]
    }
    for _ in range(N_BOOT):
        sampled = RNG.choice(task_ids, size=len(task_ids), replace=True)
        sample = pd.concat(
            [rows[rows[task_col] == task_id] for task_id in sampled], ignore_index=True
        )
        for key in boot:
            boot[key].append(float(sample[key].mean()))
    reductions = rows.budget_reduction.to_numpy()
    observed = float(reductions.mean())
    centered = reductions - observed
    null = []
    for _ in range(N_BOOT):
        signs = RNG.choice([-1.0, 1.0], size=len(centered))
        null.append(float((centered * signs).mean()))
    return {
        "n": int(len(task_ids)),
        "base_regret": float(rows.base_regret.mean()),
        "base_regret_ci": ci(boot["base_regret"]),
        "psi_regret": float(rows.psi_regret.mean()),
        "psi_regret_ci": ci(boot["psi_regret"]),
        "budget_reduction": observed,
        "budget_reduction_ci": ci(boot["budget_reduction"]),
        "paired_sign_p": float(
            (1.0 + sum(value >= observed for value in null)) / (len(null) + 1.0)
        ),
        "psi_win_rate": float(rows.psi_win.mean()),
        "same_budget_rate": float(rows.same_budget.mean()),
        "psi_aulc": float(rows.psi_aulc.mean()),
        "psi_aulc_ci": ci(boot["psi_aulc"]),
    }


def incremental_r2(data: pd.DataFrame) -> dict[str, float]:
    current = (
        data[["task_id", "label_budget", "log1p_psi"] + NONPSI_FEATURES]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    for column in ["label_budget", "log1p_psi"] + NONPSI_FEATURES:
        current[column + "_r"] = current[column] - current.groupby("task_id")[
            column
        ].transform("mean")
    target = current["label_budget_r"].to_numpy()
    base_features = current[[column + "_r" for column in NONPSI_FEATURES]].to_numpy()
    plus_features = current[
        [column + "_r" for column in NONPSI_FEATURES + ["log1p_psi"]]
    ].to_numpy()
    base_r2 = r2_score(
        target, Ridge(alpha=1.0).fit(base_features, target).predict(base_features)
    )
    plus_r2 = r2_score(
        target, Ridge(alpha=1.0).fit(plus_features, target).predict(plus_features)
    )
    psi_resid = current["log1p_psi_r"].to_numpy() - Ridge(alpha=1.0).fit(
        base_features, current["log1p_psi_r"].to_numpy()
    ).predict(base_features)
    y_resid = target - Ridge(alpha=1.0).fit(base_features, target).predict(
        base_features
    )
    return {
        "base_r2": float(base_r2),
        "plus_psi_r2": float(plus_r2),
        "delta_r2": float(plus_r2 - base_r2),
        "partial_pearson": float(pearsonr(psi_resid, y_resid)[0]),
        "partial_spearman": float(spearmanr(psi_resid, y_resid)[0]),
    }


def tiny_binary_policy() -> dict[str, float]:
    tiny_path = result_file(
        "tiny_imagenet/tiny_strict_label_budget_table.csv"
    )
    if not tiny_path.exists():
        tiny_path = result_file(
            "tiny_imagenet/tiny_strict_label_budget_table.csv"
        )
    table = pd.read_csv(tiny_path).rename(columns={"pair_id": "task_id"})
    rows = policy_rows(table, "task_id")
    rows.to_csv(
        result_file("tiny_imagenet/tiny_binary_shortlist_policy.csv"),
        index=False,
    )
    return summarize_policy(rows, "task_id")


def write_external_policy_rows(
    cifar_summary: dict[str, float],
    tiny_binary: dict[str, float],
    multiclass: dict[str, float],
) -> None:
    rows = []
    for setting, tasks, backbones, summary in [
        ("CIFAR-100 binary", 40, 4, cifar_summary),
        ("Tiny-ImageNet binary", tiny_binary["n"], 4, tiny_binary),
        ("Tiny-ImageNet 5-way", multiclass["n"], 4, multiclass),
    ]:
        rows.append(
            f"{setting} & {tasks} & {backbones} & ${summary['base_regret']:.3f}$ & "
            f"${summary['psi_regret']:.3f}$ & ${summary['budget_reduction']:.3f}$ & "
            f"${summary['paired_sign_p']:.3f}$ & ${summary['psi_aulc']:.3f}$\\\\"
        )
    (result_file("multiclass_selection/external_policy_rows.tex")).write_text(
        "\n".join(rows) + "\n"
    )


def main() -> None:
    table = compute_metrics()
    rows = policy_rows(table, "task_id")
    rows.to_csv(
        result_file("multiclass_selection/tiny_multiclass_shortlist_policy.csv"),
        index=False,
    )
    interface_rows = policy_rows(table, "task_id", "log1p_psi_interface")
    interface_rows.to_csv(
        result_file(
            "multiclass_selection/tiny_multiclass_interface_shortlist_policy.csv"
        ),
        index=False,
    )
    multiclass_summary = summarize_policy(rows, "task_id")
    interface_summary = summarize_policy(interface_rows, "task_id")
    tiny_binary_summary = tiny_binary_policy()
    cifar_path = result_file("near_tie_selection/near_tie_shortlist_stats.json")
    if not cifar_path.exists():
        cifar_path = result_file("near_tie_selection/near_tie_shortlist_stats.json")
    cifar_stats = json.loads(cifar_path.read_text())
    cifar_summary = next(row for row in cifar_stats["shortlist"] if row["k"] == 2)
    inc = incremental_r2(table)
    stats = {
        "protocol": {
            "dataset": "Tiny-ImageNet 5-way multiclass",
            "tasks": N_TASKS,
            "ways": WAYS,
            "backbones": list(CACHES),
            "policy": "leave-one-task-out non-psi Ridge top-2 shortlist, then lowest boundary-mass score",
        },
        "ovr_policy": multiclass_summary,
        "interface_policy": interface_summary,
        "tiny_binary_policy": tiny_binary_summary,
        "incremental_r2": inc,
    }
    (
        result_file("multiclass_selection/tiny_multiclass_validation_stats.json")
    ).write_text(json.dumps(stats, indent=2))
    write_external_policy_rows(cifar_summary, tiny_binary_summary, multiclass_summary)
    row = (
        f"Tiny-ImageNet 5-way & {N_TASKS} & 4 & ${multiclass_summary['base_regret']:.3f}$ & "
        f"${multiclass_summary['psi_regret']:.3f}$ & ${multiclass_summary['budget_reduction']:.3f}$ & "
        f"${multiclass_summary['paired_sign_p']:.3f}$ & ${multiclass_summary['psi_aulc']:.3f}$\\\\"
    )
    (result_file("multiclass_selection/tiny_multiclass_policy_row.tex")).write_text(
        row + "\n"
    )
    interface_row = (
        f"Tiny-ImageNet 5-way interface & {N_TASKS} & 4 & ${interface_summary['base_regret']:.3f}$ & "
        f"${interface_summary['psi_regret']:.3f}$ & ${interface_summary['budget_reduction']:.3f}$ & "
        f"${interface_summary['paired_sign_p']:.3f}$ & ${interface_summary['psi_aulc']:.3f}$\\\\"
    )
    (
        result_file("multiclass_selection/tiny_multiclass_interface_policy_row.tex")
    ).write_text(interface_row + "\n")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
