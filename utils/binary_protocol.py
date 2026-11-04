"""Shared binary-probe measurements and experiment inputs."""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from .paths import RUNS_DIR, FEATURES_DIR

DIAG_FRAC = 0.30
CACHES = {
    "R50": RUNS_DIR / "cifar100_resnet50_pretrained/feature_cache_resnet50_imagenetv2",
    "ViT-B": RUNS_DIR / "cifar100_vitb16_targeted/feature_cache_vitb16_imagenetv1",
    "DINO-MS": FEATURES_DIR / "cifar100/dino_ms",
    "SigLIP-MS": FEATURES_DIR / "cifar100/siglip_ms",
}

def load_features(path):
    import torch
    blob = torch.load(path, map_location="cpu", weights_only=True)
    return {key: blob[key].numpy() for key in ("features", "labels")}

def split_probe_diag(
    labels: np.ndarray, pair_id: int, backbone_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """Match the existing fixed 30% per-class diagnostic split."""

    rng = np.random.default_rng(88000 + 101 * pair_id + backbone_id)
    probe_parts: list[np.ndarray] = []
    diag_parts: list[np.ndarray] = []
    for class_value in (0, 1):
        class_indices = rng.permutation(np.flatnonzero(labels == class_value))
        diag_count = max(20, int(round(DIAG_FRAC * len(class_indices))))
        diag_count = min(diag_count, max(1, len(class_indices) - 60))
        diag_parts.append(class_indices[:diag_count])
        probe_parts.append(class_indices[diag_count:])
    return np.concatenate(probe_parts), np.concatenate(diag_parts)


def standardize_from_probe(
    probe: np.ndarray, diagnostic: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit normalization on the full probe pool, as in the fixed-split experiment."""

    mean = probe.mean(axis=0, keepdims=True)
    std = probe.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (
        ((probe - mean) / std).astype("float32"),
        ((diagnostic - mean) / std).astype("float32"),
    )


def fit_linear_head(features: np.ndarray, labels: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="liblinear",
        max_iter=500,
        tol=1e-4,
        random_state=0,
    )
    model.fit(features, labels)
    return model


def signed_distance(model: LogisticRegression, features: np.ndarray) -> np.ndarray:
    weight_norm = np.linalg.norm(model.coef_.reshape(-1)) + 1e-12
    return model.decision_function(features) / weight_norm


def psi_metrics(
    distances: np.ndarray, labels: np.ndarray
) -> tuple[float, float, float]:
    eps = max(0.05, float(np.quantile(np.abs(distances), 0.20)))
    beta = max(
        min(float(np.mean(labels == 1)), float(np.mean(labels == 0))), 1e-6
    )
    kernel = np.exp(-0.5 * (distances / eps) ** 2) / (
        eps * math.sqrt(2.0 * math.pi)
    )
    psi = float(kernel.mean() / beta)
    hard = float(
        np.mean(np.abs(distances) <= eps) / (2.0 * eps * beta + 1e-12)
    )
    return psi, hard, eps


def sample_indices(values: np.ndarray) -> str:
    """Store every selected index; equality can be checked directly."""
    return json.dumps(np.asarray(values, dtype=np.int64).tolist(), separators=(",", ":"))

