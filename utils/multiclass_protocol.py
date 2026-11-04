"""Five-way linear probes and active multiclass boundary measurements."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from .paths import RUNS_DIR, FEATURES_DIR
from .tiny_labels import align_labels

WAYS = 5
DIAG_FRAC = 0.30
QUANTILES = (0.10, 0.20, 0.30)
CACHES = {
    "R50": RUNS_DIR / "tiny_imagenet_resnet50_targeted/feature_cache_resnet50_imagenetv2",
    "ViT-B": FEATURES_DIR / "tiny_imagenet/ViT-B",
    "DINO-MS": FEATURES_DIR / "tiny_imagenet/DINO-MS",
    "SigLIP-MS": FEATURES_DIR / "tiny_imagenet/SigLIP-MS",
}

def load_cache(path):
    blob = torch.load(path / "train_features.pt", map_location="cpu", weights_only=True)
    return blob["features"].numpy().astype("float32"), align_labels(blob, path)

def relabel(labels: np.ndarray, classes: list[int]) -> np.ndarray:
    mapping = {class_id: offset for offset, class_id in enumerate(classes)}
    return np.asarray([mapping[int(label)] for label in labels], dtype="int64")


def standardize_from_probe(
    probe: np.ndarray, diag: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    mean = probe.mean(axis=0, keepdims=True)
    std = probe.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return (
        ((probe - mean) / std).astype("float32"),
        ((diag - mean) / std).astype("float32"),
    )


def fit_multiclass(
    features: np.ndarray, labels: np.ndarray
) -> LogisticRegression:
    model = LogisticRegression(
        C=1.0,
        penalty="l2",
        solver="lbfgs",
        max_iter=700,
        tol=1e-4,
        random_state=0,
    )
    model.fit(features, labels)
    return model


def active_interface_distances(
    scores: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(scores, axis=1)
    top1 = order[:, -1]
    top2 = order[:, -2]
    gaps = scores[np.arange(len(scores)), top1] - scores[
        np.arange(len(scores)), top2
    ]
    normals = weights[top1] - weights[top2]
    distances = gaps / np.maximum(np.linalg.norm(normals, axis=1), 1e-12)
    return top1, top2, distances


def hard_ovr(
    scores: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    eps: float,
) -> float:
    total = 0.0
    n_classes = scores.shape[1]
    for class_offset in range(n_classes):
        other_scores = scores.copy()
        other_scores[:, class_offset] = -np.inf
        rivals = np.argmax(other_scores, axis=1)
        gaps = scores[:, class_offset] - scores[
            np.arange(len(scores)), rivals
        ]
        normals = weights[class_offset] - weights[rivals]
        distances = gaps / np.maximum(
            np.linalg.norm(normals, axis=1), 1e-12
        )
        class_probability = float(np.mean(labels == class_offset))
        beta = max(
            min(class_probability, 1.0 - class_probability), 1e-12
        )
        total += float(np.mean(np.abs(distances) <= eps) / (eps * beta))
    return total


def hard_interface(
    scores: np.ndarray,
    weights: np.ndarray,
    labels: np.ndarray,
    top1: np.ndarray,
    top2: np.ndarray,
    eps: float,
) -> float:
    total = 0.0
    n_classes = scores.shape[1]
    low = np.minimum(top1, top2)
    high = np.maximum(top1, top2)
    for class_a in range(n_classes):
        for class_b in range(class_a + 1, n_classes):
            active = (low == class_a) & (high == class_b)
            normal = weights[class_a] - weights[class_b]
            distances = (scores[:, class_a] - scores[:, class_b]) / max(
                np.linalg.norm(normal), 1e-12
            )
            probability_a = float(np.mean(labels == class_a))
            probability_b = float(np.mean(labels == class_b))
            beta = max(min(probability_a, probability_b), 1e-12)
            tube_mass = np.mean(active & (np.abs(distances) <= eps))
            total += float(tube_mass / (eps * beta))
    return total

