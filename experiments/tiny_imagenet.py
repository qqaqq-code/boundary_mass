"Evaluate binary label-budget prediction on Tiny-ImageNet. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.tiny_labels import align_labels
from utils.paths import (
    PROJECT_ROOT,
    DATASETS_DIR,
    RUNS_DIR,
    FEATURES_DIR,
    MODELS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    result_file,
)

import json
import math
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, r2_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

warnings.filterwarnings("ignore")

ROOT = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
ART = RESULTS_DIR
FIG = FIGURES_DIR
TINY_ROOT = DATASETS_DIR / "tiny-imagenet-200"
ART.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

R50_CACHE = (
    RUNS_DIR / "tiny_imagenet_resnet50_targeted/feature_cache_resnet50_imagenetv2"
)
TINY_CACHE = FEATURES_DIR / "tiny_imagenet"
MS_CACHE = MODELS_DIR

N_PAIRS = 40
DIAG_FRAC = 0.30
SPCS = [2, 5, 10, 20, 50]
SEEDS = [0, 1]
N_BOOT = 2000
N_PERM = 2000
MAX_DIM = 128
RNG = np.random.default_rng(20260503)

BACKBONES = ["R50", "ViT-B", "DINO-MS", "SigLIP-MS"]
METRIC_DIRECTIONS = {
    "log1p_psi": "low",
    "psi_eff": "low",
    "median_margin": "high",
    "diag_nll": "low",
    "diag_ece": "low",
    "diag_acc": "high",
    "effective_rank": "high",
    "gsa_proxy": "low",
    "noise_flip_010": "low",
    "logme": "high",
    "h_score": "high",
    "transrate": "high",
    "nce_cluster": "high",
}
BASE_SETS = {
    "NLL + ECE": ["diag_nll", "diag_ece"],
    "margin + effective rank": ["median_margin", "effective_rank"],
    "feature transferability": ["logme", "h_score", "transrate", "nce_cluster"],
    "all non-$\\psi$ proxies": [
        "diag_nll",
        "diag_ece",
        "diag_acc",
        "median_margin",
        "effective_rank",
        "gsa_proxy",
        "noise_flip_010",
    ],
    "all non-$\\psi$ + transferability": [
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
}


class TinyImageNetDataset(Dataset):
    def __init__(
        self,
        root: Path,
        split: str,
        keep_labels: set[int] | None = None,
        transform=None,
    ):
        self.root = root
        self.split = split
        self.transform = transform
        self.wnids = [
            x.strip()
            for x in (root / "wnids.txt").read_text().splitlines()
            if x.strip()
        ]
        self.wnid_to_idx = {w: i for i, w in enumerate(self.wnids)}
        self.samples: list[tuple[Path, int]] = []
        if split == "train":
            for label, wnid in enumerate(self.wnids):
                if keep_labels is not None and label not in keep_labels:
                    continue
                img_dir = root / "train" / wnid / "images"
                for p in sorted(img_dir.glob("*.JPEG")):
                    self.samples.append((p, label))
        elif split == "val":
            ann = {}
            for line in (root / "val" / "val_annotations.txt").read_text().splitlines():
                parts = line.split("\t")
                if len(parts) >= 2:
                    ann[parts[0]] = parts[1]
            for img, wnid in sorted(ann.items()):
                label = self.wnid_to_idx[wnid]
                if keep_labels is not None and label not in keep_labels:
                    continue
                self.samples.append((root / "val" / "images" / img, label))
        else:
            raise ValueError(split)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, y


def make_pairs() -> list[tuple[int, int]]:
    # Disjoint pairs make the independent experiment easy to inspect and avoid accidental overlap.
    return [(2 * i, 2 * i + 1) for i in range(N_PAIRS)]


def collate_pil(batch):
    imgs, labels = zip(*batch)
    return list(imgs), torch.tensor(labels, dtype=torch.long)


def selected_labels() -> set[int]:
    out: set[int] = set()
    for a, b in make_pairs():
        out.add(a)
        out.add(b)
    return out


class ResNetFeature(nn.Module):
    def __init__(self):
        super().__init__()
        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.body = nn.Sequential(*list(m.children())[:-1])

    def forward(self, x):
        return self.body(x).flatten(1)


class ViTFeature(nn.Module):
    def __init__(self):
        super().__init__()
        self.m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)

    def forward(self, x):
        m = self.m
        x = m._process_input(x)
        n = x.shape[0]
        x = torch.cat([m.class_token.expand(n, -1, -1), x], dim=1)
        x = m.encoder(x)
        return x[:, 0]


@torch.inference_mode()
def extract_torchvision_features(name: str, force: bool = False) -> Path:
    outdir = TINY_CACHE / name
    outdir.mkdir(parents=True, exist_ok=True)
    if (
        (outdir / "train_features.pt").exists()
        and (outdir / "val_features.pt").exists()
        and not force
    ):
        print("cached", name, outdir, flush=True)
        return outdir
    if name == "ViT-B":
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model = ViTFeature()
        batch_size = 128
    elif name == "R50":
        weights = ResNet50_Weights.IMAGENET1K_V2
        model = ResNetFeature()
        batch_size = 256
    else:
        raise ValueError(name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    keep = selected_labels()
    for split in ["train", "val"]:
        ds = TinyImageNetDataset(TINY_ROOT, split, keep, transform=weights.transforms())
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True
        )
        feats, labs = [], []
        t0 = time.time()
        for i, (x, y) in enumerate(loader):
            z = model(x.to(device, non_blocking=True)).detach().cpu().float()
            feats.append(z)
            labs.append(y)
            if (i + 1) % 25 == 0:
                print(
                    name,
                    split,
                    i + 1,
                    "/",
                    len(loader),
                    "elapsed",
                    round(time.time() - t0, 1),
                    flush=True,
                )
        F = torch.cat(feats)
        Y = torch.cat(labs)
        torch.save(
            {
                "features": F,
                "labels": Y,
                "feature_dim": int(F.shape[1]),
                "selected_labels": sorted(keep),
            },
            outdir / f"{split}_features.pt",
        )
    del model
    torch.cuda.empty_cache()
    return outdir


def pick_features(short: str, out):
    if short.startswith("dino") and hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    if hasattr(out, "image_embeds") and out.image_embeds is not None:
        return out.image_embeds
    if hasattr(out, "vision_model_output") and out.vision_model_output is not None:
        vo = out.vision_model_output
        if hasattr(vo, "pooler_output") and vo.pooler_output is not None:
            return vo.pooler_output
        if hasattr(vo, "last_hidden_state"):
            return vo.last_hidden_state[:, 0]
    if hasattr(out, "pooler_output") and out.pooler_output is not None:
        return out.pooler_output
    if hasattr(out, "last_hidden_state"):
        return out.last_hidden_state[:, 0]
    if isinstance(out, tuple):
        return out[0][:, 0]
    raise RuntimeError("unknown output fields")


@torch.inference_mode()
def extract_modelscope_features(name: str, force: bool = False) -> Path:
    specs = {
        "DINO-MS": ("AI-ModelScope/dino-vitb16", "base"),
        "SigLIP-MS": ("AI-ModelScope/siglip-base-patch16-224", "siglip"),
    }
    model_id, backend = specs[name]
    outdir = TINY_CACHE / name
    outdir.mkdir(parents=True, exist_ok=True)
    if (
        (outdir / "train_features.pt").exists()
        and (outdir / "val_features.pt").exists()
        and not force
    ):
        print("cached", name, outdir, flush=True)
        return outdir
    from utils.paths import resolve_foundation_checkpoint
    from transformers import AutoImageProcessor, AutoModel

    local = resolve_foundation_checkpoint(model_id)
    proc = AutoImageProcessor.from_pretrained(
        local, local_files_only=True, trust_remote_code=True, use_fast=False
    )
    if backend == "siglip":
        from transformers import SiglipVisionModel

        model = (
            SiglipVisionModel.from_pretrained(local, local_files_only=True)
            .eval()
            .cuda()
        )
    else:
        model = (
            AutoModel.from_pretrained(
                local, local_files_only=True, trust_remote_code=True
            )
            .eval()
            .cuda()
        )
    keep = selected_labels()
    for split in ["train", "val"]:
        ds = TinyImageNetDataset(TINY_ROOT, split, keep, transform=None)
        loader = DataLoader(
            ds,
            batch_size=192,
            shuffle=False,
            num_workers=4,
            collate_fn=collate_pil,
            pin_memory=True,
        )
        feats, labs = [], []
        t0 = time.time()
        for i, (imgs, y) in enumerate(loader):
            inp = proc(images=imgs, return_tensors="pt")
            inp = {k: v.cuda(non_blocking=True) for k, v in inp.items()}
            z = pick_features(name.lower(), model(**inp)).detach().cpu().float()
            feats.append(z)
            labs.append(y)
            if (i + 1) % 25 == 0:
                print(
                    name,
                    split,
                    i + 1,
                    "/",
                    len(loader),
                    "elapsed",
                    round(time.time() - t0, 1),
                    flush=True,
                )
        F = torch.cat(feats)
        Y = torch.cat(labs)
        torch.save(
            {
                "features": F,
                "labels": Y,
                "modelscope_id": model_id,
                "local_dir": local,
                "feature_dim": int(F.shape[1]),
                "selected_labels": sorted(keep),
            },
            outdir / f"{split}_features.pt",
        )
    del model
    torch.cuda.empty_cache()
    return outdir


def ensure_feature_caches() -> dict[str, Path]:
    caches = {"R50": R50_CACHE}
    caches["ViT-B"] = extract_torchvision_features("ViT-B")
    caches["DINO-MS"] = extract_modelscope_features("DINO-MS")
    caches["SigLIP-MS"] = extract_modelscope_features("SigLIP-MS")
    return caches


def load_cache(d: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train = torch.load(d / "train_features.pt", map_location="cpu")
    val_name = (
        "val_features.pt" if (d / "val_features.pt").exists() else "test_features.pt"
    )
    val = torch.load(d / val_name, map_location="cpu")
    return (
        train["features"].float().numpy().astype("float32"),
        align_labels(train, d),
        val["features"].float().numpy().astype("float32"),
        align_labels(val, d),
    )


def split_probe_diag(
    y: np.ndarray, pair_id: int, backbone_id: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(99000 + 101 * pair_id + backbone_id)
    probe_parts, diag_parts = [], []
    for cls in [0, 1]:
        ids = np.flatnonzero(y == cls)
        ids = rng.permutation(ids)
        n_diag = max(30, int(round(DIAG_FRAC * len(ids))))
        n_diag = min(n_diag, max(1, len(ids) - 60))
        diag_parts.append(ids[:n_diag])
        probe_parts.append(ids[n_diag:])
    return np.concatenate(probe_parts), np.concatenate(diag_parts)


def standardize_from_probe(
    X_probe: np.ndarray, X_diag: np.ndarray, X_val: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X_probe.mean(axis=0, keepdims=True)
    sd = X_probe.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (
        ((X_probe - mu) / sd).astype("float32"),
        ((X_diag - mu) / sd).astype("float32"),
        ((X_val - mu) / sd).astype("float32"),
    )


def fit_lr(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    clf = LogisticRegression(
        C=1.0, penalty="l2", solver="liblinear", max_iter=500, tol=1e-4, random_state=0
    )
    clf.fit(X, y)
    return clf


def ece_binary(prob: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    conf = np.maximum(prob, 1 - prob)
    pred = (prob >= 0.5).astype(int)
    corr = (pred == y).astype(float)
    out = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < (hi if hi < 1 else hi + 1e-8))
        if m.any():
            out += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return float(out)


def effective_rank(X: np.ndarray) -> float:
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    eig = s * s / max(1, X.shape[0] - 1)
    sm = eig.sum()
    return float(sm * sm / (np.square(eig).sum() + 1e-12))


def signed_dist(clf: LogisticRegression, X: np.ndarray) -> np.ndarray:
    w = clf.coef_.reshape(-1)
    return clf.decision_function(X) / (np.linalg.norm(w) + 1e-12)


def psi_kernel_from_dist(
    dist: np.ndarray, y: np.ndarray, eps: float | None = None
) -> tuple[float, float]:
    if eps is None:
        eps = max(0.05, float(np.quantile(np.abs(dist), 0.20)))
    beta = max(min(float(np.mean(y == 1)), float(np.mean(y == 0))), 1e-6)
    kern = np.exp(-0.5 * (dist / eps) ** 2) / (eps * math.sqrt(2.0 * math.pi))
    return float(kern.mean() / beta), float(eps)


def gsa_proxy(clf: LogisticRegression) -> float:
    w = clf.coef_.reshape(-1)
    t = float(clf.intercept_[0]) / (np.linalg.norm(w) + 1e-12)
    return float(2.0 * math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi))


def noise_flip(
    clf: LogisticRegression, X: np.ndarray, alpha: float = 0.10, repeats: int = 3
) -> float:
    base = clf.predict(X)
    rng = np.random.default_rng(123)
    scale = np.std(X, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    vals = []
    for _ in range(repeats):
        Xn = X + rng.normal(0, alpha, size=X.shape).astype("float32") * scale
        vals.append(float(np.mean(clf.predict(Xn) != base)))
    return float(np.mean(vals))


def compact_features(X: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    X = X.astype("float64", copy=False)
    X = X - X.mean(axis=0, keepdims=True)
    if X.shape[1] <= max_dim:
        return X
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    k = min(max_dim, len(s))
    return u[:, :k] * s[:k]


def h_score(X: np.ndarray, y: np.ndarray) -> float:
    X = compact_features(X)
    n, d = X.shape
    cov = (X.T @ X) / max(n, 1)
    lam = 1e-3 * float(np.trace(cov) / max(d, 1))
    cov = cov + (lam + 1e-8) * np.eye(d)
    mu = X.mean(axis=0)
    sb = np.zeros((d, d), dtype="float64")
    for c in np.unique(y):
        Xc = X[y == c]
        diff = (Xc.mean(axis=0) - mu).reshape(-1, 1)
        sb += (len(Xc) / n) * (diff @ diff.T)
    return float(np.trace(np.linalg.solve(cov, sb)))


def coding_rate(X: np.ndarray, eps: float = 0.10) -> float:
    X = X.astype("float64", copy=False)
    n, d = X.shape
    if n <= 1 or d == 0:
        return 0.0
    X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    mat = np.eye(d) + (d / (eps * eps)) * ((X.T @ X) / max(n, 1))
    sign, logdet = np.linalg.slogdet(mat)
    return float(0.5 * logdet) if sign > 0 else float("nan")


def transrate_score(X: np.ndarray, y: np.ndarray) -> float:
    X = compact_features(X)
    total = coding_rate(X)
    cond = 0.0
    n = len(y)
    for c in np.unique(y):
        Xc = X[y == c]
        cond += (len(Xc) / n) * coding_rate(Xc)
    return float(total - cond)


def logme_evidence_single(X: np.ndarray, y: np.ndarray) -> float:
    X = compact_features(X)
    y = y.astype("float64", copy=False)
    y = y - y.mean()
    n, d = X.shape
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    uy = u.T @ y
    yy = float(y @ y)
    alpha, beta = 1.0, 1.0
    for _ in range(50):
        denom = alpha + beta * s * s
        gamma = float(np.sum((beta * s * s) / denom))
        m_coef = beta * s * uy / denom
        m2 = float(np.sum(m_coef * m_coef))
        pred = s * m_coef
        residual = max(float(yy - 2 * np.sum(pred * uy) + np.sum(pred * pred)), 1e-12)
        alpha_new = float(np.clip(gamma / max(m2, 1e-12), 1e-8, 1e8))
        beta_new = float(np.clip((n - gamma) / residual, 1e-8, 1e8))
        if abs(alpha_new - alpha) / alpha < 1e-4 and abs(beta_new - beta) / beta < 1e-4:
            alpha, beta = alpha_new, beta_new
            break
        alpha, beta = alpha_new, beta_new
    denom = alpha + beta * s * s
    gamma = float(np.sum((beta * s * s) / denom))
    m_coef = beta * s * uy / denom
    m2 = float(np.sum(m_coef * m_coef))
    pred = s * m_coef
    residual = max(float(yy - 2 * np.sum(pred * uy) + np.sum(pred * pred)), 1e-12)
    ev = (
        0.5 * d * math.log(alpha)
        + 0.5 * n * math.log(beta)
        - 0.5 * np.sum(np.log(denom))
        - 0.5 * beta * residual
        - 0.5 * alpha * m2
        - 0.5 * n * math.log(2 * math.pi)
    )
    return float(ev / n)


def logme_score(X: np.ndarray, y: np.ndarray) -> float:
    return float(
        np.mean([logme_evidence_single(X, (y == c).astype("float64")) for c in [0, 1]])
    )


def nce_cluster_score(X: np.ndarray, y: np.ndarray) -> float:
    X = compact_features(X)
    k = int(min(20, max(2, len(y) // 10)))
    labels = KMeans(n_clusters=k, n_init=5, random_state=123).fit_predict(X)
    h = 0.0
    for z in range(k):
        m = labels == z
        if not m.any():
            continue
        pz = float(m.mean())
        py = float(y[m].mean())
        if py > 1e-12:
            h -= pz * py * math.log(py)
        if py < 1 - 1e-12:
            h -= pz * (1 - py) * math.log(1 - py)
    return float(-h)


def interp_budget(
    spcs: np.ndarray, accs: np.ndarray, target: float
) -> tuple[float, bool]:
    spcs = np.asarray(spcs, dtype=float)
    accs = np.asarray(accs, dtype=float)
    order = np.argsort(spcs)
    spcs, accs = spcs[order], accs[order]
    if np.nanmax(accs) < target:
        return float(spcs.max() * 1.5), False
    if accs[0] >= target:
        return float(spcs[0]), True
    for i in range(1, len(spcs)):
        if accs[i] >= target:
            x0, x1 = np.log(spcs[i - 1]), np.log(spcs[i])
            y0, y1 = accs[i - 1], accs[i]
            t = 0.0 if abs(y1 - y0) < 1e-9 else (target - y0) / (y1 - y0)
            return float(np.exp(x0 + t * (x1 - x0))), True
    return float(spcs.max() * 1.5), False


def run_experiment() -> pd.DataFrame:
    caches = ensure_feature_caches()
    data = {name: load_cache(path) for name, path in caches.items()}
    metric_rows, lc_rows = [], []
    for pair_id, (a, b) in enumerate(make_pairs()):
        for backbone_id, name in enumerate(BACKBONES):
            Xtr_all, ytr_all, Xval_all, yval_all = data[name]
            mtr = (ytr_all == a) | (ytr_all == b)
            mval = (yval_all == a) | (yval_all == b)
            Xtr_pair = Xtr_all[mtr]
            ytr_pair = (ytr_all[mtr] == b).astype(int)
            Xval_raw = Xval_all[mval]
            yval = (yval_all[mval] == b).astype(int)
            probe_idx, diag_idx = split_probe_diag(ytr_pair, pair_id, backbone_id)
            X_probe, y_probe = Xtr_pair[probe_idx], ytr_pair[probe_idx]
            X_diag, y_diag = Xtr_pair[diag_idx], ytr_pair[diag_idx]
            X_probe, X_diag, X_val = standardize_from_probe(X_probe, X_diag, Xval_raw)
            clf = fit_lr(X_probe, y_probe)
            diag_prob = clf.predict_proba(X_diag)[:, 1]
            diag_pred = (diag_prob >= 0.5).astype(int)
            dist = signed_dist(clf, X_diag)
            psi, eps = psi_kernel_from_dist(dist, y_diag)
            full_acc = float(clf.score(X_val, yval))
            metric_rows.append(
                {
                    "pair_id": pair_id,
                    "class_a": a,
                    "class_b": b,
                    "backbone": name,
                    "probe_n": int(len(y_probe)),
                    "diag_n": int(len(y_diag)),
                    "target_full_acc": full_acc,
                    "psi_eff": psi,
                    "log1p_psi": math.log1p(psi),
                    "eps": eps,
                    "median_margin": float(np.median(np.abs(dist))),
                    "mean_margin": float(np.mean(np.abs(dist))),
                    "weight_norm": float(np.linalg.norm(clf.coef_)),
                    "gsa_proxy": gsa_proxy(clf),
                    "noise_flip_010": noise_flip(clf, X_diag),
                    "effective_rank": effective_rank(X_diag),
                    "diag_acc": float((diag_pred == y_diag).mean()),
                    "diag_nll": float(
                        log_loss(
                            y_diag,
                            np.vstack([1 - diag_prob, diag_prob]).T,
                            labels=[0, 1],
                        )
                    ),
                    "diag_ece": ece_binary(diag_prob, y_diag),
                    "logme": logme_score(X_diag, y_diag),
                    "h_score": h_score(X_diag, y_diag),
                    "transrate": transrate_score(X_diag, y_diag),
                    "nce_cluster": nce_cluster_score(X_diag, y_diag),
                }
            )
            idx0 = np.flatnonzero(y_probe == 0)
            idx1 = np.flatnonzero(y_probe == 1)
            for spc in SPCS:
                for seed in SEEDS:
                    rng = np.random.default_rng(3000 + pair_id * 31 + seed)
                    take = np.r_[
                        rng.choice(idx0, min(spc, len(idx0)), replace=False),
                        rng.choice(idx1, min(spc, len(idx1)), replace=False),
                    ]
                    sub = fit_lr(X_probe[take], y_probe[take])
                    lc_rows.append(
                        {
                            "pair_id": pair_id,
                            "backbone": name,
                            "spc": spc,
                            "seed": seed,
                            "acc": float(sub.score(X_val, yval)),
                        }
                    )
        if (pair_id + 1) % 5 == 0:
            print("processed tiny pair", pair_id + 1, "/", N_PAIRS, flush=True)
    M = pd.DataFrame(metric_rows)
    LC = (
        pd.DataFrame(lc_rows)
        .groupby(["pair_id", "backbone", "spc"], as_index=False)
        .acc.mean()
    )
    rows = []
    best_full = M.groupby("pair_id").target_full_acc.max().to_dict()
    for _, r in M.iterrows():
        cur = LC[(LC.pair_id == r.pair_id) & (LC.backbone == r.backbone)].sort_values(
            "spc"
        )
        target = 0.90 * best_full[int(r.pair_id)]
        budget, reached = interp_budget(cur.spc.values, cur.acc.values, target)
        x = np.log(cur.spc.values.astype(float))
        y = cur.acc.values.astype(float)
        rows.append(
            {
                **r.to_dict(),
                "target_acc": target,
                "label_budget": budget,
                "target_reached": reached,
                "aulc": float(np.trapz(y, x) / (x[-1] - x[0])),
                "acc_5spc": float(cur[cur.spc == 5].acc.iloc[0]),
            }
        )
    B = pd.DataFrame(rows)
    M.to_csv(
        result_file("tiny_imagenet/tiny_strict_metrics.csv"), index=False
    )
    LC.to_csv(
        result_file("tiny_imagenet/tiny_strict_learning_curves.csv"),
        index=False,
    )
    B.to_csv(
        result_file("tiny_imagenet/tiny_strict_label_budget_table.csv"),
        index=False,
    )
    return B


def residualize(df: pd.DataFrame, cols: list[str], group_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col + "_r"] = out[col] - out.groupby(group_col)[col].transform("mean")
    return out


def r2_pair(
    df: pd.DataFrame, base: list[str], group_col: str = "pair_id"
) -> tuple[float, float, float]:
    cols = ["label_budget", "log1p_psi"] + base
    d = df[[group_col] + cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = residualize(d, cols, group_col)
    y = d["label_budget_r"].to_numpy()
    X = d[[m + "_r" for m in base]].to_numpy()
    Xp = d[[m + "_r" for m in base + ["log1p_psi"]]].to_numpy()
    r2_base = r2_score(y, LinearRegression().fit(X, y).predict(X))
    r2_plus = r2_score(y, LinearRegression().fit(Xp, y).predict(Xp))
    return float(r2_base), float(r2_plus), float(r2_plus - r2_base)


def delta_r2(df: pd.DataFrame, base: list[str], group_col: str = "pair_id") -> float:
    try:
        return r2_pair(df, base, group_col)[2]
    except Exception:
        return float("nan")


def ci(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def sample_pairs(df: pd.DataFrame) -> pd.DataFrame:
    pairs = np.array(sorted(df.pair_id.unique()))
    sampled = RNG.choice(pairs, size=len(pairs), replace=True)
    chunks = []
    for boot_id, pair_id in enumerate(sampled):
        part = df[df.pair_id == pair_id].copy()
        part["boot_pair"] = boot_id
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def cluster_boot_delta(
    df: pd.DataFrame, base: list[str]
) -> tuple[tuple[float, float], float]:
    vals = [
        delta_r2(sample_pairs(df), base, group_col="boot_pair") for _ in range(N_BOOT)
    ]
    vals = [v for v in vals if np.isfinite(v)]
    p_boot = (1.0 + sum(v <= 0 for v in vals)) / (len(vals) + 1.0)
    return ci(vals), float(p_boot)


def pair_signflip_p(df: pd.DataFrame, base: list[str], observed: float) -> float:
    cols = ["label_budget", "log1p_psi"] + base
    d = df[["pair_id"] + cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = residualize(d, cols, "pair_id")
    y = d["label_budget_r"].to_numpy()
    X = d[[m + "_r" for m in base]].to_numpy()
    psi = d[["log1p_psi_r"]].to_numpy()
    r2_base = r2_score(y, LinearRegression().fit(X, y).predict(X))
    blocks = [idx.to_numpy() for _, idx in d.groupby("pair_id").groups.items()]
    null = []
    for _ in range(N_PERM):
        pp = psi.copy()
        for sign, idx in zip(RNG.choice([-1.0, 1.0], size=len(blocks)), blocks):
            pp[idx, 0] *= sign
        Xp = np.hstack([X, pp])
        null.append(
            float(r2_score(y, LinearRegression().fit(Xp, y).predict(Xp)) - r2_base)
        )
    return float((1.0 + sum(v >= observed for v in null)) / (len(null) + 1.0))


def residual_corr_after_base(df: pd.DataFrame, base: list[str]) -> dict[str, float]:
    cols = ["label_budget", "log1p_psi"] + base
    d = df[["pair_id"] + cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = residualize(d, cols, "pair_id")
    y = d["label_budget_r"].to_numpy()
    X = d[[m + "_r" for m in base]].to_numpy()
    psi = d["log1p_psi_r"].to_numpy()
    y_res = y - LinearRegression().fit(X, y).predict(X)
    psi_res = psi - LinearRegression().fit(X, psi).predict(X)
    return {
        "n": int(len(d)),
        "partial_pearson": float(pearsonr(psi_res, y_res)[0]),
        "partial_spearman": float(spearmanr(psi_res, y_res)[0]),
    }


def corr_pair(df: pd.DataFrame, metric: str) -> dict[str, float]:
    d = (
        df[[metric, "label_budget", "pair_id"]]
        .replace([np.inf, -np.inf], np.nan)
        .dropna()
        .copy()
    )
    d[metric + "_r"] = d[metric] - d.groupby("pair_id")[metric].transform("mean")
    d["label_budget_r"] = d["label_budget"] - d.groupby("pair_id")[
        "label_budget"
    ].transform("mean")
    return {
        "n": int(len(d)),
        "pearson": float(pearsonr(d[metric + "_r"], d["label_budget_r"])[0]),
        "spearman": float(spearmanr(d[metric + "_r"], d["label_budget_r"])[0]),
    }


def selection_summary(df: pd.DataFrame, metric: str) -> dict[str, float]:
    direction = METRIC_DIRECTIONS[metric]
    rows = []
    for pair_id, sub in df.groupby("pair_id"):
        best = sub.loc[sub.label_budget.idxmin()]
        chosen = sub.sort_values(metric, ascending=(direction == "low")).iloc[0]
        rows.append(
            {
                "pair_id": pair_id,
                "correct": float(chosen.backbone == best.backbone),
                "regret": float(chosen.label_budget - best.label_budget),
                "rel_regret": float(
                    (chosen.label_budget - best.label_budget)
                    / (best.label_budget + 1e-9)
                ),
                "aulc": float(chosen.aulc),
            }
        )
    S = pd.DataFrame(rows)
    pairs = np.array(sorted(S.pair_id.unique()))
    vals = {"acc": [], "regret": [], "rel": [], "aulc": []}
    for _ in range(N_BOOT):
        sampled = RNG.choice(pairs, size=len(pairs), replace=True)
        b = pd.concat([S[S.pair_id == p] for p in sampled])
        vals["acc"].append(float(b.correct.mean()))
        vals["regret"].append(float(b.regret.mean()))
        vals["rel"].append(float(b.rel_regret.mean()))
        vals["aulc"].append(float(b.aulc.mean()))
    acc_ci, reg_ci, rel_ci, aulc_ci = (
        ci(vals["acc"]),
        ci(vals["regret"]),
        ci(vals["rel"]),
        ci(vals["aulc"]),
    )
    return {
        "metric": metric,
        "selection_acc": float(S.correct.mean()),
        "selection_acc_ci": acc_ci,
        "mean_regret": float(S.regret.mean()),
        "mean_regret_ci": reg_ci,
        "mean_rel_regret": float(S.rel_regret.mean()),
        "mean_rel_regret_ci": rel_ci,
        "mean_aulc": float(S.aulc.mean()),
        "mean_aulc_ci": aulc_ci,
    }


def fmt_p(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def analyze(B: pd.DataFrame) -> dict[str, object]:
    incremental = []
    for name, base in BASE_SETS.items():
        r2_base, r2_plus, obs = r2_pair(B, base)
        boot_ci, p_boot = cluster_boot_delta(B, base)
        p_perm = pair_signflip_p(B, base, obs)
        partial = residual_corr_after_base(B, base)
        loco = np.asarray(
            [
                delta_r2(B[B.pair_id != pair_id], base)
                for pair_id in sorted(B.pair_id.unique())
            ]
        )
        loco = loco[np.isfinite(loco)]
        incremental.append(
            {
                "base": name,
                "r2_base": r2_base,
                "r2_plus_psi": r2_plus,
                "delta_r2": obs,
                "delta_r2_ci_low": boot_ci[0],
                "delta_r2_ci_high": boot_ci[1],
                "bootstrap_p_delta_le_zero": p_boot,
                "paired_signflip_p": p_perm,
                "loco_positive": int(np.sum(loco > 0)),
                "loco_n": int(len(loco)),
                **partial,
            }
        )
    correlations = {
        m: corr_pair(B, m)
        for m in [
            "log1p_psi",
            "logme",
            "h_score",
            "transrate",
            "nce_cluster",
            "diag_nll",
            "diag_ece",
            "median_margin",
        ]
    }
    selection = [
        selection_summary(B, m)
        for m in [
            "log1p_psi",
            "logme",
            "h_score",
            "diag_nll",
            "diag_ece",
            "median_margin",
            "effective_rank",
        ]
    ]
    stats = {
        "protocol": {
            "dataset": "Tiny-ImageNet",
            "pairs": N_PAIRS,
            "backbones": BACKBONES,
            "diagnostics": "probe-train/diagnostic-heldout, no val labels for diagnostics",
            "target": "samples per class to reach 90% of best full-shot validation accuracy",
        },
        "incremental": incremental,
        "correlations": correlations,
        "selection": selection,
    }
    (
        result_file("tiny_imagenet/tiny_strict_label_budget_stats.json")
    ).write_text(json.dumps(stats, indent=2))
    rows = []
    for row in incremental:
        rows.append(
            f"{row['base']} & ${row['r2_base']:.3f}$ & ${row['r2_plus_psi']:.3f}$ & ${row['delta_r2']:.3f}[{row['delta_r2_ci_low']:.3f},{row['delta_r2_ci_high']:.3f}]$ & {fmt_p(row['paired_signflip_p'])} & ${row['partial_pearson']:.3f}/{row['partial_spearman']:.3f}$\\\\"
        )
    (result_file("tiny_imagenet/tiny_incremental_rows.tex")).write_text(
        "\n".join(rows) + "\n"
    )
    sel_rows = []
    labels = {
        "log1p_psi": r"$\log(1+\hat\psi_{\rm eff})$",
        "logme": "LogME",
        "h_score": "H-score",
        "diag_nll": "NLL",
        "diag_ece": "ECE",
        "median_margin": "margin",
        "effective_rank": "effective rank",
    }
    for row in selection:
        sel_rows.append(
            f"{labels[row['metric']]} & ${100 * row['selection_acc']:.1f}[{100 * row['selection_acc_ci'][0]:.1f},{100 * row['selection_acc_ci'][1]:.1f}]$ & ${row['mean_regret']:.2f}[{row['mean_regret_ci'][0]:.2f},{row['mean_regret_ci'][1]:.2f}]$ & ${100 * row['mean_rel_regret']:.1f}[{100 * row['mean_rel_regret_ci'][0]:.1f},{100 * row['mean_rel_regret_ci'][1]:.1f}]$ & ${row['mean_aulc']:.3f}$\\\\"
        )
    (result_file("tiny_imagenet/tiny_selection_rows.tex")).write_text(
        "\n".join(sel_rows) + "\n"
    )
    return stats


def main() -> None:
    B = run_experiment()
    stats = analyze(B)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
