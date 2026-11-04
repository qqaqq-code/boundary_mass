"Evaluate transferability diagnostics based on source-model logits. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import (
    PROJECT_ROOT,
    DATASETS_DIR,
    RUNS_DIR,
    RESULTS_DIR,
    FIGURES_DIR,
    result_file,
)

import json
import math
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.special import logsumexp, softmax
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights, resnet50, vit_b_16

warnings.filterwarnings("ignore")

ROOT = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
ART = RESULTS_DIR
FIG = FIGURES_DIR
ART.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

DATA_ROOT = DATASETS_DIR / "cifar100"
N_PAIRS = 40
DIAG_FRAC = 0.30
N_BOOT = 2000
N_PERM = 2000
RNG = np.random.default_rng(20260502)

CACHE_LABELS = {
    "R50": RUNS_DIR
    / "cifar100_resnet50_pretrained/feature_cache_resnet50_imagenetv2/train_features.pt",
    "ViT-B": RUNS_DIR
    / "cifar100_vitb16_targeted/feature_cache_vitb16_imagenetv1/train_features.pt",
}

MODELS = {
    "R50": {
        "weights": ResNet50_Weights.IMAGENET1K_V2,
        "builder": resnet50,
        "batch_size": 256,
        "out": result_file("source_logits_cifar100_r50_train.pt"),
    },
    "ViT-B": {
        "weights": ViT_B_16_Weights.IMAGENET1K_V1,
        "builder": vit_b_16,
        "batch_size": 128,
        "out": result_file("source_logits_cifar100_vitb_train.pt"),
    },
}

SOURCE_METRICS = ["leep", "nce", "source_logme"]
OLD_NONPSI = [
    "diag_nll",
    "diag_ece",
    "diag_acc",
    "median_margin",
    "effective_rank",
    "gsa_proxy",
    "noise_flip_010",
]
FEATURE_TRANSFER = ["logme", "h_score", "transrate", "nce_cluster"]
BASE_SETS = {
    "LEEP + NCE": ["leep", "nce"],
    "source-logit set": SOURCE_METRICS,
    "source logits + feature transferability": SOURCE_METRICS + FEATURE_TRANSFER,
    "all non-$\\psi$ + source/feature transferability": OLD_NONPSI
    + SOURCE_METRICS
    + FEATURE_TRANSFER,
}


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


def dataset_labels() -> np.ndarray:
    ds = CIFAR100(root=str(DATA_ROOT), train=True, download=False)
    return np.asarray(ds.targets, dtype=np.int64)


def cache_labels(name: str) -> np.ndarray:
    obj = torch.load(CACHE_LABELS[name], map_location="cpu")
    return obj["labels"].numpy().astype("int64")


def verify_label_order() -> None:
    y_ds = dataset_labels()
    for name in CACHE_LABELS:
        y_cache = cache_labels(name)
        if len(y_cache) != len(y_ds) or not np.array_equal(y_cache, y_ds):
            raise RuntimeError(
                f"CIFAR-100 dataset order does not match feature cache for {name}"
            )


@torch.inference_mode()
def extract_logits(name: str, force: bool = False) -> Path:
    spec = MODELS[name]
    out: Path = spec["out"]
    if out.exists() and not force:
        print(f"using cached source logits for {name}: {out}", flush=True)
        return out

    weights = spec["weights"]
    transform = weights.transforms()
    ds = CIFAR100(root=str(DATA_ROOT), train=True, download=False, transform=transform)
    loader = DataLoader(
        ds, batch_size=spec["batch_size"], shuffle=False, num_workers=4, pin_memory=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = spec["builder"](weights=weights).to(device).eval()
    outs = []
    t0 = time.time()
    for i, (x, _) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        logits = model(x).detach().cpu().to(torch.float16)
        outs.append(logits)
        if (i + 1) % 25 == 0:
            print(
                f"{name} logits batch {i + 1}/{len(loader)} in {time.time() - t0:.1f}s",
                flush=True,
            )
    logits = torch.cat(outs, dim=0)
    torch.save(
        {
            "logits": logits,
            "labels": torch.as_tensor(ds.targets, dtype=torch.long),
            "model": name,
            "source_classes": 1000,
            "weights": str(weights),
        },
        out,
    )
    print(f"saved {name} logits {tuple(logits.shape)} to {out}", flush=True)
    return out


def logme_evidence_single(X: np.ndarray, y: np.ndarray) -> float:
    X = X.astype("float64", copy=False)
    X = X - X.mean(axis=0, keepdims=True)
    y = y.astype("float64", copy=False)
    y = y - y.mean()
    n, d = X.shape
    if n < 4 or d < 1:
        return float("nan")
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    uy = u.T @ y
    yy = float(y @ y)
    alpha, beta = 1.0, 1.0
    for _ in range(50):
        denom = alpha + beta * s * s
        gamma = float(np.sum((beta * s * s) / denom))
        m_coef = beta * s * uy / denom
        m2 = float(np.sum(m_coef * m_coef))
        pred_coef = s * m_coef
        residual = max(
            float(yy - 2 * np.sum(pred_coef * uy) + np.sum(pred_coef * pred_coef)),
            1e-12,
        )
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
    pred_coef = s * m_coef
    residual = max(
        float(yy - 2 * np.sum(pred_coef * uy) + np.sum(pred_coef * pred_coef)), 1e-12
    )
    evidence = (
        0.5 * d * math.log(alpha)
        + 0.5 * n * math.log(beta)
        - 0.5 * np.sum(np.log(denom))
        - 0.5 * beta * residual
        - 0.5 * alpha * m2
        - 0.5 * n * math.log(2 * math.pi)
    )
    return float(evidence / n)


def source_logme(probs: np.ndarray, y: np.ndarray) -> float:
    vals = []
    # Compress softmax outputs to the active source classes for numerical speed.
    keep = np.argsort(probs.mean(axis=0))[-256:]
    X = np.log(np.maximum(probs[:, keep], 1e-12))
    for c in [0, 1]:
        vals.append(logme_evidence_single(X, (y == c).astype("float64")))
    return float(np.mean(vals))


def leep_score(logits: np.ndarray, y: np.ndarray) -> float:
    probs = softmax(logits.astype("float64"), axis=1)
    n, k = probs.shape
    pyz = np.zeros((2, k), dtype="float64")
    denom = probs.sum(axis=0) + 1e-12
    for c in [0, 1]:
        pyz[c] = probs[y == c].sum(axis=0) / denom
    pred = probs @ pyz.T
    return float(np.mean(np.log(np.maximum(pred[np.arange(n), y], 1e-12))))


def nce_score(logits: np.ndarray, y: np.ndarray) -> float:
    z = logits.argmax(axis=1)
    n = len(y)
    h = 0.0
    for cls in np.unique(z):
        m = z == cls
        pz = float(m.mean())
        py = float(y[m].mean())
        if py > 1e-12:
            h -= pz * py * math.log(py)
        if py < 1 - 1e-12:
            h -= pz * (1 - py) * math.log(1 - py)
    return float(-h)


def compute_metric_rows() -> pd.DataFrame:
    verify_label_order()
    pairs = make_pairs()
    label_ref = dataset_labels()
    logits = {}
    for name in MODELS:
        path = extract_logits(name)
        obj = torch.load(path, map_location="cpu")
        logits[name] = obj["logits"].float().numpy()
    rows = []
    for pair_id, (a, b) in enumerate(pairs):
        for backbone_id, name in enumerate(["R50", "ViT-B"]):
            y_all = cache_labels(name)
            mtr = (y_all == a) | (y_all == b)
            pair_global = np.flatnonzero(mtr)
            y_pair = (y_all[mtr] == b).astype(int)
            _, diag_idx = split_probe_diag(y_pair, pair_id, backbone_id)
            diag_global = pair_global[diag_idx]
            y_diag = y_pair[diag_idx]
            cur_logits = logits[name][diag_global]
            probs = softmax(cur_logits.astype("float64"), axis=1)
            rows.append(
                {
                    "pair_id": pair_id,
                    "class_a": a,
                    "class_b": b,
                    "backbone": name,
                    "diag_n": int(len(y_diag)),
                    "leep": leep_score(cur_logits, y_diag),
                    "nce": nce_score(cur_logits, y_diag),
                    "source_logme": source_logme(probs, y_diag),
                    "source_top1_entropy": float(
                        -np.mean(np.log(np.maximum(probs.max(axis=1), 1e-12)))
                    ),
                    "source_top1_match_pair": float(
                        np.mean(np.isin(label_ref[diag_global], [a, b]))
                    ),
                }
            )
    M = pd.DataFrame(rows)
    M.to_csv(
        result_file(
            "source_logits/source_logit_transferability_metrics.csv"
        ),
        index=False,
    )
    return M


def residualize(df: pd.DataFrame, cols: list[str], group_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[col + "_r"] = out[col] - out.groupby(group_col)[col].transform("mean")
    return out


def corr_pair_resid(
    df: pd.DataFrame, x: str, y: str = "label_budget"
) -> dict[str, float]:
    d = df[[x, y, "pair_id"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d[x + "_r"] = d[x] - d.groupby("pair_id")[x].transform("mean")
    d[y + "_r"] = d[y] - d.groupby("pair_id")[y].transform("mean")
    return {
        "n": int(len(d)),
        "pearson": float(pearsonr(d[x + "_r"], d[y + "_r"])[0]),
        "spearman": float(spearmanr(d[x + "_r"], d[y + "_r"])[0]),
    }


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


def ci(vals: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return float("nan"), float("nan")
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
    vals = []
    for _ in range(N_BOOT):
        vals.append(delta_r2(sample_pairs(df), base, group_col="boot_pair"))
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
    pair_blocks = [idx.to_numpy() for _, idx in d.groupby("pair_id").groups.items()]
    null = []
    for _ in range(N_PERM):
        psi_perm = psi.copy()
        for sign, idx in zip(
            RNG.choice([-1.0, 1.0], size=len(pair_blocks)), pair_blocks
        ):
            psi_perm[idx, 0] *= sign
        Xp = np.hstack([X, psi_perm])
        r2_plus = r2_score(y, LinearRegression().fit(Xp, y).predict(Xp))
        null.append(float(r2_plus - r2_base))
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


def fmt_p(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def analyze(B: pd.DataFrame) -> dict[str, object]:
    correlations = {m: corr_pair_resid(B, m) for m in SOURCE_METRICS}
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
                "loco_min_delta": float(np.min(loco)),
                "loco_max_delta": float(np.max(loco)),
                "loco_positive": int(np.sum(loco > 0)),
                "loco_n": int(len(loco)),
                **partial,
            }
        )
    return {
        "protocol": {
            "scope": "R50 and ViT-B/16 subset with official torchvision ImageNet source classifiers",
            "n_rows": int(len(B)),
            "n_pairs": int(B.pair_id.nunique()),
            "source_logits": "downloaded/loaded torchvision ImageNet heads and forwarded CIFAR-100 train images",
            "not_included": "DINO-MS and SigLIP-MS lack comparable 1000-way ImageNet source-classifier heads in the cached setup",
        },
        "correlations": correlations,
        "incremental": incremental,
    }


def write_tex(stats: dict[str, object]) -> None:
    rows = []
    for row in stats["incremental"]:
        rows.append(
            f"{row['base']} & ${row['r2_base']:.3f}$ & ${row['r2_plus_psi']:.3f}$ & "
            f"${row['delta_r2']:.3f}[{row['delta_r2_ci_low']:.3f},{row['delta_r2_ci_high']:.3f}]$ & "
            f"{fmt_p(row['paired_signflip_p'])} & ${row['partial_pearson']:.3f}/{row['partial_spearman']:.3f}$\\\\"
        )
    (
        result_file("source_logits/source_logit_incremental_rows.tex")
    ).write_text("\n".join(rows) + "\n")
    single = []
    labels = {"leep": "LEEP", "nce": "NCE", "source_logme": "source-logit LogME"}
    for metric in SOURCE_METRICS:
        c = stats["correlations"][metric]
        single.append(
            f"{labels[metric]} & ${c['pearson']:.3f}/{c['spearman']:.3f}$\\\\"
        )
    (result_file("source_logits/source_logit_single_rows.tex")).write_text(
        "\n".join(single) + "\n"
    )


def maybe_plot(stats: dict[str, object]) -> None:
    try:
        import matplotlib.pyplot as plt

        rows = stats["incremental"]
        labels = [r["base"].replace(" + ", "\n+ ") for r in rows]
        vals = [r["delta_r2"] for r in rows]
        lo = [r["delta_r2"] - r["delta_r2_ci_low"] for r in rows]
        hi = [r["delta_r2_ci_high"] - r["delta_r2"] for r in rows]
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.bar(np.arange(len(vals)), vals, color="#59A14F")
        ax.errorbar(
            np.arange(len(vals)),
            vals,
            yerr=[lo, hi],
            fmt="none",
            color="black",
            capsize=3,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel(r"Incremental $\Delta R^2$")
        ax.set_xticks(np.arange(len(vals)))
        ax.set_xticklabels(labels, fontsize=8)
        ax.set_title(r"$\psi$ beyond downloaded ImageNet source-logit baselines")
        fig.tight_layout()
        fig.savefig(FIG / "source_logit_transferability_experiment.png", dpi=220)
        fig.savefig(FIG / "source_logit_transferability_experiment.pdf")
        plt.close(fig)
    except Exception as exc:
        print("plot failed", exc, flush=True)


def main() -> None:
    source_rows = compute_metric_rows()
    strict = pd.read_csv(
        result_file(
            "backbone_selection/strict_label_budget_table.csv"
        )
    )
    feature = pd.read_csv(
        result_file(
            "transferability/transferability_baseline_metrics.csv"
        )
    )
    B = strict.merge(
        feature, on=["pair_id", "class_a", "class_b", "backbone"], how="left"
    )
    B = B.merge(
        source_rows, on=["pair_id", "class_a", "class_b", "backbone"], how="inner"
    )
    B.to_csv(
        result_file("source_logits/source_logit_label_budget_table.csv"),
        index=False,
    )
    stats = analyze(B)
    (
        result_file("source_logits/source_logit_transferability_stats.json")
    ).write_text(json.dumps(stats, indent=2))
    write_tex(stats)
    maybe_plot(stats)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
