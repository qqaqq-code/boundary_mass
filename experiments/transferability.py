"Compare transferability baselines and incremental boundary-mass predictions. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import (
    PROJECT_ROOT,
    RUNS_DIR,
    FEATURES_DIR,
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
from scipy.stats import pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

warnings.filterwarnings("ignore")

ROOT = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
ART = RESULTS_DIR
FIG = FIGURES_DIR
ART.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

CANDIDATES = {
    "R50": RUNS_DIR / "cifar100_resnet50_pretrained/feature_cache_resnet50_imagenetv2",
    "ViT-B": RUNS_DIR / "cifar100_vitb16_targeted/feature_cache_vitb16_imagenetv1",
    "DINO-MS": FEATURES_DIR / "cifar100/dino_ms",
    "SigLIP-MS": FEATURES_DIR / "cifar100/siglip_ms",
}

N_PAIRS = 40
DIAG_FRAC = 0.30
N_BOOT = 2000
N_PERM = 2000
MAX_DIM = 128
RNG = np.random.default_rng(20260502)

OLD_NONPSI = [
    "diag_nll",
    "diag_ece",
    "diag_acc",
    "median_margin",
    "effective_rank",
    "gsa_proxy",
    "noise_flip_010",
]

TRANSFER_METRICS = ["logme", "h_score", "transrate", "nce_cluster"]
TRANSFER_DIRECTIONS = {
    "logme": "high",
    "h_score": "high",
    "transrate": "high",
    "nce_cluster": "high",
    "log1p_psi": "low",
}

TRANSFER_LABELS = {
    "logme": "LogME",
    "h_score": "H-score",
    "transrate": "TransRate",
    "nce_cluster": "NCE-style cluster",
    "log1p_psi": r"$\log(1+\hat\psi_{\rm eff})$",
}

BASE_SETS = {
    "LogME + H-score": ["logme", "h_score"],
    "LogME + H-score + TransRate": ["logme", "h_score", "transrate"],
    "feature transferability set": TRANSFER_METRICS,
    "all non-$\\psi$ + transferability": OLD_NONPSI + TRANSFER_METRICS,
}

POLICIES = {
    "transferability set": TRANSFER_METRICS,
    "transferability set + $\\psi$": TRANSFER_METRICS + ["log1p_psi"],
    "all non-$\\psi$ + transferability": OLD_NONPSI + TRANSFER_METRICS,
    "all non-$\\psi$ + transferability + $\\psi$": OLD_NONPSI
    + TRANSFER_METRICS
    + ["log1p_psi"],
}


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


def standardize_from_probe(X_probe: np.ndarray, X_diag: np.ndarray) -> np.ndarray:
    mu = X_probe.mean(axis=0, keepdims=True)
    sd = X_probe.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return ((X_diag - mu) / sd).astype("float32")


def compact_features(X: np.ndarray, max_dim: int = MAX_DIM) -> np.ndarray:
    X = X.astype("float64", copy=False)
    X = X - X.mean(axis=0, keepdims=True)
    if X.shape[1] <= max_dim:
        return X.astype("float64", copy=False)
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    k = min(max_dim, len(s))
    return (u[:, :k] * s[:k]).astype("float64", copy=False)


def h_score(X: np.ndarray, y: np.ndarray) -> float:
    X = compact_features(X)
    n, d = X.shape
    if n < 4 or d < 1:
        return float("nan")
    cov = (X.T @ X) / max(n, 1)
    lam = 1e-3 * float(np.trace(cov) / max(d, 1))
    cov = cov + (lam + 1e-8) * np.eye(d)
    mu = X.mean(axis=0)
    sb = np.zeros((d, d), dtype="float64")
    for c in np.unique(y):
        Xc = X[y == c]
        if len(Xc) == 0:
            continue
        diff = (Xc.mean(axis=0) - mu).reshape(-1, 1)
        sb += (len(Xc) / n) * (diff @ diff.T)
    return float(np.trace(np.linalg.solve(cov, sb)))


def coding_rate(X: np.ndarray, eps: float = 0.10) -> float:
    X = X.astype("float64", copy=False)
    n, d = X.shape
    if n <= 1 or d == 0:
        return 0.0
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    X = X / np.maximum(norms, 1e-12)
    cov = (X.T @ X) / max(n, 1)
    mat = np.eye(d) + (d / (eps * eps)) * cov
    sign, logdet = np.linalg.slogdet(mat)
    if sign <= 0:
        return float("nan")
    return float(0.5 * logdet)


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
    n, d = X.shape
    if n < 4 or d < 1:
        return float("nan")
    y = y.astype("float64", copy=False)
    y = y - y.mean()
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
        alpha_new = gamma / max(m2, 1e-12)
        beta_new = (n - gamma) / residual
        alpha_new = float(np.clip(alpha_new, 1e-8, 1e8))
        beta_new = float(np.clip(beta_new, 1e-8, 1e8))
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


def logme_score(X: np.ndarray, y: np.ndarray) -> float:
    X = compact_features(X)
    vals = []
    for c in [0, 1]:
        vals.append(logme_evidence_single(X, (y == c).astype("float64")))
    return float(np.mean(vals))


def nce_cluster_score(X: np.ndarray, y: np.ndarray) -> float:
    X = compact_features(X)
    n = len(y)
    if n < 10:
        return float("nan")
    k = int(min(20, max(2, n // 10)))
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


def compute_transferability_rows() -> pd.DataFrame:
    pairs = make_pairs()
    backbones = {k: load_cache(v) for k, v in CANDIDATES.items() if has_cache(v)}
    rows = []
    print("BACKBONES", list(backbones), flush=True)
    t0 = time.time()
    for pair_id, (a, b) in enumerate(pairs):
        for backbone_id, (name, (Xtr_all, ytr_all)) in enumerate(backbones.items()):
            mtr = (ytr_all == a) | (ytr_all == b)
            X_pair = Xtr_all[mtr]
            y_pair = (ytr_all[mtr] == b).astype(int)
            probe_idx, diag_idx = split_probe_diag(y_pair, pair_id, backbone_id)
            X_diag = standardize_from_probe(X_pair[probe_idx], X_pair[diag_idx])
            y_diag = y_pair[diag_idx]
            rows.append(
                {
                    "pair_id": pair_id,
                    "class_a": a,
                    "class_b": b,
                    "backbone": name,
                    "diag_n": int(len(y_diag)),
                    "logme": logme_score(X_diag, y_diag),
                    "h_score": h_score(X_diag, y_diag),
                    "transrate": transrate_score(X_diag, y_diag),
                    "nce_cluster": nce_cluster_score(X_diag, y_diag),
                }
            )
        if (pair_id + 1) % 5 == 0:
            print(
                f"processed {pair_id + 1}/{len(pairs)} pairs in {time.time() - t0:.1f}s",
                flush=True,
            )
    T = pd.DataFrame(rows)
    T.to_csv(
        result_file(
            "transferability/transferability_baseline_metrics.csv"
        ),
        index=False,
    )
    return T


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


def policy_summary(rows: pd.DataFrame) -> dict[str, float]:
    pairs = np.array(sorted(rows.pair_id.unique()))
    regret_vals, rel_vals, acc_vals, aulc_vals = [], [], [], []
    for _ in range(N_BOOT):
        sampled = RNG.choice(pairs, size=len(pairs), replace=True)
        b = pd.concat([rows[rows.pair_id == p] for p in sampled], ignore_index=True)
        regret_vals.append(float(b.regret.mean()))
        rel_vals.append(float(b.relative_regret.mean()))
        acc_vals.append(float(b.correct.mean()))
        aulc_vals.append(float(b.chosen_aulc.mean()))
    regret_ci = ci(regret_vals)
    rel_ci = ci(rel_vals)
    acc_ci = ci(acc_vals)
    aulc_ci = ci(aulc_vals)
    return {
        "n": int(rows.pair_id.nunique()),
        "selection_acc": float(rows.correct.mean()),
        "selection_acc_ci_low": acc_ci[0],
        "selection_acc_ci_high": acc_ci[1],
        "mean_regret": float(rows.regret.mean()),
        "mean_regret_ci_low": regret_ci[0],
        "mean_regret_ci_high": regret_ci[1],
        "mean_rel_regret": float(rows.relative_regret.mean()),
        "mean_rel_regret_ci_low": rel_ci[0],
        "mean_rel_regret_ci_high": rel_ci[1],
        "mean_aulc": float(rows.chosen_aulc.mean()),
        "mean_aulc_ci_low": aulc_ci[0],
        "mean_aulc_ci_high": aulc_ci[1],
    }


def choose_by_single(df: pd.DataFrame, metric: str, direction: str) -> pd.DataFrame:
    rows = []
    for pair_id, sub in df.groupby("pair_id"):
        best = sub.loc[sub.label_budget.idxmin()]
        chosen = sub.sort_values(metric, ascending=(direction == "low")).iloc[0]
        rows.append(
            {
                "pair_id": int(pair_id),
                "policy": TRANSFER_LABELS.get(metric, metric),
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


def choose_by_lopo_regression(
    df: pd.DataFrame, policy: str, predictors: list[str]
) -> pd.DataFrame:
    rows = []
    for pair_id, sub in df.groupby("pair_id"):
        train = df[df.pair_id != pair_id].copy()
        test = sub.copy()
        cols = predictors + ["label_budget"]
        train = (
            train[["pair_id", "backbone", "aulc"] + cols]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        test = (
            test[["pair_id", "backbone", "aulc"] + cols]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )
        if len(test) == 0:
            continue
        mu = train[predictors].mean(axis=0)
        sd = train[predictors].std(axis=0).replace(0, 1.0)
        Xtr = ((train[predictors] - mu) / sd).to_numpy()
        ytr = train["label_budget"].to_numpy()
        Xte = ((test[predictors] - mu) / sd).to_numpy()
        pred = LinearRegression().fit(Xtr, ytr).predict(Xte)
        test = test.copy()
        test["pred_budget"] = pred
        chosen = test.sort_values("pred_budget", ascending=True).iloc[0]
        best = sub.loc[sub.label_budget.idxmin()]
        rows.append(
            {
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


def analyze(B: pd.DataFrame) -> dict[str, object]:
    correlations = {m: corr_pair_resid(B, m) for m in TRANSFER_METRICS}

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

    policy_frames = [
        choose_by_single(B, "log1p_psi", "low"),
        choose_by_single(B, "logme", "high"),
        choose_by_single(B, "h_score", "high"),
        choose_by_single(B, "transrate", "high"),
    ]
    for policy, predictors in POLICIES.items():
        policy_frames.append(choose_by_lopo_regression(B, policy, predictors))
    P = pd.concat(policy_frames, ignore_index=True)
    P.to_csv(
        result_file(
            "transferability/transferability_policy_selection.csv"
        ),
        index=False,
    )
    policies = []
    for policy, sub in P.groupby("policy"):
        row = {"policy": policy, **policy_summary(sub)}
        policies.append(row)

    policy_map = {r["policy"]: r for r in policies}
    paired = {}
    for base, plus in [
        ("transferability set", "transferability set + $\\psi$"),
        (
            "all non-$\\psi$ + transferability",
            "all non-$\\psi$ + transferability + $\\psi$",
        ),
    ]:
        a = P[P.policy == base].set_index("pair_id")
        b = P[P.policy == plus].set_index("pair_id")
        idx = sorted(set(a.index) & set(b.index))
        dif = (
            a.loc[idx, "chosen_budget"].to_numpy()
            - b.loc[idx, "chosen_budget"].to_numpy()
        )
        paired[f"{base} -> {plus}"] = {
            "n": int(len(idx)),
            "psi_lower_budget_rate": float(np.mean(dif > 0)),
            "mean_budget_reduction": float(np.mean(dif)),
        }

    return {
        "protocol": {
            "base": "strict study probe-train/diagnostic-heldout split reused in study",
            "feature_metrics": TRANSFER_METRICS,
            "pca_cap": MAX_DIM,
            "source_logit_metrics": "computed separately by script 17 for the R50/ViT subset with torchvision ImageNet classifiers",
        },
        "correlations": correlations,
        "incremental": incremental,
        "policies": policies,
        "paired_policy_deltas": paired,
    }


def write_tex(stats: dict[str, object]) -> None:
    inc_rows = []
    for row in stats["incremental"]:
        inc_rows.append(
            f"{row['base']} & ${row['r2_base']:.3f}$ & ${row['r2_plus_psi']:.3f}$ & "
            f"${row['delta_r2']:.3f}[{row['delta_r2_ci_low']:.3f},{row['delta_r2_ci_high']:.3f}]$ & "
            f"{fmt_p(row['paired_signflip_p'])} & ${row['partial_pearson']:.3f}/{row['partial_spearman']:.3f}$\\\\"
        )
    (
        result_file(
            "transferability/transferability_incremental_rows.tex"
        )
    ).write_text("\n".join(inc_rows) + "\n")

    single_rows = []
    for metric in TRANSFER_METRICS:
        c = stats["correlations"][metric]
        single_rows.append(
            f"{TRANSFER_LABELS[metric]} & ${c['pearson']:.3f}/{c['spearman']:.3f}$\\\\"
        )
    (
        result_file("transferability/transferability_single_rows.tex")
    ).write_text("\n".join(single_rows) + "\n")

    policy_order = [
        r"$\log(1+\hat\psi_{\rm eff})$",
        "LogME",
        "H-score",
        "TransRate",
        "transferability set",
        "transferability set + $\\psi$",
        "all non-$\\psi$ + transferability",
        "all non-$\\psi$ + transferability + $\\psi$",
    ]
    policies = {row["policy"]: row for row in stats["policies"]}
    policy_rows = []
    for name in policy_order:
        if name not in policies:
            continue
        row = policies[name]
        policy_rows.append(
            f"{name} & ${100 * row['selection_acc']:.1f}[{100 * row['selection_acc_ci_low']:.1f},{100 * row['selection_acc_ci_high']:.1f}]$ & "
            f"${row['mean_regret']:.2f}[{row['mean_regret_ci_low']:.2f},{row['mean_regret_ci_high']:.2f}]$ & "
            f"${100 * row['mean_rel_regret']:.1f}[{100 * row['mean_rel_regret_ci_low']:.1f},{100 * row['mean_rel_regret_ci_high']:.1f}]$ & "
            f"${row['mean_aulc']:.3f}$\\\\"
        )
    (
        result_file("transferability/transferability_policy_rows.tex")
    ).write_text("\n".join(policy_rows) + "\n")


def maybe_plot(B: pd.DataFrame, stats: dict[str, object]) -> None:
    try:
        import matplotlib.pyplot as plt

        rows = stats["incremental"]
        labels = [r["base"].replace(" + ", "\n+ ") for r in rows]
        vals = [r["delta_r2"] for r in rows]
        lo = [r["delta_r2"] - r["delta_r2_ci_low"] for r in rows]
        hi = [r["delta_r2_ci_high"] - r["delta_r2"] for r in rows]
        fig, ax = plt.subplots(figsize=(7.0, 3.0))
        ax.bar(np.arange(len(vals)), vals, color="#4C78A8")
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
        ax.set_title(r"$\log(1+\hat\psi_{\rm eff})$ beyond transferability baselines")
        fig.tight_layout()
        fig.savefig(FIG / "transferability_baseline_experiment.png", dpi=220)
        fig.savefig(FIG / "transferability_baseline_experiment.pdf")
        plt.close(fig)
    except Exception as exc:
        print("plot failed", exc, flush=True)


def main() -> None:
    strict_path = result_file(
        "backbone_selection/strict_label_budget_table.csv"
    )
    if not strict_path.exists():
        raise FileNotFoundError(f"missing strict experiment table: {strict_path}")
    B0 = pd.read_csv(strict_path)
    T = compute_transferability_rows()
    B = B0.merge(T, on=["pair_id", "class_a", "class_b", "backbone"], how="inner")
    B.to_csv(
        result_file(
            "transferability/transferability_label_budget_table.csv"
        ),
        index=False,
    )
    stats = analyze(B)
    (
        result_file(
            "transferability/transferability_baseline_stats.json"
        )
    ).write_text(json.dumps(stats, indent=2))
    write_tex(stats)
    maybe_plot(B, stats)
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
