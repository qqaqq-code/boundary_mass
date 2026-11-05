"Compare boundary mass and label budgets across frozen backbones. See docs/EXPERIMENTS.md for usage."

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
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, r2_score

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

SPCS = [2, 5, 10, 20, 50]
SEEDS = [0, 1]
N_PAIRS = 40
DIAG_FRAC = 0.30
N_BOOT = 2000
N_PERM = 2000
RNG = np.random.default_rng(20260501)

METRIC_DIRECTIONS = {
    "log1p_psi": "low",
    "psi_eff": "low",
    "gsa_proxy": "low",
    "noise_flip_010": "low",
    "median_margin": "high",
    "mean_margin": "high",
    "diag_nll": "low",
    "diag_ece": "low",
    "diag_acc": "high",
    "effective_rank": "high",
    "weight_norm": "low",
}

LABELS = {
    "log1p_psi": r"$\log(1+\hat\psi_{\rm eff})$",
    "psi_eff": r"$\hat\psi_{\rm eff}$",
    "gsa_proxy": "Gaussian-area proxy",
    "noise_flip_010": "noise flip proxy",
    "median_margin": "median margin",
    "diag_nll": "held-out NLL",
    "diag_ece": "held-out ECE",
    "diag_acc": "held-out acc.",
    "effective_rank": "effective rank",
}

BASE_SETS = {
    "NLL + ECE": ["diag_nll", "diag_ece"],
    "margin + effective rank": ["median_margin", "effective_rank"],
    "NLL + ECE + margin + rank": [
        "diag_nll",
        "diag_ece",
        "median_margin",
        "effective_rank",
    ],
    "all non-$\\psi$ proxies": [
        "diag_nll",
        "diag_ece",
        "diag_acc",
        "median_margin",
        "effective_rank",
        "gsa_proxy",
        "noise_flip_010",
    ],
}


def has_cache(d: Path) -> bool:
    return (d / "train_features.pt").exists() and (d / "test_features.pt").exists()


def load_cache(d: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tr = torch.load(d / "train_features.pt", map_location="cpu")
    te = torch.load(d / "test_features.pt", map_location="cpu")
    return (
        tr["features"].float().numpy().astype("float32"),
        tr["labels"].numpy().astype("int64"),
        te["features"].float().numpy().astype("float32"),
        te["labels"].numpy().astype("int64"),
    )


def standardize_from_probe(
    X_probe: np.ndarray, X_diag: np.ndarray, X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = X_probe.mean(axis=0, keepdims=True)
    sd = X_probe.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return (
        ((X_probe - mu) / sd).astype("float32"),
        ((X_diag - mu) / sd).astype("float32"),
        ((X_test - mu) / sd).astype("float32"),
    )


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
    if X.shape[0] < 3:
        return float("nan")
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


def hard_tube_from_dist(dist: np.ndarray, y: np.ndarray, eps: float) -> float:
    beta = max(min(float(np.mean(y == 1)), float(np.mean(y == 0))), 1e-6)
    return float(np.mean(np.abs(dist) <= eps) / (2.0 * eps * beta + 1e-12))


def gsa_proxy(clf: LogisticRegression) -> float:
    w = clf.coef_.reshape(-1)
    nw = np.linalg.norm(w) + 1e-12
    t = float(clf.intercept_[0]) / nw
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


def interp_budget(
    spcs: np.ndarray, accs: np.ndarray, target: float
) -> tuple[float, bool]:
    spcs = np.asarray(spcs, dtype=float)
    accs = np.asarray(accs, dtype=float)
    order = np.argsort(spcs)
    spcs = spcs[order]
    accs = accs[order]
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


def corr_pair_resid(df: pd.DataFrame, x: str, y: str) -> dict[str, float] | None:
    if x == y:
        return None
    d = df[[x, y, "pair_id"]].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < 8 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return None
    d[x + "_r"] = d[x] - d.groupby("pair_id")[x].transform("mean")
    d[y + "_r"] = d[y] - d.groupby("pair_id")[y].transform("mean")
    if d[x + "_r"].std() == 0 or d[y + "_r"].std() == 0:
        return None
    return {
        "n": int(len(d)),
        "pearson": float(pearsonr(d[x + "_r"], d[y + "_r"])[0]),
        "spearman": float(spearmanr(d[x + "_r"], d[y + "_r"])[0]),
    }


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
        r2_plus = r2_score(
            y,
            LinearRegression()
            .fit(np.hstack([X, psi_perm]), y)
            .predict(np.hstack([X, psi_perm])),
        )
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
        "pearson": float(pearsonr(psi_res, y_res)[0]),
        "spearman": float(spearmanr(psi_res, y_res)[0]),
    }


def selection_summary_with_ci(sel: pd.DataFrame, metric: str) -> dict[str, float]:
    sub = sel[sel.metric == metric].copy()
    pairs = np.array(sorted(sub.pair_id.unique()))
    acc_vals, regret_vals, rel_vals = [], [], []
    for _ in range(N_BOOT):
        sampled = RNG.choice(pairs, size=len(pairs), replace=True)
        b = pd.concat([sub[sub.pair_id == p] for p in sampled], ignore_index=True)
        acc_vals.append(float(b.correct.mean()))
        regret_vals.append(float(b.regret.mean()))
        rel_vals.append(float(b.relative_regret.mean()))
    acc_ci = ci(acc_vals)
    regret_ci = ci(regret_vals)
    rel_ci = ci(rel_vals)
    return {
        "metric": metric,
        "n": int(sub.pair_id.nunique()),
        "selection_acc": float(sub.correct.mean()),
        "selection_acc_ci_low": acc_ci[0],
        "selection_acc_ci_high": acc_ci[1],
        "mean_regret": float(sub.regret.mean()),
        "mean_regret_ci_low": regret_ci[0],
        "mean_regret_ci_high": regret_ci[1],
        "mean_rel_regret": float(sub.relative_regret.mean()),
        "mean_rel_regret_ci_low": rel_ci[0],
        "mean_rel_regret_ci_high": rel_ci[1],
        "median_regret": float(sub.regret.median()),
    }


def fmt_p(p: float) -> str:
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def matched_utility(df: pd.DataFrame, acc_tol: float) -> pd.DataFrame:
    rows = []
    for pair_id, sub in df.groupby("pair_id"):
        recs = sub.to_dict("records")
        for i in range(len(recs)):
            for j in range(i + 1, len(recs)):
                a, b = recs[i], recs[j]
                if abs(a["target_full_acc"] - b["target_full_acc"]) > acc_tol:
                    continue
                low, high = (a, b) if a["log1p_psi"] <= b["log1p_psi"] else (b, a)
                diff = float(high["label_budget"] - low["label_budget"])
                win = 0.5 if abs(diff) < 1e-9 else (1.0 if diff > 0 else 0.0)
                rows.append(
                    {
                        "pair_id": int(pair_id),
                        "acc_tol": acc_tol,
                        "low_psi_backbone": low["backbone"],
                        "high_psi_backbone": high["backbone"],
                        "win": win,
                        "label_saving": diff,
                        "relative_saving": diff / (float(high["label_budget"]) + 1e-9),
                        "full_acc_gap": abs(
                            a["target_full_acc"] - b["target_full_acc"]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def matched_summary_ci(matched: pd.DataFrame) -> dict[str, float]:
    if matched.empty:
        return {"n_comparisons": 0}
    pairs = np.array(sorted(matched.pair_id.unique()))
    win_vals, mean_save_vals = [], []
    for _ in range(N_BOOT):
        sampled = RNG.choice(pairs, size=len(pairs), replace=True)
        b = pd.concat(
            [matched[matched.pair_id == p] for p in sampled], ignore_index=True
        )
        win_vals.append(float(b.win.mean()))
        mean_save_vals.append(float(b.relative_saving.mean()))
    win_ci = ci(win_vals)
    mean_ci = ci(mean_save_vals)
    return {
        "n_comparisons": int(len(matched)),
        "n_pairs": int(matched.pair_id.nunique()),
        "win_rate": float(matched.win.mean()),
        "win_rate_ci_low": win_ci[0],
        "win_rate_ci_high": win_ci[1],
        "mean_relative_saving": float(matched.relative_saving.mean()),
        "mean_relative_saving_ci_low": mean_ci[0],
        "mean_relative_saving_ci_high": mean_ci[1],
    }


def run_experiment() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    pairs = make_pairs()
    backbones = {k: load_cache(v) for k, v in CANDIDATES.items() if has_cache(v)}
    print("BACKBONES", list(backbones), flush=True)

    metric_rows, lc_rows = [], []
    t0 = time.time()
    for pair_id, (a, b) in enumerate(pairs):
        for backbone_id, (name, (Xtr_all, ytr_all, Xte_all, yte_all)) in enumerate(
            backbones.items()
        ):
            mtr = (ytr_all == a) | (ytr_all == b)
            mte = (yte_all == a) | (yte_all == b)
            Xtr_pair = Xtr_all[mtr]
            ytr_pair = (ytr_all[mtr] == b).astype(int)
            Xte = Xte_all[mte]
            yte = (yte_all[mte] == b).astype(int)
            probe_idx, diag_idx = split_probe_diag(ytr_pair, pair_id, backbone_id)
            X_probe_raw, y_probe = Xtr_pair[probe_idx], ytr_pair[probe_idx]
            X_diag_raw, y_diag = Xtr_pair[diag_idx], ytr_pair[diag_idx]
            X_probe, X_diag, X_test = standardize_from_probe(
                X_probe_raw, X_diag_raw, Xte
            )

            clf = fit_lr(X_probe, y_probe)
            diag_prob = clf.predict_proba(X_diag)[:, 1]
            diag_pred = (diag_prob >= 0.5).astype(int)
            dist = signed_dist(clf, X_diag)
            psi, eps = psi_kernel_from_dist(dist, y_diag)
            hard = hard_tube_from_dist(dist, y_diag, eps)
            test_acc = float(clf.score(X_test, yte))
            metric_rows.append(
                {
                    "pair_id": pair_id,
                    "class_a": a,
                    "class_b": b,
                    "backbone": name,
                    "probe_n": int(len(y_probe)),
                    "diag_n": int(len(y_diag)),
                    "target_full_acc": test_acc,
                    "psi_eff": psi,
                    "log1p_psi": math.log1p(psi),
                    "hard_tube_psi": hard,
                    "log1p_hard_tube_psi": math.log1p(hard),
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
                }
            )

            idx0 = np.flatnonzero(y_probe == 0)
            idx1 = np.flatnonzero(y_probe == 1)
            for spc in SPCS:
                for seed in SEEDS:
                    rng = np.random.default_rng(1000 + pair_id * 31 + seed)
                    take = np.r_[
                        rng.choice(idx0, min(spc, len(idx0)), replace=False),
                        rng.choice(idx1, min(spc, len(idx1)), replace=False),
                    ]
                    sub = fit_lr(X_probe[take], y_probe[take])
                    lc_rows.append(
                        {
                            "pair_id": pair_id,
                            "class_a": a,
                            "class_b": b,
                            "backbone": name,
                            "spc": spc,
                            "seed": seed,
                            "acc": float(sub.score(X_test, yte)),
                        }
                    )
        if (pair_id + 1) % 10 == 0:
            print(
                "done pairs",
                pair_id + 1,
                "elapsed",
                round(time.time() - t0, 1),
                flush=True,
            )

    M = pd.DataFrame(metric_rows)
    LC = pd.DataFrame(lc_rows)
    M.to_csv(
        result_file(
            "backbone_selection/strict_label_budget_metrics.csv"
        ),
        index=False,
    )
    LC.to_csv(
        result_file(
            "backbone_selection/strict_label_budget_learning_curves.csv"
        ),
        index=False,
    )

    lc_mean = LC.groupby(["pair_id", "backbone", "spc"]).acc.mean().reset_index()
    rows = []
    for pair_id, subM in M.groupby("pair_id"):
        target = 0.90 * float(subM.target_full_acc.max())
        for _, r in subM.iterrows():
            cur = lc_mean[
                (lc_mean.pair_id == pair_id) & (lc_mean.backbone == r.backbone)
            ].sort_values("spc")
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
                    "acc_5spc": float(cur[cur.spc == 5].acc.iloc[0])
                    if (cur.spc == 5).any()
                    else float("nan"),
                }
            )
    B = pd.DataFrame(rows)
    B.to_csv(
        result_file(
            "backbone_selection/strict_label_budget_table.csv"
        ),
        index=False,
    )
    return M, LC, B, list(backbones)


def analyze(B: pd.DataFrame, backbones: list[str]) -> dict[str, object]:
    metrics = list(METRIC_DIRECTIONS)
    targets = ["label_budget", "aulc", "acc_5spc", "target_full_acc"]
    correlations = {t: {m: corr_pair_resid(B, m, t) for m in metrics} for t in targets}

    incremental = []
    for name, base in BASE_SETS.items():
        r2_base, r2_plus, obs = r2_pair(B, base)
        boot_ci, p_boot = cluster_boot_delta(B, base)
        p_perm = pair_signflip_p(B, base, obs)
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
            }
        )

    sel_rows = []
    for pair_id, sub in B.groupby("pair_id"):
        best = sub.loc[sub.label_budget.idxmin()]
        for metric, direction in METRIC_DIRECTIONS.items():
            chosen = sub.sort_values(metric, ascending=(direction == "low")).iloc[0]
            sel_rows.append(
                {
                    "pair_id": int(pair_id),
                    "metric": metric,
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
                }
            )
    S = pd.DataFrame(sel_rows)
    S.to_csv(
        result_file(
            "backbone_selection/strict_label_budget_selection.csv"
        ),
        index=False,
    )
    selection = [
        selection_summary_with_ci(S, m) for m in LABELS if m in S.metric.unique()
    ]

    matched = []
    matched_rows = []
    for tol in [0.01, 0.02]:
        m = matched_utility(B, tol)
        if not m.empty:
            m.to_csv(
                result_file(f"strict_matched_fullacc_{int(tol * 100)}pct.csv"),
                index=False,
            )
        s = matched_summary_ci(m)
        s["acc_tol"] = tol
        matched.append(s)
        if s.get("n_comparisons", 0):
            matched_rows.append(
                f"$\\le {100 * tol:.0f}\\%$ & {s['n_comparisons']} & "
                f"${100 * s['win_rate']:.1f}[{100 * s['win_rate_ci_low']:.1f},{100 * s['win_rate_ci_high']:.1f}]$ & "
                f"${100 * s['mean_relative_saving']:.1f}[{100 * s['mean_relative_saving_ci_low']:.1f},{100 * s['mean_relative_saving_ci_high']:.1f}]$\\\\"
            )

    inc_rows = []
    for r in incremental:
        inc_rows.append(
            f"{r['base']} & ${r['r2_base']:.3f}$ & ${r['r2_plus_psi']:.3f}$ & "
            f"${r['delta_r2']:.3f}[{r['delta_r2_ci_low']:.3f},{r['delta_r2_ci_high']:.3f}]$ & "
            f"{fmt_p(r['paired_signflip_p'])}\\\\"
        )
    (
        result_file(
            "backbone_selection/strict_incremental_ci_rows.tex"
        )
    ).write_text("\n".join(inc_rows) + "\n")

    sel_by_metric = {r["metric"]: r for r in selection}
    selection_rows = []
    for metric in [
        "log1p_psi",
        "psi_eff",
        "gsa_proxy",
        "noise_flip_010",
        "median_margin",
        "diag_nll",
        "diag_ece",
        "diag_acc",
        "effective_rank",
    ]:
        r = sel_by_metric[metric]
        c = correlations["label_budget"][metric]
        corr_txt = "--" if c is None else f"{c['pearson']:.3f}/{c['spearman']:.3f}"
        selection_rows.append(
            f"{LABELS[metric]} & {corr_txt} & "
            f"${100 * r['selection_acc']:.1f}[{100 * r['selection_acc_ci_low']:.1f},{100 * r['selection_acc_ci_high']:.1f}]$ & "
            f"${r['mean_regret']:.2f}[{r['mean_regret_ci_low']:.2f},{r['mean_regret_ci_high']:.2f}]$ & "
            f"${100 * r['mean_rel_regret']:.1f}[{100 * r['mean_rel_regret_ci_low']:.1f},{100 * r['mean_rel_regret_ci_high']:.1f}]$\\\\"
        )
    (
        result_file("backbone_selection/strict_selection_ci_rows.tex")
    ).write_text("\n".join(selection_rows) + "\n")
    (
        result_file(
            "backbone_selection/strict_matched_utility_rows.tex"
        )
    ).write_text("\n".join(matched_rows) + "\n")

    all_nonpsi = BASE_SETS["all non-$\\psi$ proxies"]
    partial = residual_corr_after_base(B, all_nonpsi)

    # Plot strict experiment for the appendix.
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        incdf = pd.DataFrame(incremental)
        fig, axs = plt.subplots(1, 2, figsize=(8.5, 3.3))
        for bb, sub in B.groupby("backbone"):
            axs[0].scatter(sub.log1p_psi, sub.label_budget, s=18, alpha=0.7, label=bb)
        axs[0].set_xlabel(r"held-out-train $\log(1+D_{\rm ker})$")
        axs[0].set_ylabel("test label budget")
        axs[0].legend(frameon=False, fontsize=7)
        axs[1].bar(range(len(incdf)), incdf.delta_r2, color="#4c78a8")
        axs[1].set_xticks(range(len(incdf)))
        axs[1].set_xticklabels(incdf.base, rotation=25, ha="right", fontsize=8)
        axs[1].set_ylabel(r"$\Delta R^2$ from adding $D_{\rm ker}$")
        fig.tight_layout()
        fig.savefig(FIG / "fig06_detailed_backbone_selection.png", dpi=220)
        fig.savefig(FIG / "strict_label_budget_decision_experiment.pdf")
    except Exception as exc:
        print("plot failed", repr(exc), flush=True)

    summary = {
        "protocol": {
            "diagnostics": "probe trained on train/probe split; psi, margin, NLL, ECE, rank, noise flip, GSA computed only on train/diagnostic-heldout split",
            "target": "test split used only for full-shot target and learning-curve label budget",
            "diag_frac": DIAG_FRAC,
            "spcs": SPCS,
            "seeds": SEEDS,
        },
        "backbones": backbones,
        "n_pairs": int(B.pair_id.nunique()),
        "n_rows": int(len(B)),
        "correlations": correlations,
        "incremental": incremental,
        "partial_after_all_nonpsi": partial,
        "selection": selection,
        "matched_utility": matched,
    }
    (
        result_file(
            "backbone_selection/strict_label_budget_decision_stats.json"
        )
    ).write_text(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    _, _, B, backbones = run_experiment()
    summary = analyze(B, backbones)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
