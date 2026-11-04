"Evaluate the linear PACTran comparison. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, RUNS_DIR, FEATURES_DIR, RESULTS_DIR, result_file

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss, r2_score

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
MAX_DIM = 128
N_BOOT = 2000
N_PERM = 2000
RNG = np.random.default_rng(20260503)

OLD_NONPSI = [
    "diag_nll",
    "diag_ece",
    "diag_acc",
    "median_margin",
    "effective_rank",
    "gsa_proxy",
    "noise_flip_010",
]
TRANSFER = ["logme", "h_score", "transrate", "nce_cluster"]


def make_pairs():
    rng = np.random.default_rng(20260429)
    pairs = [(a, a + 1) for a in range(0, 40, 2)]
    seen = set(tuple(sorted(p)) for p in pairs)
    while len(pairs) < N_PAIRS:
        a, b = rng.choice(100, 2, replace=False)
        p = tuple(sorted((int(a), int(b))))
        if p not in seen:
            seen.add(p)
            pairs.append(p)
    return pairs


def load_cache(d: Path):
    tr = torch.load(d / "train_features.pt", map_location="cpu")
    return tr["features"].float().numpy().astype("float32"), tr[
        "labels"
    ].numpy().astype("int64")


def split_probe_diag(y, pair_id, backbone_id):
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


def standardize_from_probe(X_probe, X_diag):
    mu = X_probe.mean(axis=0, keepdims=True)
    sd = X_probe.std(axis=0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return ((X_diag - mu) / sd).astype("float32")


def compact(X):
    X = X.astype("float64", copy=False)
    X = X - X.mean(axis=0, keepdims=True)
    if X.shape[1] <= MAX_DIM:
        return X
    u, s, _ = np.linalg.svd(X, full_matrices=False)
    k = min(MAX_DIM, len(s))
    return u[:, :k] * s[:k]


def pactran_linear_score(X, y):
    # PACTran-style linear probe score: negative regularized target evidence.
    # It is a practical proxy, not claimed as a faithful reimplementation of every PACTran variant.
    Xc = compact(X)
    clf = LogisticRegression(
        C=1.0, penalty="l2", solver="lbfgs", max_iter=500, random_state=0
    )
    clf.fit(Xc, y)
    prob = clf.predict_proba(Xc)
    ce = float(log_loss(y, prob, labels=[0, 1]))
    w2 = float(np.sum(clf.coef_ * clf.coef_))
    complexity = 0.5 * w2 / max(1, len(y))
    return float(-(ce + complexity))


def compute_rows():
    pairs = make_pairs()
    caches = {k: load_cache(v) for k, v in CANDIDATES.items()}
    rows = []
    for pair_id, (a, b) in enumerate(pairs):
        for backbone_id, (name, (Xtr_all, ytr_all)) in enumerate(caches.items()):
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
                    "pactran_linear": pactran_linear_score(X_diag, y_diag),
                }
            )
    out = pd.DataFrame(rows)
    out.to_csv(result_file("pactran/pactran_linear_metrics.csv"), index=False)
    return out


def residualize(df, cols, group_col):
    out = df.copy()
    for col in cols:
        out[col + "_r"] = out[col] - out.groupby(group_col)[col].transform("mean")
    return out


def r2_pair(df, base, group_col="pair_id"):
    cols = ["label_budget", "log1p_psi"] + base
    d = df[[group_col] + cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = residualize(d, cols, group_col)
    y = d["label_budget_r"].to_numpy()
    X = d[[m + "_r" for m in base]].to_numpy()
    Xp = d[[m + "_r" for m in base + ["log1p_psi"]]].to_numpy()
    rb = r2_score(y, LinearRegression().fit(X, y).predict(X))
    rp = r2_score(y, LinearRegression().fit(Xp, y).predict(Xp))
    return float(rb), float(rp), float(rp - rb)


def delta_r2(df, base, group_col="pair_id"):
    try:
        return r2_pair(df, base, group_col)[2]
    except Exception:
        return float("nan")


def ci(vals):
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def sample_pairs(df):
    pairs = np.array(sorted(df.pair_id.unique()))
    sampled = RNG.choice(pairs, size=len(pairs), replace=True)
    chunks = []
    for boot_id, pair_id in enumerate(sampled):
        part = df[df.pair_id == pair_id].copy()
        part["boot_pair"] = boot_id
        chunks.append(part)
    return pd.concat(chunks, ignore_index=True)


def cluster_boot_delta(df, base):
    vals = [delta_r2(sample_pairs(df), base, "boot_pair") for _ in range(N_BOOT)]
    vals = [v for v in vals if np.isfinite(v)]
    p_boot = (1.0 + sum(v <= 0 for v in vals)) / (len(vals) + 1.0)
    return ci(vals), float(p_boot)


def pair_signflip_p(df, base, observed):
    cols = ["label_budget", "log1p_psi"] + base
    d = df[["pair_id"] + cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = residualize(d, cols, "pair_id")
    y = d["label_budget_r"].to_numpy()
    X = d[[m + "_r" for m in base]].to_numpy()
    psi = d[["log1p_psi_r"]].to_numpy()
    rb = r2_score(y, LinearRegression().fit(X, y).predict(X))
    blocks = [idx.to_numpy() for _, idx in d.groupby("pair_id").groups.items()]
    null = []
    for _ in range(N_PERM):
        pp = psi.copy()
        for sign, idx in zip(RNG.choice([-1.0, 1.0], size=len(blocks)), blocks):
            pp[idx, 0] *= sign
        Xp = np.hstack([X, pp])
        null.append(float(r2_score(y, LinearRegression().fit(Xp, y).predict(Xp)) - rb))
    return float((1.0 + sum(v >= observed for v in null)) / (len(null) + 1.0))


def partial_corr(df, base):
    cols = ["label_budget", "log1p_psi"] + base
    d = df[["pair_id"] + cols].replace([np.inf, -np.inf], np.nan).dropna().copy()
    d = residualize(d, cols, "pair_id")
    y = d["label_budget_r"].to_numpy()
    X = d[[m + "_r" for m in base]].to_numpy()
    psi = d["log1p_psi_r"].to_numpy()
    yr = y - LinearRegression().fit(X, y).predict(X)
    pr = psi - LinearRegression().fit(X, psi).predict(X)
    return float(pearsonr(pr, yr)[0]), float(spearmanr(pr, yr)[0])


def fmt_p(p):
    return "$<0.001$" if p < 0.001 else f"${p:.3f}$"


def main():
    rows = compute_rows()
    base = pd.read_csv(
        result_file(
            "transferability/transferability_label_budget_table.csv"
        )
    )
    B = base.merge(rows, on=["pair_id", "class_a", "class_b", "backbone"], how="inner")
    B.to_csv(
        result_file("pactran/pactran_transferability_label_budget_table.csv"),
        index=False,
    )
    base_sets = {
        "PACTran-linear": ["pactran_linear"],
        "feature transferability + PACTran-linear": TRANSFER + ["pactran_linear"],
        "all non-$\\psi$ + transferability + PACTran-linear": OLD_NONPSI
        + TRANSFER
        + ["pactran_linear"],
    }
    inc = []
    tex = []
    for name, cols in base_sets.items():
        rb, rp, dlt = r2_pair(B, cols)
        boot, _ = cluster_boot_delta(B, cols)
        p = pair_signflip_p(B, cols, dlt)
        pc = partial_corr(B, cols)
        loco = np.asarray(
            [delta_r2(B[B.pair_id != pid], cols) for pid in sorted(B.pair_id.unique())]
        )
        row = {
            "base": name,
            "r2_base": rb,
            "r2_plus_psi": rp,
            "delta_r2": dlt,
            "delta_r2_ci_low": boot[0],
            "delta_r2_ci_high": boot[1],
            "paired_signflip_p": p,
            "partial_pearson": pc[0],
            "partial_spearman": pc[1],
            "loco_positive": int(np.sum(loco > 0)),
            "loco_n": int(np.isfinite(loco).sum()),
        }
        inc.append(row)
        tex.append(
            f"{name} & ${rb:.3f}$ & ${rp:.3f}$ & ${dlt:.3f}[{boot[0]:.3f},{boot[1]:.3f}]$ & {fmt_p(p)} & ${pc[0]:.3f}/{pc[1]:.3f}$\\\\"
        )
    stats = {
        "protocol": {
            "metric": "PACTran-linear/PAC-Bayes-style regularized target-head score on diagnostic-heldout features",
            "not_claimed": "not an exact reimplementation of every PACTran variant",
        },
        "incremental": inc,
    }
    (result_file("pactran/pactran_linear_stats.json")).write_text(
        json.dumps(stats, indent=2)
    )
    (result_file("pactran/pactran_linear_rows.tex")).write_text(
        "\n".join(tex) + "\n"
    )
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
