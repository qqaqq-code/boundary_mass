"Evaluate backbone-selection policies from pilot labels. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, RUNS_DIR, FEATURES_DIR, RESULTS_DIR, result_file

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import log_loss

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
PILOT_SPCS = [2, 5, 10]
N_BOOT = 2000
RNG = np.random.default_rng(20260503)
DIAG_FRAC = 0.30

SINGLE_POLICIES = {
    "choose by $\\psi$": ("log1p_psi", "low"),
    "choose by margin": ("median_margin", "high"),
    "choose by LogME": ("logme", "high"),
    "choose by H-score": ("h_score", "high"),
}
REG_POLICIES = {
    "NLL/ECE": ["diag_nll", "diag_ece"],
    "LogME/H-score": ["logme", "h_score"],
    "transferability set": ["logme", "h_score", "transrate", "nce_cluster"],
    "combined non-$\\psi$": [
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
    "combined non-$\\psi$ + $\\psi$": [
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
        "log1p_psi",
    ],
}


def make_pairs() -> list[tuple[int, int]]:
    rng = np.random.default_rng(20260429)
    pairs: list[tuple[int, int]] = []
    for a in range(0, 40, 2):
        pairs.append((a, a + 1))
    seen = set(tuple(sorted(p)) for p in pairs)
    while len(pairs) < 40:
        a, b = rng.choice(100, 2, replace=False)
        p = tuple(sorted((int(a), int(b))))
        if p not in seen:
            seen.add(p)
            pairs.append(p)
    return pairs


def load_cache(d: Path):
    tr = torch.load(d / "train_features.pt", map_location="cpu")
    te = torch.load(d / "test_features.pt", map_location="cpu")
    return (
        tr["features"].float().numpy().astype("float32"),
        tr["labels"].numpy().astype("int64"),
        te["features"].float().numpy().astype("float32"),
        te["labels"].numpy().astype("int64"),
    )


def split_probe_diag(y: np.ndarray, pair_id: int, backbone_id: int):
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
    return ((X_probe - mu) / sd).astype("float32"), ((X_diag - mu) / sd).astype(
        "float32"
    )


def fit_lr(X, y):
    clf = LogisticRegression(
        C=1.0, penalty="l2", solver="liblinear", max_iter=500, tol=1e-4, random_state=0
    )
    clf.fit(X, y)
    return clf


def ece_binary(prob, y, bins=15):
    conf = np.maximum(prob, 1 - prob)
    pred = (prob >= 0.5).astype(int)
    corr = (pred == y).astype(float)
    out = 0.0
    for lo, hi in zip(
        np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]
    ):
        m = (conf >= lo) & (conf < (hi if hi < 1 else hi + 1e-8))
        if m.any():
            out += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return float(out)


def effective_rank(X):
    X = X - X.mean(axis=0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    eig = s * s / max(1, X.shape[0] - 1)
    sm = eig.sum()
    return float(sm * sm / (np.square(eig).sum() + 1e-12))


def psi_from_dist(dist, y):
    eps = max(0.05, float(np.quantile(np.abs(dist), 0.20)))
    beta = max(min(float(np.mean(y == 1)), float(np.mean(y == 0))), 1e-6)
    kern = np.exp(-0.5 * (dist / eps) ** 2) / (eps * math.sqrt(2.0 * math.pi))
    return float(kern.mean() / beta)


def gsa_proxy(clf):
    w = clf.coef_.reshape(-1)
    t = float(clf.intercept_[0]) / (np.linalg.norm(w) + 1e-12)
    return float(2.0 * math.exp(-0.5 * t * t) / math.sqrt(2.0 * math.pi))


def noise_flip(clf, X, alpha=0.10, repeats=3):
    base = clf.predict(X)
    rng = np.random.default_rng(123)
    scale = np.std(X, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    return float(
        np.mean(
            [
                np.mean(
                    clf.predict(
                        X + rng.normal(0, alpha, size=X.shape).astype("float32") * scale
                    )
                    != base
                )
                for _ in range(repeats)
            ]
        )
    )


def compute_pilot_metrics() -> pd.DataFrame:
    transfer = pd.read_csv(
        result_file(
            "transferability/transferability_baseline_metrics.csv"
        )
    )
    pairs = make_pairs()
    caches = {k: load_cache(v) for k, v in CANDIDATES.items()}
    rows = []
    for pair_id, (a, b) in enumerate(pairs):
        for backbone_id, (name, (Xtr_all, ytr_all, _, _)) in enumerate(caches.items()):
            mtr = (ytr_all == a) | (ytr_all == b)
            Xtr_pair = Xtr_all[mtr]
            ytr_pair = (ytr_all[mtr] == b).astype(int)
            probe_idx, diag_idx = split_probe_diag(ytr_pair, pair_id, backbone_id)
            X_probe_raw, y_probe = Xtr_pair[probe_idx], ytr_pair[probe_idx]
            X_diag_raw, y_diag = Xtr_pair[diag_idx], ytr_pair[diag_idx]
            X_probe, X_diag = standardize_from_probe(X_probe_raw, X_diag_raw)
            idx0 = np.flatnonzero(y_probe == 0)
            idx1 = np.flatnonzero(y_probe == 1)
            for spc in PILOT_SPCS:
                vals = []
                for seed in [0, 1, 2]:
                    rng = np.random.default_rng(
                        5000 + pair_id * 37 + backbone_id * 11 + spc * 3 + seed
                    )
                    take = np.r_[
                        rng.choice(idx0, min(spc, len(idx0)), replace=False),
                        rng.choice(idx1, min(spc, len(idx1)), replace=False),
                    ]
                    clf = fit_lr(X_probe[take], y_probe[take])
                    prob = clf.predict_proba(X_diag)[:, 1]
                    pred = (prob >= 0.5).astype(int)
                    w = clf.coef_.reshape(-1)
                    dist = clf.decision_function(X_diag) / (np.linalg.norm(w) + 1e-12)
                    psi = psi_from_dist(dist, y_diag)
                    vals.append(
                        {
                            "psi_eff": psi,
                            "log1p_psi": math.log1p(psi),
                            "median_margin": float(np.median(np.abs(dist))),
                            "diag_nll": float(
                                log_loss(
                                    y_diag, np.vstack([1 - prob, prob]).T, labels=[0, 1]
                                )
                            ),
                            "diag_ece": ece_binary(prob, y_diag),
                            "diag_acc": float((pred == y_diag).mean()),
                            "effective_rank": effective_rank(X_diag),
                            "gsa_proxy": gsa_proxy(clf),
                            "noise_flip_010": noise_flip(clf, X_diag),
                        }
                    )
                avg = pd.DataFrame(vals).mean().to_dict()
                rows.append(
                    {
                        "pair_id": pair_id,
                        "class_a": a,
                        "class_b": b,
                        "backbone": name,
                        "pilot_spc": spc,
                        **avg,
                    }
                )
    P = pd.DataFrame(rows)
    P = P.merge(transfer, on=["pair_id", "class_a", "class_b", "backbone"], how="left")
    P.to_csv(
        result_file("pilot_selection/pilot_policy_metrics.csv"), index=False
    )
    return P


def ci(vals):
    arr = np.asarray(vals, dtype=float)
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def summarize(rows: pd.DataFrame):
    pairs = np.array(sorted(rows.pair_id.unique()))
    vals = {"acc": [], "regret": [], "rel": [], "aulc": []}
    for _ in range(N_BOOT):
        sampled = RNG.choice(pairs, size=len(pairs), replace=True)
        b = pd.concat([rows[rows.pair_id == p] for p in sampled], ignore_index=True)
        vals["acc"].append(float(b.correct.mean()))
        vals["regret"].append(float(b.regret.mean()))
        vals["rel"].append(float(b.relative_regret.mean()))
        vals["aulc"].append(float(b.chosen_aulc.mean()))
    return {
        "n": int(rows.pair_id.nunique()),
        "selection_acc": float(rows.correct.mean()),
        "selection_acc_ci": ci(vals["acc"]),
        "mean_regret": float(rows.regret.mean()),
        "mean_regret_ci": ci(vals["regret"]),
        "mean_rel_regret": float(rows.relative_regret.mean()),
        "mean_rel_regret_ci": ci(vals["rel"]),
        "mean_aulc": float(rows.chosen_aulc.mean()),
        "mean_aulc_ci": ci(vals["aulc"]),
    }


def choose_single(df, policy, metric, direction):
    rows = []
    for (spc, pair_id), sub in df.groupby(["pilot_spc", "pair_id"]):
        best = sub.loc[sub.label_budget.idxmin()]
        chosen = sub.sort_values(metric, ascending=(direction == "low")).iloc[0]
        rows.append(
            {
                "pilot_spc": spc,
                "pair_id": pair_id,
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


def choose_lopo(df, policy, predictors):
    rows = []
    for (spc, pair_id), sub in df.groupby(["pilot_spc", "pair_id"]):
        train = df[(df.pilot_spc == spc) & (df.pair_id != pair_id)].copy()
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
        mu = train[predictors].mean(axis=0)
        sd = train[predictors].std(axis=0).replace(0, 1.0)
        pred = (
            LinearRegression()
            .fit(
                ((train[predictors] - mu) / sd).to_numpy(),
                train.label_budget.to_numpy(),
            )
            .predict(((test[predictors] - mu) / sd).to_numpy())
        )
        test = test.copy()
        test["pred_budget"] = pred
        chosen = test.sort_values("pred_budget").iloc[0]
        best = sub.loc[sub.label_budget.idxmin()]
        rows.append(
            {
                "pilot_spc": spc,
                "pair_id": pair_id,
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


def main() -> None:
    metrics = compute_pilot_metrics()
    truth = pd.read_csv(
        result_file(
            "backbone_selection/strict_label_budget_table.csv"
        )
    )[["pair_id", "backbone", "label_budget", "aulc"]]
    D = metrics.merge(truth, on=["pair_id", "backbone"], how="left")
    frames = [
        choose_single(D, name, metric, direction)
        for name, (metric, direction) in SINGLE_POLICIES.items()
    ]
    frames += [choose_lopo(D, name, preds) for name, preds in REG_POLICIES.items()]
    R = pd.concat(frames, ignore_index=True)
    R.to_csv(
        result_file("pilot_selection/pilot_policy_selection.csv"), index=False
    )
    summaries = []
    for (spc, policy), sub in R.groupby(["pilot_spc", "policy"]):
        summaries.append({"pilot_spc": int(spc), "policy": policy, **summarize(sub)})
    stats = {
        "protocol": {
            "dataset": "CIFAR-100 strict 40-pair experiment",
            "pilot_spc": PILOT_SPCS,
            "diagnostics": "linear heads trained on pilot labels, diagnostics on diagnostic-heldout train features",
            "target": "study strict label-budget table",
        },
        "summaries": summaries,
    }
    # Paired savings of combined + psi relative to combined non-psi.
    paired = []
    for spc in PILOT_SPCS:
        a = R[(R.pilot_spc == spc) & (R.policy == "combined non-$\\psi$")].set_index(
            "pair_id"
        )
        b = R[
            (R.pilot_spc == spc) & (R.policy == "combined non-$\\psi$ + $\\psi$")
        ].set_index("pair_id")
        idx = sorted(set(a.index) & set(b.index))
        diff = (
            a.loc[idx, "chosen_budget"].to_numpy()
            - b.loc[idx, "chosen_budget"].to_numpy()
        )
        paired.append(
            {
                "pilot_spc": spc,
                "n": len(idx),
                "psi_reduces_budget_rate": float(np.mean(diff > 0)),
                "mean_budget_reduction": float(np.mean(diff)),
                "median_budget_reduction": float(np.median(diff)),
            }
        )
    stats["paired_nonpsi_to_nonpsi_plus_psi"] = paired
    (result_file("pilot_selection/pilot_policy_stats.json")).write_text(
        json.dumps(stats, indent=2)
    )
    labels = summaries
    rows = []
    for spc in PILOT_SPCS:
        for policy in [
            "choose by $\\psi$",
            "choose by margin",
            "NLL/ECE",
            "LogME/H-score",
            "transferability set",
            "combined non-$\\psi$",
            "combined non-$\\psi$ + $\\psi$",
        ]:
            row = next(
                x for x in labels if x["pilot_spc"] == spc and x["policy"] == policy
            )
            rows.append(
                f"{spc} & {policy} & ${100 * row['selection_acc']:.1f}[{100 * row['selection_acc_ci'][0]:.1f},{100 * row['selection_acc_ci'][1]:.1f}]$ & ${row['mean_regret']:.2f}[{row['mean_regret_ci'][0]:.2f},{row['mean_regret_ci'][1]:.2f}]$ & ${100 * row['mean_rel_regret']:.1f}[{100 * row['mean_rel_regret_ci'][0]:.1f},{100 * row['mean_rel_regret_ci'][1]:.1f}]$ & ${row['mean_aulc']:.3f}$\\\\"
            )
    (result_file("pilot_selection/pilot_policy_rows.tex")).write_text(
        "\n".join(rows) + "\n"
    )
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
