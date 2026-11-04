"Measure label demand under boundary-tangent transformations. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, RUNS_DIR, RESULTS_DIR, FIGURES_DIR, result_file

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

warnings.filterwarnings("ignore")

ROOT = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
ART = RESULTS_DIR
FIG = FIGURES_DIR
ART.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)

BENCHES = [
    (
        "CIFAR-100/R50",
        RUNS_DIR
        / "cifar100_resnet50_pretrained/feature_cache_resnet50_imagenetv2/train_features.pt",
        RUNS_DIR
        / "cifar100_resnet50_pretrained/feature_cache_resnet50_imagenetv2/test_features.pt",
    ),
    (
        "CIFAR-100/ViT-B",
        RUNS_DIR
        / "cifar100_vitb16_targeted/feature_cache_vitb16_imagenetv1/train_features.pt",
        RUNS_DIR
        / "cifar100_vitb16_targeted/feature_cache_vitb16_imagenetv1/test_features.pt",
    ),
]
PAIRS = [(0, 1), (2, 3), (4, 5), (6, 7), (8, 9), (10, 11), (12, 13), (14, 15)]
SCALES = [0.5, 1.0, 2.0, 4.0]
SPCS = [2, 5, 10, 20, 50, 100]
SEEDS = [0, 1, 2]
N_BOOT = 2000
RNG = np.random.default_rng(20260501)


def load_cache(path: Path) -> tuple[np.ndarray, np.ndarray]:
    obj = torch.load(path, map_location="cpu")
    X = obj["features"].float().numpy().astype("float32")
    y = obj["labels"].long().numpy().astype("int64")
    return X, y


def standardize_pair(Xtr: np.ndarray, Xte: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mu = Xtr.mean(axis=0, keepdims=True)
    sig = Xtr.std(axis=0, keepdims=True)
    sig[sig < 1e-6] = 1.0
    return ((Xtr - mu) / sig).astype("float32"), ((Xte - mu) / sig).astype("float32")


def fit_lr(X: np.ndarray, y: np.ndarray, C: float = 1.0) -> LogisticRegression:
    clf = LogisticRegression(
        C=C, penalty="l2", solver="liblinear", max_iter=1000, tol=1e-5, random_state=0
    )
    clf.fit(X, y)
    return clf


def transform_axis(X: np.ndarray, u: np.ndarray, scale: float) -> np.ndarray:
    proj = X @ u
    return (X + ((scale - 1.0) * proj)[:, None] * u[None, :]).astype("float32")


def tangent_axis(X: np.ndarray, y: np.ndarray, u: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=u.shape).astype("float64")
    bases = [u.astype("float64")]
    mean_diff = (X[y == 1].mean(axis=0) - X[y == 0].mean(axis=0)).astype("float64")
    mean_diff = mean_diff - float(mean_diff @ bases[0]) * bases[0]
    nd = float(np.linalg.norm(mean_diff))
    if nd > 1e-8:
        bases.append(mean_diff / nd)
    for b in bases:
        v = v - float(v @ b) * b
    nv = float(np.linalg.norm(v))
    if nv < 1e-10:
        v = rng.normal(size=u.shape).astype("float64")
        v = v - float(v @ bases[0]) * bases[0]
        nv = float(np.linalg.norm(v))
    return (v / (nv + 1e-12)).astype("float32")


def ece_binary(prob: np.ndarray, y: np.ndarray, bins: int = 10) -> float:
    conf = np.maximum(prob, 1.0 - prob)
    pred = (prob >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    out = 0.0
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < (hi if hi < 1 else hi + 1e-8))
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def psi_kernel(
    clf: LogisticRegression, X: np.ndarray, y: np.ndarray, eps: float
) -> tuple[float, np.ndarray]:
    w = clf.coef_.reshape(-1)
    nw = float(np.linalg.norm(w) + 1e-12)
    dist = clf.decision_function(X) / nw
    p = float(np.mean(y == 1))
    beta = max(min(p, 1.0 - p), 1e-6)
    kern = np.exp(-0.5 * (dist / eps) ** 2) / (eps * math.sqrt(2.0 * math.pi))
    return float(kern.mean() / beta), dist


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
            if abs(y1 - y0) < 1e-8:
                return float(spcs[i]), True
            t = (target - y0) / (y1 - y0)
            return float(np.exp(x0 + t * (x1 - x0))), True
    return float(spcs.max() * 1.5), False


def residual_corr(
    df: pd.DataFrame, x: str, y: str, groups: list[str]
) -> dict[str, float] | None:
    d = df[[x, y] + groups].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < 8 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return None
    for col in [x, y]:
        d[col + "_r"] = d[col] - d.groupby(groups)[col].transform("mean")
    if d[x + "_r"].std() == 0 or d[y + "_r"].std() == 0:
        return None
    return {
        "n": int(len(d)),
        "pearson": float(pearsonr(d[x + "_r"], d[y + "_r"])[0]),
        "spearman": float(spearmanr(d[x + "_r"], d[y + "_r"])[0]),
    }


def ci(vals: list[float] | np.ndarray) -> tuple[float, float]:
    arr = np.asarray(vals, dtype=float)
    arr = arr[np.isfinite(arr)]
    return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))


def effect_summary(D: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    pairs = []
    for (bench, pair_id, direction), sub in D.groupby(
        ["benchmark", "pair_id", "direction"]
    ):
        one = sub[sub.scale == 1.0].iloc[0]
        two = sub[sub.scale == 2.0].iloc[0]
        rec = {"benchmark": bench, "pair_id": pair_id, "direction": direction}
        for metric in [
            "log1p_psi_eff",
            "early_acc_5spc",
            "aulc",
            "label_budget_to_90pct_full",
            "full_acc",
        ]:
            rec["delta_" + metric] = float(two[metric] - one[metric])
        pairs.append(rec)
    E = pd.DataFrame(pairs)

    rows = []
    for direction, sub in E.groupby("direction"):
        row = {"direction": direction, "n": int(len(sub))}
        for metric in [
            "delta_log1p_psi_eff",
            "delta_early_acc_5spc",
            "delta_aulc",
            "delta_label_budget_to_90pct_full",
            "delta_full_acc",
        ]:
            vals = sub[metric].to_numpy()
            boot = []
            for _ in range(N_BOOT):
                idx = RNG.integers(0, len(vals), size=len(vals))
                boot.append(float(vals[idx].mean()))
            lo, hi = ci(boot)
            row[metric + "_mean"] = float(vals.mean())
            row[metric + "_ci_low"] = lo
            row[metric + "_ci_high"] = hi
        rows.append(row)

    comparison = {}
    merged = E.pivot_table(
        index=["benchmark", "pair_id"],
        columns="direction",
        values=[
            "delta_log1p_psi_eff",
            "delta_early_acc_5spc",
            "delta_aulc",
            "delta_label_budget_to_90pct_full",
        ],
    )
    for metric in [
        "delta_log1p_psi_eff",
        "delta_early_acc_5spc",
        "delta_aulc",
        "delta_label_budget_to_90pct_full",
    ]:
        normal = merged[(metric, "normal")].to_numpy()
        tangent = merged[(metric, "tangent")].to_numpy()
        stat = np.abs(normal) - np.abs(tangent)
        obs = float(stat.mean())
        null = []
        for _ in range(N_BOOT):
            signs = RNG.choice([-1.0, 1.0], size=len(stat))
            null.append(float((signs * stat).mean()))
        p = float((1.0 + sum(v >= obs for v in null)) / (len(null) + 1.0))
        comparison[metric] = {
            "mean_abs_normal": float(np.abs(normal).mean()),
            "mean_abs_tangent": float(np.abs(tangent).mean()),
            "normal_tangent_abs_ratio": float(
                np.abs(normal).mean() / (np.abs(tangent).mean() + 1e-12)
            ),
            "paired_signflip_p_abs_normal_gt_tangent": p,
        }
    return pd.DataFrame(rows), {
        "paired_effects": E.to_dict(orient="records"),
        "normal_vs_tangent": comparison,
    }


def main() -> None:
    raw_rows = []
    full_rows = []
    for bench_idx, (bench, tr_path, te_path) in enumerate(BENCHES):
        print("loading", bench, flush=True)
        Xtr_all, ytr_all = load_cache(tr_path)
        Xte_all, yte_all = load_cache(te_path)
        for pair_id, (a, b) in enumerate(PAIRS):
            mtr = (ytr_all == a) | (ytr_all == b)
            mte = (yte_all == a) | (yte_all == b)
            if mtr.sum() < 20 or mte.sum() < 20:
                continue
            Xtr = Xtr_all[mtr].copy()
            ytr = (ytr_all[mtr] == b).astype(int)
            Xte = Xte_all[mte].copy()
            yte = (yte_all[mte] == b).astype(int)
            Xtr, Xte = standardize_pair(Xtr, Xte)
            base_clf = fit_lr(Xtr, ytr, C=1.0)
            u = base_clf.coef_.reshape(-1).astype("float32")
            u = u / (np.linalg.norm(u) + 1e-12)
            v = tangent_axis(Xtr, ytr, u, seed=20260501 + 101 * bench_idx + pair_id)
            _, base_dist = psi_kernel(base_clf, Xte, yte, eps=1.0)
            eps = float(max(0.05, np.quantile(np.abs(base_dist), 0.20)))

            for direction, axis in [("normal", u), ("tangent", v)]:
                for scale in SCALES:
                    Xtr_s = transform_axis(Xtr, axis, scale)
                    Xte_s = transform_axis(Xte, axis, scale)
                    full_clf = fit_lr(Xtr_s, ytr, C=1.0)
                    proba = full_clf.predict_proba(Xte_s)[:, 1]
                    pred = (proba >= 0.5).astype(int)
                    full_acc = float((pred == yte).mean())
                    full_nll = float(
                        log_loss(yte, np.vstack([1 - proba, proba]).T, labels=[0, 1])
                    )
                    full_ece = ece_binary(proba, yte)
                    psi, dist = psi_kernel(full_clf, Xte_s, yte, eps=eps)
                    full = {
                        "benchmark": bench,
                        "pair_id": pair_id,
                        "class_a": a,
                        "class_b": b,
                        "direction": direction,
                        "scale": scale,
                        "full_acc": full_acc,
                        "psi_eff": psi,
                        "log1p_psi_eff": math.log1p(psi),
                        "predicted_relative_psi": (
                            1.0 / scale if direction == "normal" else 1.0
                        ),
                        "full_nll": full_nll,
                        "full_ece": full_ece,
                        "full_margin_median": float(np.median(np.abs(dist))),
                        "full_weight_norm": float(np.linalg.norm(full_clf.coef_)),
                        "eps": eps,
                    }
                    full_rows.append(full)
                    for spc in SPCS:
                        for seed in SEEDS:
                            rng = np.random.default_rng(
                                1000 * pair_id
                                + 100 * bench_idx
                                + 19 * seed
                                + int(10 * scale)
                            )
                            idxs = []
                            for cls in [0, 1]:
                                ids = np.flatnonzero(ytr == cls)
                                ids = rng.choice(
                                    ids, size=min(spc, len(ids)), replace=False
                                )
                                idxs.extend(ids.tolist())
                            idxs = np.array(idxs)
                            clf = fit_lr(Xtr_s[idxs], ytr[idxs], C=1.0)
                            acc = float((clf.predict(Xte_s) == yte).mean())
                            raw_rows.append(
                                {**full, "spc": spc, "seed": seed, "acc": acc}
                            )
            print("done", bench, pair_id, flush=True)

    raw = pd.DataFrame(raw_rows)
    full = pd.DataFrame(full_rows)
    raw.to_csv(
        result_file("normal_tangent/controlled_direction_learning_curves.csv"),
        index=False,
    )
    full.to_csv(
        result_file(
            "normal_tangent/controlled_direction_transform_summary.csv"
        ),
        index=False,
    )

    lc = (
        raw.groupby(["benchmark", "pair_id", "direction", "scale", "spc"])
        .agg(acc=("acc", "mean"), acc_std=("acc", "std"))
        .reset_index()
    )
    targets = (
        full.groupby(["benchmark", "pair_id"])["full_acc"]
        .max()
        .mul(0.90)
        .rename("target_acc")
        .reset_index()
    )
    records = []
    for (bench, pair_id, direction, scale), sub in lc.groupby(
        ["benchmark", "pair_id", "direction", "scale"]
    ):
        sub = sub.sort_values("spc")
        frow = (
            full[
                (full.benchmark == bench)
                & (full.pair_id == pair_id)
                & (full.direction == direction)
                & (full.scale == scale)
            ]
            .iloc[0]
            .to_dict()
        )
        x = np.log(sub.spc.to_numpy(dtype=float))
        y = sub.acc.to_numpy(dtype=float)
        target = float(
            targets[(targets.benchmark == bench) & (targets.pair_id == pair_id)][
                "target_acc"
            ].iloc[0]
        )
        budget, reached = interp_budget(sub.spc.to_numpy(dtype=float), y, target)
        rec = {
            "benchmark": bench,
            "pair_id": int(pair_id),
            "direction": direction,
            "scale": float(scale),
            "early_acc_5spc": float(sub[sub.spc == 5]["acc"].mean()),
            "aulc": float(np.trapz(y, x) / (x[-1] - x[0])),
            "slope_acc_logspc": float(np.polyfit(x, y, 1)[0]),
            "label_budget_to_90pct_full": float(budget),
            "target_reached": bool(reached),
            "target_acc": target,
        }
        for key in [
            "psi_eff",
            "log1p_psi_eff",
            "predicted_relative_psi",
            "full_acc",
            "full_nll",
            "full_ece",
            "full_margin_median",
            "full_weight_norm",
        ]:
            rec[key] = float(frow[key])
        records.append(rec)
    D = pd.DataFrame(records)
    D.to_csv(
        result_file(
            "normal_tangent/controlled_direction_sample_difficulty.csv"
        ),
        index=False,
    )

    corr_targets = ["early_acc_5spc", "aulc", "label_budget_to_90pct_full"]
    predictors = [
        "log1p_psi_eff",
        "predicted_relative_psi",
        "full_margin_median",
        "full_nll",
        "full_ece",
        "full_acc",
    ]
    correlations = {}
    for direction in ["normal", "tangent"]:
        correlations[direction] = {}
        sub = D[D.direction == direction]
        for target in corr_targets:
            correlations[direction][target] = {}
            for pred in predictors:
                correlations[direction][target][pred] = residual_corr(
                    sub, pred, target, ["benchmark", "pair_id"]
                )

    effects, effect_json = effect_summary(D)
    effects.to_csv(
        result_file("normal_tangent/controlled_direction_effect_summary.csv"),
        index=False,
    )

    # Mean-by-scale rows for appendix.
    scale_rows = []
    mean_scale = (
        D.groupby(["direction", "scale"])
        .agg(
            logpsi=("log1p_psi_eff", "mean"),
            acc5=("early_acc_5spc", "mean"),
            aulc=("aulc", "mean"),
            budget=("label_budget_to_90pct_full", "mean"),
        )
        .reset_index()
    )
    for _, r in mean_scale.iterrows():
        scale_rows.append(
            f"{r.direction} & {r.scale:.1f} & {r.logpsi:.3f} & {100 * r.acc5:.2f} & {100 * r.aulc:.2f} & {r.budget:.1f} \\\\"
        )
    (
        result_file("normal_tangent/controlled_direction_scale_rows.tex")
    ).write_text("\n".join(scale_rows) + "\n")

    effect_rows = []
    for _, r in effects.sort_values("direction").iterrows():
        effect_rows.append(
            f"{r.direction} & "
            f"{r.delta_log1p_psi_eff_mean:.3f} [{r.delta_log1p_psi_eff_ci_low:.3f},{r.delta_log1p_psi_eff_ci_high:.3f}] & "
            f"{100 * r.delta_early_acc_5spc_mean:.2f} [{100 * r.delta_early_acc_5spc_ci_low:.2f},{100 * r.delta_early_acc_5spc_ci_high:.2f}] & "
            f"{100 * r.delta_aulc_mean:.2f} [{100 * r.delta_aulc_ci_low:.2f},{100 * r.delta_aulc_ci_high:.2f}] & "
            f"{r.delta_label_budget_to_90pct_full_mean:.2f} [{r.delta_label_budget_to_90pct_full_ci_low:.2f},{r.delta_label_budget_to_90pct_full_ci_high:.2f}] \\\\"
        )
    (
        result_file("normal_tangent/controlled_direction_effect_rows.tex")
    ).write_text("\n".join(effect_rows) + "\n")

    summary = {
        "n_conditions": int(len(D)),
        "n_learning_rows": int(len(raw)),
        "correlations": correlations,
        "effect_summary": effects.to_dict(orient="records"),
        **effect_json,
    }
    (
        result_file("normal_tangent/controlled_direction_summary.json")
    ).write_text(json.dumps(summary, indent=2))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axs = plt.subplots(1, 2, figsize=(8.8, 3.2))
        for direction, sub in D.groupby("direction"):
            m = (
                sub.groupby("scale")
                .agg(
                    logpsi=("log1p_psi_eff", "mean"),
                    budget=("label_budget_to_90pct_full", "mean"),
                )
                .reset_index()
            )
            axs[0].plot(m.scale, m.logpsi, marker="o", label=direction)
            axs[1].plot(m.scale, m.budget, marker="o", label=direction)
        for ax in axs:
            ax.set_xscale("log", base=2)
            ax.set_xlabel("scale")
            ax.grid(alpha=0.25)
            ax.legend(frameon=False)
        axs[0].set_ylabel("mean log(1+psi)")
        axs[1].set_ylabel("mean label budget")
        fig.tight_layout()
        fig.savefig(FIG / "controlled_direction_controls.png", dpi=220)
        fig.savefig(FIG / "controlled_direction_controls.pdf")
    except Exception as exc:
        summary["plot_error"] = repr(exc)
        (
            result_file("normal_tangent/controlled_direction_summary.json")
        ).write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
