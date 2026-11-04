"Measure label demand under boundary-normal scaling. See docs/EXPERIMENTS.md for usage."

from utils.paths import PROJECT_ROOT, RUNS_DIR, RESULTS_DIR, FIGURES_DIR, result_file
import json, math, os, warnings
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


def load_cache(path):
    obj = torch.load(path, map_location="cpu")
    X = obj["features"].float().numpy().astype("float32")
    y = obj["labels"].long().numpy().astype("int64")
    return X, y


def standardize_pair(Xtr, Xte):
    mu = Xtr.mean(axis=0, keepdims=True)
    sig = Xtr.std(axis=0, keepdims=True)
    sig[sig < 1e-6] = 1.0
    Xtr = (Xtr - mu) / sig
    Xte = (Xte - mu) / sig
    return Xtr.astype("float32"), Xte.astype("float32")


def fit_lr(X, y, C=1.0):
    clf = LogisticRegression(
        C=C, penalty="l2", solver="liblinear", max_iter=1000, tol=1e-5, random_state=0
    )
    clf.fit(X, y)
    return clf


def transform_normal(X, u, scale):
    proj = X @ u
    return X + ((scale - 1.0) * proj)[:, None] * u[None, :]


def ece_binary(prob, y, bins=10):
    conf = np.maximum(prob, 1.0 - prob)
    pred = (prob >= 0.5).astype(int)
    correct = (pred == y).astype(float)
    out = 0.0
    for lo, hi in zip(
        np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]
    ):
        m = (conf >= lo) & (conf < hi if hi < 1 else conf <= hi)
        if m.any():
            out += m.mean() * abs(conf[m].mean() - correct[m].mean())
    return float(out)


def psi_kernel(clf, X, y, eps):
    w = clf.coef_.reshape(-1)
    nw = float(np.linalg.norm(w) + 1e-12)
    dist = clf.decision_function(X) / nw
    p = float(np.mean(y == 1))
    beta = max(min(p, 1.0 - p), 1e-6)
    kern = np.exp(-0.5 * (dist / eps) ** 2) / (eps * math.sqrt(2.0 * math.pi))
    return float(kern.mean() / beta), dist


def interp_budget(spcs, accs, target):
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


def residual_corr(df, x, y, groups):
    d = df[[x, y] + groups].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < 8 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return None
    for col in [x, y]:
        d[col + "_r"] = d[col] - d.groupby(groups)[col].transform("mean")
    if d[x + "_r"].std() == 0 or d[y + "_r"].std() == 0:
        return None
    return dict(
        n=int(len(d)),
        pearson=float(pearsonr(d[x + "_r"], d[y + "_r"])[0]),
        spearman=float(spearmanr(d[x + "_r"], d[y + "_r"])[0]),
    )


raw_rows = []
summary_rows = []
for bench, tr_path, te_path in BENCHES:
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
        u = base_clf.coef_.reshape(-1)
        u = u / (np.linalg.norm(u) + 1e-12)
        # fixed pair-level tube width in identity metric
        _, base_dist = psi_kernel(base_clf, Xte, yte, eps=1.0)
        eps = float(max(0.05, np.quantile(np.abs(base_dist), 0.20)))
        for scale in SCALES:
            Xtr_s = transform_normal(Xtr, u, scale).astype("float32")
            Xte_s = transform_normal(Xte, u, scale).astype("float32")
            full_clf = fit_lr(Xtr_s, ytr, C=1.0)
            proba = full_clf.predict_proba(Xte_s)[:, 1]
            pred = (proba >= 0.5).astype(int)
            full_acc = float((pred == yte).mean())
            full_nll = float(
                log_loss(yte, np.vstack([1 - proba, proba]).T, labels=[0, 1])
            )
            full_ece = ece_binary(proba, yte)
            psi, dist = psi_kernel(full_clf, Xte_s, yte, eps=eps)
            margin_median = float(np.median(np.abs(dist)))
            weight_norm = float(np.linalg.norm(full_clf.coef_))
            for spc in SPCS:
                for seed in SEEDS:
                    rng = np.random.default_rng(
                        1000 * pair_id + 17 * seed + int(10 * scale)
                    )
                    idxs = []
                    for cls in [0, 1]:
                        ids = np.flatnonzero(ytr == cls)
                        ids = rng.choice(ids, size=min(spc, len(ids)), replace=False)
                        idxs.extend(ids.tolist())
                    idxs = np.array(idxs)
                    clf = fit_lr(Xtr_s[idxs], ytr[idxs], C=1.0)
                    pr = clf.predict(Xte_s)
                    acc = float((pr == yte).mean())
                    raw_rows.append(
                        dict(
                            benchmark=bench,
                            pair_id=pair_id,
                            class_a=a,
                            class_b=b,
                            normal_scale=scale,
                            spc=spc,
                            seed=seed,
                            acc=acc,
                            full_acc=full_acc,
                            psi_eff=psi,
                            log1p_psi_eff=math.log1p(psi),
                            full_nll=full_nll,
                            full_ece=full_ece,
                            full_margin_median=margin_median,
                            full_weight_norm=weight_norm,
                            eps=eps,
                        )
                    )
            summary_rows.append(
                dict(
                    benchmark=bench,
                    pair_id=pair_id,
                    class_a=a,
                    class_b=b,
                    normal_scale=scale,
                    full_acc=full_acc,
                    psi_eff=psi,
                    log1p_psi_eff=math.log1p(psi),
                    predicted_relative_psi=1.0 / scale,
                    full_nll=full_nll,
                    full_ece=full_ece,
                    full_margin_median=margin_median,
                    full_weight_norm=weight_norm,
                    eps=eps,
                )
            )
        print("done", bench, pair_id, flush=True)

raw = pd.DataFrame(raw_rows)
summ = pd.DataFrame(summary_rows)
raw.to_csv(
    result_file("normal_tangent/controlled_real_feature_learning_curves.csv"),
    index=False,
)
summ.to_csv(
    result_file("normal_tangent/controlled_real_feature_transform_summary.csv"),
    index=False,
)

# Mean learning curve per transformation.
lc = (
    raw.groupby(["benchmark", "pair_id", "normal_scale", "spc"])
    .agg(acc=("acc", "mean"), acc_std=("acc", "std"))
    .reset_index()
)
# Per pair target based on the best full-data transformed classifier; use 90% to keep label budget observable.
targets = (
    summ.groupby(["benchmark", "pair_id"])["full_acc"]
    .max()
    .mul(0.90)
    .rename("target_acc")
    .reset_index()
)
records = []
for (bench, pair, scale), sub in lc.groupby(["benchmark", "pair_id", "normal_scale"]):
    sub = sub.sort_values("spc")
    srow = (
        summ[
            (summ.benchmark == bench)
            & (summ.pair_id == pair)
            & (summ.normal_scale == scale)
        ]
        .iloc[0]
        .to_dict()
    )
    x = np.log(sub["spc"].values.astype(float))
    y = sub["acc"].values.astype(float)
    target = float(
        targets[(targets.benchmark == bench) & (targets.pair_id == pair)][
            "target_acc"
        ].iloc[0]
    )
    budget, reached = interp_budget(sub["spc"].values, y, target)
    rec = dict(
        benchmark=bench,
        pair_id=int(pair),
        normal_scale=float(scale),
        early_acc_5spc=float(sub[sub.spc == 5]["acc"].mean())
        if (sub.spc == 5).any()
        else np.nan,
        aulc=float(np.trapz(y, x) / (x[-1] - x[0])),
        slope_acc_logspc=float(np.polyfit(x, y, 1)[0]),
        label_budget_to_90pct_full=float(budget),
        target_reached=bool(reached),
        target_acc=target,
    )
    for k in [
        "psi_eff",
        "log1p_psi_eff",
        "predicted_relative_psi",
        "full_acc",
        "full_nll",
        "full_ece",
        "full_margin_median",
        "full_weight_norm",
    ]:
        rec[k] = float(srow[k])
    records.append(rec)
D = pd.DataFrame(records)
D.to_csv(
    result_file("normal_tangent/controlled_real_feature_sample_difficulty.csv"),
    index=False,
)

metrics = [
    "log1p_psi_eff",
    "psi_eff",
    "predicted_relative_psi",
    "full_margin_median",
    "full_weight_norm",
    "full_nll",
    "full_ece",
    "full_acc",
]
target_cols = [
    "early_acc_5spc",
    "aulc",
    "slope_acc_logspc",
    "label_budget_to_90pct_full",
]
summary = {}
for target in target_cols:
    summary[target] = {}
    for metric in metrics:
        res = residual_corr(D, metric, target, ["benchmark", "pair_id"])
        if res:
            summary[target][metric] = res
with open(
    result_file(
        "normal_tangent/controlled_real_feature_sample_difficulty_summary.json"
    ),
    "w",
) as f:
    json.dump(summary, f, indent=2)

# compact table rows: use signs so higher is better for target except label budget.
rows = []
metric_labels = {
    "log1p_psi_eff": r"$\log(1+\hat\psi_{\rm eff})$",
    "predicted_relative_psi": "predicted $1/s$",
    "full_margin_median": "median margin",
    "full_nll": "NLL",
    "full_ece": "ECE",
    "full_acc": "full-shot acc.",
}
for metric in [
    "log1p_psi_eff",
    "predicted_relative_psi",
    "full_margin_median",
    "full_nll",
    "full_ece",
    "full_acc",
]:
    vals = []
    for target in ["early_acc_5spc", "aulc", "label_budget_to_90pct_full"]:
        res = summary.get(target, {}).get(metric)
        vals.append(
            "--" if not res else f"{res['pearson']:.3f} / {res['spearman']:.3f}"
        )
    rows.append(f"{metric_labels[metric]} & {vals[0]} & {vals[1]} & {vals[2]} \\")
Path(
    result_file(
        "normal_tangent/controlled_real_feature_sample_difficulty_rows.tex"
    )
).write_text("\n".join(rows) + "\n")

# transformation table: average psi and low-shot acc by scale.
scale_tbl = (
    D.groupby("normal_scale")
    .agg(
        logpsi=("log1p_psi_eff", "mean"),
        psi=("psi_eff", "mean"),
        early=("early_acc_5spc", "mean"),
        aulc=("aulc", "mean"),
        budget=("label_budget_to_90pct_full", "mean"),
    )
    .reset_index()
)
scale_rows = []
for _, r in scale_tbl.iterrows():
    scale_rows.append(
        f"{r['normal_scale']:.1f} & {r['logpsi']:.3f} & {r['early'] * 100:.2f} & {r['aulc'] * 100:.2f} & {r['budget']:.1f} \\"
    )
Path(
    result_file("normal_tangent/controlled_real_feature_scale_rows.tex")
).write_text("\n".join(scale_rows) + "\n")
with open(
    result_file("normal_tangent/controlled_real_feature_scale_summary.json"),
    "w",
) as f:
    json.dump(scale_tbl.to_dict(orient="records"), f, indent=2)

# Figure
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
for bench, bdf in raw.groupby("benchmark"):
    mean = bdf.groupby(["normal_scale", "spc"])["acc"].mean().reset_index()
    for scale, sdf in mean.groupby("normal_scale"):
        axes[0].plot(
            sdf["spc"],
            sdf["acc"],
            marker="o",
            label=f"{bench} s={scale:g}" if bench.endswith("R50") else None,
            alpha=0.75,
        )
axes[0].set_xscale("log")
axes[0].set_xlabel("samples per class")
axes[0].set_ylabel("binary test accuracy")
axes[0].set_title("learning curves")
axes[0].grid(True, alpha=0.25)
axes[1].scatter(
    D["log1p_psi_eff"], D["aulc"], c=D["normal_scale"], cmap="viridis", s=28
)
axes[1].set_xlabel("log(1+psi_eff)")
axes[1].set_ylabel("AULC")
axes[1].set_title("sample efficiency vs boundary mass")
axes[1].grid(True, alpha=0.25)
axes[2].scatter(
    D["log1p_psi_eff"],
    D["label_budget_to_90pct_full"],
    c=D["normal_scale"],
    cmap="viridis",
    s=28,
)
axes[2].set_xlabel("log(1+psi_eff)")
axes[2].set_ylabel("label budget to target")
axes[2].set_title("label budget")
axes[2].grid(True, alpha=0.25)
fig.tight_layout()
fig.savefig(FIG / "controlled_real_feature_sample_difficulty.png", dpi=220)

print("wrote controlled real-feature artifacts")
print(json.dumps(summary, indent=2)[:4000])
