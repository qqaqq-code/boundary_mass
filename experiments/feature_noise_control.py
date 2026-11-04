"Evaluate feature-noise controls for boundary-mass interventions. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, RUNS_DIR, RESULTS_DIR, FIGURES_DIR, result_file
import json, math, warnings, time
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
BACKBONES = {
    "R50": RUNS_DIR / "cifar100_resnet50_pretrained/feature_cache_resnet50_imagenetv2",
    "ViT-B": RUNS_DIR / "cifar100_vitb16_targeted/feature_cache_vitb16_imagenetv1",
}
N_PAIRS = 40
RNG = np.random.default_rng(20260429)
ALPHAS = [0.05, 0.10, 0.20]
REPEATS = 5


def load_cache(d: Path):
    tr = torch.load(d / "train_features.pt", map_location="cpu")
    te = torch.load(d / "test_features.pt", map_location="cpu")
    return (
        tr["features"].float().numpy().astype("float32"),
        tr["labels"].numpy().astype("int64"),
        te["features"].float().numpy().astype("float32"),
        te["labels"].numpy().astype("int64"),
    )


def standardize(Xtr, Xte):
    mu = Xtr.mean(0, keepdims=True)
    sd = Xtr.std(0, keepdims=True)
    sd[sd < 1e-6] = 1.0
    return ((Xtr - mu) / sd).astype("float32"), ((Xte - mu) / sd).astype("float32")


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
    edges = np.linspace(0, 1, bins + 1)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf >= lo) & (conf < (hi if hi < 1 else hi + 1e-8))
        if m.any():
            out += m.mean() * abs(conf[m].mean() - corr[m].mean())
    return float(out)


def effective_rank(X):
    X = X - X.mean(0, keepdims=True)
    s = np.linalg.svd(X, full_matrices=False, compute_uv=False)
    eig = s * s / max(1, (X.shape[0] - 1))
    sm = eig.sum()
    return float(sm * sm / (np.square(eig).sum() + 1e-12))


def psi_kernel(clf, X, y):
    w = clf.coef_.reshape(-1)
    nw = np.linalg.norm(w) + 1e-12
    dist = clf.decision_function(X) / nw
    eps = max(0.05, float(np.quantile(np.abs(dist), 0.20)))
    beta = max(min(float(np.mean(y == 1)), float(np.mean(y == 0))), 1e-6)
    kern = np.exp(-0.5 * (dist / eps) ** 2) / (eps * math.sqrt(2 * math.pi))
    return float(kern.mean() / beta), dist, eps


def gsa_proxy(clf):
    w = clf.coef_.reshape(-1)
    nw = np.linalg.norm(w) + 1e-12
    b = float(clf.intercept_[0])
    t = b / nw
    return float(2.0 * math.exp(-0.5 * t * t) / math.sqrt(2 * math.pi))


def noise_drop(clf, X, y, alpha):
    base_acc = float(clf.score(X, y))
    rng = np.random.default_rng(991 + int(alpha * 1000))
    accs = []
    flips = []
    base = clf.predict(X)
    scale = np.std(X, axis=0, keepdims=True)
    scale[scale < 1e-6] = 1.0
    for r in range(REPEATS):
        Xn = X + rng.normal(0, alpha, size=X.shape).astype("float32") * scale
        pred = clf.predict(Xn)
        accs.append(float(np.mean(pred == y)))
        flips.append(float(np.mean(pred != base)))
    return (
        base_acc,
        float(np.mean(accs)),
        float(base_acc - np.mean(accs)),
        float(np.mean(flips)),
    )


def corr_resid(df, x, y, groups):
    d = df[[x, y] + groups].replace([np.inf, -np.inf], np.nan).dropna().copy()
    if len(d) < 8 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return None
    for c in [x, y]:
        d[c + "_r"] = d[c] - d.groupby(groups)[c].transform("mean")
    if d[x + "_r"].std() < 1e-12 or d[y + "_r"].std() < 1e-12:
        return None
    return {
        "n": int(len(d)),
        "pearson": float(pearsonr(d[x + "_r"], d[y + "_r"])[0]),
        "spearman": float(spearmanr(d[x + "_r"], d[y + "_r"])[0]),
    }


pairs = []
for a in range(0, 40, 2):
    pairs.append((a, a + 1))
seen = set(tuple(sorted(p)) for p in pairs)
while len(pairs) < N_PAIRS:
    a, b = RNG.choice(100, 2, replace=False)
    p = tuple(sorted((int(a), int(b))))
    if p not in seen:
        seen.add(p)
        pairs.append(p)

caches = {k: load_cache(v) for k, v in BACKBONES.items()}
rows = []
t0 = time.time()
for pi, (a, b) in enumerate(pairs):
    for name, (Xtr_all, ytr_all, Xte_all, yte_all) in caches.items():
        mtr = (ytr_all == a) | (ytr_all == b)
        mte = (yte_all == a) | (yte_all == b)
        Xtr = Xtr_all[mtr]
        ytr = (ytr_all[mtr] == b).astype(int)
        Xte = Xte_all[mte]
        yte = (yte_all[mte] == b).astype(int)
        Xtr, Xte = standardize(Xtr, Xte)
        clf = fit_lr(Xtr, ytr)
        prob = clf.predict_proba(Xte)[:, 1]
        psi, dist, eps = psi_kernel(clf, Xtr, ytr)
        row = {
            "pair_id": pi,
            "class_a": a,
            "class_b": b,
            "backbone": name,
            "psi_eff": psi,
            "log1p_psi": math.log1p(psi),
            "median_margin": float(np.median(np.abs(dist))),
            "mean_margin": float(np.mean(np.abs(dist))),
            "weight_norm": float(np.linalg.norm(clf.coef_)),
            "gsa_proxy": gsa_proxy(clf),
            "full_acc": float(clf.score(Xte, yte)),
            "full_nll": float(
                log_loss(yte, np.vstack([1 - prob, prob]).T, labels=[0, 1])
            ),
            "full_ece": ece_binary(prob, yte),
            "effective_rank": effective_rank(Xtr),
        }
        for alpha in ALPHAS:
            base, ra, drop, flip = noise_drop(clf, Xte, yte, alpha)
            tag = f"{int(alpha * 100):02d}"
            row[f"robust_acc_{tag}"] = ra
            row[f"acc_drop_{tag}"] = drop
            row[f"noise_flip_{tag}"] = flip
        rows.append(row)
    if (pi + 1) % 10 == 0:
        print("done pairs", pi + 1, "elapsed", round(time.time() - t0, 1), flush=True)

D = pd.DataFrame(rows)
D.to_csv(
    result_file("feature_noise/feature_brittleness_metrics.csv"), index=False
)
metrics = [
    "log1p_psi",
    "psi_eff",
    "gsa_proxy",
    "noise_flip_10",
    "median_margin",
    "mean_margin",
    "weight_norm",
    "full_nll",
    "full_ece",
    "full_acc",
    "effective_rank",
]
targets = [
    "acc_drop_05",
    "acc_drop_10",
    "acc_drop_20",
    "robust_acc_10",
    "robust_acc_20",
]
summary = []
for tgt in targets:
    for m in metrics:
        if m in D.columns:
            c = corr_resid(D, m, tgt, ["backbone"])
            if c:
                summary.append({"target": tgt, "metric": m, **c})
S = pd.DataFrame(summary)
S.to_csv(
    result_file("feature_noise/feature_brittleness_correlations.csv"),
    index=False,
)
# Pairwise model-selection under noise: choose lower predicted brittleness / higher robust accuracy.
sel = []
for pi, sub in D.groupby("pair_id"):
    for tag in ["10", "20"]:
        best = sub.loc[sub[f"robust_acc_{tag}"].idxmax()]
        for metric in metrics:
            if metric not in sub.columns:
                continue
            ascending = metric not in [
                "median_margin",
                "mean_margin",
                "full_acc",
                "effective_rank",
            ]
            chosen = sub.sort_values(metric, ascending=ascending).iloc[0]
            regret = float(best[f"robust_acc_{tag}"] - chosen[f"robust_acc_{tag}"])
            sel.append(
                {
                    "pair_id": pi,
                    "target": f"robust_acc_{tag}",
                    "metric": metric,
                    "chosen_backbone": chosen.backbone,
                    "best_backbone": best.backbone,
                    "correct": chosen.backbone == best.backbone,
                    "regret_acc": regret,
                }
            )
SEL = pd.DataFrame(sel)
SEL.to_csv(
    result_file("feature_noise/feature_brittleness_selection.csv"), index=False
)
SEL_SUM = (
    SEL.groupby(["target", "metric"])
    .agg(
        n=("pair_id", "count"),
        selection_acc=("correct", "mean"),
        mean_regret=("regret_acc", "mean"),
        median_regret=("regret_acc", "median"),
    )
    .reset_index()
)
SEL_SUM.to_csv(
    result_file("feature_noise/feature_brittleness_selection_summary.csv"),
    index=False,
)
with open(
    result_file("feature_noise/feature_brittleness_summary.json"), "w"
) as f:
    json.dump(
        {
            "correlations": summary,
            "selection": SEL_SUM.to_dict(orient="records"),
            "alphas": ALPHAS,
            "n_pairs": N_PAIRS,
        },
        f,
        indent=2,
    )
# TeX rows for alpha=.10 drop correlations and robust selection.
label = {
    "log1p_psi": "$\\log(1+\\hat\\psi_{\\rm eff})$",
    "psi_eff": "$\\hat\\psi_{\\rm eff}$",
    "gsa_proxy": "Gaussian-area proxy",
    "noise_flip_10": "noise flip proxy",
    "median_margin": "median margin",
    "mean_margin": "mean margin",
    "weight_norm": "weight norm",
    "full_nll": "NLL",
    "full_ece": "ECE",
    "full_acc": "clean acc.",
    "effective_rank": "effective rank",
}
rows_tex = []
subcorr = S[S.target == "acc_drop_10"].set_index("metric")
subsel = SEL_SUM[SEL_SUM.target == "robust_acc_10"].set_index("metric")
for m in [
    "log1p_psi",
    "psi_eff",
    "gsa_proxy",
    "noise_flip_10",
    "median_margin",
    "full_nll",
    "full_ece",
    "full_acc",
    "effective_rank",
]:
    if m not in subcorr.index or m not in subsel.index:
        continue
    c = subcorr.loc[m]
    s = subsel.loc[m]
    rows_tex.append(
        f"{label[m]} & {c.pearson:.3f}/{c.spearman:.3f} & {100 * s.selection_acc:.1f} & {100 * s.mean_regret:.2f}"
        + r"\% \\"
    )
(result_file("feature_noise/feature_brittleness_rows.tex")).write_text(
    "\n".join(rows_tex) + "\n"
)
# Plot residual scatter for psi vs alpha=.10 drop.
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = D.copy()
for col in ["log1p_psi", "acc_drop_10"]:
    P[col + "_r"] = P[col] - P.groupby("backbone")[col].transform("mean")
fig, ax = plt.subplots(figsize=(4.6, 3.4))
for bb, sub in P.groupby("backbone"):
    ax.scatter(sub["log1p_psi_r"], sub["acc_drop_10_r"], s=22, alpha=0.75, label=bb)
coef = np.polyfit(P["log1p_psi_r"], P["acc_drop_10_r"], 1)
xs = np.linspace(P["log1p_psi_r"].min(), P["log1p_psi_r"].max(), 100)
ax.plot(xs, coef[0] * xs + coef[1], color="black", lw=1.4)
ax.axhline(0, color="0.85", lw=0.8)
ax.axvline(0, color="0.85", lw=0.8)
ax.set_xlabel("residual log(1+$\\hat\\psi_{\\rm eff}$)")
ax.set_ylabel("residual acc. drop under noise")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout()
fig.savefig(FIG / "feature_brittleness_psi_drop.pdf")
fig.savefig(FIG / "feature_brittleness_psi_drop.png", dpi=220)
print(
    json.dumps(
        {
            "rows": len(D),
            "summary_csv": str(
                result_file(
                    "feature_noise/feature_brittleness_correlations.csv"
                )
            ),
        },
        indent=2,
    )
)
