"Measure estimator sensitivity and computational cost. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, RUNS_DIR, RESULTS_DIR, result_file
import csv, json, math, random
from pathlib import Path
from collections import defaultdict
import torch

PAPER = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
ART = RESULTS_DIR
ART.mkdir(parents=True, exist_ok=True)
OLD_PREFIX = "."
RUN_SPECS = [
    (
        "CIFAR-100/R50",
        RUNS_DIR / "cifar100_resnet50_pretrained",
        "feature_cache_resnet50_imagenetv2",
    ),
    (
        "CIFAR-100/ViT-B",
        RUNS_DIR / "cifar100_vitb16_targeted",
        "feature_cache_vitb16_imagenetv1",
    ),
    (
        "Tiny-IN/R50",
        RUNS_DIR / "tiny_imagenet_resnet50_targeted",
        "feature_cache_resnet50_imagenetv2",
    ),
]


def fl(x, d=float("nan")):
    try:
        return float(x)
    except:
        return d


def it(x, d=0):
    try:
        return int(float(x))
    except:
        return d


def ranks(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        r = (i + j - 1) / 2 + 1
        for k in range(i, j):
            out[order[k]] = r
        i = j
    return out


def pearson(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    mx = sum(x for x, y in pairs) / len(pairs)
    my = sum(y for x, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, y in pairs)
    vy = sum((y - my) ** 2 for x, y in pairs)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


def spearman(xs, ys):
    pairs = [(x, y) for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y)]
    if len(pairs) < 3:
        return None
    return pearson(ranks([x for x, y in pairs]), ranks([y for x, y in pairs]))


def split_indices(n, holdout, seed):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    return idx[holdout:], idx[:holdout]


def resolve(s, root, run_name):
    if s.startswith(OLD_PREFIX):
        q = PAPER / s[len(OLD_PREFIX) :].lstrip("/")
        if q.exists():
            return q
    p = Path(s)
    try:
        if p.exists():
            return p
    except PermissionError:
        pass
    m = list(root.glob(f"**/{run_name}"))
    return m[0] if m else p


def ckpath(rd):
    for p in [rd / "checkpoints/best.pt", rd / "best.pt"]:
        if p.exists():
            return p
    raise FileNotFoundError(rd)


def smooth_psi(features, W, b, eps0, tau, rho, device, batch=2048):
    W = W.to(device).float()
    b = b.to(device).float()
    total_tube = 0.0
    total_p = 0.0
    n = 0
    wnorm = torch.linalg.vector_norm(W, dim=1).clamp_min(1e-8)
    for st in range(0, features.shape[0], batch):
        z = features[st : st + batch].to(device).float()
        m = z.shape[0]
        logits = z @ W.T + b
        s = logits / wnorm.unsqueeze(0)
        tube = torch.sigmoid((eps0 - s.abs()) / tau).mean(0)
        pplus = torch.sigmoid(s / tau).mean(0)
        total_tube += tube.cpu() * m
        total_p += pplus.cpu() * m
        n += m
    tube = total_tube / n
    pplus = total_p / n
    bal = (1 - torch.sqrt((2 * pplus - 1) ** 2 + rho**2) + rho) / 2
    val = (tube / (eps0 * bal.clamp_min(1e-8))).sum()
    return float(val.item())


# Sensitivity on representative completed runs: all modes, spc 10/20/50, seed 42 per benchmark.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
grid = [
    (0.05, 0.025, 1e-6),
    (0.05, 0.05, 1e-6),
    (0.10, 0.025, 1e-6),
    (0.10, 0.05, 1e-6),
    (0.10, 0.10, 1e-6),
    (0.20, 0.05, 1e-6),
    (0.10, 0.05, 1e-5),
]
base_params = (0.10, 0.05, 1e-6)
records = []
for bench, root, cache_name in RUN_SPECS:
    eval_csv = root / "summary_reports/evaluation_best_runs.csv"
    if not eval_csv.exists():
        continue
    cache = root / cache_name
    train_blob = torch.load(
        cache / "train_features.pt", map_location="cpu", weights_only=False
    )
    features = train_blob["features"].float()
    with eval_csv.open(newline="") as fh:
        rows = [
            r
            for r in csv.DictReader(fh)
            if r.get("status") == "ok" and r.get("run_name") != "cache_warmup"
        ]
    # bounded subset for speed, but covers modes/spc.
    selected = []
    seen = set()
    for r in rows:
        key = (r.get("mode"), r.get("train_samples_per_class"), r.get("seed"))
        if (
            r.get("seed") in ("42", "43")
            and r.get("train_samples_per_class") in ("10", "20", "50")
            and key not in seen
        ):
            selected.append(r)
            seen.add(key)
    for r in selected:
        rd = resolve(r.get("run_dir", ""), root, r.get("run_name", ""))
        try:
            ck = torch.load(ckpath(rd), map_location="cpu", weights_only=False)
        except Exception as e:
            print("skip sensitivity", bench, r.get("run_name"), e)
            continue
        W = ck["head_state"]["classifier.weight"].detach().cpu()
        b = ck["head_state"]["classifier.bias"].detach().cpu()
        seed = it(r.get("seed"))
        holdout = it(r.get("num_holdout_examples"), 2048)
        _, hold = split_indices(int(features.shape[0]), holdout, seed)
        hold_features = features[hold]
        vals = {}
        for eps, tau, rho in grid:
            vals[f"eps{eps:g}_tau{tau:g}_rho{rho:g}"] = smooth_psi(
                hold_features, W, b, eps, tau, rho, device
            )
        rec = {
            "benchmark": bench,
            "run_name": r.get("run_name"),
            "mode": r.get("mode"),
            "spc": r.get("train_samples_per_class"),
            "seed": r.get("seed"),
            "reported_psieff": fl(r.get("heldout_psieff")),
        }
        rec.update(vals)
        records.append(rec)
# Sensitivity summary relative to base key.
base_key = f"eps{base_params[0]:g}_tau{base_params[1]:g}_rho{base_params[2]:g}"
sens = []
base_vals = [fl(r.get(base_key)) for r in records]
for eps, tau, rho in grid:
    key = f"eps{eps:g}_tau{tau:g}_rho{rho:g}"
    vals = [fl(r.get(key)) for r in records]
    rel = [
        abs(v - b) / max(abs(b), 1e-8)
        for v, b in zip(vals, base_vals)
        if math.isfinite(v) and math.isfinite(b)
    ]
    sens.append(
        {
            "setting": key,
            "n": len(rel),
            "spearman_vs_default": spearman(vals, base_vals),
            "median_relative_shift": float(sorted(rel)[len(rel) // 2]) if rel else None,
            "mean_relative_shift": sum(rel) / len(rel) if rel else None,
        }
    )
# Cost from metrics.csv.
cost_rows = []
for bench, root, _ in RUN_SPECS:
    eval_csv = root / "summary_reports/evaluation_best_runs.csv"
    if not eval_csv.exists():
        continue
    with eval_csv.open(newline="") as fh:
        rows = [
            r
            for r in csv.DictReader(fh)
            if r.get("status") == "ok" and r.get("run_name") != "cache_warmup"
        ]
    vals = defaultdict(list)
    for r in rows:
        rd = resolve(r.get("run_dir", ""), root, r.get("run_name", ""))
        m = rd / "metrics.csv"
        if not m.exists():
            continue
        with m.open(newline="") as fh:
            for row in csv.DictReader(fh):
                sec = fl(row.get("epoch_seconds"))
                ep = it(row.get("epoch"))
                if math.isfinite(sec) and ep > 3:
                    vals[r.get("mode")].append(sec)

    def mean(xs):
        return sum(xs) / len(xs) if xs else float("nan")

    st = mean(vals["standard"])
    ps = mean(vals["psieff"])
    wd = mean(vals["weight_decay"])
    cost_rows.append(
        {
            "benchmark": bench,
            "standard_sec": st,
            "weight_decay_sec": wd,
            "psieff_sec": ps,
            "psieff_overhead_pct": 100 * (ps - st) / st
            if math.isfinite(st) and st
            else float("nan"),
        }
    )
(result_file("estimator_cost/sensitivity_records.json")).write_text(
    json.dumps(records, indent=2, sort_keys=True), encoding="utf-8"
)
(result_file("estimator_cost/sensitivity_summary.json")).write_text(
    json.dumps({"sensitivity": sens, "cost": cost_rows}, indent=2, sort_keys=True),
    encoding="utf-8",
)
with (result_file("estimator_cost/sensitivity_table_rows.tex")).open(
    "w", encoding="utf-8"
) as f:
    for x in sens:
        f.write(
            f"{x['setting'].replace('_', '/')} & {x['n']} & ${x['spearman_vs_default']:.3f}$ & ${100 * x['median_relative_shift']:.1f}\\%$\\\\\n"
        )
with (result_file("estimator_cost/cost_table_rows.tex")).open(
    "w", encoding="utf-8"
) as f:
    for x in cost_rows:
        f.write(
            f"{x['benchmark']} & ${x['standard_sec']:.2f}$ & ${x['psieff_sec']:.2f}$ & ${x['psieff_overhead_pct']:.1f}\\%$\\\\\n"
        )
print(json.dumps({"sensitivity": sens, "cost": cost_rows}, indent=2))
