"Compare boundary mass with alternative representation metrics. See docs/EXPERIMENTS.md for usage."

from utils.paths import PROJECT_ROOT, RUNS_DIR, RESULTS_DIR, result_file
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, List, Tuple

import torch

PAPER = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
RUN_ROOT = RUNS_DIR / "cifar100_resnet50_pretrained"
OUT_DIR = RESULTS_DIR
OUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_CSV = RUN_ROOT / "summary_reports" / "evaluation_best_runs.csv"
TRAIN_CACHE = RUN_ROOT / "feature_cache_resnet50_imagenetv2" / "train_features.pt"
TEST_CACHE = RUN_ROOT / "feature_cache_resnet50_imagenetv2" / "test_features.pt"
OLD_PREFIX = "."


def to_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def to_int(x, default=0):
    try:
        return int(float(x))
    except Exception:
        return default


def pearson(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return None
    x = [p[0] for p in pairs]
    y = [p[1] for p in pairs]
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    if vx <= 0 or vy <= 0:
        return None
    return sum((a - mx) * (b - my) for a, b in pairs) / math.sqrt(vx * vy)


def ranks(vals):
    vals = list(vals)
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    out = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and vals[order[j]] == vals[order[i]]:
            j += 1
        r = (i + j - 1) / 2.0 + 1.0
        for k in range(i, j):
            out[order[k]] = r
        i = j
    return out


def spearman(xs, ys):
    pairs = [
        (float(x), float(y))
        for x, y in zip(xs, ys)
        if math.isfinite(float(x)) and math.isfinite(float(y))
    ]
    if len(pairs) < 3:
        return None
    return pearson(ranks([p[0] for p in pairs]), ranks([p[1] for p in pairs]))


def split_indices(
    num_examples: int, holdout_size: int, seed: int
) -> Tuple[List[int], List[int]]:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_examples, generator=generator).tolist()
    return indices[holdout_size:], indices[:holdout_size]


def select_balanced_train_indices(
    train_indices, train_labels, samples_per_class, seed, num_classes=100
):
    if samples_per_class <= 0:
        return train_indices
    rng = random.Random(seed)
    by_class = {c: [] for c in range(num_classes)}
    for idx in train_indices:
        by_class[int(train_labels[idx].item())].append(idx)
    selected = []
    for c in range(num_classes):
        candidates = by_class[c]
        rng.shuffle(candidates)
        selected.extend(candidates[:samples_per_class])
    rng.shuffle(selected)
    return selected


def effective_rank_subset(features: torch.Tensor) -> float:
    x = features.float()
    x = x - x.mean(dim=0, keepdim=True)
    if x.shape[0] <= 1:
        return float("nan")
    # Participation-ratio effective rank of the covariance: tr(C)^2 / ||C||_F^2.
    c = (x.T @ x) / max(1, x.shape[0] - 1)
    tr = torch.trace(c)
    fro2 = (c * c).sum()
    if float(fro2) <= 0:
        return float("nan")
    return float((tr * tr / fro2).item())


def resolve_run_dir(row):
    s = str(row["run_dir"])
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
    # Last fallback: search by run name.
    matches = list(RUN_ROOT.glob(f"**/{row['run_name']}"))
    if matches:
        return matches[0]
    return p


def load_checkpoint(run_dir: Path):
    ck = run_dir / "checkpoints" / "best.pt"
    if not ck.exists():
        ck = run_dir / "best.pt"
    return torch.load(ck, map_location="cpu", weights_only=False), ck


def evaluate_head_proxies(
    W,
    b,
    test_features,
    test_labels,
    device,
    noise_alpha=0.10,
    noise_repeats=3,
    batch_size=2048,
):
    W = W.to(device).float()
    b = b.to(device).float()
    denom = torch.cdist(W, W).clamp_min(1e-8)
    n = test_features.shape[0]
    margins = []
    raw_margins = []
    confs = []
    correct = []
    clean_preds = []
    for start in range(0, n, batch_size):
        z = test_features[start : start + batch_size].to(device).float()
        y = test_labels[start : start + batch_size].to(device).long()
        logits = z @ W.T + b
        probs = torch.softmax(logits, dim=1)
        pred = logits.argmax(dim=1)
        clean_preds.append(pred.cpu())
        correct.append((pred == y).float().cpu())
        confs.append(probs.max(dim=1).values.cpu())
        true_logits = logits.gather(1, y[:, None]).squeeze(1)
        diffs = true_logits[:, None] - logits
        diffs.scatter_(1, y[:, None], float("inf"))
        raw = diffs.min(dim=1).values
        den = denom[y]
        ndiff = diffs / den
        norm_margin = ndiff.min(dim=1).values
        margins.append(norm_margin.cpu())
        raw_margins.append(raw.cpu())
    clean_preds = torch.cat(clean_preds)
    feature_scale = test_features.float().std(dim=0, unbiased=False).clamp_min(1e-8)
    flips = []
    gen = torch.Generator(device="cpu").manual_seed(20260427)
    for rep in range(noise_repeats):
        noisy_preds = []
        for start in range(0, n, batch_size):
            z = test_features[start : start + batch_size].float()
            noise = (
                torch.randn(z.shape, generator=gen, dtype=z.dtype)
                * feature_scale.unsqueeze(0)
                * noise_alpha
            )
            z = (z + noise).to(device)
            logits = z @ W.T + b
            noisy_preds.append(logits.argmax(dim=1).cpu())
        noisy_preds = torch.cat(noisy_preds)
        flips.append(float((noisy_preds != clean_preds).float().mean().item()))
    return {
        "weight_fro_norm": float(torch.linalg.vector_norm(W).item()),
        "weight_spectral_norm": float(torch.linalg.svdvals(W).max().item()),
        "weight_mean_row_norm": float(torch.linalg.vector_norm(W, dim=1).mean().item()),
        "margin_norm_mean": float(torch.cat(margins).mean().item()),
        "margin_norm_median": float(torch.cat(margins).median().item()),
        "margin_raw_mean": float(torch.cat(raw_margins).mean().item()),
        "test_confidence_mean": float(torch.cat(confs).mean().item()),
        "test_acc_recomputed": float(torch.cat(correct).mean().item()),
        "noise_flip_rate_a010": sum(flips) / len(flips),
    }


def main():
    with EVAL_CSV.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    rows = [
        r
        for r in rows
        if r.get("status") == "ok" and r.get("run_name") != "cache_warmup"
    ]
    # Keep the real completed low-shot sweep and psi tuning; exclude failed/missing checkpoints later.
    train_blob = torch.load(TRAIN_CACHE, map_location="cpu", weights_only=False)
    test_blob = torch.load(TEST_CACHE, map_location="cpu", weights_only=False)
    train_features = train_blob["features"].float()
    train_labels = train_blob["labels"].long()
    test_features = test_blob["features"].float()
    test_labels = test_blob["labels"].long()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device={device}, runs={len(rows)}")
    erank_cache = {}
    out = []
    for idx, row in enumerate(rows, start=1):
        run_dir = resolve_run_dir(row)
        if not run_dir.exists():
            print("missing run dir", row.get("run_name"), row.get("run_dir"))
            continue
        try:
            ck, ck_path = load_checkpoint(run_dir)
        except Exception as e:
            print("skip checkpoint", run_dir, e)
            continue
        W = ck["head_state"]["classifier.weight"].detach().cpu()
        b = ck["head_state"]["classifier.bias"].detach().cpu()
        seed = to_int(row.get("seed"))
        spc = to_int(row.get("train_samples_per_class"))
        holdout = to_int(row.get("num_holdout_examples"), 2048)
        erank_key = (spc, seed)
        if erank_key not in erank_cache:
            train_idx, _ = split_indices(int(train_features.shape[0]), holdout, seed)
            selected = select_balanced_train_indices(train_idx, train_labels, spc, seed)
            erank_cache[erank_key] = effective_rank_subset(train_features[selected])
        proxies = evaluate_head_proxies(W, b, test_features, test_labels, device)
        rec = dict(row)
        rec["run_dir_resolved"] = str(run_dir)
        rec["checkpoint_resolved"] = str(ck_path)
        rec["feature_effective_rank_labeled"] = erank_cache[erank_key]
        rec.update(proxies)
        rec["log1p_psieff"] = math.log1p(to_float(row.get("heldout_psieff")))
        out.append(rec)
        if idx % 10 == 0:
            print(f"processed {idx}/{len(rows)}")
    csv_path = result_file("proxy_metrics/proxy_experiment_completed_runs.csv")
    if out:
        fields = list(out[0].keys())
        # append any later keys
        for r in out:
            for k in r.keys():
                if k not in fields:
                    fields.append(k)
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(out)
        (
            result_file("proxy_metrics/proxy_experiment_completed_runs.json")
        ).write_text(json.dumps(out, indent=2, sort_keys=True), encoding="utf-8")
    metrics = {
        "psi_eff": ("heldout_psieff", -1),
        "log1p_psi_eff": ("log1p_psieff", -1),
        "weight_norm": ("weight_fro_norm", None),
        "spectral_norm": ("weight_spectral_norm", None),
        "normalized_margin": ("margin_norm_mean", +1),
        "feature_effective_rank": ("feature_effective_rank_labeled", None),
        "noise_flip_rate": ("noise_flip_rate_a010", -1),
        "NLL": ("eval_nll", -1),
        "ECE": ("eval_ece", -1),
        "AURC": ("eval_aurc", -1),
    }
    ys = [to_float(r["eval_acc"]) for r in out]
    summary = []
    for name, (field, expected) in metrics.items():
        xs = [to_float(r.get(field)) for r in out]
        usable = sum(1 for x, y in zip(xs, ys) if math.isfinite(x) and math.isfinite(y))
        pr = pearson(xs, ys)
        sr = spearman(xs, ys)
        summary.append(
            {
                "metric": name,
                "field": field,
                "n": usable,
                "pearson_with_top1": pr,
                "spearman_with_top1": sr,
                "expected_direction": expected,
                "abs_pearson": None if pr is None else abs(pr),
                "abs_spearman": None if sr is None else abs(sr),
                "status": "computed" if usable else "missing",
            }
        )
    summary = sorted(
        summary,
        key=lambda r: r["abs_spearman"] if r["abs_spearman"] is not None else -1,
        reverse=True,
    )
    (
        result_file("proxy_metrics/proxy_experiment_completed_summary.json")
    ).write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (
        result_file("proxy_metrics/proxy_experiment_completed_summary.csv")
    ).open("w", newline="") as fh:
        fields = [
            "metric",
            "field",
            "n",
            "pearson_with_top1",
            "spearman_with_top1",
            "expected_direction",
            "abs_pearson",
            "abs_spearman",
            "status",
        ]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(summary)
    # Compact LaTeX rows for Table 2.
    keep = [
        "psi_eff",
        "log1p_psi_eff",
        "weight_norm",
        "normalized_margin",
        "feature_effective_rank",
        "noise_flip_rate",
        "NLL",
        "ECE",
        "AURC",
    ]
    by = {r["metric"]: r for r in summary}
    lines = []
    for k in keep:
        r = by[k]
        label = {
            "psi_eff": r"$\\hat\\psieff$",
            "log1p_psi_eff": r"$\\log(1+\\hat\\psieff)$",
            "weight_norm": "Weight norm",
            "normalized_margin": "Normalized margin",
            "feature_effective_rank": "Feature effective rank",
            "noise_flip_rate": "Noise flip rate",
            "NLL": "NLL",
            "ECE": "ECE",
            "AURC": "AURC",
        }[k]
        p = "--" if r["pearson_with_top1"] is None else f"{r['pearson_with_top1']:.3f}"
        s = (
            "--"
            if r["spearman_with_top1"] is None
            else f"{r['spearman_with_top1']:.3f}"
        )
        lines.append(f"{label} & ${p}$ & ${s}$ & {r['n']}\\\\")
    (
        result_file("proxy_metrics/proxy_experiment_completed_table_rows.tex")
    ).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote", csv_path)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
