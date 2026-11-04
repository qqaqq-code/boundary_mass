"Compute alternative diagnostics across multiple backbones. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, RUNS_DIR, RESULTS_DIR, result_file
import csv, json, math, random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
import torch

PAPER = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
ART = RESULTS_DIR
ART.mkdir(parents=True, exist_ok=True)
OLD_PREFIX = "."
RUN_SPECS = [
    {
        "name": "CIFAR-100/R50",
        "root": RUNS_DIR / "cifar100_resnet50_pretrained",
        "cache": "feature_cache_resnet50_imagenetv2",
    },
    {
        "name": "CIFAR-100/ViT-B",
        "root": RUNS_DIR / "cifar100_vitb16_targeted",
        "cache": "feature_cache_vitb16_imagenetv1",
    },
    {
        "name": "Tiny-IN/R50",
        "root": RUNS_DIR / "tiny_imagenet_resnet50_targeted",
        "cache": "feature_cache_resnet50_imagenetv2",
    },
]


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
    mx = sum(x for x, y in pairs) / len(pairs)
    my = sum(y for x, y in pairs) / len(pairs)
    vx = sum((x - mx) ** 2 for x, y in pairs)
    vy = sum((y - my) ** 2 for x, y in pairs)
    if vx <= 0 or vy <= 0:
        return None
    return sum((x - mx) * (y - my) for x, y in pairs) / math.sqrt(vx * vy)


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


def split_indices(num_examples: int, holdout_size: int, seed: int):
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(num_examples, generator=g).tolist()
    return idx[holdout_size:], idx[:holdout_size]


def select_balanced(train_idx, labels, spc, seed, num_classes):
    if spc <= 0:
        return train_idx
    rng = random.Random(seed)
    by = {c: [] for c in range(num_classes)}
    for idx in train_idx:
        y = int(labels[idx].item())
        if y in by:
            by[y].append(idx)
    out = []
    for c in range(num_classes):
        cand = by[c]
        if len(cand) < spc:
            continue
        rng.shuffle(cand)
        out.extend(cand[:spc])
    rng.shuffle(out)
    return out


def effective_rank(features):
    x = features.float()
    x = x - x.mean(0, keepdim=True)
    if x.shape[0] < 2:
        return float("nan")
    c = (x.T @ x) / max(1, x.shape[0] - 1)
    tr = torch.trace(c)
    fro2 = (c * c).sum()
    return float((tr * tr / fro2).item()) if float(fro2) > 0 else float("nan")


def resolve_run_dir(row, root):
    s = str(row.get("run_dir", ""))
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
    matches = list(root.glob(f"**/{row.get('run_name', '')}"))
    return matches[0] if matches else p


def checkpoint_path(run_dir):
    for p in [run_dir / "checkpoints/best.pt", run_dir / "best.pt"]:
        if p.exists():
            return p
    raise FileNotFoundError(run_dir)


def eval_proxies(
    W, b, test_features, test_labels, device, noise_alpha=0.10, batch_size=4096
):
    W = W.to(device).float()
    b = b.to(device).float()
    denom = torch.cdist(W, W).clamp_min(1e-8)
    n = test_features.shape[0]
    margins = []
    rawm = []
    clean = []
    correct = []
    conf = []
    for st in range(0, n, batch_size):
        z = test_features[st : st + batch_size].to(device).float()
        y = test_labels[st : st + batch_size].to(device).long()
        logits = z @ W.T + b
        probs = torch.softmax(logits, 1)
        pred = logits.argmax(1)
        clean.append(pred.cpu())
        correct.append((pred == y).float().cpu())
        conf.append(probs.max(1).values.cpu())
        true = logits.gather(1, y[:, None]).squeeze(1)
        diffs = true[:, None] - logits
        diffs.scatter_(1, y[:, None], float("inf"))
        rawm.append(diffs.min(1).values.cpu())
        margins.append((diffs / denom[y]).min(1).values.cpu())
    clean = torch.cat(clean)
    scale = test_features.float().std(0, unbiased=False).clamp_min(1e-8)
    gen = torch.Generator(device="cpu").manual_seed(20260427)
    flips = []
    for rep in range(2):
        preds = []
        for st in range(0, n, batch_size):
            z = test_features[st : st + batch_size].float()
            z = (
                z
                + torch.randn(z.shape, generator=gen, dtype=z.dtype)
                * scale.unsqueeze(0)
                * noise_alpha
            )
            logits = z.to(device) @ W.T + b
            preds.append(logits.argmax(1).cpu())
        preds = torch.cat(preds)
        flips.append(float((preds != clean).float().mean().item()))
    return {
        "weight_fro_norm": float(torch.linalg.vector_norm(W).item()),
        "weight_spectral_norm": float(torch.linalg.svdvals(W).max().item()),
        "margin_norm_mean": float(torch.cat(margins).mean().item()),
        "margin_norm_median": float(torch.cat(margins).median().item()),
        "margin_raw_mean": float(torch.cat(rawm).mean().item()),
        "noise_flip_rate_a010": sum(flips) / len(flips),
        "test_acc_recomputed": float(torch.cat(correct).mean().item()),
        "test_confidence_mean": float(torch.cat(conf).mean().item()),
    }


def residual_corr(
    rows, field, yfield="eval_acc", groupfields=("benchmark", "train_samples_per_class")
):
    groups = defaultdict(list)
    for r in rows:
        groups[tuple(r[g] for g in groupfields)].append(r)
    xs = []
    ys = []
    for _, items in groups.items():
        pairs = [(to_float(r.get(field)), to_float(r.get(yfield))) for r in items]
        pairs = [p for p in pairs if math.isfinite(p[0]) and math.isfinite(p[1])]
        if len(pairs) < 2:
            continue
        mx = sum(p[0] for p in pairs) / len(pairs)
        my = sum(p[1] for p in pairs) / len(pairs)
        xs += [p[0] - mx for p in pairs]
        ys += [p[1] - my for p in pairs]
    return pearson(xs, ys), spearman(xs, ys), len(xs)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    all_rows = []
    bench_summary = []
    for spec in RUN_SPECS:
        root = spec["root"]
        eval_csv = root / "summary_reports/evaluation_best_runs.csv"
        if not eval_csv.exists():
            print("missing eval csv", eval_csv)
            continue
        cache = root / spec["cache"]
        train_blob = torch.load(
            cache / "train_features.pt", map_location="cpu", weights_only=False
        )
        eval_cache = cache / "test_features.pt"
        if not eval_cache.exists():
            eval_cache = cache / "val_features.pt"
        test_blob = torch.load(eval_cache, map_location="cpu", weights_only=False)
        train_features = train_blob["features"].float()
        train_labels = train_blob["labels"].long()
        test_features = test_blob["features"].float()
        test_labels = test_blob["labels"].long()
        with eval_csv.open(newline="") as fh:
            rows = [
                r
                for r in csv.DictReader(fh)
                if r.get("status") == "ok" and r.get("run_name") != "cache_warmup"
            ]
        erank_cache = {}
        count = 0
        for row in rows:
            rd = resolve_run_dir(row, root)
            try:
                ck = torch.load(
                    checkpoint_path(rd), map_location="cpu", weights_only=False
                )
            except Exception as e:
                print("skip", spec["name"], row.get("run_name"), e)
                continue
            W = ck["head_state"]["classifier.weight"].detach().cpu()
            b = ck["head_state"]["classifier.bias"].detach().cpu()
            num_classes = W.shape[0]
            seed = to_int(row.get("seed"))
            spc = to_int(row.get("train_samples_per_class"))
            holdout = to_int(row.get("num_holdout_examples"), 2048)
            key = (spc, seed, num_classes)
            if key not in erank_cache:
                tr, _ = split_indices(int(train_features.shape[0]), holdout, seed)
                sel = select_balanced(tr, train_labels, spc, seed, num_classes)
                erank_cache[key] = (
                    effective_rank(train_features[sel]) if sel else float("nan")
                )
            rec = dict(row)
            rec["benchmark"] = spec["name"]
            rec["feature_effective_rank_labeled"] = erank_cache[key]
            rec.update(eval_proxies(W, b, test_features, test_labels, device))
            psi = to_float(row.get("heldout_psieff"))
            rec["log1p_psieff"] = (
                math.log1p(psi) if math.isfinite(psi) and psi > 0 else float("nan")
            )
            all_rows.append(rec)
            count += 1
            if count % 10 == 0:
                print(spec["name"], count, "/", len(rows))
        bench_summary.append(
            {
                "benchmark": spec["name"],
                "runs": count,
                "feature_dim": int(train_features.shape[1]),
                "classes": int(
                    max(
                        int(train_labels.max().item()) + 1,
                        int(test_labels.max().item()) + 1,
                    )
                ),
            }
        )
    out_csv = result_file("proxy_metrics/multi_backbone_proxy_runs.csv")
    if all_rows:
        fields = []
        for r in all_rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
        with out_csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(all_rows)
        (
            result_file("proxy_metrics/multi_backbone_proxy_runs.json")
        ).write_text(json.dumps(all_rows, indent=2, sort_keys=True), encoding="utf-8")
    metrics = [
        ("psi_eff", "heldout_psieff"),
        ("log1p_psi_eff", "log1p_psieff"),
        ("weight_norm", "weight_fro_norm"),
        ("spectral_norm", "weight_spectral_norm"),
        ("normalized_margin", "margin_norm_mean"),
        ("feature_effective_rank", "feature_effective_rank_labeled"),
        ("noise_flip_rate", "noise_flip_rate_a010"),
        ("NLL", "eval_nll"),
        ("ECE", "eval_ece"),
        ("AURC", "eval_aurc"),
    ]
    summary = []
    ys = [to_float(r.get("eval_acc")) for r in all_rows]
    for name, field in metrics:
        xs = [to_float(r.get(field)) for r in all_rows]
        rp, rs, n = residual_corr(all_rows, field)
        summary.append(
            {
                "metric": name,
                "field": field,
                "n": n,
                "mixed_pearson": pearson(xs, ys),
                "mixed_spearman": spearman(xs, ys),
                "spc_residual_pearson": rp,
                "spc_residual_spearman": rs,
            }
        )
    (
        result_file("proxy_metrics/multi_backbone_proxy_summary.json")
    ).write_text(
        json.dumps(
            {"benchmarks": bench_summary, "metrics": summary}, indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )
    labels = {
        "psi_eff": r"$\hat\psieff$",
        "log1p_psi_eff": r"$\log(1+\hat\psieff)$",
        "weight_norm": "Weight norm",
        "spectral_norm": "Spectral norm",
        "normalized_margin": "Normalized margin",
        "feature_effective_rank": "Feature eff. rank",
        "noise_flip_rate": "Noise flip rate",
        "NLL": "NLL",
        "ECE": "ECE",
        "AURC": "AURC",
    }
    keep = [
        "psi_eff",
        "log1p_psi_eff",
        "weight_norm",
        "normalized_margin",
        "feature_effective_rank",
        "noise_flip_rate",
        "NLL",
        "AURC",
    ]
    by = {r["metric"]: r for r in summary}
    with (
        result_file("proxy_metrics/multi_backbone_proxy_table_rows.tex")
    ).open("w", encoding="utf-8") as f:
        for k in keep:
            r = by[k]
            fmt = lambda x: "--" if x is None else f"{x:.3f}"
            f.write(
                f"{labels[k]} & ${fmt(r['spc_residual_pearson'])}$ & ${fmt(r['spc_residual_spearman'])}$ & ${fmt(r['mixed_pearson'])}$ & {r['n']}\\\\\n"
            )
    with (
        result_file("proxy_metrics/multi_backbone_benchmark_rows.tex")
    ).open("w", encoding="utf-8") as f:
        for b in bench_summary:
            f.write(
                f"{b['benchmark']} & {b['runs']} & {b['classes']} & {b['feature_dim']}\\\\\n"
            )
    print(json.dumps({"benchmarks": bench_summary, "metrics": summary}, indent=2))


if __name__ == "__main__":
    main()
