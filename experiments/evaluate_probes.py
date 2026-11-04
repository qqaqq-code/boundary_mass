"Evaluate saved probes and summarize their prediction metrics. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_cifar100_probe import (
    LinearHead as CifarLinearHead,
    build_backbone as build_cifar_backbone,
    build_datasets as build_cifar_datasets,
    load_or_compute_feature_cache as load_cifar_feature_cache,
)
from train_encoder import psieff_surrogate
from train_imagefolder_probe import (
    LinearHead as ImageFolderLinearHead,
    build_backbone as build_imagefolder_backbone,
    build_datasets as build_imagefolder_datasets,
    load_or_compute_feature_cache as load_imagefolder_feature_cache,
)
from utils.metrics import evaluate_classifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURE_CACHE_MEMO: Dict[
    str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]
] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-hoc evaluation for frozen-backbone linear-probe runs."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ece-bins", type=int, default=15)
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def resolve_maybe_relative(path_str: str | None, run_dir: Path) -> Optional[Path]:
    if not path_str:
        return None

    path = Path(path_str)
    candidates = [path]
    if not path.is_absolute():
        candidates.append((run_dir / path).resolve())
        candidates.append((PROJECT_ROOT / path).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[-1]


def find_run_dirs(run_root: Path) -> List[Path]:
    if (run_root / "metrics.jsonl").exists() and (run_root / "args.json").exists():
        return [run_root.resolve()]

    run_dirs = {
        metrics_path.parent.resolve()
        for metrics_path in run_root.rglob("metrics.jsonl")
        if "summary_reports" not in metrics_path.parts
        and (metrics_path.parent / "args.json").exists()
    }
    return sorted(run_dirs)


def split_indices(
    num_examples: int, holdout_size: int, seed: int
) -> Tuple[List[int], List[int]]:
    if holdout_size <= 0 or holdout_size >= num_examples:
        raise ValueError("holdout_size must be between 1 and num_examples - 1")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_examples, generator=generator).tolist()
    return indices[holdout_size:], indices[:holdout_size]


def select_balanced_train_indices(
    train_indices: List[int],
    train_labels: torch.Tensor,
    samples_per_class: int,
    seed: int,
    num_classes: int,
) -> List[int]:
    if samples_per_class <= 0:
        return train_indices

    rng = torch.Generator().manual_seed(seed)
    by_class: Dict[int, List[int]] = {class_idx: [] for class_idx in range(num_classes)}
    for index in train_indices:
        by_class[int(train_labels[index].item())].append(index)

    selected: List[int] = []
    for class_idx in range(num_classes):
        candidates = by_class[class_idx]
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"class {class_idx} has only {len(candidates)} examples after holdout; need {samples_per_class}"
            )
        permutation = torch.randperm(len(candidates), generator=rng).tolist()
        selected.extend(candidates[i] for i in permutation[:samples_per_class])

    final_order = torch.randperm(len(selected), generator=rng).tolist()
    return [selected[i] for i in final_order]


def make_loader(
    features: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> DataLoader:
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )


def evaluate_psieff(
    head: nn.Module,
    holdout_features: torch.Tensor,
    device: torch.device,
    tau: float,
    rho: float,
    eps0: float,
) -> float:
    head.eval()
    with torch.no_grad():
        value, _ = psieff_surrogate(
            features=holdout_features.to(device, non_blocking=True),
            weight=head.classifier.weight,
            bias=head.classifier.bias,
            tau=tau,
            rho=rho,
            eps0=eps0,
            reduction="sum",
        )
    return float(value.item())


def detect_dataset_kind(args_dict: Dict) -> str:
    if "data_dir" in args_dict:
        return "cifar100_pretrained"
    if "train_dir" in args_dict and "val_dir" in args_dict:
        return "imagefolder"
    raise ValueError("Could not infer dataset kind from args.json")


def load_cached_features(
    feature_cache_dir: Path, dataset_kind: str
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    train_cache = feature_cache_dir / "train_features.pt"
    eval_cache_name = (
        "test_features.pt"
        if dataset_kind == "cifar100_pretrained"
        else "val_features.pt"
    )
    eval_cache = feature_cache_dir / eval_cache_name
    if not train_cache.exists() or not eval_cache.exists():
        return None

    train_blob = torch.load(train_cache, map_location="cpu", weights_only=False)
    eval_blob = torch.load(eval_cache, map_location="cpu", weights_only=False)
    max_label = torch.max(
        train_blob["labels"].max().long(),
        eval_blob["labels"].max().long(),
    )
    num_classes = int(
        eval_blob.get(
            "num_classes",
            train_blob.get("num_classes", int(max_label.item() + 1)),
        )
    )
    return (
        train_blob["features"].float(),
        train_blob["labels"].long(),
        eval_blob["features"].float(),
        eval_blob["labels"].long(),
        num_classes,
    )


def load_features_with_fallback(
    run_dir: Path,
    args_dict: Dict,
    device: torch.device,
    eval_batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int, int]:
    dataset_kind = detect_dataset_kind(args_dict)
    cache_dir = resolve_maybe_relative(args_dict.get("feature_cache_dir"), run_dir)
    cache_key_parts = [dataset_kind]
    if cache_dir is not None:
        cache_key_parts.append(str(cache_dir.resolve()))
    else:
        cache_key_parts.append(str(sorted(args_dict.items())))
    cache_key = "||".join(cache_key_parts)

    if cache_key in FEATURE_CACHE_MEMO:
        return FEATURE_CACHE_MEMO[cache_key]

    if cache_dir is not None:
        cached = load_cached_features(cache_dir, dataset_kind)
        if cached is not None:
            (
                train_features_all,
                train_labels_all,
                eval_features,
                eval_labels,
                num_classes,
            ) = cached
            feature_dim = int(train_features_all.shape[1])
            result = (
                train_features_all,
                train_labels_all,
                eval_features,
                eval_labels,
                num_classes,
                feature_dim,
            )
            FEATURE_CACHE_MEMO[cache_key] = result
            return result

    if dataset_kind == "cifar100_pretrained":
        namespace = argparse.Namespace(**args_dict)
        namespace.data_dir = resolve_maybe_relative(args_dict["data_dir"], run_dir)
        namespace.feature_cache_dir = cache_dir
        backbone, feature_dim, transform, _ = build_cifar_backbone(namespace, device)
        train_dataset, eval_dataset = build_cifar_datasets(namespace, transform)
        train_features_all, train_labels_all, eval_features, eval_labels = (
            load_cifar_feature_cache(
                backbone=backbone,
                train_dataset=train_dataset,
                test_dataset=eval_dataset,
                device=device,
                batch_size=eval_batch_size,
                num_workers=num_workers,
                cache_dir=cache_dir,
                feature_dim=feature_dim,
                backbone_model=namespace.backbone_model,
                weights_name=namespace.weights,
            )
        )
        num_classes = 100
        result = (
            train_features_all,
            train_labels_all,
            eval_features,
            eval_labels,
            num_classes,
            feature_dim,
        )
        FEATURE_CACHE_MEMO[cache_key] = result
        return result

    namespace = argparse.Namespace(**args_dict)
    namespace.train_dir = resolve_maybe_relative(args_dict["train_dir"], run_dir)
    namespace.val_dir = resolve_maybe_relative(args_dict["val_dir"], run_dir)
    namespace.feature_cache_dir = cache_dir
    backbone, feature_dim, transform = build_imagefolder_backbone(namespace, device)
    train_dataset, eval_dataset = build_imagefolder_datasets(namespace, transform)
    num_classes = len(train_dataset.classes)
    train_features_all, train_labels_all, eval_features, eval_labels = (
        load_imagefolder_feature_cache(
            backbone=backbone,
            train_dataset=train_dataset,
            val_dataset=eval_dataset,
            device=device,
            batch_size=eval_batch_size,
            num_workers=num_workers,
            cache_dir=cache_dir,
            feature_dim=feature_dim,
            num_classes=num_classes,
        )
    )
    result = (
        train_features_all,
        train_labels_all,
        eval_features,
        eval_labels,
        num_classes,
        feature_dim,
    )
    FEATURE_CACHE_MEMO[cache_key] = result
    return result


def load_checkpoint_path(run_dir: Path, summary: Dict, checkpoint_tag: str) -> Path:
    summary_key = "best_checkpoint" if checkpoint_tag == "best" else "last_checkpoint"
    summary_path = resolve_maybe_relative(summary.get(summary_key), run_dir)
    fallback = run_dir / "checkpoints" / f"{checkpoint_tag}.pt"
    return (
        summary_path if summary_path is not None and summary_path.exists() else fallback
    )


def make_config_group(row: Dict) -> str:
    pieces = [
        f"dataset={row['dataset_kind']}",
        f"backbone={row['backbone_model']}",
        f"checkpoint={row['checkpoint_tag']}",
        f"mode={row['mode']}",
        f"spc={row['train_samples_per_class']}",
    ]
    if row["mode"] == "weight_decay":
        pieces.append(f"wd={row['weight_decay']}")
    elif row["mode"] == "psieff":
        pieces.append(f"lam={row['lambda_psieff']}")
    return "|".join(pieces)


def summarize_group_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(make_config_group(row), []).append(row)

    def mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
        if not values:
            return None, None
        if len(values) == 1:
            return mean(values), 0.0
        return mean(values), pstdev(values)

    summary_rows: List[Dict] = []
    for group_key, items in sorted(grouped.items()):
        first = items[0]
        eval_acc_mean, eval_acc_std = mean_std([row["eval_acc"] for row in items])
        eval_nll_mean, eval_nll_std = mean_std([row["eval_nll"] for row in items])
        eval_ece_mean, eval_ece_std = mean_std([row["eval_ece"] for row in items])
        eval_brier_mean, eval_brier_std = mean_std([row["eval_brier"] for row in items])
        eval_aurc_mean, eval_aurc_std = mean_std([row["eval_aurc"] for row in items])
        psi_mean, psi_std = mean_std([row["heldout_psieff"] for row in items])

        summary_rows.append(
            {
                "config_group": group_key,
                "dataset_kind": first["dataset_kind"],
                "backbone_model": first["backbone_model"],
                "checkpoint_tag": first["checkpoint_tag"],
                "mode": first["mode"],
                "train_samples_per_class": first["train_samples_per_class"],
                "weight_decay": first["weight_decay"],
                "lambda_psieff": first["lambda_psieff"],
                "num_runs": len(items),
                "eval_acc_mean": eval_acc_mean,
                "eval_acc_std": eval_acc_std,
                "eval_nll_mean": eval_nll_mean,
                "eval_nll_std": eval_nll_std,
                "eval_ece_mean": eval_ece_mean,
                "eval_ece_std": eval_ece_std,
                "eval_brier_mean": eval_brier_mean,
                "eval_brier_std": eval_brier_std,
                "eval_aurc_mean": eval_aurc_mean,
                "eval_aurc_std": eval_aurc_std,
                "heldout_psieff_mean": psi_mean,
                "heldout_psieff_std": psi_std,
            }
        )

    return summary_rows


def evaluate_run(
    run_dir: Path,
    checkpoint_tag: str,
    device: torch.device,
    eval_batch_size: int,
    num_workers: int,
    ece_bins: int,
) -> Dict:
    args_dict = read_json(run_dir / "args.json")
    summary = (
        read_json(run_dir / "summary.json")
        if (run_dir / "summary.json").exists()
        else {}
    )
    dataset_kind = detect_dataset_kind(args_dict)

    (
        train_features_all,
        train_labels_all,
        eval_features,
        eval_labels,
        num_classes,
        feature_dim,
    ) = load_features_with_fallback(
        run_dir=run_dir,
        args_dict=args_dict,
        device=device,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )

    train_indices, holdout_indices = split_indices(
        num_examples=int(train_features_all.shape[0]),
        holdout_size=int(args_dict["holdout_size"]),
        seed=int(args_dict["seed"]),
    )
    selected_train_indices = select_balanced_train_indices(
        train_indices=train_indices,
        train_labels=train_labels_all,
        samples_per_class=int(args_dict.get("train_samples_per_class", 0)),
        seed=int(args_dict["seed"]),
        num_classes=num_classes,
    )

    train_features = train_features_all[selected_train_indices]
    train_labels = train_labels_all[selected_train_indices]
    holdout_features = train_features_all[holdout_indices]

    checkpoint_path = load_checkpoint_path(run_dir, summary, checkpoint_tag)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found for {run_dir}: {checkpoint_path}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    head_cls = (
        CifarLinearHead
        if dataset_kind == "cifar100_pretrained"
        else ImageFolderLinearHead
    )
    head = head_cls(in_dim=feature_dim, num_classes=num_classes).to(device)
    head.load_state_dict(checkpoint["head_state"])

    criterion = nn.CrossEntropyLoss()
    train_loader = make_loader(train_features, train_labels, batch_size=eval_batch_size)
    eval_loader = make_loader(eval_features, eval_labels, batch_size=eval_batch_size)
    train_metrics = evaluate_classifier(
        head=head,
        loader=train_loader,
        device=device,
        criterion=criterion,
        num_bins=ece_bins,
    )
    eval_metrics = evaluate_classifier(
        head=head,
        loader=eval_loader,
        device=device,
        criterion=criterion,
        num_bins=ece_bins,
    )
    heldout_psieff = evaluate_psieff(
        head=head,
        holdout_features=holdout_features,
        device=device,
        tau=float(args_dict.get("tau", 0.05)),
        rho=float(args_dict.get("rho", 1e-6)),
        eps0=float(args_dict.get("eps0", 0.1)),
    )

    eval_split = "test" if dataset_kind == "cifar100_pretrained" else "val"
    result = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "dataset_kind": dataset_kind,
        "eval_split": eval_split,
        "checkpoint_tag": checkpoint_tag,
        "checkpoint_path": str(checkpoint_path),
        "mode": summary.get("mode", args_dict.get("mode")),
        "seed": int(summary.get("seed", args_dict.get("seed"))),
        "backbone_model": summary.get(
            "backbone_model", args_dict.get("backbone_model")
        ),
        "weights": summary.get("weights", args_dict.get("weights")),
        "train_samples_per_class": int(
            summary.get(
                "train_samples_per_class", args_dict.get("train_samples_per_class", 0)
            )
        ),
        "weight_decay": safe_float(args_dict.get("weight_decay")),
        "lambda_psieff": safe_float(args_dict.get("lambda_psieff")),
        "num_train_examples": int(train_features.shape[0]),
        "num_holdout_examples": int(holdout_features.shape[0]),
        f"train_loss": train_metrics.loss,
        f"train_acc": train_metrics.acc,
        f"train_nll": train_metrics.nll,
        f"train_brier": train_metrics.brier,
        f"train_ece": train_metrics.ece,
        f"train_aurc": train_metrics.aurc,
        f"train_mean_confidence": train_metrics.mean_confidence,
        f"eval_loss": eval_metrics.loss,
        f"eval_acc": eval_metrics.acc,
        f"eval_nll": eval_metrics.nll,
        f"eval_brier": eval_metrics.brier,
        f"eval_ece": eval_metrics.ece,
        f"eval_aurc": eval_metrics.aurc,
        f"eval_mean_confidence": eval_metrics.mean_confidence,
        "heldout_psieff": heldout_psieff,
        "best_epoch": summary.get("best_epoch"),
        "final_epoch": summary.get("final_epoch"),
    }
    return result


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir else run_root / "summary_reports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    run_rows: List[Dict] = []

    for run_dir in find_run_dirs(run_root):
        print(f"Evaluating {run_dir}")
        try:
            row = evaluate_run(
                run_dir=run_dir,
                checkpoint_tag=args.checkpoint,
                device=device,
                eval_batch_size=args.eval_batch_size,
                num_workers=args.num_workers,
                ece_bins=args.ece_bins,
            )
        except Exception as exc:
            row = {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "dataset_kind": "unknown",
                "checkpoint_tag": args.checkpoint,
                "status": f"error: {exc}",
            }
        else:
            row["status"] = "ok"
            write_json(run_dir / f"evaluation_{args.checkpoint}.json", row)

        run_rows.append(row)

    ok_rows = [row for row in run_rows if row.get("status") == "ok"]
    group_rows = summarize_group_rows(ok_rows)

    write_csv(output_dir / f"evaluation_{args.checkpoint}_runs.csv", run_rows)
    write_csv(output_dir / f"evaluation_{args.checkpoint}_groups.csv", group_rows)
    write_json(output_dir / f"evaluation_{args.checkpoint}_runs.json", run_rows)
    write_json(output_dir / f"evaluation_{args.checkpoint}_groups.json", group_rows)

    print(f"Evaluated {len(ok_rows)} / {len(run_rows)} runs from {run_root}")
    print(f"Wrote {output_dir / f'evaluation_{args.checkpoint}_runs.csv'}")
    print(f"Wrote {output_dir / f'evaluation_{args.checkpoint}_groups.csv'}")


if __name__ == "__main__":
    main()
