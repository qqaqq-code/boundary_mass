"Evaluate fitted probes under feature perturbations. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from train_cifar100_probe import LinearHead as CifarLinearHead
from experiments.evaluate_probes import (
    detect_dataset_kind,
    evaluate_psieff,
    find_run_dirs,
    load_checkpoint_path,
    load_features_with_fallback,
    read_json,
    safe_float,
    split_indices,
)
from train_imagefolder_probe import LinearHead as ImageFolderLinearHead
from utils.metrics import evaluate_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen linear probes under feature-space Gaussian perturbations."
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
    parser.add_argument("--noise-levels", type=str, default="0.05,0.1,0.2,0.3")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--base-seed", type=int, default=2026)
    parser.add_argument("--run-name-contains", type=str, default="")
    return parser.parse_args()


def build_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(str(output_dir))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(
        output_dir / "feature_noise_eval.log", encoding="utf-8"
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def parse_csv_tokens(raw: str, cast_fn) -> List:
    values: List = []
    for token in raw.split(","):
        token = token.strip()
        if token:
            values.append(cast_fn(token))
    return values


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


def make_loader(
    features: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> DataLoader:
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )


@torch.no_grad()
def predict_classes(
    head: nn.Module, loader: DataLoader, device: torch.device
) -> torch.Tensor:
    head.eval()
    pred_chunks: List[torch.Tensor] = []
    for features, _labels in loader:
        features = features.to(device, non_blocking=True)
        logits = head(features)
        pred_chunks.append(torch.argmax(logits, dim=1).cpu())
    return torch.cat(pred_chunks, dim=0)


def mean_std(values: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    values = list(values)
    if not values:
        return None, None
    if len(values) == 1:
        return mean(values), 0.0
    return mean(values), pstdev(values)


def make_base_group(row: Dict) -> str:
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


def summarize_alpha_groups(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        key = make_base_group(row) + f"|alpha={row['noise_alpha']}"
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict] = []
    for group_key, items in sorted(grouped.items()):
        first = items[0]
        acc_mean, acc_std = mean_std([row["noise_acc"] for row in items])
        drop_mean, drop_std = mean_std([row["acc_drop_vs_clean"] for row in items])
        flip_mean, flip_std = mean_std([row["prediction_flip_rate"] for row in items])
        ece_mean, ece_std = mean_std([row["noise_ece"] for row in items])
        aurc_mean, aurc_std = mean_std([row["noise_aurc"] for row in items])
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
                "noise_alpha": first["noise_alpha"],
                "num_rows": len(items),
                "noise_acc_mean": acc_mean,
                "noise_acc_std": acc_std,
                "acc_drop_vs_clean_mean": drop_mean,
                "acc_drop_vs_clean_std": drop_std,
                "prediction_flip_rate_mean": flip_mean,
                "prediction_flip_rate_std": flip_std,
                "noise_ece_mean": ece_mean,
                "noise_ece_std": ece_std,
                "noise_aurc_mean": aurc_mean,
                "noise_aurc_std": aurc_std,
            }
        )
    return summary_rows


def summarize_overall_groups(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(make_base_group(row), []).append(row)

    summary_rows: List[Dict] = []
    for group_key, items in sorted(grouped.items()):
        first = items[0]
        clean_acc_mean, clean_acc_std = mean_std([row["clean_acc"] for row in items])
        noise_acc_mean, noise_acc_std = mean_std([row["noise_acc"] for row in items])
        drop_mean, drop_std = mean_std([row["acc_drop_vs_clean"] for row in items])
        flip_mean, flip_std = mean_std([row["prediction_flip_rate"] for row in items])
        ece_mean, ece_std = mean_std([row["noise_ece"] for row in items])
        aurc_mean, aurc_std = mean_std([row["noise_aurc"] for row in items])
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
                "num_eval_conditions": len(items),
                "num_unique_runs": len({row["run_name"] for row in items}),
                "clean_acc_mean": clean_acc_mean,
                "clean_acc_std": clean_acc_std,
                "noise_acc_mean_over_conditions": noise_acc_mean,
                "noise_acc_std_over_conditions": noise_acc_std,
                "acc_drop_vs_clean_mean_over_conditions": drop_mean,
                "acc_drop_vs_clean_std_over_conditions": drop_std,
                "prediction_flip_rate_mean_over_conditions": flip_mean,
                "prediction_flip_rate_std_over_conditions": flip_std,
                "noise_ece_mean_over_conditions": ece_mean,
                "noise_ece_std_over_conditions": ece_std,
                "noise_aurc_mean_over_conditions": aurc_mean,
                "noise_aurc_std_over_conditions": aurc_std,
            }
        )
    return summary_rows


def load_run_bundle(
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

    train_features_all, _, eval_features, eval_labels, num_classes, feature_dim = (
        load_features_with_fallback(
            run_dir=run_dir,
            args_dict=args_dict,
            device=device,
            eval_batch_size=eval_batch_size,
            num_workers=num_workers,
        )
    )
    _, holdout_indices = split_indices(
        num_examples=int(train_features_all.shape[0]),
        holdout_size=int(args_dict["holdout_size"]),
        seed=int(args_dict["seed"]),
    )
    holdout_features = train_features_all[holdout_indices]

    checkpoint_path = load_checkpoint_path(run_dir, summary, checkpoint_tag)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    head_cls = (
        CifarLinearHead
        if dataset_kind == "cifar100_pretrained"
        else ImageFolderLinearHead
    )
    head = head_cls(in_dim=feature_dim, num_classes=num_classes).to(device)
    head.load_state_dict(checkpoint["head_state"])

    criterion = nn.CrossEntropyLoss()
    clean_loader = make_loader(eval_features, eval_labels, batch_size=eval_batch_size)
    clean_metrics = evaluate_classifier(
        head=head,
        loader=clean_loader,
        device=device,
        criterion=criterion,
        num_bins=ece_bins,
    )
    clean_predictions = predict_classes(head=head, loader=clean_loader, device=device)
    heldout_psieff = evaluate_psieff(
        head=head,
        holdout_features=holdout_features,
        device=device,
        tau=float(args_dict.get("tau", 0.05)),
        rho=float(args_dict.get("rho", 1e-6)),
        eps0=float(args_dict.get("eps0", 0.1)),
    )

    return {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "dataset_kind": dataset_kind,
        "backbone_model": summary.get(
            "backbone_model", args_dict.get("backbone_model")
        ),
        "weights": summary.get("weights", args_dict.get("weights")),
        "checkpoint_tag": checkpoint_tag,
        "mode": summary.get("mode", args_dict.get("mode")),
        "seed": int(summary.get("seed", args_dict.get("seed"))),
        "train_samples_per_class": int(
            summary.get(
                "train_samples_per_class", args_dict.get("train_samples_per_class", 0)
            )
        ),
        "weight_decay": safe_float(args_dict.get("weight_decay")),
        "lambda_psieff": safe_float(args_dict.get("lambda_psieff")),
        "eval_features": eval_features,
        "eval_labels": eval_labels,
        "feature_scale": eval_features.std(dim=0, unbiased=False),
        "criterion": criterion,
        "head": head,
        "clean_metrics": clean_metrics,
        "clean_predictions": clean_predictions,
        "heldout_psieff": heldout_psieff,
    }


def evaluate_noisy_bundle(
    bundle: Dict,
    noise_alpha: float,
    repeat_idx: int,
    device: torch.device,
    eval_batch_size: int,
    ece_bins: int,
    base_seed: int,
) -> Dict:
    generator = torch.Generator().manual_seed(
        base_seed + 1000 * int(bundle["seed"]) + repeat_idx
    )
    noisy_features = bundle["eval_features"] + noise_alpha * bundle[
        "feature_scale"
    ].unsqueeze(0) * torch.randn(
        bundle["eval_features"].shape,
        generator=generator,
        dtype=bundle["eval_features"].dtype,
    )
    noise_loader = make_loader(
        noisy_features, bundle["eval_labels"], batch_size=eval_batch_size
    )
    noise_metrics = evaluate_classifier(
        head=bundle["head"],
        loader=noise_loader,
        device=device,
        criterion=bundle["criterion"],
        num_bins=ece_bins,
    )
    noise_predictions = predict_classes(
        head=bundle["head"], loader=noise_loader, device=device
    )
    flip_rate = float(
        (noise_predictions != bundle["clean_predictions"]).float().mean().item()
    )

    return {
        "run_name": bundle["run_name"],
        "run_dir": bundle["run_dir"],
        "dataset_kind": bundle["dataset_kind"],
        "backbone_model": bundle["backbone_model"],
        "weights": bundle["weights"],
        "checkpoint_tag": bundle["checkpoint_tag"],
        "mode": bundle["mode"],
        "seed": bundle["seed"],
        "train_samples_per_class": bundle["train_samples_per_class"],
        "weight_decay": bundle["weight_decay"],
        "lambda_psieff": bundle["lambda_psieff"],
        "noise_alpha": noise_alpha,
        "repeat_idx": repeat_idx,
        "clean_acc": bundle["clean_metrics"].acc,
        "noise_acc": noise_metrics.acc,
        "acc_drop_vs_clean": bundle["clean_metrics"].acc - noise_metrics.acc,
        "prediction_flip_rate": flip_rate,
        "noise_ece": noise_metrics.ece,
        "noise_aurc": noise_metrics.aurc,
        "heldout_psieff": bundle["heldout_psieff"],
    }


def main() -> None:
    args = parse_args()
    output_dir = (args.output_dir or (args.run_root / "summary_reports")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(output_dir)

    device = torch.device(args.device)
    noise_levels = parse_csv_tokens(args.noise_levels, float)
    run_dirs = find_run_dirs(args.run_root.resolve())
    if args.run_name_contains:
        run_dirs = [
            run_dir for run_dir in run_dirs if args.run_name_contains in run_dir.name
        ]

    logger.info("Found %d run directories under %s", len(run_dirs), args.run_root)
    logger.info(
        "Noise levels=%s | Repeats=%d | Device=%s", noise_levels, args.repeats, device
    )

    run_rows: List[Dict] = []
    for run_index, run_dir in enumerate(run_dirs, start=1):
        logger.info("[%d/%d] Evaluating %s", run_index, len(run_dirs), run_dir.name)
        bundle = load_run_bundle(
            run_dir=run_dir,
            checkpoint_tag=args.checkpoint,
            device=device,
            eval_batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            ece_bins=args.ece_bins,
        )
        for noise_alpha in noise_levels:
            for repeat_idx in range(args.repeats):
                row = evaluate_noisy_bundle(
                    bundle=bundle,
                    noise_alpha=noise_alpha,
                    repeat_idx=repeat_idx,
                    device=device,
                    eval_batch_size=args.eval_batch_size,
                    ece_bins=args.ece_bins,
                    base_seed=args.base_seed,
                )
                run_rows.append(row)
            alpha_rows = [
                row
                for row in run_rows
                if row["run_name"] == run_dir.name and row["noise_alpha"] == noise_alpha
            ]
            acc_mean = mean(row["noise_acc"] for row in alpha_rows)
            drop_mean = mean(row["acc_drop_vs_clean"] for row in alpha_rows)
            logger.info(
                "  alpha=%.3f | clean=%.4f noisy=%.4f drop=%.4f",
                noise_alpha,
                alpha_rows[0]["clean_acc"],
                acc_mean,
                drop_mean,
            )

    alpha_rows = summarize_alpha_groups(run_rows)
    overall_rows = summarize_overall_groups(run_rows)

    write_csv(output_dir / f"feature_noise_{args.checkpoint}_runs.csv", run_rows)
    write_csv(
        output_dir / f"feature_noise_{args.checkpoint}_alpha_groups.csv", alpha_rows
    )
    write_csv(
        output_dir / f"feature_noise_{args.checkpoint}_overall_groups.csv", overall_rows
    )
    write_json(output_dir / f"feature_noise_{args.checkpoint}_runs.json", run_rows)
    write_json(
        output_dir / f"feature_noise_{args.checkpoint}_alpha_groups.json", alpha_rows
    )
    write_json(
        output_dir / f"feature_noise_{args.checkpoint}_overall_groups.json",
        overall_rows,
    )
    logger.info(
        "Wrote %s", output_dir / f"feature_noise_{args.checkpoint}_overall_groups.csv"
    )


if __name__ == "__main__":
    main()
