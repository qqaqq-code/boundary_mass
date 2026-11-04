"Evaluate fitted probes under image distribution shifts. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations

import argparse
import csv
import io
import json
import logging
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets

from train_cifar100_probe import (
    LinearHead as CifarLinearHead,
    build_backbone as build_cifar_backbone,
)
from experiments.evaluate_probes import (
    detect_dataset_kind,
    evaluate_psieff,
    find_run_dirs,
    load_checkpoint_path,
    load_features_with_fallback,
    read_json,
    resolve_maybe_relative,
    safe_float,
    split_indices,
)
from train_imagefolder_probe import (
    LinearHead as ImageFolderLinearHead,
    build_backbone as build_imagefolder_backbone,
)
from utils.metrics import evaluate_classifier


SHIFT_FEATURE_MEMO: Dict[str, Tuple[torch.Tensor, torch.Tensor, int, int]] = {}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen linear probes under image-space shift."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", choices=["best", "last"], default="best")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument(
        "--corruptions", type=str, default="gaussian_noise,gaussian_blur,contrast,jpeg"
    )
    parser.add_argument("--severities", type=str, default="2")
    parser.add_argument("--run-name-contains", type=str, default="")
    return parser.parse_args()


def build_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(str(output_dir))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "shift_eval.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def parse_csv_tokens(raw: str, cast_fn) -> List:
    values: List = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
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


def pil_gaussian_noise(image: Image.Image, severity: int, seed: int) -> Image.Image:
    sigma_map = {1: 8.0, 2: 16.0, 3: 24.0}
    sigma = sigma_map.get(severity, 16.0)
    rng = np.random.default_rng(seed)
    array = np.asarray(image).astype(np.float32)
    noisy = np.clip(
        array + rng.normal(0.0, sigma, size=array.shape), 0.0, 255.0
    ).astype(np.uint8)
    return Image.fromarray(noisy)


def pil_gaussian_blur(image: Image.Image, severity: int) -> Image.Image:
    radius_map = {1: 0.8, 2: 1.5, 3: 2.5}
    return image.filter(ImageFilter.GaussianBlur(radius=radius_map.get(severity, 1.5)))


def pil_contrast(image: Image.Image, severity: int) -> Image.Image:
    factor_map = {1: 0.75, 2: 0.5, 3: 0.3}
    return ImageEnhance.Contrast(image).enhance(factor_map.get(severity, 0.5))


def pil_jpeg(image: Image.Image, severity: int) -> Image.Image:
    quality_map = {1: 50, 2: 25, 3: 10}
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=quality_map.get(severity, 25))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def apply_corruption(
    image: Image.Image, corruption: str, severity: int, seed: int
) -> Image.Image:
    if corruption == "gaussian_noise":
        return pil_gaussian_noise(image, severity=severity, seed=seed)
    if corruption == "gaussian_blur":
        return pil_gaussian_blur(image, severity=severity)
    if corruption == "contrast":
        return pil_contrast(image, severity=severity)
    if corruption == "jpeg":
        return pil_jpeg(image, severity=severity)
    raise ValueError(f"Unsupported corruption: {corruption}")


class CorruptedVisionDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        post_transform,
        corruption: str,
        severity: int,
        base_seed: int = 2026,
    ) -> None:
        self.base_dataset = base_dataset
        self.post_transform = post_transform
        self.corruption = corruption
        self.severity = severity
        self.base_seed = base_seed

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int):
        image, label = self.base_dataset[index]
        image = apply_corruption(
            image=image,
            corruption=self.corruption,
            severity=self.severity,
            seed=self.base_seed + 10007 * index,
        )
        return self.post_transform(image), label


@torch.no_grad()
def extract_features(
    backbone: nn.Module,
    dataset: Dataset,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    feature_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        features = backbone(images)
        feature_chunks.append(features.cpu())
        label_chunks.append(labels.cpu())
    return torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0)


def feature_cache_path(
    output_dir: Path,
    dataset_kind: str,
    backbone_model: str,
    weights: str,
    corruption: str,
    severity: int,
) -> Path:
    safe_weights = weights.replace("/", "_")
    return (
        output_dir
        / "shift_feature_cache"
        / f"{dataset_kind}__{backbone_model}__{safe_weights}__{corruption}__s{severity}.pt"
    )


def load_or_compute_shift_features(
    run_dir: Path,
    args_dict: Dict,
    dataset_kind: str,
    corruption: str,
    severity: int,
    device: torch.device,
    eval_batch_size: int,
    num_workers: int,
    output_dir: Path,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    cache_key = "||".join(
        [
            dataset_kind,
            str(args_dict.get("backbone_model")),
            str(args_dict.get("weights")),
            corruption,
            str(severity),
        ]
    )
    if cache_key in SHIFT_FEATURE_MEMO:
        return SHIFT_FEATURE_MEMO[cache_key]

    cache_path = feature_cache_path(
        output_dir=output_dir,
        dataset_kind=dataset_kind,
        backbone_model=str(args_dict.get("backbone_model")),
        weights=str(args_dict.get("weights", "")),
        corruption=corruption,
        severity=severity,
    )
    if cache_path.exists():
        blob = torch.load(cache_path, map_location="cpu", weights_only=False)
        result = (
            blob["features"].float(),
            blob["labels"].long(),
            int(blob["num_classes"]),
            int(blob["feature_dim"]),
        )
        SHIFT_FEATURE_MEMO[cache_key] = result
        return result

    if dataset_kind == "cifar100_pretrained":
        namespace = argparse.Namespace(**args_dict)
        namespace.data_dir = resolve_maybe_relative(args_dict["data_dir"], run_dir)
        backbone, feature_dim, post_transform, _ = build_cifar_backbone(
            namespace, device
        )
        raw_dataset = datasets.CIFAR100(
            root=namespace.data_dir, train=False, transform=None, download=False
        )
        shifted_dataset = CorruptedVisionDataset(
            base_dataset=raw_dataset,
            post_transform=post_transform,
            corruption=corruption,
            severity=severity,
        )
        features, labels = extract_features(
            backbone, shifted_dataset, device, eval_batch_size, num_workers
        )
        num_classes = 100
    else:
        namespace = argparse.Namespace(**args_dict)
        namespace.val_dir = resolve_maybe_relative(args_dict["val_dir"], run_dir)
        backbone, feature_dim, post_transform = build_imagefolder_backbone(
            namespace, device
        )
        raw_dataset = datasets.ImageFolder(root=namespace.val_dir, transform=None)
        shifted_dataset = CorruptedVisionDataset(
            base_dataset=raw_dataset,
            post_transform=post_transform,
            corruption=corruption,
            severity=severity,
        )
        features, labels = extract_features(
            backbone, shifted_dataset, device, eval_batch_size, num_workers
        )
        num_classes = len(raw_dataset.classes)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "features": features,
            "labels": labels,
            "num_classes": num_classes,
            "feature_dim": feature_dim,
            "corruption": corruption,
            "severity": severity,
        },
        cache_path,
    )
    result = (features, labels, num_classes, feature_dim)
    SHIFT_FEATURE_MEMO[cache_key] = result
    return result


def make_loader(
    features: torch.Tensor, labels: torch.Tensor, batch_size: int
) -> DataLoader:
    return DataLoader(
        TensorDataset(features, labels),
        batch_size=batch_size,
        shuffle=False,
        pin_memory=True,
    )


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


def mean_std(values: Iterable[float]) -> Tuple[Optional[float], Optional[float]]:
    values = list(values)
    if not values:
        return None, None
    if len(values) == 1:
        return mean(values), 0.0
    return mean(values), pstdev(values)


def summarize_condition_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        key = (
            make_base_group(row)
            + f"|corruption={row['corruption']}|severity={row['severity']}"
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: List[Dict] = []
    for group_key, items in sorted(grouped.items()):
        first = items[0]
        acc_mean, acc_std = mean_std([row["shift_acc"] for row in items])
        nll_mean, nll_std = mean_std([row["shift_nll"] for row in items])
        ece_mean, ece_std = mean_std([row["shift_ece"] for row in items])
        brier_mean, brier_std = mean_std([row["shift_brier"] for row in items])
        aurc_mean, aurc_std = mean_std([row["shift_aurc"] for row in items])
        drop_mean, drop_std = mean_std([row["acc_drop_vs_clean"] for row in items])
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
                "corruption": first["corruption"],
                "severity": first["severity"],
                "num_runs": len(items),
                "shift_acc_mean": acc_mean,
                "shift_acc_std": acc_std,
                "acc_drop_vs_clean_mean": drop_mean,
                "acc_drop_vs_clean_std": drop_std,
                "shift_nll_mean": nll_mean,
                "shift_nll_std": nll_std,
                "shift_ece_mean": ece_mean,
                "shift_ece_std": ece_std,
                "shift_brier_mean": brier_mean,
                "shift_brier_std": brier_std,
                "shift_aurc_mean": aurc_mean,
                "shift_aurc_std": aurc_std,
            }
        )
    return summary_rows


def summarize_overall_rows(rows: List[Dict]) -> List[Dict]:
    grouped: Dict[str, List[Dict]] = {}
    for row in rows:
        grouped.setdefault(make_base_group(row), []).append(row)

    summary_rows: List[Dict] = []
    for group_key, items in sorted(grouped.items()):
        first = items[0]
        shift_acc_mean, shift_acc_std = mean_std([row["shift_acc"] for row in items])
        drop_mean, drop_std = mean_std([row["acc_drop_vs_clean"] for row in items])
        shift_ece_mean, shift_ece_std = mean_std([row["shift_ece"] for row in items])
        shift_aurc_mean, shift_aurc_std = mean_std([row["shift_aurc"] for row in items])
        clean_acc_mean, clean_acc_std = mean_std([row["clean_acc"] for row in items])
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
                "shift_acc_mean_over_conditions": shift_acc_mean,
                "shift_acc_std_over_conditions": shift_acc_std,
                "acc_drop_vs_clean_mean_over_conditions": drop_mean,
                "acc_drop_vs_clean_std_over_conditions": drop_std,
                "shift_ece_mean_over_conditions": shift_ece_mean,
                "shift_ece_std_over_conditions": shift_ece_std,
                "shift_aurc_mean_over_conditions": shift_aurc_mean,
                "shift_aurc_std_over_conditions": shift_aurc_std,
            }
        )
    return summary_rows


def evaluate_shifted_run(
    run_dir: Path,
    corruption: str,
    severity: int,
    checkpoint_tag: str,
    device: torch.device,
    eval_batch_size: int,
    num_workers: int,
    ece_bins: int,
    output_dir: Path,
) -> Dict:
    args_dict = read_json(run_dir / "args.json")
    summary = (
        read_json(run_dir / "summary.json")
        if (run_dir / "summary.json").exists()
        else {}
    )
    dataset_kind = detect_dataset_kind(args_dict)

    (
        clean_train_features_all,
        clean_train_labels_all,
        clean_eval_features,
        clean_eval_labels,
        num_classes,
        feature_dim,
    ) = load_features_with_fallback(
        run_dir=run_dir,
        args_dict=args_dict,
        device=device,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
    )
    _, holdout_indices = split_indices(
        num_examples=int(clean_train_features_all.shape[0]),
        holdout_size=int(args_dict["holdout_size"]),
        seed=int(args_dict["seed"]),
    )
    holdout_features = clean_train_features_all[holdout_indices]
    (
        shifted_eval_features,
        shifted_eval_labels,
        shifted_num_classes,
        shifted_feature_dim,
    ) = load_or_compute_shift_features(
        run_dir=run_dir,
        args_dict=args_dict,
        dataset_kind=dataset_kind,
        corruption=corruption,
        severity=severity,
        device=device,
        eval_batch_size=eval_batch_size,
        num_workers=num_workers,
        output_dir=output_dir,
    )

    if shifted_num_classes != num_classes or shifted_feature_dim != feature_dim:
        raise ValueError(
            f"Shift feature metadata mismatch for {run_dir}: clean=({num_classes}, {feature_dim}) "
            f"shift=({shifted_num_classes}, {shifted_feature_dim})"
        )

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
    clean_loader = make_loader(
        clean_eval_features, clean_eval_labels, batch_size=eval_batch_size
    )
    shift_loader = make_loader(
        shifted_eval_features, shifted_eval_labels, batch_size=eval_batch_size
    )
    clean_metrics = evaluate_classifier(
        head=head,
        loader=clean_loader,
        device=device,
        criterion=criterion,
        num_bins=ece_bins,
    )
    shift_metrics = evaluate_classifier(
        head=head,
        loader=shift_loader,
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
        "corruption": corruption,
        "severity": severity,
        "clean_acc": clean_metrics.acc,
        "shift_acc": shift_metrics.acc,
        "acc_drop_vs_clean": clean_metrics.acc - shift_metrics.acc,
        "shift_nll": shift_metrics.nll,
        "shift_ece": shift_metrics.ece,
        "shift_brier": shift_metrics.brier,
        "shift_aurc": shift_metrics.aurc,
        "shift_mean_confidence": shift_metrics.mean_confidence,
        "heldout_psieff": heldout_psieff,
    }


def main() -> None:
    args = parse_args()
    output_dir = (args.output_dir or (args.run_root / "summary_reports")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_logger(output_dir)

    device = torch.device(args.device)
    corruptions = parse_csv_tokens(args.corruptions, str)
    severities = parse_csv_tokens(args.severities, int)
    run_dirs = find_run_dirs(args.run_root.resolve())
    if args.run_name_contains:
        run_dirs = [
            run_dir for run_dir in run_dirs if args.run_name_contains in run_dir.name
        ]

    logger.info("Found %d run directories under %s", len(run_dirs), args.run_root)
    logger.info(
        "Corruptions=%s | Severities=%s | Device=%s", corruptions, severities, device
    )

    run_rows: List[Dict] = []
    for run_index, run_dir in enumerate(run_dirs, start=1):
        logger.info("[%d/%d] Evaluating %s", run_index, len(run_dirs), run_dir.name)
        for corruption in corruptions:
            for severity in severities:
                row = evaluate_shifted_run(
                    run_dir=run_dir,
                    corruption=corruption,
                    severity=severity,
                    checkpoint_tag=args.checkpoint,
                    device=device,
                    eval_batch_size=args.eval_batch_size,
                    num_workers=args.num_workers,
                    ece_bins=args.ece_bins,
                    output_dir=output_dir,
                )
                run_rows.append(row)
                logger.info(
                    "  %s s=%d | clean=%.4f shift=%.4f drop=%.4f",
                    corruption,
                    severity,
                    row["clean_acc"],
                    row["shift_acc"],
                    row["acc_drop_vs_clean"],
                )

    condition_rows = summarize_condition_rows(run_rows)
    overall_rows = summarize_overall_rows(run_rows)

    write_csv(output_dir / f"shift_{args.checkpoint}_runs.csv", run_rows)
    write_csv(
        output_dir / f"shift_{args.checkpoint}_condition_groups.csv", condition_rows
    )
    write_csv(output_dir / f"shift_{args.checkpoint}_overall_groups.csv", overall_rows)
    write_json(output_dir / f"shift_{args.checkpoint}_runs.json", run_rows)
    write_json(
        output_dir / f"shift_{args.checkpoint}_condition_groups.json", condition_rows
    )
    write_json(
        output_dir / f"shift_{args.checkpoint}_overall_groups.json", overall_rows
    )

    logger.info("Wrote %s", output_dir / f"shift_{args.checkpoint}_overall_groups.csv")


if __name__ == "__main__":
    main()
