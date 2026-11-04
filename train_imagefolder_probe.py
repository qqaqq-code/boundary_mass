"Train frozen pretrained probes on ImageFolder datasets. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, models

from train_encoder import psieff_surrogate
from utils.metrics import ClassifierMetrics, evaluate_classifier


@dataclass
class EpochMetrics:
    epoch: int
    lr: float
    train_loss: float
    train_ce: float
    train_reg: float
    train_acc: float
    val_loss: float
    val_acc: float
    val_nll: float
    val_brier: float
    val_ece: float
    val_aurc: float
    val_mean_confidence: float
    heldout_psieff: float
    epoch_seconds: float


class LinearHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(in_dim, num_classes)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.classifier(features)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train frozen-backbone linear probes with psi_eff regularization on ImageFolder datasets."
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--backbone-model", choices=["resnet50"], default="resnet50")
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V2")
    parser.add_argument(
        "--mode", choices=["standard", "weight_decay", "psieff"], default="standard"
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-psieff", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=1e-6)
    parser.add_argument("--eps0", type=float, default=0.1)
    parser.add_argument("--holdout-size", type=int, default=4000)
    parser.add_argument("--train-samples-per-class", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--ece-bins", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_logger(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger(str(output_dir))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def save_json(path: Path, payload: Dict) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def append_jsonl(path: Path, payload: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def append_csv(path: Path, metrics: EpochMetrics) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(metrics).keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(asdict(metrics))


def build_backbone(
    args: argparse.Namespace, device: torch.device
) -> Tuple[nn.Module, int, object]:
    if args.backbone_model != "resnet50":
        raise ValueError(f"Unsupported backbone model: {args.backbone_model}")

    if not hasattr(models.ResNet50_Weights, args.weights):
        raise ValueError(f"Unknown weights enum for resnet50: {args.weights}")

    weights = getattr(models.ResNet50_Weights, args.weights)
    model = models.resnet50(weights=weights).to(device)
    feature_dim = model.fc.in_features
    model.fc = nn.Identity()
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, feature_dim, weights.transforms()


def build_datasets(
    args: argparse.Namespace, transform: object
) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    train_dataset = datasets.ImageFolder(root=args.train_dir, transform=transform)
    val_dataset = datasets.ImageFolder(root=args.val_dir, transform=transform)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Train/val class_to_idx mappings do not match.")
    return train_dataset, val_dataset


@torch.no_grad()
def extract_features(
    model: nn.Module,
    dataset: datasets.ImageFolder,
    indices: List[int],
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )

    feature_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        features = model(images)
        feature_chunks.append(features.cpu())
        label_chunks.append(targets.cpu())
    return torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0)


def load_or_compute_feature_cache(
    backbone: nn.Module,
    train_dataset: datasets.ImageFolder,
    val_dataset: datasets.ImageFolder,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cache_dir: Path | None,
    feature_dim: int,
    num_classes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_cache = cache_dir / "train_features.pt" if cache_dir is not None else None
    val_cache = cache_dir / "val_features.pt" if cache_dir is not None else None

    if (
        train_cache is not None
        and val_cache is not None
        and train_cache.exists()
        and val_cache.exists()
    ):
        train_blob = torch.load(train_cache, map_location="cpu")
        val_blob = torch.load(val_cache, map_location="cpu")
        return (
            train_blob["features"].float(),
            train_blob["labels"].long(),
            val_blob["features"].float(),
            val_blob["labels"].long(),
        )

    train_features, train_labels = extract_features(
        backbone,
        train_dataset,
        list(range(len(train_dataset))),
        device,
        batch_size,
        num_workers,
    )
    val_features, val_labels = extract_features(
        backbone,
        val_dataset,
        list(range(len(val_dataset))),
        device,
        batch_size,
        num_workers,
    )

    if train_cache is not None and val_cache is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        train_payload = {
            "features": train_features,
            "labels": train_labels,
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "num_examples": len(train_dataset),
        }
        val_payload = {
            "features": val_features,
            "labels": val_labels,
            "feature_dim": feature_dim,
            "num_classes": num_classes,
            "num_examples": len(val_dataset),
        }
        torch.save(train_payload, train_cache)
        torch.save(val_payload, val_cache)

    return train_features, train_labels, val_features, val_labels


def split_indices(
    num_examples: int, holdout_size: int, seed: int
) -> Tuple[List[int], List[int]]:
    if holdout_size <= 0 or holdout_size >= num_examples:
        raise ValueError("--holdout-size must be between 1 and len(train_dataset)-1")
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_examples, generator=generator).tolist()
    holdout_indices = indices[:holdout_size]
    train_indices = indices[holdout_size:]
    return train_indices, holdout_indices


def select_balanced_train_indices(
    train_indices: List[int],
    train_labels: torch.Tensor,
    samples_per_class: int,
    seed: int,
    num_classes: int,
) -> List[int]:
    if samples_per_class <= 0:
        return train_indices

    rng = random.Random(seed)
    by_class: Dict[int, List[int]] = {c: [] for c in range(num_classes)}
    for idx in train_indices:
        by_class[int(train_labels[idx].item())].append(idx)

    selected: List[int] = []
    for c in range(num_classes):
        candidates = by_class[c]
        if len(candidates) < samples_per_class:
            raise ValueError(
                f"class {c} has only {len(candidates)} train examples after holdout; need {samples_per_class}"
            )
        rng.shuffle(candidates)
        selected.extend(candidates[:samples_per_class])

    rng.shuffle(selected)
    return selected


def make_loader(
    features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


@torch.no_grad()
def evaluate(
    head: LinearHead,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    num_bins: int,
) -> ClassifierMetrics:
    return evaluate_classifier(
        head=head, loader=loader, device=device, criterion=criterion, num_bins=num_bins
    )


@torch.no_grad()
def evaluate_psieff(
    head: LinearHead,
    holdout_features: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    head.eval()
    features = holdout_features.to(device, non_blocking=True)
    value, _ = psieff_surrogate(
        features=features,
        weight=head.classifier.weight,
        bias=head.classifier.bias,
        tau=args.tau,
        rho=args.rho,
        eps0=args.eps0,
        reduction="sum",
    )
    return float(value.item())


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    logger = build_logger(output_dir)
    save_json(
        output_dir / "args.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )

    device = torch.device(args.device)
    backbone, feature_dim, transform = build_backbone(args, device)
    train_dataset, val_dataset = build_datasets(args, transform)
    num_classes = len(train_dataset.classes)

    cache_dir = (
        args.feature_cache_dir.resolve() if args.feature_cache_dir is not None else None
    )

    logger.info(
        "Extracting frozen %s features with weights=%s",
        args.backbone_model,
        args.weights,
    )
    train_features_all, train_labels_all, val_features, val_labels = (
        load_or_compute_feature_cache(
            backbone=backbone,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            cache_dir=cache_dir,
            feature_dim=feature_dim,
            num_classes=num_classes,
        )
    )

    train_indices, holdout_indices = split_indices(
        len(train_dataset), args.holdout_size, args.seed
    )
    selected_train_indices = select_balanced_train_indices(
        train_indices=train_indices,
        train_labels=train_labels_all,
        samples_per_class=args.train_samples_per_class,
        seed=args.seed,
        num_classes=num_classes,
    )

    train_features = train_features_all[selected_train_indices]
    train_labels = train_labels_all[selected_train_indices]
    holdout_features = train_features_all[holdout_indices]

    train_loader = make_loader(
        train_features, train_labels, args.batch_size, shuffle=True
    )
    val_loader = make_loader(
        val_features, val_labels, args.eval_batch_size, shuffle=False
    )
    holdout_features_device = holdout_features.to(device, non_blocking=True)

    head = LinearHead(in_dim=feature_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        head.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay if args.mode == "weight_decay" else 0.0,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = -math.inf
    best_epoch = -1
    best_metrics: EpochMetrics | None = None
    final_metrics: EpochMetrics | None = None
    best_path = checkpoints_dir / "best.pt"
    last_path = checkpoints_dir / "last.pt"

    logger.info(
        "Dataset stats: classes=%d train=%d holdout=%d val=%d feature_dim=%d",
        num_classes,
        train_features.shape[0],
        holdout_features.shape[0],
        val_features.shape[0],
        feature_dim,
    )
    logger.info("Mode: %s", args.mode)
    logger.info(
        "Train samples per class: %s",
        args.train_samples_per_class if args.train_samples_per_class > 0 else "all",
    )

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        head.train()

        train_loss = 0.0
        train_ce = 0.0
        train_reg = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = head(features)
            ce_loss = criterion(logits, labels)

            reg_loss = torch.tensor(0.0, device=device)
            if args.mode == "psieff":
                reg_loss, _ = psieff_surrogate(
                    features=holdout_features_device,
                    weight=head.classifier.weight,
                    bias=head.classifier.bias,
                    tau=args.tau,
                    rho=args.rho,
                    eps0=args.eps0,
                    reduction="sum",
                )

            loss = ce_loss + args.lambda_psieff * reg_loss
            loss.backward()
            optimizer.step()

            batch_size = labels.size(0)
            train_loss += loss.item() * batch_size
            train_ce += ce_loss.item() * batch_size
            train_reg += reg_loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += batch_size

        scheduler.step()

        val_metrics = evaluate(
            head, val_loader, device, criterion, num_bins=args.ece_bins
        )
        heldout_psieff = evaluate_psieff(head, holdout_features, device, args)
        epoch_seconds = time.time() - start_time

        metrics = EpochMetrics(
            epoch=epoch,
            lr=float(optimizer.param_groups[0]["lr"]),
            train_loss=train_loss / train_total,
            train_ce=train_ce / train_total,
            train_reg=train_reg / train_total,
            train_acc=train_correct / train_total,
            val_loss=val_metrics.loss,
            val_acc=val_metrics.acc,
            val_nll=val_metrics.nll,
            val_brier=val_metrics.brier,
            val_ece=val_metrics.ece,
            val_aurc=val_metrics.aurc,
            val_mean_confidence=val_metrics.mean_confidence,
            heldout_psieff=heldout_psieff,
            epoch_seconds=epoch_seconds,
        )
        final_metrics = metrics

        append_jsonl(output_dir / "metrics.jsonl", asdict(metrics))
        append_csv(output_dir / "metrics.csv", metrics)

        checkpoint = {
            "epoch": epoch,
            "head_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metrics": asdict(metrics),
            "args": vars(args),
            "classes": train_dataset.classes,
        }
        torch.save(checkpoint, last_path)
        if metrics.val_acc > best_acc:
            best_acc = metrics.val_acc
            best_epoch = epoch
            best_metrics = metrics
            torch.save(checkpoint, best_path)

        logger.info(
            "epoch=%03d lr=%.6f train_loss=%.4f train_ce=%.4f train_reg=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f val_ece=%.4f val_aurc=%.4f heldout_psieff=%.4f time=%.1fs",
            metrics.epoch,
            metrics.lr,
            metrics.train_loss,
            metrics.train_ce,
            metrics.train_reg,
            metrics.train_acc,
            metrics.val_loss,
            metrics.val_acc,
            metrics.val_ece,
            metrics.val_aurc,
            metrics.heldout_psieff,
            metrics.epoch_seconds,
        )

    if best_metrics is None or final_metrics is None:
        raise RuntimeError("Training finished without recording metrics.")

    summary = {
        "best_epoch": best_epoch,
        "best_val_acc": best_acc,
        "best_val_loss": best_metrics.val_loss,
        "best_val_nll": best_metrics.val_nll,
        "best_val_brier": best_metrics.val_brier,
        "best_val_ece": best_metrics.val_ece,
        "best_val_aurc": best_metrics.val_aurc,
        "best_val_mean_confidence": best_metrics.val_mean_confidence,
        "best_heldout_psieff": best_metrics.heldout_psieff,
        "final_epoch": args.epochs,
        "final_val_acc": final_metrics.val_acc,
        "final_val_loss": final_metrics.val_loss,
        "final_val_nll": final_metrics.val_nll,
        "final_val_brier": final_metrics.val_brier,
        "final_val_ece": final_metrics.val_ece,
        "final_val_aurc": final_metrics.val_aurc,
        "final_val_mean_confidence": final_metrics.val_mean_confidence,
        "final_heldout_psieff": final_metrics.heldout_psieff,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "mode": args.mode,
        "seed": args.seed,
        "device": str(device),
        "backbone_model": args.backbone_model,
        "weights": args.weights,
        "train_samples_per_class": args.train_samples_per_class,
        "num_classes": num_classes,
        "num_train_examples": int(train_features.shape[0]),
        "num_holdout_examples": int(holdout_features.shape[0]),
        "num_val_examples": int(val_features.shape[0]),
    }
    save_json(output_dir / "summary.json", summary)
    logger.info("Finished. Best epoch=%d best_val_acc=%.4f", best_epoch, best_acc)


if __name__ == "__main__":
    main()
