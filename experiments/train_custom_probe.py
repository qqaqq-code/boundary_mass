"Train a linear probe on a learned CIFAR-100 encoder. See docs/EXPERIMENTS.md for usage."

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
from torchvision import datasets

from train_encoder import CIFARResNet18, default_transforms, psieff_surrogate


@dataclass
class EpochMetrics:
    epoch: int
    lr: float
    train_loss: float
    train_ce: float
    train_reg: float
    train_acc: float
    test_loss: float
    test_acc: float
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
        description="Train linear probes with psi_eff regularization on frozen CIFAR-100 features."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--backbone-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument(
        "--mode", choices=["standard", "weight_decay", "psieff"], default="standard"
    )
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lambda-psieff", type=float, default=1e-4)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=1e-6)
    parser.add_argument("--eps0", type=float, default=0.1)
    parser.add_argument("--holdout-size", type=int, default=2048)
    parser.add_argument("--train-samples-per-class", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--download", action="store_true")
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


def load_backbone(args: argparse.Namespace, device: torch.device) -> CIFARResNet18:
    model = CIFARResNet18(num_classes=100).to(device)
    checkpoint = torch.load(
        args.backbone_checkpoint, map_location=device, weights_only=False
    )
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_datasets(
    args: argparse.Namespace,
) -> Tuple[datasets.CIFAR100, datasets.CIFAR100]:
    _, eval_transform = default_transforms()
    train_dataset = datasets.CIFAR100(
        root=args.data_dir, train=True, transform=eval_transform, download=args.download
    )
    test_dataset = datasets.CIFAR100(
        root=args.data_dir,
        train=False,
        transform=eval_transform,
        download=args.download,
    )
    return train_dataset, test_dataset


@torch.no_grad()
def extract_features(
    model: CIFARResNet18,
    dataset: datasets.CIFAR100,
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
        features = model.backbone(images)
        feature_chunks.append(features.cpu())
        label_chunks.append(targets.cpu())
    return torch.cat(feature_chunks, dim=0), torch.cat(label_chunks, dim=0)


def load_or_compute_feature_cache(
    backbone: CIFARResNet18,
    train_dataset: datasets.CIFAR100,
    test_dataset: datasets.CIFAR100,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    cache_dir: Path | None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_cache = cache_dir / "train_features.pt" if cache_dir is not None else None
    test_cache = cache_dir / "test_features.pt" if cache_dir is not None else None

    if (
        train_cache is not None
        and test_cache is not None
        and train_cache.exists()
        and test_cache.exists()
    ):
        train_blob = torch.load(train_cache, map_location="cpu")
        test_blob = torch.load(test_cache, map_location="cpu")
        return (
            train_blob["features"].float(),
            train_blob["labels"].long(),
            test_blob["features"].float(),
            test_blob["labels"].long(),
        )

    train_features, train_labels = extract_features(
        backbone,
        train_dataset,
        list(range(len(train_dataset))),
        device,
        batch_size,
        num_workers,
    )
    test_features, test_labels = extract_features(
        backbone,
        test_dataset,
        list(range(len(test_dataset))),
        device,
        batch_size,
        num_workers,
    )

    if train_cache is not None and test_cache is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"features": train_features, "labels": train_labels}, train_cache)
        torch.save({"features": test_features, "labels": test_labels}, test_cache)

    return train_features, train_labels, test_features, test_labels


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
    num_classes: int = 100,
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
                f"class {c} has only {len(candidates)} samples after holdout; need {samples_per_class}"
            )
        rng.shuffle(candidates)
        selected.extend(candidates[:samples_per_class])

    rng.shuffle(selected)
    return selected


def make_loader(
    features: torch.Tensor, labels: torch.Tensor, batch_size: int, shuffle: bool
) -> DataLoader:
    dataset = TensorDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


@torch.no_grad()
def evaluate(
    head: LinearHead, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> Tuple[float, float]:
    head.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = head(features)
        loss = criterion(logits, labels)
        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)
    return total_loss / total, total_correct / total


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
    backbone = load_backbone(args, device)
    train_dataset, test_dataset = build_datasets(args)

    cache_dir = None
    if args.feature_cache_dir is not None:
        cache_dir = args.feature_cache_dir.resolve()

    logger.info("Extracting frozen features from %s", args.backbone_checkpoint)
    train_features_all, train_labels_all, test_features, test_labels = (
        load_or_compute_feature_cache(
            backbone=backbone,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            device=device,
            batch_size=args.eval_batch_size,
            num_workers=args.num_workers,
            cache_dir=cache_dir,
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
    )

    train_features = train_features_all[selected_train_indices]
    train_labels = train_labels_all[selected_train_indices]
    holdout_features = train_features_all[holdout_indices]

    train_loader = make_loader(
        train_features, train_labels, args.batch_size, shuffle=True
    )
    test_loader = make_loader(
        test_features, test_labels, args.eval_batch_size, shuffle=False
    )
    holdout_features_device = holdout_features.to(device, non_blocking=True)

    in_dim = train_features.shape[1]
    head = LinearHead(in_dim=in_dim, num_classes=100).to(device)
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
    best_path = checkpoints_dir / "best.pt"
    last_path = checkpoints_dir / "last.pt"

    logger.info(
        "Feature shapes: train=%s holdout=%s test=%s",
        tuple(train_features.shape),
        tuple(holdout_features.shape),
        tuple(test_features.shape),
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

        test_loss, test_acc = evaluate(head, test_loader, device, criterion)
        heldout_psieff = evaluate_psieff(head, holdout_features, device, args)
        epoch_seconds = time.time() - start_time

        metrics = EpochMetrics(
            epoch=epoch,
            lr=float(optimizer.param_groups[0]["lr"]),
            train_loss=train_loss / train_total,
            train_ce=train_ce / train_total,
            train_reg=train_reg / train_total,
            train_acc=train_correct / train_total,
            test_loss=test_loss,
            test_acc=test_acc,
            heldout_psieff=heldout_psieff,
            epoch_seconds=epoch_seconds,
        )

        append_jsonl(output_dir / "metrics.jsonl", asdict(metrics))
        append_csv(output_dir / "metrics.csv", metrics)

        checkpoint = {
            "epoch": epoch,
            "head_state": head.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(),
            "metrics": asdict(metrics),
            "args": vars(args),
        }
        torch.save(checkpoint, last_path)
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(checkpoint, best_path)

        logger.info(
            "epoch=%03d lr=%.6f train_loss=%.4f train_ce=%.4f train_reg=%.4f train_acc=%.4f test_loss=%.4f test_acc=%.4f heldout_psieff=%.4f time=%.1fs",
            metrics.epoch,
            metrics.lr,
            metrics.train_loss,
            metrics.train_ce,
            metrics.train_reg,
            metrics.train_acc,
            metrics.test_loss,
            metrics.test_acc,
            metrics.heldout_psieff,
            metrics.epoch_seconds,
        )

    summary = {
        "best_epoch": best_epoch,
        "best_test_acc": best_acc,
        "final_epoch": args.epochs,
        "best_checkpoint": str(best_path),
        "last_checkpoint": str(last_path),
        "mode": args.mode,
        "seed": args.seed,
        "device": str(device),
        "backbone_checkpoint": str(args.backbone_checkpoint),
        "train_samples_per_class": args.train_samples_per_class,
        "num_train_examples": int(train_features.shape[0]),
        "num_holdout_examples": int(holdout_features.shape[0]),
    }
    save_json(output_dir / "summary.json", summary)
    logger.info("Finished. Best epoch=%d best_test_acc=%.4f", best_epoch, best_acc)


if __name__ == "__main__":
    main()
