"Train CIFAR-100 encoders with boundary-mass regularization. See docs/EXPERIMENTS.md for usage."

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
from typing import Dict, Iterable, Iterator, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms


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


class CIFARResNet18(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward(
        self, x: torch.Tensor, return_features: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        logits = self.classifier(features)
        if return_features:
            return logits, features
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CIFAR-100 baselines and psi_eff-regularized models."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--mode", choices=["standard", "weight_decay", "psieff"], default="standard"
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--lambda-psieff", type=float, default=1e-3)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=1e-6)
    parser.add_argument("--eps0", type=float, default=0.1)
    parser.add_argument("--holdout-size", type=int, default=2048)
    parser.add_argument("--reg-batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


def make_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        return args.output_dir.resolve()
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"cifar100_{args.mode}_seed{args.seed}_{timestamp}"
    return (Path("runs") / name).resolve()


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


def default_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    mean = (0.5071, 0.4867, 0.4408)
    std = (0.2675, 0.2565, 0.2761)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    return train_transform, eval_transform


def build_dataloaders(args: argparse.Namespace) -> Dict[str, DataLoader]:
    train_transform, eval_transform = default_transforms()
    train_aug = datasets.CIFAR100(
        root=args.data_dir,
        train=True,
        transform=train_transform,
        download=args.download,
    )
    train_eval = datasets.CIFAR100(
        root=args.data_dir, train=True, transform=eval_transform, download=args.download
    )
    test_dataset = datasets.CIFAR100(
        root=args.data_dir,
        train=False,
        transform=eval_transform,
        download=args.download,
    )

    if args.holdout_size <= 0 or args.holdout_size >= len(train_aug):
        raise ValueError("--holdout-size must be between 1 and len(train_dataset)-1")

    generator = torch.Generator().manual_seed(args.seed)
    indices = torch.randperm(len(train_aug), generator=generator).tolist()
    holdout_indices = indices[: args.holdout_size]
    train_indices = indices[args.holdout_size :]

    train_dataset = Subset(train_aug, train_indices)
    holdout_dataset = Subset(train_eval, holdout_indices)

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        ),
        "holdout": DataLoader(
            holdout_dataset,
            batch_size=args.reg_batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        ),
        "holdout_eval": DataLoader(
            holdout_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        ),
        "test": DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True,
            persistent_workers=args.num_workers > 0,
        ),
    }
    return loaders


def infinite_loader(loader: DataLoader) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def smooth_balance(p_plus: torch.Tensor, rho: float) -> torch.Tensor:
    rho_tensor = torch.tensor(rho, dtype=p_plus.dtype, device=p_plus.device)
    return 0.5 * (
        1.0 - torch.sqrt((2.0 * p_plus - 1.0) ** 2 + rho_tensor**2) + rho_tensor
    )


def psieff_surrogate(
    features: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    tau: float,
    rho: float,
    eps0: float,
    reduction: str = "sum",
) -> Tuple[torch.Tensor, torch.Tensor]:
    weight_norm = weight.norm(dim=1).clamp_min(1e-12)
    margins = (features @ weight.t() + bias.unsqueeze(0)) / weight_norm.unsqueeze(0)
    tube = torch.sigmoid((eps0 - margins.abs()) / tau).mean(dim=0)
    p_plus = torch.sigmoid(margins / tau).mean(dim=0)
    balance = smooth_balance(p_plus, rho).clamp_min(1e-6)
    per_class = tube / (eps0 * balance)

    if reduction == "sum":
        value = per_class.sum()
    elif reduction == "mean":
        value = per_class.mean()
    else:
        raise ValueError(f"Unsupported reduction: {reduction}")
    return value, per_class


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total = 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, targets)
        total_loss += loss.item() * targets.size(0)
        total_correct += (logits.argmax(dim=1) == targets).sum().item()
        total += targets.size(0)
    return total_loss / total, total_correct / total


@torch.no_grad()
def evaluate_psieff(
    model: CIFARResNet18,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    model.eval()
    feature_chunks: List[torch.Tensor] = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        _, features = model(images, return_features=True)
        feature_chunks.append(features)
    features = torch.cat(feature_chunks, dim=0)
    value, _ = psieff_surrogate(
        features=features,
        weight=model.classifier.weight,
        bias=model.classifier.bias,
        tau=args.tau,
        rho=args.rho,
        eps0=args.eps0,
        reduction="sum",
    )
    return float(value.item())


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


def main() -> None:
    args = parse_args()
    output_dir = make_output_dir(args)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    logger = build_logger(output_dir)
    save_json(
        output_dir / "args.json",
        {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    )

    device = torch.device(args.device)
    loaders = build_dataloaders(args)

    model = CIFARResNet18(num_classes=100).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay if args.mode == "weight_decay" else 0.0,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    holdout_iter = infinite_loader(loaders["holdout"])

    best_acc = -math.inf
    best_epoch = -1
    best_path = checkpoints_dir / "best.pt"
    last_path = checkpoints_dir / "last.pt"

    logger.info("Output directory: %s", output_dir)
    logger.info("Device: %s", device)
    logger.info("Mode: %s", args.mode)
    logger.info(
        "Train batches: %d | Holdout batches: %d | Test batches: %d",
        len(loaders["train"]),
        len(loaders["holdout_eval"]),
        len(loaders["test"]),
    )

    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        model.train()

        train_loss = 0.0
        train_ce = 0.0
        train_reg = 0.0
        train_correct = 0
        train_total = 0

        for images, targets in loaders["train"]:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            ce_loss = criterion(logits, targets)

            reg_loss = torch.tensor(0.0, device=device)
            if args.mode == "psieff":
                reg_images, _ = next(holdout_iter)
                reg_images = reg_images.to(device, non_blocking=True)
                _, reg_features = model(reg_images, return_features=True)
                reg_loss, _ = psieff_surrogate(
                    features=reg_features,
                    weight=model.classifier.weight,
                    bias=model.classifier.bias,
                    tau=args.tau,
                    rho=args.rho,
                    eps0=args.eps0,
                    reduction="sum",
                )

            loss = ce_loss + args.lambda_psieff * reg_loss
            loss.backward()
            optimizer.step()

            batch_size = targets.size(0)
            train_loss += loss.item() * batch_size
            train_ce += ce_loss.item() * batch_size
            train_reg += reg_loss.item() * batch_size
            train_correct += (logits.argmax(dim=1) == targets).sum().item()
            train_total += batch_size

        scheduler.step()

        test_loss, test_acc = evaluate_classifier(
            model, loaders["test"], device, criterion
        )
        heldout_psieff = evaluate_psieff(model, loaders["holdout_eval"], device, args)
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
            "model_state": model.state_dict(),
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
    }
    save_json(output_dir / "summary.json", summary)
    logger.info("Finished. Best epoch=%d best_test_acc=%.4f", best_epoch, best_acc)


if __name__ == "__main__":
    main()
