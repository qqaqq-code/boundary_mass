"Train end-to-end interventions and downstream low-label probes. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


class BaseCNN(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.flatten = nn.Flatten()
        self.fc_last = nn.Sequential(nn.Linear(128 * 2 * 2, 256), nn.ReLU())
        self.classifier = nn.Linear(256, num_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(
                    module.weight, mode="fan_in", nonlinearity="relu"
                )
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.flatten(x)
        return self.fc_last(x)

    def get_features_and_logits(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        return features, self.classifier(features)


class CIFARResNet18(nn.Module):
    def __init__(self, num_classes: int = 100) -> None:
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        backbone.maxpool = nn.Identity()
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.features(x)
        return self.classifier(features)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def get_features_and_logits(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self.features(x)
        return features, self.classifier(features)


def build_model(arch: str, num_classes: int = 100) -> nn.Module:
    if arch == "compact":
        return BaseCNN(num_classes=num_classes)
    if arch == "resnet18":
        return CIFARResNet18(num_classes=num_classes)
    raise ValueError(f"unknown arch: {arch}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="End-to-end psi experiment plus downstream low-label probe."
    )
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dataset", type=str, default="cifar100", choices=["cifar100", "cifar10"]
    )
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument(
        "--arch", type=str, default="compact", choices=["compact", "resnet18"]
    )
    parser.add_argument("--modes", type=str, default="standard,weight_decay,psieff")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--probe-epochs", type=int, default=160)
    parser.add_argument("--probe-budgets", type=str, default="5,10,20,50")
    parser.add_argument("--probe-seeds", type=str, default="42,43,44")
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--psi-lr", type=float, default=3.5e-3)
    parser.add_argument("--lambda-psi", type=float, default=0.12)
    parser.add_argument("--lambda-margin", type=float, default=0.1)
    parser.add_argument("--lambda-entropy", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--psi-eps", type=float, default=0.1)
    parser.add_argument("--psi-samples", type=int, default=2000)
    parser.add_argument("--margin-eps", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def build_datasets(data_dir: Path, download: bool, dataset: str):
    if dataset == "cifar100":
        dataset_cls = datasets.CIFAR100
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
        num_classes = 100
    elif dataset == "cifar10":
        dataset_cls = datasets.CIFAR10
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2470, 0.2435, 0.2616)
        num_classes = 10
    else:
        raise ValueError(f"unknown dataset: {dataset}")
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    eval_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)]
    )
    train_aug = dataset_cls(
        root=data_dir, train=True, download=download, transform=train_transform
    )
    train_eval = dataset_cls(
        root=data_dir, train=True, download=download, transform=eval_transform
    )
    test_eval = dataset_cls(
        root=data_dir, train=False, download=download, transform=eval_transform
    )
    return train_aug, train_eval, test_eval, num_classes


def make_loader(
    dataset, batch_size: int, shuffle: bool, num_workers: int
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def compute_psi_surrogate_loss(
    features: torch.Tensor,
    logits: torch.Tensor,
    labels: torch.Tensor,
    weight: torch.Tensor,
    margin_eps: float = 0.2,
) -> torch.Tensor:
    correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    masked_logits = logits.masked_fill(
        F.one_hot(labels, num_classes=logits.shape[1]).bool(), -torch.inf
    )
    max_wrong_logits = masked_logits.max(dim=1).values
    margin = correct_logits - max_wrong_logits
    margin_loss = torch.relu(margin_eps - margin).mean()
    weight_loss = weight.norm()
    return 0.5 * margin_loss + 0.05 * weight_loss


def compute_margin_loss(
    logits: torch.Tensor, labels: torch.Tensor, margin_eps: float
) -> torch.Tensor:
    correct_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
    masked_logits = logits.masked_fill(
        F.one_hot(labels, num_classes=logits.shape[1]).bool(), -torch.inf
    )
    max_wrong_logits = masked_logits.max(dim=1).values
    margin = correct_logits - max_wrong_logits
    return torch.relu(margin_eps - margin).mean()


def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=1).mean()


@torch.no_grad()
def evaluate_classifier(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total = 0
    confidences: List[torch.Tensor] = []
    correctness: List[torch.Tensor] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = model(images)
        total_loss += criterion(logits, targets).item()
        probs = logits.softmax(dim=1)
        conf, pred = probs.max(dim=1)
        total_correct += (pred == targets).sum().item()
        total += targets.numel()
        confidences.append(conf.cpu())
        correctness.append((pred == targets).float().cpu())
    return {
        "loss": total_loss / total,
        "acc": total_correct / total,
        "ece": expected_calibration_error(
            torch.cat(confidences), torch.cat(correctness)
        ),
    }


def expected_calibration_error(
    confidences: torch.Tensor, correct: torch.Tensor, bins: int = 15
) -> float:
    ece = torch.tensor(0.0)
    boundaries = torch.linspace(0.0, 1.0, bins + 1)
    for lo, hi in zip(boundaries[:-1], boundaries[1:]):
        if hi == 1.0:
            mask = (confidences >= lo) & (confidences <= hi)
        else:
            mask = (confidences >= lo) & (confidences < hi)
        if mask.any():
            ece += (
                mask.float().mean()
                * (confidences[mask].mean() - correct[mask].mean()).abs()
            )
    return float(ece.item())


@torch.no_grad()
def estimate_boundary_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float,
    max_samples: int,
) -> Dict[str, float]:
    model.eval()
    feature_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    collected = 0
    for images, labels in loader:
        if collected >= max_samples:
            break
        images = images.to(device, non_blocking=True)
        features = model.features(images)
        feature_chunks.append(features.cpu())
        label_chunks.append(labels.cpu())
        collected += labels.numel()
    features = torch.cat(feature_chunks, dim=0)[:max_samples].numpy()
    labels = torch.cat(label_chunks, dim=0)[:max_samples].numpy()
    weight = model.classifier.weight.detach().cpu().numpy()
    bias = model.classifier.bias.detach().cpu().numpy()
    hard_values: List[float] = []
    kernel_values: List[float] = []
    for class_idx in range(weight.shape[0]):
        w = weight[class_idx]
        norm = np.linalg.norm(w) + 1e-8
        distances = np.abs(features @ w + bias[class_idx]) / norm
        tube_prob = np.mean(distances <= eps)
        kernel_prob = np.mean(np.maximum(0.0, 1.0 - distances / (eps + 1e-8)))
        p_pos = np.mean(labels == class_idx)
        balance = max(min(p_pos, 1.0 - p_pos), 0.01)
        hard_values.append((tube_prob / (eps + 1e-8)) / balance)
        kernel_values.append((kernel_prob / (eps + 1e-8)) / balance)
    logits = features @ weight.T + bias
    true_logits = logits[np.arange(labels.shape[0]), labels]
    logits[np.arange(labels.shape[0]), labels] = -np.inf
    margin = true_logits - np.max(logits, axis=1)
    feature_std = features.std(axis=0)
    return {
        "psi_eff": float(np.mean(hard_values)),
        "psi_kernel": float(np.mean(kernel_values)),
        "weight_norm": float(np.linalg.norm(weight)),
        "feature_norm_mean": float(np.linalg.norm(features, axis=1).mean()),
        "feature_std_mean": float(feature_std.mean()),
        "margin_mean": float(np.mean(margin)),
        "margin_p10": float(np.quantile(margin, 0.10)),
    }


@torch.no_grad()
def estimate_psi_eff(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    eps: float,
    max_samples: int,
) -> float:
    return estimate_boundary_metrics(model, loader, device, eps, max_samples)["psi_eff"]


def train_mode(
    mode: str,
    initial_state: Dict[str, torch.Tensor],
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    output_dir: Path,
) -> Dict[str, float]:
    model = build_model(args.arch, num_classes=args.num_classes)
    model.load_state_dict(copy.deepcopy(initial_state))
    model.to(device)
    if mode == "psieff":
        optimizer = optim.SGD(
            model.parameters(), lr=args.psi_lr, momentum=0.9, weight_decay=0.0
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[22, 32], gamma=0.5
        )
    elif mode == "weight_decay":
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 30], gamma=0.5
        )
    else:
        optimizer = optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0
        )
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[20, 30], gamma=0.5
        )

    criterion = nn.CrossEntropyLoss()
    rows: List[Dict[str, float]] = []
    mode_dir = output_dir / mode
    mode_dir.mkdir(parents=True, exist_ok=True)
    start = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        total_ce = 0.0
        total_correct = 0
        total = 0
        for images, targets in tqdm(
            train_loader, desc=f"{mode} epoch {epoch}/{args.epochs}", leave=False
        ):
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if mode == "psieff":
                features, logits = model.get_features_and_logits(images)
                ce = criterion(logits, targets)
                reg = compute_psi_surrogate_loss(
                    features, logits, targets, model.classifier.weight
                )
                loss = ce + args.lambda_psi * reg
            elif mode == "margin":
                logits = model(images)
                ce = criterion(logits, targets)
                loss = ce + args.lambda_margin * compute_margin_loss(
                    logits, targets, args.margin_eps
                )
            elif mode == "entropy":
                logits = model(images)
                ce = criterion(logits, targets)
                loss = ce - args.lambda_entropy * compute_entropy(logits)
            else:
                logits = model(images)
                ce = criterion(logits, targets)
                loss = ce
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * targets.numel()
            total_ce += ce.item() * targets.numel()
            total_correct += (logits.argmax(dim=1) == targets).sum().item()
            total += targets.numel()
        scheduler.step()
        test_metrics = evaluate_classifier(model, test_loader, device)
        boundary_metrics = estimate_boundary_metrics(
            model, test_loader, device, args.psi_eps, args.psi_samples
        )
        row = {
            "epoch": epoch,
            "train_loss": total_loss / total,
            "train_ce": total_ce / total,
            "train_acc": total_correct / total,
            "test_loss": test_metrics["loss"],
            "test_acc": test_metrics["acc"],
            "test_ece": test_metrics["ece"],
            "elapsed_seconds": time.time() - start,
        }
        row.update(boundary_metrics)
        rows.append(row)
        append_csv(mode_dir / "metrics.csv", row)
        print(
            f"{mode} epoch={epoch:02d} train_acc={row['train_acc']:.4f} "
            f"test_acc={row['test_acc']:.4f} psi={row['psi_eff']:.4f} ece={row['test_ece']:.4f}",
            flush=True,
        )
    torch.save(
        {
            "model_state": model.state_dict(),
            "metrics": rows[-1],
            "args": vars(args),
            "mode": mode,
            "arch": args.arch,
        },
        mode_dir / "last.pt",
    )
    return rows[-1]


def append_csv(path: Path, row: Dict[str, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def extract_features(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    features: List[torch.Tensor] = []
    labels: List[torch.Tensor] = []
    for images, targets in tqdm(loader, desc="extract", leave=False):
        images = images.to(device, non_blocking=True)
        features.append(model.features(images).cpu())
        labels.append(targets.cpu())
    return torch.cat(features).float(), torch.cat(labels).long()


def select_balanced_indices(
    labels: torch.Tensor, samples_per_class: int, seed: int, num_classes: int = 100
) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    selected: List[int] = []
    for class_idx in range(num_classes):
        candidates = torch.nonzero(labels == class_idx, as_tuple=False).flatten()
        order = torch.randperm(candidates.numel(), generator=generator)[
            :samples_per_class
        ]
        selected.extend(candidates[order].tolist())
    final_order = torch.randperm(len(selected), generator=generator)
    return torch.tensor(selected, dtype=torch.long)[final_order]


def train_probe(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    samples_per_class: int,
    seed: int,
    epochs: int,
    device: torch.device,
    num_classes: int,
) -> float:
    idx = select_balanced_indices(
        train_labels, samples_per_class, seed, num_classes=num_classes
    )
    x_train = train_features[idx]
    y_train = train_labels[idx]
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True).clamp_min(1e-6)
    x_train = (x_train - mean) / std
    x_test = (test_features - mean) / std
    head = nn.Linear(x_train.shape[1], num_classes).to(device)
    optimizer = optim.AdamW(head.parameters(), lr=5e-3, weight_decay=1e-4)
    batch_size = min(256, x_train.shape[0])
    generator = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        order = torch.randperm(x_train.shape[0], generator=generator)
        for start in range(0, x_train.shape[0], batch_size):
            batch = order[start : start + batch_size]
            logits = head(x_train[batch].to(device, non_blocking=True))
            loss = F.cross_entropy(logits, y_train[batch].to(device, non_blocking=True))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    head.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for start in range(0, x_test.shape[0], 1024):
            logits = head(x_test[start : start + 1024].to(device, non_blocking=True))
            pred = logits.argmax(dim=1).cpu()
            targets = test_labels[start : start + 1024]
            correct += (pred == targets).sum().item()
            total += targets.numel()
    return correct / total


def downstream_experiment(
    output_dir: Path,
    train_eval,
    test_eval,
    device: torch.device,
    args: argparse.Namespace,
) -> List[Dict[str, float]]:
    train_loader = make_loader(
        train_eval, args.eval_batch_size, False, args.num_workers
    )
    test_loader = make_loader(test_eval, args.eval_batch_size, False, args.num_workers)
    budgets = [int(x) for x in args.probe_budgets.split(",") if x.strip()]
    seeds = [int(x) for x in args.probe_seeds.split(",") if x.strip()]
    rows: List[Dict[str, float]] = []
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    for mode in modes:
        checkpoint = torch.load(
            output_dir / mode / "last.pt", map_location="cpu", weights_only=False
        )
        model = build_model(args.arch, num_classes=args.num_classes).to(device)
        model.load_state_dict(checkpoint["model_state"])
        train_features, train_labels = extract_features(model, train_loader, device)
        test_features, test_labels = extract_features(model, test_loader, device)
        for budget in budgets:
            accs = [
                train_probe(
                    train_features,
                    train_labels,
                    test_features,
                    test_labels,
                    budget,
                    seed,
                    args.probe_epochs,
                    device,
                    args.num_classes,
                )
                for seed in seeds
            ]
            row = {
                "mode": mode,
                "samples_per_class": budget,
                "probe_acc_mean": float(np.mean(accs)),
                "probe_acc_std": float(np.std(accs, ddof=0)),
                "probe_seeds": len(seeds),
            }
            rows.append(row)
            append_csv(output_dir / "downstream_probe.csv", row)
            print(
                f"downstream mode={mode} spc={budget} acc={row['probe_acc_mean']:.4f}+/-{row['probe_acc_std']:.4f}",
                flush=True,
            )
    return rows


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    train_aug, train_eval, test_eval, num_classes = build_datasets(
        args.data_dir, args.download, args.dataset
    )
    args.num_classes = num_classes
    train_loader = make_loader(train_aug, args.batch_size, True, args.num_workers)
    test_loader = make_loader(test_eval, args.eval_batch_size, False, args.num_workers)

    summary: Dict[str, object] = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    }
    if not args.skip_train:
        base_model = build_model(args.arch, num_classes=args.num_classes)
        initial_state = copy.deepcopy(base_model.state_dict())
        torch.save(initial_state, args.output_dir / "initial_state.pt")
        train_rows = {}
        modes = [x.strip() for x in args.modes.split(",") if x.strip()]
        for mode in modes:
            train_rows[mode] = train_mode(
                mode,
                initial_state,
                train_loader,
                test_loader,
                device,
                args,
                args.output_dir,
            )
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary["training"] = train_rows

    probe_rows = downstream_experiment(args.output_dir, train_eval, test_eval, device, args)
    summary["downstream_probe"] = probe_rows
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
