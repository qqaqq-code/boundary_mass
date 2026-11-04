"""共享准确率、校准和边界质量评估函数。

运行入口及输入说明见 docs/EXPERIMENTS.md。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


@dataclass
class ClassifierMetrics:
    loss: float
    acc: float
    nll: float
    brier: float
    ece: float
    aurc: float
    mean_confidence: float

    def as_dict(self, prefix: str = "") -> Dict[str, float]:
        if not prefix:
            return asdict(self)
        return {f"{prefix}{key}": value for key, value in asdict(self).items()}


def expected_calibration_error(
    confidences: torch.Tensor,
    correct: torch.Tensor,
    num_bins: int = 15,
) -> float:
    if confidences.numel() == 0:
        return 0.0

    bins = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=confidences.device)
    total = confidences.numel()
    ece = confidences.new_tensor(0.0)

    for bin_idx in range(num_bins):
        left = bins[bin_idx]
        right = bins[bin_idx + 1]
        if bin_idx == 0:
            mask = (confidences >= left) & (confidences <= right)
        else:
            mask = (confidences > left) & (confidences <= right)
        if not torch.any(mask):
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece = ece + mask.float().mean() * torch.abs(bin_acc - bin_conf)

    return float(ece.item())


def area_under_risk_coverage(confidences: torch.Tensor, correct: torch.Tensor) -> float:
    if confidences.numel() == 0:
        return 0.0

    order = torch.argsort(confidences, descending=True)
    errors = 1.0 - correct[order]
    cumulative_errors = torch.cumsum(errors, dim=0)
    coverage_counts = torch.arange(
        1, errors.numel() + 1, device=errors.device, dtype=torch.float32
    )
    selective_risk = cumulative_errors / coverage_counts
    return float(selective_risk.mean().item())


@torch.no_grad()
def evaluate_classifier(
    head: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
    num_bins: int = 15,
) -> ClassifierMetrics:
    head.eval()

    total_loss = 0.0
    total_nll = 0.0
    total_brier = 0.0
    total_correct = 0
    total = 0

    confidence_chunks = []
    correct_chunks = []

    for features, labels in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = head(features)
        probs = torch.softmax(logits, dim=1)
        confidences, predictions = probs.max(dim=1)
        correct = (predictions == labels).float()

        batch_size = labels.size(0)
        total_loss += criterion(logits, labels).item() * batch_size
        total_nll += F.cross_entropy(logits, labels, reduction="sum").item()
        one_hot = F.one_hot(labels, num_classes=probs.size(1)).float()
        total_brier += (probs - one_hot).pow(2).sum(dim=1).sum().item()
        total_correct += int(correct.sum().item())
        total += batch_size

        confidence_chunks.append(confidences.detach().cpu())
        correct_chunks.append(correct.detach().cpu())

    if total == 0:
        return ClassifierMetrics(
            loss=0.0,
            acc=0.0,
            nll=0.0,
            brier=0.0,
            ece=0.0,
            aurc=0.0,
            mean_confidence=0.0,
        )

    all_confidences = torch.cat(confidence_chunks, dim=0)
    all_correct = torch.cat(correct_chunks, dim=0)
    return ClassifierMetrics(
        loss=total_loss / total,
        acc=total_correct / total,
        nll=total_nll / total,
        brier=total_brier / total,
        ece=expected_calibration_error(all_confidences, all_correct, num_bins=num_bins),
        aurc=area_under_risk_coverage(all_confidences, all_correct),
        mean_confidence=float(all_confidences.mean().item()),
    )
