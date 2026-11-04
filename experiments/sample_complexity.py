"Measure synthetic learning curves across representation scales. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.neighbors import KNeighborsClassifier


@dataclass
class CurveRow:
    scale_normal: float
    psi_eff: float
    n_train: int
    seed: int
    k: int
    test_error: float
    test_accuracy: float
    n_over_psi: float
    n_over_psi_d: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthetic sample-complexity collapse under representation-induced psi_eff changes."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--scales", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0]
    )
    parser.add_argument(
        "--n-train-values",
        type=int,
        nargs="+",
        default=[50, 100, 200, 400, 800, 1600, 3200],
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--n-test", type=int, default=50_000)
    parser.add_argument("--k", type=int, default=7)
    parser.add_argument("--dimension", type=int, default=2)
    return parser.parse_args()


def sample_uniform_halfspace(
    rng: np.random.Generator,
    n: int,
    scale_normal: float,
    dimension: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = rng.uniform(-1.0, 1.0, size=(n, dimension))
    z = x.copy()
    z[:, 0] *= scale_normal
    y = (x[:, 0] >= 0.0).astype(np.int64)
    return z, y


def write_csv(path: Path, rows: Iterable[CurveRow]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def fit_log_model(x: np.ndarray, y: np.ndarray) -> dict:
    mask = (x > 0) & (y > 0)
    lx = np.log(x[mask])
    ly = np.log(y[mask])
    design = np.column_stack([np.ones_like(lx), lx])
    coef, *_ = np.linalg.lstsq(design, ly, rcond=None)
    pred = design @ coef
    ss_res = float(np.sum((ly - pred) ** 2))
    ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return {
        "intercept": float(coef[0]),
        "slope": float(coef[1]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0,
    }


def save_learning_curves(rows: List[CurveRow], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for scale in sorted({row.scale_normal for row in rows}):
        subset = [row for row in rows if row.scale_normal == scale]
        xs = sorted({row.n_train for row in subset})
        means = [
            np.mean([row.test_error for row in subset if row.n_train == x]) for x in xs
        ]
        ax.plot(xs, means, marker="o", label=f"scale={scale:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training samples n")
    ax.set_ylabel("Test error")
    ax.set_title("Raw learning curves")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "raw_learning_curves.png", dpi=220)
    plt.close(fig)


def save_collapse(
    rows: List[CurveRow], output_dir: Path, use_power_d: bool = False
) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    key = "n_over_psi_d" if use_power_d else "n_over_psi"
    for scale in sorted({row.scale_normal for row in rows}):
        subset = [row for row in rows if row.scale_normal == scale]
        xs = sorted({getattr(row, key) for row in subset})
        means = [
            np.mean([row.test_error for row in subset if getattr(row, key) == x])
            for x in xs
        ]
        ax.plot(xs, means, marker="o", label=f"scale={scale:g}")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("n / psi_eff^d" if use_power_d else "n / psi_eff")
    ax.set_ylabel("Test error")
    ax.set_title("Collapsed learning curves")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(
        output_dir
        / ("collapse_n_over_psi_d.png" if use_power_d else "collapse_n_over_psi.png"),
        dpi=220,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[CurveRow] = []
    for scale in args.scales:
        psi_eff = 2.0 / scale
        test_rng = np.random.default_rng(10_000 + int(scale * 1000))
        z_test, y_test = sample_uniform_halfspace(
            test_rng, args.n_test, scale, args.dimension
        )
        for n_train in args.n_train_values:
            for seed in args.seeds:
                rng = np.random.default_rng(
                    seed + 1_000_000 * int(scale * 1000) + n_train
                )
                z_train, y_train = sample_uniform_halfspace(
                    rng, n_train, scale, args.dimension
                )
                k = min(args.k, n_train)
                if k % 2 == 0:
                    k = max(1, k - 1)
                clf = KNeighborsClassifier(n_neighbors=k)
                clf.fit(z_train, y_train)
                acc = float(clf.score(z_test, y_test))
                rows.append(
                    CurveRow(
                        scale_normal=scale,
                        psi_eff=psi_eff,
                        n_train=n_train,
                        seed=seed,
                        k=k,
                        test_error=max(1.0 - acc, 1.0 / args.n_test),
                        test_accuracy=acc,
                        n_over_psi=n_train / psi_eff,
                        n_over_psi_d=n_train / (psi_eff**args.dimension),
                    )
                )

    write_csv(output_dir / "sample_complexity_rows.csv", rows)
    errors = np.asarray([row.test_error for row in rows])
    n_values = np.asarray([row.n_train for row in rows], dtype=np.float64)
    n_over_psi = np.asarray([row.n_over_psi for row in rows], dtype=np.float64)
    n_over_psi_d = np.asarray([row.n_over_psi_d for row in rows], dtype=np.float64)
    summary = {
        "dimension": args.dimension,
        "n_rows": len(rows),
        "n_test": args.n_test,
        "fit_log_error_vs_n": fit_log_model(1.0 / n_values, errors),
        "fit_log_error_vs_psi_over_n": fit_log_model(1.0 / n_over_psi, errors),
        "fit_log_error_vs_psi_d_over_n": fit_log_model(1.0 / n_over_psi_d, errors),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    save_learning_curves(rows, output_dir)
    save_collapse(rows, output_dir, use_power_d=False)
    save_collapse(rows, output_dir, use_power_d=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
