"Evaluate the boundary-mass pushforward law under controlled transformations. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class PushforwardRow:
    group: str
    case: str
    eps: float
    n_samples: int
    normal_x: float
    normal_y: float
    matrix: str
    predicted_psieff: float
    measured_psieff: float
    relative_error: float
    balance: float
    tube_probability: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Controlled synthetic validation of the psi_eff pushforward law."
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-samples", type=int, default=500_000)
    parser.add_argument("--eps", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--orientation-points", type=int, default=25)
    parser.add_argument(
        "--eps-values", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.04, 0.08]
    )
    return parser.parse_args()


def normalize(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v)
    if norm <= 0:
        raise ValueError("zero vector")
    return v / norm


def line_length_in_square(normal: np.ndarray) -> float:
    normal = normalize(normal)
    tangent = np.array([-normal[1], normal[0]], dtype=np.float64)
    return 2.0 / max(abs(tangent[0]), abs(tangent[1]), 1e-12)


def predicted_psieff(matrix: np.ndarray, normal: np.ndarray) -> float:
    normal = normalize(normal)
    image_normal = np.linalg.solve(matrix.T, normal)
    return line_length_in_square(normal) * float(np.linalg.norm(image_normal))


def hard_tube_estimate(
    rng: np.random.Generator,
    matrix: np.ndarray,
    normal: np.ndarray,
    eps: float,
    n_samples: int,
) -> tuple[float, float, float]:
    normal = normalize(normal)
    x = rng.uniform(-1.0, 1.0, size=(n_samples, 2))
    z = x @ matrix.T
    image_normal = np.linalg.solve(matrix.T, normal)
    signed = z @ image_normal
    distance = np.abs(signed) / np.linalg.norm(image_normal)
    positive = (x @ normal) >= 0.0
    p_plus = float(positive.mean())
    balance = min(p_plus, 1.0 - p_plus)
    tube_probability = float((distance <= eps).mean())
    return tube_probability / max(eps * balance, 1e-12), balance, tube_probability


def matrix_to_json(matrix: np.ndarray) -> str:
    return json.dumps(matrix.round(8).tolist(), separators=(",", ":"))


def add_case(
    rows: List[PushforwardRow],
    group: str,
    case: str,
    matrix: np.ndarray,
    normal: np.ndarray,
    eps: float,
    n_samples: int,
    seed: int,
) -> None:
    rng = np.random.default_rng(seed)
    pred = predicted_psieff(matrix, normal)
    measured, balance, tube_probability = hard_tube_estimate(
        rng, matrix, normal, eps, n_samples
    )
    rows.append(
        PushforwardRow(
            group=group,
            case=case,
            eps=eps,
            n_samples=n_samples,
            normal_x=float(normalize(normal)[0]),
            normal_y=float(normalize(normal)[1]),
            matrix=matrix_to_json(matrix),
            predicted_psieff=pred,
            measured_psieff=measured,
            relative_error=abs(measured - pred) / max(abs(pred), 1e-12),
            balance=balance,
            tube_probability=tube_probability,
        )
    )


def write_csv(path: Path, rows: Iterable[PushforwardRow]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def save_scatter(rows: List[PushforwardRow], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.4))
    groups = sorted({row.group for row in rows if row.group != "eps_convergence"})
    for group in groups:
        subset = [row for row in rows if row.group == group]
        ax.scatter(
            [row.predicted_psieff for row in subset],
            [row.measured_psieff for row in subset],
            label=group,
            s=32,
            alpha=0.85,
        )
    vals = [row.predicted_psieff for row in rows if row.group != "eps_convergence"]
    lo, hi = min(vals), max(vals)
    pad = 0.05 * (hi - lo)
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="black", lw=1, ls="--")
    ax.set_xlabel("Predicted psi_eff")
    ax.set_ylabel("Measured tube psi_eff")
    ax.set_title("Pushforward law validation")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "pushforward_scatter.png", dpi=220)
    plt.close(fig)


def save_orientation(rows: List[PushforwardRow], output_dir: Path) -> None:
    subset = [row for row in rows if row.group == "orientation"]
    angles = [math.degrees(math.atan2(row.normal_y, row.normal_x)) for row in subset]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(angles, [row.predicted_psieff for row in subset], label="predicted", lw=2)
    ax.scatter(angles, [row.measured_psieff for row in subset], label="measured", s=24)
    ax.set_xlabel("Boundary normal angle (degrees)")
    ax.set_ylabel("psi_eff")
    ax.set_title("Orientation dependence under anisotropic scaling")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_dir / "orientation_curve.png", dpi=220)
    plt.close(fig)


def save_anisotropy(rows: List[PushforwardRow], output_dir: Path) -> None:
    subset = [row for row in rows if row.group == "anisotropy"]
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    for prefix in ["normal_scale", "tangent_scale"]:
        ss = []
        pred = []
        meas = []
        for row in subset:
            if row.case.startswith(prefix):
                ss.append(float(row.case.split("=")[1]))
                pred.append(row.predicted_psieff)
                meas.append(row.measured_psieff)
        order = np.argsort(ss)
        ss_arr = np.asarray(ss)[order]
        ax.plot(ss_arr, np.asarray(pred)[order], label=f"{prefix} predicted", lw=2)
        ax.scatter(ss_arr, np.asarray(meas)[order], label=f"{prefix} measured", s=24)
    ax.set_xscale("log")
    ax.set_xlabel("Scale factor")
    ax.set_ylabel("psi_eff")
    ax.set_title("Normal vs tangent scaling")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "anisotropy_sweep.png", dpi=220)
    plt.close(fig)


def save_eps_convergence(rows: List[PushforwardRow], output_dir: Path) -> None:
    subset = [row for row in rows if row.group == "eps_convergence"]
    subset.sort(key=lambda row: row.eps)
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    ax.plot(
        [row.eps for row in subset], [row.relative_error for row in subset], marker="o"
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tube width eps")
    ax.set_ylabel("Relative error")
    ax.set_title("Tube-estimator convergence")
    fig.tight_layout()
    fig.savefig(output_dir / "eps_convergence.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: List[PushforwardRow] = []
    n0 = np.array([1.0, 0.0])

    for c in [0.5, 1.0, 2.0, 4.0]:
        add_case(
            rows,
            "isotropic",
            f"scale={c}",
            np.diag([c, c]),
            n0,
            args.eps,
            args.n_samples,
            args.seed + len(rows),
        )

    for s in [0.5, 1.0, 2.0, 4.0, 8.0]:
        add_case(
            rows,
            "anisotropy",
            f"normal_scale={s}",
            np.diag([s, 1.0]),
            n0,
            args.eps,
            args.n_samples,
            args.seed + len(rows),
        )
        add_case(
            rows,
            "anisotropy",
            f"tangent_scale={s}",
            np.diag([1.0, s]),
            n0,
            args.eps,
            args.n_samples,
            args.seed + len(rows),
        )

    for k in [-1.5, -0.75, 0.0, 0.75, 1.5]:
        add_case(
            rows,
            "shear",
            f"shear={k}",
            np.array([[1.0, k], [0.0, 1.0]]),
            n0,
            args.eps,
            args.n_samples,
            args.seed + len(rows),
        )

    anisotropic = np.diag([4.0, 1.0])
    for theta in np.linspace(0.0, math.pi, args.orientation_points):
        normal = np.array([math.cos(theta), math.sin(theta)])
        add_case(
            rows,
            "orientation",
            f"theta={theta:.4f}",
            anisotropic,
            normal,
            args.eps,
            args.n_samples,
            args.seed + len(rows),
        )

    eps_matrix = np.diag([4.0, 1.0])
    eps_normal = normalize(np.array([1.0, 1.0]))
    for eps in args.eps_values:
        add_case(
            rows,
            "eps_convergence",
            f"eps={eps}",
            eps_matrix,
            eps_normal,
            eps,
            args.n_samples,
            args.seed + len(rows),
        )

    write_csv(output_dir / "pushforward_results.csv", rows)
    summary_rows = [row for row in rows if row.group != "eps_convergence"]
    summary = {
        "n_rows": len(rows),
        "n_samples": args.n_samples,
        "eps": args.eps,
        "mean_relative_error": float(
            np.mean([row.relative_error for row in summary_rows])
        ),
        "median_relative_error": float(
            np.median([row.relative_error for row in summary_rows])
        ),
        "max_relative_error": float(
            np.max([row.relative_error for row in summary_rows])
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )

    save_scatter(rows, output_dir)
    save_orientation(rows, output_dir)
    save_anisotropy(rows, output_dir)
    save_eps_convergence(rows, output_dir)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
