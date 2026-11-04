"Collect per-epoch boundary-mass and utility measurements. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build Boundary-Utility Plane reports from existing linear-probe runs."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-trajectories", type=int, default=80)
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def utility_from_row(row: Dict) -> tuple[str, Optional[float]]:
    for key in ["test_acc", "val_acc", "acc"]:
        value = safe_float(row.get(key))
        if value is not None:
            return key, value
    return "", None


def iter_metric_dirs(run_root: Path) -> Iterable[Path]:
    for path in sorted(run_root.rglob("metrics.jsonl")):
        yield path.parent


def collect_rows(run_root: Path) -> List[Dict]:
    out: List[Dict] = []
    for run_dir in iter_metric_dirs(run_root):
        metrics = read_jsonl(run_dir / "metrics.jsonl")
        if not metrics:
            continue
        args = read_json(run_dir / "args.json")
        rel = run_dir.relative_to(run_root)
        for row in metrics:
            psi = safe_float(row.get("heldout_psieff"))
            utility_key, utility = utility_from_row(row)
            if psi is None or utility is None:
                continue
            out.append(
                {
                    "run_path": str(rel),
                    "run_name": run_dir.name,
                    "epoch": row.get("epoch"),
                    "mode": args.get("mode", ""),
                    "seed": args.get("seed", ""),
                    "train_samples_per_class": args.get("train_samples_per_class", ""),
                    "weight_decay": args.get("weight_decay", ""),
                    "lambda_psieff": args.get("lambda_psieff", ""),
                    "utility_key": utility_key,
                    "utility": utility,
                    "heldout_psieff": psi,
                    "log1p_psieff": math.log1p(psi),
                    "nll": safe_float(row.get("test_nll", row.get("val_nll"))),
                    "ece": safe_float(row.get("test_ece", row.get("val_ece"))),
                    "aurc": safe_float(row.get("test_aurc", row.get("val_aurc"))),
                }
            )
    return out


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def final_rows(rows: List[Dict]) -> List[Dict]:
    by_run: Dict[str, List[Dict]] = {}
    for row in rows:
        by_run.setdefault(row["run_path"], []).append(row)
    out = []
    for items in by_run.values():
        items.sort(key=lambda r: int(r["epoch"]))
        out.append(items[-1])
    return out


def pearson(x: List[float], y: List[float]) -> Optional[float]:
    if len(x) < 2:
        return None
    x_arr = np.asarray(x, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.float64)
    if np.std(x_arr) == 0 or np.std(y_arr) == 0:
        return None
    return float(np.corrcoef(x_arr, y_arr)[0, 1])


def save_plane(rows: List[Dict], output_dir: Path, max_trajectories: int) -> None:
    colors = {
        "standard": "#1f77b4",
        "weight_decay": "#ff7f0e",
        "psieff": "#2ca02c",
        "": "#7f7f7f",
    }
    by_run: Dict[str, List[Dict]] = {}
    for row in rows:
        by_run.setdefault(row["run_path"], []).append(row)
    selected = list(by_run.items())[:max_trajectories]

    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for _, items in selected:
        items.sort(key=lambda r: int(r["epoch"]))
        mode = str(items[0]["mode"])
        ax.plot(
            [r["log1p_psieff"] for r in items],
            [r["utility"] for r in items],
            color=colors.get(mode, "#7f7f7f"),
            alpha=0.28,
            lw=1.0,
        )
        ax.scatter(
            [items[-1]["log1p_psieff"]],
            [items[-1]["utility"]],
            color=colors.get(mode, "#7f7f7f"),
            alpha=0.75,
            s=14,
        )
    ax.set_xlabel("log(1 + heldout psi_eff)")
    ax.set_ylabel("Validation/Test accuracy")
    ax.set_title("Boundary-Utility Plane")
    handles = [
        plt.Line2D([0], [0], color=color, lw=2, label=mode or "unknown")
        for mode, color in colors.items()
        if any(str(row["mode"]) == mode for row in rows)
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "boundary_utility_plane.png", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(run_root)
    finals = final_rows(rows)
    write_csv(output_dir / "boundary_utility_epochs.csv", rows)
    write_csv(output_dir / "boundary_utility_final.csv", finals)

    correlations = {
        "num_epoch_rows": len(rows),
        "num_runs": len(finals),
        "final_corr_log1p_psieff_utility": pearson(
            [r["log1p_psieff"] for r in finals],
            [r["utility"] for r in finals],
        ),
        "final_corr_psieff_utility": pearson(
            [r["heldout_psieff"] for r in finals],
            [r["utility"] for r in finals],
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(correlations, indent=2, sort_keys=True), encoding="utf-8"
    )
    if rows:
        save_plane(rows, output_dir, args.max_trajectories)
    print(json.dumps(correlations, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
