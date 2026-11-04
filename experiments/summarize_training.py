"Summarize recorded training runs. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize run outputs under a run root directory."
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def read_json(path: Path) -> Dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def safe_float(x: object) -> Optional[float]:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def find_run_dirs(group_dir: Path) -> List[Path]:
    runs: List[Path] = []
    for child in sorted(group_dir.iterdir()):
        if child.is_dir() and (child / "metrics.jsonl").exists():
            runs.append(child)
    return runs


def summarize_single_run(group: str, run_dir: Path) -> Optional[Dict]:
    metrics_path = run_dir / "metrics.jsonl"
    metrics = read_jsonl(metrics_path)
    if not metrics:
        return None

    summary_path = run_dir / "summary.json"
    summary: Dict = read_json(summary_path) if summary_path.exists() else {}
    args_path = run_dir / "args.json"
    args: Dict = read_json(args_path) if args_path.exists() else {}

    final_row = metrics[-1]
    metric_key = "test_acc" if "test_acc" in final_row else "val_acc"
    loss_key = "test_loss" if "test_loss" in final_row else "val_loss"
    best_row = max(metrics, key=lambda r: float(r.get(metric_key, -math.inf)))

    out = {
        "group": group,
        "run_name": run_dir.name,
        "mode": summary.get("mode", args.get("mode")),
        "seed": summary.get("seed", args.get("seed")),
        "train_samples_per_class": summary.get(
            "train_samples_per_class", args.get("train_samples_per_class")
        ),
        "weight_decay": safe_float(args.get("weight_decay")),
        "lambda_psieff": safe_float(args.get("lambda_psieff")),
        "primary_metric": metric_key,
        "epochs_completed": final_row.get("epoch"),
        "best_epoch": best_row.get("epoch"),
        "best_test_acc": safe_float(best_row.get(metric_key)),
        "final_test_acc": safe_float(final_row.get(metric_key)),
        "final_test_loss": safe_float(final_row.get(loss_key)),
        "final_train_acc": safe_float(final_row.get("train_acc")),
        "final_train_loss": safe_float(final_row.get("train_loss")),
        "best_heldout_psieff": safe_float(best_row.get("heldout_psieff")),
        "final_heldout_psieff": safe_float(final_row.get("heldout_psieff")),
        "summary_best_test_acc": safe_float(
            summary.get("best_test_acc", summary.get("best_val_acc"))
        ),
        "summary_best_epoch": summary.get("best_epoch"),
        "run_dir": str(run_dir),
        "status": "complete" if summary_path.exists() else "running",
    }
    return out


def make_config_group(row: Dict) -> str:
    mode = str(row.get("mode"))
    spc = row.get("train_samples_per_class")
    pieces = [str(row.get("group")), f"mode={mode}", f"spc={spc}"]
    if mode == "weight_decay":
        pieces.append(f"wd={row.get('weight_decay')}")
    elif mode == "psieff":
        pieces.append(f"lam={row.get('lambda_psieff')}")
    return "|".join(pieces)


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_group_stats(rows: List[Dict]) -> List[Dict]:
    by_group: Dict[str, List[Dict]] = {}
    for row in rows:
        by_group.setdefault(make_config_group(row), []).append(row)

    stats_rows: List[Dict] = []
    for group, items in sorted(by_group.items()):
        best_vals = [
            r["best_test_acc"] for r in items if r["best_test_acc"] is not None
        ]
        final_vals = [
            r["final_test_acc"] for r in items if r["final_test_acc"] is not None
        ]
        psi_vals = [
            r["final_heldout_psieff"]
            for r in items
            if r["final_heldout_psieff"] is not None
        ]
        complete_n = sum(1 for r in items if r["status"] == "complete")
        first = items[0]

        def m_and_s(vals: List[float]) -> Tuple[Optional[float], Optional[float]]:
            if not vals:
                return None, None
            if len(vals) == 1:
                return mean(vals), 0.0
            return mean(vals), pstdev(vals)

        best_mean, best_std = m_and_s(best_vals)
        final_mean, final_std = m_and_s(final_vals)
        psi_mean, psi_std = m_and_s(psi_vals)

        stats_rows.append(
            {
                "config_group": group,
                "group": first.get("group"),
                "mode": first.get("mode"),
                "train_samples_per_class": first.get("train_samples_per_class"),
                "weight_decay": first.get("weight_decay"),
                "lambda_psieff": first.get("lambda_psieff"),
                "num_runs": len(items),
                "num_complete": complete_n,
                "best_test_acc_mean": best_mean,
                "best_test_acc_std": best_std,
                "final_test_acc_mean": final_mean,
                "final_test_acc_std": final_std,
                "final_heldout_psieff_mean": psi_mean,
                "final_heldout_psieff_std": psi_std,
            }
        )
    return stats_rows


def main() -> None:
    args = parse_args()
    run_root = args.run_root.resolve()
    output_dir = (
        args.output_dir.resolve() if args.output_dir else run_root / "summary_reports"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    run_rows: List[Dict] = []
    for group_dir in sorted(run_root.iterdir()):
        if not group_dir.is_dir():
            continue
        group = group_dir.name
        for run_dir in find_run_dirs(group_dir):
            row = summarize_single_run(group=group, run_dir=run_dir)
            if row is not None:
                run_rows.append(row)

    run_rows = sorted(run_rows, key=lambda r: (str(r["group"]), str(r["run_name"])))
    group_rows = build_group_stats(run_rows)

    runs_csv = output_dir / "summary_runs.csv"
    groups_csv = output_dir / "summary_groups.csv"
    runs_json = output_dir / "summary_runs.json"
    groups_json = output_dir / "summary_groups.json"

    write_csv(runs_csv, run_rows)
    write_csv(groups_csv, group_rows)
    runs_json.write_text(
        json.dumps(run_rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    groups_json.write_text(
        json.dumps(group_rows, indent=2, sort_keys=True), encoding="utf-8"
    )

    print(f"Wrote {runs_csv}")
    print(f"Wrote {groups_csv}")
    print(f"Wrote {runs_json}")
    print(f"Wrote {groups_json}")


if __name__ == "__main__":
    main()
