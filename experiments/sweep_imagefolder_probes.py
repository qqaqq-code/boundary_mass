"Run a parameter sweep over ImageFolder probes. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import CODE_DIR

import argparse
import csv
import json
import logging
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch multi-seed ImageFolder linear-probe sweeps with persistent logs."
    )
    parser.add_argument("--python", type=str, default=sys.executable)
    parser.add_argument(
        "--train-script", type=Path, default=CODE_DIR / "train_imagefolder_probe.py"
    )
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--base-output", type=Path, required=True)
    parser.add_argument("--feature-cache-dir", type=Path, default=None)
    parser.add_argument("--backbone-model", choices=["resnet50"], default="resnet50")
    parser.add_argument("--weights", type=str, default="IMAGENET1K_V2")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument(
        "--modes",
        nargs="+",
        choices=["standard", "weight_decay", "psieff"],
        default=["standard", "weight_decay", "psieff"],
    )
    parser.add_argument(
        "--weight-decay-values", type=float, nargs="+", default=[1e-4, 5e-4]
    )
    parser.add_argument(
        "--lambda-psieff-values", type=float, nargs="+", default=[5e-5, 1e-4, 2e-4]
    )
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--tau", type=float, default=0.05)
    parser.add_argument("--rho", type=float, default=1e-6)
    parser.add_argument("--eps0", type=float, default=0.1)
    parser.add_argument("--holdout-size", type=int, default=4000)
    parser.add_argument(
        "--train-samples-per-class-values", type=int, nargs="+", default=[5, 10, 20, 50]
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_logger(base_output: Path) -> logging.Logger:
    logger = logging.getLogger(str(base_output))
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(base_output / "launcher.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    logger.propagate = False
    return logger


def append_jsonl(path: Path, payload: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, sort_keys=True) + "\n")


def append_csv(path: Path, row: Dict) -> None:
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def build_runs(args: argparse.Namespace) -> List[Dict]:
    runs: List[Dict] = []
    for train_samples_per_class in args.train_samples_per_class_values:
        for seed in args.seeds:
            if "standard" in args.modes:
                runs.append(
                    {
                        "mode": "standard",
                        "seed": seed,
                        "weight_decay": 0.0,
                        "lambda_psieff": 0.0,
                        "train_samples_per_class": train_samples_per_class,
                    }
                )
            if "weight_decay" in args.modes:
                for wd in args.weight_decay_values:
                    runs.append(
                        {
                            "mode": "weight_decay",
                            "seed": seed,
                            "weight_decay": wd,
                            "lambda_psieff": 0.0,
                            "train_samples_per_class": train_samples_per_class,
                        }
                    )
            if "psieff" in args.modes:
                for lam in args.lambda_psieff_values:
                    runs.append(
                        {
                            "mode": "psieff",
                            "seed": seed,
                            "weight_decay": 0.0,
                            "lambda_psieff": lam,
                            "train_samples_per_class": train_samples_per_class,
                        }
                    )
    return runs


def main() -> None:
    args = parse_args()
    base_output = args.base_output.resolve()
    base_output.mkdir(parents=True, exist_ok=True)
    logger = build_logger(base_output)

    runs = build_runs(args)
    logger.info("Prepared %d runs", len(runs))

    for run in runs:
        mode = run["mode"]
        seed = run["seed"]
        train_samples_per_class = run["train_samples_per_class"]
        subset_tag = f"spc_{train_samples_per_class:03d}"
        if mode == "weight_decay":
            tag = f"wd_{run['weight_decay']:.0e}".replace("+0", "")
        elif mode == "psieff":
            tag = f"psi_{run['lambda_psieff']:.0e}".replace("+0", "")
        else:
            tag = "baseline"

        output_dir = base_output / f"{subset_tag}_{mode}_{tag}_seed{seed}"
        command = [
            args.python,
            str(args.train_script),
            "--train-dir",
            str(args.train_dir),
            "--val-dir",
            str(args.val_dir),
            "--output-dir",
            str(output_dir),
            "--backbone-model",
            args.backbone_model,
            "--weights",
            args.weights,
            "--mode",
            mode,
            "--epochs",
            str(args.epochs),
            "--batch-size",
            str(args.batch_size),
            "--eval-batch-size",
            str(args.eval_batch_size),
            "--num-workers",
            str(args.num_workers),
            "--seed",
            str(seed),
            "--lr",
            str(args.lr),
            "--momentum",
            str(args.momentum),
            "--holdout-size",
            str(args.holdout_size),
            "--tau",
            str(args.tau),
            "--rho",
            str(args.rho),
            "--eps0",
            str(args.eps0),
            "--weight-decay",
            str(run["weight_decay"]),
            "--lambda-psieff",
            str(run["lambda_psieff"]),
            "--train-samples-per-class",
            str(train_samples_per_class),
        ]
        if args.feature_cache_dir is not None:
            command.extend(["--feature-cache-dir", str(args.feature_cache_dir)])

        append_jsonl(
            base_output / "commands.jsonl",
            {
                "mode": mode,
                "seed": seed,
                "train_samples_per_class": train_samples_per_class,
                "command": command,
                "output_dir": str(output_dir),
            },
        )
        logger.info("Launching: %s", shlex.join(command))

        start = time.time()
        if args.dry_run:
            returncode = 0
        else:
            completed = subprocess.run(command, check=False)
            returncode = completed.returncode
        duration = time.time() - start

        row = {
            "mode": mode,
            "seed": seed,
            "weight_decay": run["weight_decay"],
            "lambda_psieff": run["lambda_psieff"],
            "train_samples_per_class": train_samples_per_class,
            "returncode": returncode,
            "duration_seconds": f"{duration:.1f}",
            "output_dir": str(output_dir),
        }
        append_csv(base_output / "sweep_summary.csv", row)

        if returncode != 0:
            logger.error("Run failed with return code %s: %s", returncode, output_dir)
        else:
            logger.info("Run completed in %.1fs: %s", duration, output_dir)


if __name__ == "__main__":
    main()
