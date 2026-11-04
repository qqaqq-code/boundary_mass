"""Run a BoundaryMass experiment from the project root."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REGISTRY = json.loads((ROOT / "configs/experiments.json").read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", nargs="?", choices=sorted(REGISTRY))
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--input-results", type=Path,
                        help="Read previously computed tables from this directory.")
    args, extra = parser.parse_known_args()
    if args.list or args.experiment is None:
        for name, task in REGISTRY.items():
            print(f"{name:28s} {task['purpose']}")
        return 0
    task = REGISTRY[args.experiment]
    if extra[:1] == ["--"]:
        extra = extra[1:]
    if extra and not task["supports_cli"]:
        parser.error("This experiment uses the settings at the top of its source file.")
    missing = [m for m in task["dependencies"] if importlib.util.find_spec(m) is None]
    if missing:
        parser.error("Install the required packages: " + ", ".join(missing))
    output = (args.output_dir or ROOT / "outputs" / args.experiment).resolve()
    protected = [ROOT / p for p in ["paper", "results", "configs", "data", "docs",
                                    "experiments", "utils", "local_data", "figures"]]
    if output == ROOT or any(output == p or p in output.parents for p in protected):
        parser.error("Choose a directory under outputs/ or outside the project.")
    results = output / "results"
    results.mkdir(parents=True, exist_ok=True)
    if args.input_results:
        source_root = args.input_results.expanduser().resolve()
        if not source_root.is_dir():
            parser.error(f"Input directory does not exist: {source_root}")
        if source_root != results:
            for source in source_root.iterdir():
                dest = results / source.name
                if dest.exists():
                    continue
                if source.is_dir():
                    shutil.copytree(source, dest)
                else:
                    shutil.copy2(source, dest)
    absent = [p for p in task["required_results"] if not (results / p).is_file()]
    if absent:
        parser.error("Required results: " + ", ".join(absent) +
                     ". Run upstream experiments with the same --output-dir, or provide --input-results.")
    figures = output / "figures"
    figures.mkdir(exist_ok=True)
    env = dict(os.environ, BOUNDARY_RESULTS_DIR=str(results),
               BOUNDARY_FIGURES_DIR=str(figures), BOUNDARY_LOGS_DIR=str(output / "logs"),
               MPLBACKEND="Agg", PYTHONDONTWRITEBYTECODE="1")
    defaults = [x.replace("{output}", str(output)) for x in task["argv"]]
    print(f"Experiment: {args.experiment}\nOutput: {output}", flush=True)
    return subprocess.call([sys.executable, "-m", task["module"], *defaults, *extra],
                           cwd=ROOT, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
