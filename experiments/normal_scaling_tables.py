"""Summarize both halves of the normal/tangent comparison from one data set."""
import argparse
from pathlib import Path
import pandas as pd
from utils.paths import result_file, RESULTS_DIR


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=result_file(
        "normal_tangent/controlled_direction_sample_difficulty.csv"))
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "normal_tangent")
    args = parser.parse_args()
    data = pd.read_csv(args.input)
    keys = ["benchmark", "pair_id", "direction"]
    metrics = ["log1p_psi_eff", "early_acc_5spc", "aulc", "label_budget_to_90pct_full"]
    if data.duplicated(keys + ["scale"]).any():
        raise ValueError("Each task, backbone, direction and scale must occur once")
    base = data[data.scale == 1].set_index(keys)[metrics]
    expanded = data[data.scale == 2].set_index(keys)[metrics]
    if set(base.index) != set(expanded.index):
        raise ValueError("The two scales must contain the same configurations")
    effects = (expanded - base).reset_index()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data.groupby(["direction", "scale"])[metrics].mean().to_csv(args.output_dir / "paired_scale_means.csv")
    effects.groupby("direction")[metrics].mean().to_csv(args.output_dir / "paired_scale_effects.csv")
    effects.to_csv(args.output_dir / "paired_scale_differences.csv", index=False)


if __name__ == "__main__":
    main()
