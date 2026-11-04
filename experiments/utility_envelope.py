"Compute the boundary-utility envelope. See docs/EXPERIMENTS.md for usage."

from __future__ import annotations
from utils.paths import PROJECT_ROOT, RESULTS_DIR, FIGURES_DIR, result_file
import csv, json, math, statistics
from pathlib import Path
from collections import defaultdict
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = PROJECT_ROOT
STUDY_ROOT = PROJECT_ROOT
ART = RESULTS_DIR
FIG = FIGURES_DIR
ART.mkdir(parents=True, exist_ok=True)
FIG.mkdir(parents=True, exist_ok=True)
CSV = result_file("boundary_utility/boundary_utility_epochs.csv")


def fl(x):
    try:
        return float(x)
    except Exception:
        return float("nan")


def dataset_from_path(path: str) -> str:
    s = path.lower()
    if "tiny" in s:
        return "Tiny-ImageNet/R50"
    if "vit" in s:
        return "CIFAR-100/ViT"
    if "cifar100" in s:
        return "CIFAR-100/R50"
    return "frozen runs"


rows = []
with CSV.open(newline="") as fh:
    reader = csv.DictReader(fh)
    for r in reader:
        acc = fl(r.get("utility"))
        psi = fl(r.get("heldout_psieff"))
        if not (math.isfinite(acc) and math.isfinite(psi) and psi > 0):
            continue
        r["_acc"] = acc
        r["_psi"] = psi
        r["_logpsi"] = math.log1p(psi)
        r["_dataset"] = dataset_from_path(r.get("run_path", ""))
        rows.append(r)
# Envelope by 1 point accuracy bins.
bins = defaultdict(list)
for r in rows:
    b = math.floor(r["_acc"] * 100) / 100.0
    bins[b].append(r)
env = []
for b, items in sorted(bins.items()):
    if len(items) >= 5:
        best = min(items, key=lambda x: x["_logpsi"])
        env.append(
            {
                "acc_bin": b,
                "n": len(items),
                "min_logpsi": best["_logpsi"],
                "mode": best.get("mode", ""),
                "spc": best.get("train_samples_per_class", ""),
                "run_name": best.get("run_name", ""),
                "acc": best["_acc"],
            }
        )
# Group into trajectories.
runs = defaultdict(list)
for r in rows:
    key = (
        r.get("run_path", ""),
        r.get("run_name", ""),
        r.get("mode", ""),
        r.get("train_samples_per_class", ""),
        r.get("seed", ""),
    )
    runs[key].append(r)
patterns = []
for key, items in runs.items():
    items = sorted(items, key=lambda r: fl(r.get("epoch")))
    if len(items) < 10:
        continue
    ps = [x["_psi"] for x in items]
    acc = [x["_acc"] for x in items]
    p0, pmax, pend = ps[0], max(ps), ps[-1]
    imax = ps.index(pmax)
    early_expand = pmax > 1.05 * p0 and imax <= max(2, int(0.25 * len(ps)))
    compress = pend < 0.95 * pmax
    patterns.append(
        {
            "key": key,
            "items": items,
            "mode": key[2],
            "spc": key[3],
            "seed": key[4],
            "early_expand": early_expand,
            "compress": compress,
            "p0": p0,
            "pmax": pmax,
            "pend": pend,
            "imax": imax,
            "acc0": acc[0],
            "accmax": max(acc),
            "accend": acc[-1],
            "score": (pmax / max(p0, 1e-9)) - (pend / max(pmax, 1e-9)),
        }
    )
rep = max(patterns, key=lambda x: x["score"]) if patterns else None


# Distance to envelope for final points.
def env_value(acc: float):
    if not env:
        return float("nan")
    b = math.floor(acc * 100) / 100.0
    candidates = [e for e in env if abs(e["acc_bin"] - b) <= 1e-9]
    if not candidates:
        candidates = sorted(env, key=lambda e: abs(e["acc_bin"] - b))[:1]
    return candidates[0]["min_logpsi"]


finals = []
for p in patterns:
    last = p["items"][-1]
    ev = env_value(last["_acc"])
    finals.append(
        {
            "mode": p["mode"],
            "spc": p["spc"],
            "run_name": p["key"][1],
            "acc": last["_acc"],
            "logpsi": last["_logpsi"],
            "distance": last["_logpsi"] - ev,
            "early_expand": p["early_expand"],
            "compress": p["compress"],
        }
    )
mode_summary = []
for mode in sorted(set(f["mode"] for f in finals)):
    vals = [
        f["distance"]
        for f in finals
        if f["mode"] == mode and math.isfinite(f["distance"])
    ]
    comps = [f for f in finals if f["mode"] == mode]
    if vals:
        mode_summary.append(
            {
                "mode": mode,
                "n": len(vals),
                "distance_mean": sum(vals) / len(vals),
                "distance_median": statistics.median(vals),
                "compression_fraction": sum(
                    1 for f in comps if f["early_expand"] and f["compress"]
                )
                / len(comps),
            }
        )
# Figures.
plt.rcParams.update({"font.size": 8})
fig, axs = plt.subplots(1, 3, figsize=(13.0, 3.35))
colors = {"standard": "#4C78A8", "weight_decay": "#F58518", "psieff": "#54A24B"}
# Envelope scatter, subsample for readability.
sub = rows[:: max(1, len(rows) // 4000)]
for mode in ["standard", "weight_decay", "psieff"]:
    xs = [r["_acc"] for r in sub if r.get("mode") == mode]
    ys = [r["_logpsi"] for r in sub if r.get("mode") == mode]
    axs[0].scatter(xs, ys, s=4, alpha=0.18, color=colors.get(mode, "gray"), label=mode)
axs[0].plot(
    [e["acc_bin"] + 0.005 for e in env],
    [e["min_logpsi"] for e in env],
    color="black",
    lw=1.6,
    label="empirical envelope",
)
axs[0].set_xlabel("accuracy")
axs[0].set_ylabel(r"$\log(1+\hat\psi_{eff})$")
axs[0].set_title("Envelope")
axs[0].legend(frameon=False, fontsize=7)
if rep:
    its = rep["items"]
    xs = [r["_acc"] for r in its]
    ys = [r["_logpsi"] for r in its]
    axs[1].plot(xs, ys, color=colors.get(rep["mode"], "black"), lw=1.6)
    axs[1].scatter([xs[0]], [ys[0]], s=18, color="black", label="start")
    axs[1].scatter(
        [xs[rep["imax"]]], [ys[rep["imax"]]], s=18, color="red", label="peak"
    )
    axs[1].scatter(
        [xs[-1]], [ys[-1]], s=18, color="white", edgecolor="black", label="final"
    )
    axs[1].set_title(f"single run: {rep['mode']}, spc={rep['spc']}")
    axs[1].legend(frameon=False, fontsize=7)
axs[1].set_xlabel("accuracy")
axs[1].set_ylabel(r"$\log(1+\hat\psi_{eff})$")
labels = [m["mode"] for m in mode_summary]
vals = [m["distance_median"] for m in mode_summary]
axs[2].bar(labels, vals, color=[colors.get(x, "gray") for x in labels])
axs[2].axhline(0, color="black", lw=0.8)
axs[2].set_ylabel("median final distance to envelope")
axs[2].set_title("Final proximity")
for tick in axs[2].get_xticklabels():
    tick.set_rotation(20)
fig.tight_layout()
fig.savefig(FIG / "ib_state_plane_panel.png", dpi=240)
# Also save separate panels for appendix.
for name, axidx in []:
    pass
summary = {
    "num_epoch_rows": len(rows),
    "num_runs": len(patterns),
    "envelope_bins": len(env),
    "compression_runs": sum(1 for p in patterns if p["early_expand"] and p["compress"]),
    "compression_fraction": sum(
        1 for p in patterns if p["early_expand"] and p["compress"]
    )
    / max(1, len(patterns)),
    "representative": None
    if rep is None
    else {
        k: rep[k]
        for k in [
            "mode",
            "spc",
            "seed",
            "p0",
            "pmax",
            "pend",
            "acc0",
            "accmax",
            "accend",
            "imax",
        ]
    },
    "mode_summary": mode_summary,
}
(result_file("boundary_utility/ib_state_plane_analysis.json")).write_text(
    json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
)
with (result_file("boundary_utility/ib_state_plane_mode_table.tex")).open(
    "w", encoding="utf-8"
) as f:
    for m in mode_summary:
        f.write(
            f"{m['mode'].replace('_', ' ')} & {m['n']} & ${m['distance_median']:.3f}$ & ${m['distance_mean']:.3f}$ & ${100 * m['compression_fraction']:.1f}\\%$\\\\\n"
        )
print(json.dumps(summary, indent=2))
