"Run a small boundary-mass example with bundled features. See docs/EXPERIMENTS.md for usage."

from utils.paths import PROJECT_ROOT, FIGURES_DIR
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from utils.boundary import estimate_boundary_mass

ROOT = PROJECT_ROOT
DATA = PROJECT_ROOT / "examples/toy_boundary_features.csv"
OUT = FIGURES_DIR / "demo_boundary_mass.png"
OUT.parent.mkdir(parents=True, exist_ok=True)
EPS = 0.25

rows = []
with DATA.open(newline="", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        rows.append(row)
X = np.array([[float(r["feature_1"]), float(r["feature_2"])] for r in rows])
y = np.array([int(r["label"]) for r in rows])
split = np.array([r["split"] for r in rows])

train = split == "probe_train"
diag = split == "diagnostic"
mu0 = X[train & (y == 0)].mean(axis=0)
mu1 = X[train & (y == 1)].mean(axis=0)
w = mu1 - mu0
mid = 0.5 * (mu0 + mu1)
b = -float(w @ mid)
norm_w = float(np.linalg.norm(w))
if norm_w <= 1e-12:
    raise RuntimeError("Degenerate demo boundary")

dist = (X[diag] @ w + b) / norm_w
tube_mass = float(np.mean(np.abs(dist) <= EPS))
predicted_positive_mass = float(np.mean(dist >= 0))
class_balance = min(predicted_positive_mass, 1.0 - predicted_positive_mass)
if class_balance <= 0:
    raise RuntimeError("The diagnostic pool must occupy both predicted classes")
psi_hat = estimate_boundary_mass(dist, EPS)

print(f"demo epsilon: {EPS:.3f}")
print(f"diagnostic tube mass: {tube_mass:.4f}")
print(f"diagnostic predicted-class balance: {class_balance:.4f}")
print(f"estimated effective boundary complexity: {psi_hat:.4f}")

fig, ax = plt.subplots(figsize=(5.2, 3.9), dpi=150)
ax.scatter(X[y == 0, 0], X[y == 0, 1], s=18, c="#246BFE", label="class 0", alpha=0.85)
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=18, c="#FF6A21", label="class 1", alpha=0.85)
xx = np.linspace(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3, 200)
if abs(w[1]) > 1e-12:
    yy = -(w[0] * xx + b) / w[1]
    ax.plot(xx, yy, c="black", lw=1.4, label="probe boundary")
    for sign in [-1, 1]:
        yy_tube = -(w[0] * xx + b - sign * EPS * norm_w) / w[1]
        ax.plot(xx, yy_tube, c="#D95F5F", lw=1.0, alpha=0.7)
ax.set_title(f"effective boundary complexity = {psi_hat:.3f}")
ax.set_xlabel("feature 1")
ax.set_ylabel("feature 2")
ax.legend(frameon=False, loc="best", fontsize=8)
fig.tight_layout()
fig.savefig(OUT)
print(f"wrote {OUT}")
