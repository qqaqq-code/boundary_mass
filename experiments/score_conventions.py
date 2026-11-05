"""Evaluate kernel and tube estimates of boundary mass on frozen features.

Run from the project root with ``python -m experiments.score_conventions``.
Scoring mode fits heads on probe-training features and measures boundary mass
on diagnostic-heldout features. Geometry mode evaluates normal scaling.
Paths follow BOUNDARY_DATA_DIR and BOUNDARY_RESULTS_DIR.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from utils import binary_protocol as protocol
from utils.boundary import binary_score_conventions
from utils.paths import RESULTS_DIR
from experiments.pilot_refitting import make_pairs
from experiments.acquisition import balanced_pilot, initial_rng


def measure(model, features, labels, epsilon=None):
    distances = protocol.signed_distance(model, features)
    if epsilon is None:
        epsilon = max(0.05, float(np.quantile(np.abs(distances), 0.20)))
    return binary_score_conventions(distances, labels, epsilon)


def scoring(n_pairs, n_seeds, out):
    rows = []
    for backbone_id, (backbone, directory) in enumerate(protocol.CACHES.items()):
        blob = protocol.load_features(directory / "train_features.pt")
        for pair_id, (a, b) in enumerate(make_pairs()[:n_pairs]):
            mask = (blob["labels"] == a) | (blob["labels"] == b)
            X = np.asarray(blob["features"][mask], dtype="float32")
            y = np.asarray(blob["labels"][mask] == b, dtype="int64")
            probe, diag = protocol.split_probe_diag(y, pair_id, backbone_id)
            Xp, Xd = protocol.standardize_from_probe(X[probe], X[diag])
            yp, yd = y[probe], y[diag]
            meta = dict(pair_id=pair_id, class_a=a, class_b=b, backbone=backbone,
                        diag_n=len(yd))
            model = protocol.fit_linear_head(Xp, yp)
            rows.append(dict(**meta, protocol="full", pilot_spc=0, seed=-1,
                             **measure(model, Xd, yd)))
            groups = [np.flatnonzero(yp == c) for c in (0, 1)]
            for spc in (2, 5, 10):
                for seed in range(n_seeds):
                    rng = np.random.default_rng(5000 + pair_id * 37
                                                + backbone_id * 11 + spc * 3 + seed)
                    selected = np.r_[tuple(rng.choice(ids, spc, replace=False)
                                            for ids in groups)]
                    model = protocol.fit_linear_head(Xp[selected], yp[selected])
                    rows.append(dict(**meta, protocol="pilot", pilot_spc=spc,
                                     seed=seed, **measure(model, Xd, yd)))
            for seed in range(n_seeds):
                selected = balanced_pilot(yp, 2, initial_rng(pair_id, backbone_id, seed))
                model = protocol.fit_linear_head(Xp[selected], yp[selected])
                rows.append(dict(**meta, protocol="acquisition", pilot_spc=2,
                                 seed=seed, **measure(model, Xd, yd)))
            if (pair_id + 1) % 10 == 0 or pair_id + 1 == n_pairs:
                pd.DataFrame(rows).to_csv(out / "score_variants.csv", index=False)
                print(f"{backbone}: {pair_id + 1}/{n_pairs} tasks; {len(rows)} heads", flush=True)
        del blob
    return pd.DataFrame(rows)


def geometry(out):
    rows = []
    for backbone_id, (backbone, directory) in enumerate(list(protocol.CACHES.items())[:2]):
        train = protocol.load_features(directory / "train_features.pt")
        test = protocol.load_features(directory / "test_features.pt")
        for pair_id, (a, b) in enumerate(make_pairs()[:8]):
            m = (train["labels"] == a) | (train["labels"] == b)
            t = (test["labels"] == a) | (test["labels"] == b)
            X, Xd = protocol.standardize_from_probe(train["features"][m], test["features"][t])
            y, yd = (train["labels"][m] == b).astype(int), (test["labels"][t] == b).astype(int)
            def fit(features):
                return LogisticRegression(C=1.0, penalty="l2", solver="liblinear",
                    max_iter=1000, tol=1e-5, random_state=0).fit(features, y)
            base = fit(X)
            w = base.coef_.reshape(-1)
            u = w / np.linalg.norm(w)
            rng = np.random.default_rng(20260501 + 101 * backbone_id + pair_id)
            v = rng.normal(size=u.shape)
            md = (X[y == 1].mean(0) - X[y == 0].mean(0)).astype(float)
            md -= (md @ u) * u
            bases = [u] + ([md / np.linalg.norm(md)] if np.linalg.norm(md) > 1e-8 else [])
            for axis in bases:
                v -= (v @ axis) * axis
            v /= np.linalg.norm(v)
            d0 = protocol.signed_distance(base, Xd)
            epsilon = max(0.05, float(np.quantile(np.abs(d0), .2)))
            for direction, axis in (("normal", u), ("tangent", v)):
                for scale in (.5, 1., 2., 4.):
                    def transform(features):
                        return (features + (scale - 1) * (features @ axis)[:, None] * axis).astype("float32")
                    Xt, Xdt = transform(X), transform(Xd)
                    # Inverse transpose of the symmetric rank-one scaling map.
                    wt = w + (1 / scale - 1) * float(w @ axis) * axis
                    transported = (Xdt @ wt + base.intercept_[0]) / np.linalg.norm(wt)
                    model = fit(Xt)
                    refitted = protocol.signed_distance(model, Xdt)
                    distance_factor = scale if direction == "normal" else 1.
                    for head, distances, width in (
                        ("transported_fixed_width", transported, epsilon),
                        ("transported_scaled_width", transported, distance_factor * epsilon),
                        ("refitted_fixed_width", refitted, epsilon),
                    ):
                        values = binary_score_conventions(distances, yd, width)
                        rows.append(dict(backbone=backbone, pair_id=pair_id,
                            direction=direction, scale=scale, head=head,
                            predicted_ratio=1 / distance_factor,
                            transport_distance_error=float(np.max(np.abs(transported-distance_factor*d0))),
                            **values))
            print(f"geometry {backbone}: {pair_id + 1}/8", flush=True)
        del train, test
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "geometry_scores.csv", index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("scoring", "geometry"), default="scoring")
    parser.add_argument("--n-pairs", type=int, default=40)
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "score_conventions")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "scoring":
        scoring(args.n_pairs, args.n_seeds, args.output_dir)
    else:
        geometry(args.output_dir)


if __name__ == "__main__":
    main()
