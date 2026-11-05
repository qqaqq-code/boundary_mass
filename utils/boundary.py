"""Two-sided boundary-tube estimates for a fitted binary separator."""

import numpy as np


def binary_score_conventions(signed_distances, labels, epsilon):
    """Estimate boundary density and two-sided boundary-tube coefficients.

    ``kernel_label`` and ``hard_label`` estimate signed-distance density at
    zero, normalized by observed label balance. ``kernel_predicted`` and
    ``hard_predicted`` estimate the two-sided coefficient, normalized by
    predicted-class balance. A zero balance denominator gives NaN.
    """
    distances = np.asarray(signed_distances, dtype=float)
    labels = np.asarray(labels)
    if distances.ndim != 1 or not distances.size or labels.shape != distances.shape:
        raise ValueError("distances and labels must be matching nonempty vectors")
    if not np.isfinite(distances).all() or not np.isin(labels, [0, 1]).all():
        raise ValueError("distances must be finite and labels must be binary")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    positive = float(np.mean(distances >= 0))
    beta = min(positive, 1.0 - positive)
    alpha_y = min(float(np.mean(labels == 0)), float(np.mean(labels == 1)))
    tube = float(np.mean(np.abs(distances) <= epsilon))
    with np.errstate(over="ignore"):
        density = float(np.exp(-0.5 * (distances / epsilon) ** 2).mean()
                        / (np.sqrt(2.0 * np.pi) * epsilon))
    return dict(epsilon=float(epsilon), predicted_positive=positive,
                beta=beta, alpha_y=alpha_y, tube_fraction=tube,
                kernel_label=density / alpha_y if alpha_y > 0 else np.nan,
                hard_label=tube / (2 * epsilon * alpha_y) if alpha_y > 0 else np.nan,
                kernel_predicted=2 * density / beta if beta > 0 else np.nan,
                hard_predicted=tube / (epsilon * beta) if beta > 0 else np.nan)


def estimate_boundary_mass(signed_distances, epsilon, *, kernel="hard"):
    """Estimate effective boundary complexity using predicted-class balance.

    ``signed_distances`` contains independent diagnostic distances to a fixed
    fitted head. ``hard`` estimates tube probability divided by width and
    balance; ``gaussian`` estimates twice the density at zero divided by balance.
    Both target the two-sided coefficient in the paper as bandwidth shrinks.
    """
    distances = np.asarray(signed_distances, dtype=float)
    if distances.ndim != 1 or distances.size == 0:
        raise ValueError("signed_distances must be a nonempty one-dimensional array")
    if not np.all(np.isfinite(distances)):
        raise ValueError("signed_distances must contain finite values")
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    positive_mass = float(np.mean(distances >= 0))
    balance = min(positive_mass, 1.0 - positive_mass)
    if balance <= 0:
        raise ValueError("The diagnostic pool must cover both predicted classes")
    if kernel == "hard":
        coefficient = np.mean(np.abs(distances) <= epsilon) / epsilon
    elif kernel == "gaussian":
        with np.errstate(over="ignore"):
            weights = np.exp(-0.5 * (distances / epsilon) ** 2)
        coefficient = 2.0 * weights.mean() / (np.sqrt(2.0 * np.pi) * epsilon)
    else:
        raise ValueError("kernel must be 'hard' or 'gaussian'")
    return float(coefficient / balance)
