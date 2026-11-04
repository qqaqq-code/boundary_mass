"""Two-sided boundary-tube estimates for a fitted binary separator."""

import numpy as np


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
