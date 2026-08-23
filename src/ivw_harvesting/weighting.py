"""
Count-only weighting schemes for intervention-harvesting position-bias estimation.

This module implements the weighting schemes studied in the CIKM 2026 short paper
"Surrogate Inverse-Variance Weighting for Intervention Harvesting on Heavy-Tailed
Click Logs" (Magnani & Xie, Coupang).

Each weight function takes arrays (N_k, N_kp) of impression counts at two positions
for a set of (query, document) pairs and returns (omega_k, omega_kp), the weights
applied to the numerator and denominator sums respectively.

The paper's contribution is a family of *count-only* (surrogate inverse-variance)
weights that depend only on the per-pair impression counts, not on clicks, and so
preserve ratio-unbiasedness while reducing variance on heavy-tailed logs:

    original / uniform  omega_i = 1                          (Agarwal et al. 2019 baseline)
    min                 omega_i = min(N_k^i, N_kp^i)         (proposed)
    harmonic            omega_i = N_k^i N_kp^i / (N_k^i + N_kp^i)   (proposed)

For symmetric schemes (original, min, harmonic, adaptive) omega_k == omega_kp.
For clipped, omega_k != omega_kp (breaks ratio-unbiasedness; lower variance, biased).

The ratio estimator uses:
    R = sum_i omega_kp_i * CTR_kp_i / sum_i omega_k_i * CTR_k_i

In practice the build functions compute per-pair factors:
    factor_0 = omega_k / N_k    (multiply raw clicks_k by this)
    factor_1 = omega_kp / N_kp  (multiply raw clicks_kp by this)
"""

from typing import Callable, Tuple

import numpy as np


def weight_original(N_k: np.ndarray, N_kp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Unit weights: omega_i = 1 (uniform baseline of Agarwal et al. 2019)."""
    ones = np.ones(len(N_k), dtype=float)
    return ones, ones.copy()


def weight_min(N_k: np.ndarray, N_kp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Min-impressions weights: omega_i = min(N_k^i, N_kp^i)."""
    omega = np.minimum(N_k, N_kp).astype(float)
    return omega, omega.copy()


def weight_harmonic(N_k: np.ndarray, N_kp: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Harmonic weights: omega_i = N_k^i * N_kp^i / (N_k^i + N_kp^i).

    Optimal when alpha == beta (equal propensity-derived coefficients).
    Ratio-unbiased: omega depends only on impression counts, not clicks.
    """
    omega = (N_k * N_kp).astype(float) / (N_k + N_kp)
    return omega, omega.copy()


def weight_clipped(
    N_k: np.ndarray, N_kp: np.ndarray, tau: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Clipped weights: subsample majority position to enforce max ratio tau.

    NOTE: omega_k != omega_kp -> breaks ratio-unbiasedness.
    Each position uses its own effective count after clipping:
        omega_k^i  = min(N_k^i,  tau * N_kp^i)
        omega_kp^i = min(N_kp^i, tau * N_k^i)
    """
    omega_k = np.minimum(N_k.astype(float), tau * N_kp)
    omega_kp = np.minimum(N_kp.astype(float), tau * N_k)
    return omega_k, omega_kp


def weight_adaptive(
    N_k: np.ndarray, N_kp: np.ndarray, alpha: float, beta: float
) -> Tuple[np.ndarray, np.ndarray]:
    """Variance-optimal weights given propensity-derived coefficients alpha, beta.

    omega_i* = N_k^i * N_kp^i / (alpha * N_kp^i + beta * N_k^i)

    When alpha == beta this reduces to harmonic weighting.
    Use cross-fitting to preserve unbiasedness.
    """
    omega = (N_k * N_kp).astype(float) / (alpha * N_kp + beta * N_k)
    return omega, omega.copy()


# ---------------------------------------------------------------------------
# Registry: map string scheme names to weight functions
# ---------------------------------------------------------------------------

def get_weight_fn(scheme: str) -> Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Return a weight function for the given scheme name.

    Supported names:
        "original"            - unit weights (uniform baseline)
        "variance_reduced"    - alias for "min" (backward compat)
        "min"                 - min-impressions weights
        "harmonic"            - harmonic weights
        "clipped_2"           - clipped weights with tau=2
        "clipped_5"           - clipped weights with tau=5
        "clipped_10"          - clipped weights with tau=10
    """
    _registry = {
        "original": weight_original,
        "variance_reduced": weight_min,  # backward compat alias
        "min": weight_min,
        "harmonic": weight_harmonic,
        "clipped_2": lambda N_k, N_kp: weight_clipped(N_k, N_kp, tau=2.0),
        "clipped_5": lambda N_k, N_kp: weight_clipped(N_k, N_kp, tau=5.0),
        "clipped_10": lambda N_k, N_kp: weight_clipped(N_k, N_kp, tau=10.0),
    }
    if scheme not in _registry:
        raise ValueError(
            f"Unknown weighting scheme {scheme!r}. "
            f"Choose from: {sorted(_registry)}"
        )
    return _registry[scheme]


# ---------------------------------------------------------------------------
# Variance diagnostics
# ---------------------------------------------------------------------------

def variance_term(omega: np.ndarray, N_r: np.ndarray) -> float:
    """Compute V_r(omega) = sum_i omega_i^2 / N_r^i / (sum_i omega_i)^2.

    Used in the delta-method conditional variance:
        Var(R | N) ~= R^2 * [alpha * V_k(omega) + beta * V_kp(omega)]
    """
    W = omega.sum()
    if W == 0:
        return 0.0
    return float(np.sum(omega**2 / N_r) / W**2)


def variance_decomposition(
    N_k: np.ndarray,
    N_kp: np.ndarray,
    alpha: float,
    beta: float,
) -> dict:
    """Compute the variance-reduction components for harmonic vs original weights.

    Returns a dict with keys:
        D_k, D_kp          - V_k(harmonic) - V_k(original), idem for k'
        term1              - beta * (D_k + D_kp)   (always <= 0 by Cauchy-Schwarz)
        term2              - -(beta - alpha) * D_k  (<= 0 when D_k >= 0)
        total              - term1 + term2 = variance reduction
    """
    omega_h, _ = weight_harmonic(N_k, N_kp)
    omega_1, _ = weight_original(N_k, N_kp)

    V_k_h = variance_term(omega_h, N_k)
    V_kp_h = variance_term(omega_h, N_kp)
    V_k_1 = variance_term(omega_1, N_k)
    V_kp_1 = variance_term(omega_1, N_kp)

    D_k = V_k_h - V_k_1
    D_kp = V_kp_h - V_kp_1
    term1 = beta * (D_k + D_kp)
    term2 = -(beta - alpha) * D_k

    return {
        "D_k": D_k,
        "D_kp": D_kp,
        "term1": term1,
        "term2": term2,
        "total": term1 + term2,
    }
