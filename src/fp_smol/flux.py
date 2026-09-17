"""
Scharfetter-Gummel flux discretization for the drift-diffusion equation.

At each interior face, the flux is written as

    J_{i+1/2} = alpha_i * N_i + beta_i * N_{i+1}

using exponentially-fitted coefficients derived from the Bernoulli
function B(z) = z / (e^z - 1). This scheme is robust across the full
range of Peclet numbers (drift-dominated, diffusion-dominated, and
stiff/large-|Pe| regimes) and preserves positivity of the solution.
"""

import numpy as np


def bernoulli(z):
    """
    Bernoulli function B(z) = z / (e^z - 1), evaluated stably.

    Uses a Taylor expansion near z=0 (to avoid 0/0), and asymptotic
    limits for large |z| (to avoid overflow in exp(z)).

    Parameters
    ----------
    z : float or ndarray
        Input value(s).

    Returns
    -------
    float or ndarray
        B(z), same shape as input.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)

    small = np.abs(z) < 1.0e-3
    large_pos = z > 50
    large_neg = z < -50
    normal = ~small & ~large_pos & ~large_neg

    # Taylor expansion: B(z) ~ 1 - z/2 + z^2/12
    out[small] = 1 - z[small] / 2 + z[small] ** 2 / 12

    # Asymptotic limits
    out[large_pos] = 0.0
    out[large_neg] = -z[large_neg]

    # General case, via expm1 for accuracy near z=0 boundary
    out[normal] = z[normal] / np.expm1(z[normal])

    return out


def sg_face_coefficients(Ax_face, dx, D_x):
    """
    Scharfetter-Gummel face coefficients for the flux
    J_{i+1/2} = alpha_i * N_i + beta_i * N_{i+1}.

    Parameters
    ----------
    Ax_face : ndarray, shape (Ncells-1,)
        Drift velocity at interior faces, m^2/s (or consistent units).
    dx : float
        Uniform cell spacing.
    D_x : float
        Diffusivity (constant across the domain).

    Returns
    -------
    alpha_face : ndarray, shape (Ncells-1,)
        Coefficient multiplying the upwind (left) cell value.
    beta_face : ndarray, shape (Ncells-1,)
        Coefficient multiplying the downwind (right) cell value.
    """
    Ax_face = np.asarray(Ax_face, dtype=float)

    Pe_face = Ax_face * dx / D_x

    Bp = bernoulli(Pe_face)    # B(Pe)
    Bm = bernoulli(-Pe_face)   # B(-Pe)

    alpha_face = (D_x / dx) * Bm
    beta_face = -(D_x / dx) * Bp

    return alpha_face, beta_face


def face_average(values):
    """
    Simple arithmetic average of cell-center values onto interior faces.

    Parameters
    ----------
    values : ndarray, shape (N,)
        Cell-center values (e.g. drift Ax_c).

    Returns
    -------
    ndarray, shape (N-1,)
        Face-averaged values.
    """
    values = np.asarray(values, dtype=float)
    return 0.5 * (values[:-1] + values[1:])