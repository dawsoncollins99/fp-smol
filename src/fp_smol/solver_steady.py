"""
Steady-state finite-volume solver for the 1D Fokker-Planck equation in
x = r^2 space, using the Scharfetter-Gummel flux discretization.

Solves the steady conservation law

    dJ/dx + lambda(x) N(x) = 0

with an injection flux Jinj at the left boundary and zero flux at the
right boundary, giving a tridiagonal linear system A*N = b.
"""

import numpy as np
import scipy.sparse as sp
from scipy.linalg import solve_banded

from fp_smol.flux import sg_face_coefficients, face_average


def settling_loss_rate(xc, C_stk, h):
    """
    Stokes settling loss rate lambda(x) = (C_stk/h) * x.

    Parameters
    ----------
    xc : ndarray
        Cell-center locations in x = r^2 space, m^2.
    C_stk : float
        Stokes settling coefficient, 1/(m s).
    h : float
        Chamber height, m.

    Returns
    -------
    ndarray
        Settling loss rate at each cell center, 1/s.
    """
    return (C_stk / h) * np.asarray(xc, dtype=float)


def _build_diagonals(grid, Ax_c, D_x, lam, Jinj):
    """
    Build the main/lower/upper diagonals and RHS for the steady FV system.

    Shared by both the banded (direct steady solve) and sparse (reused
    for time-stepping) system assembly below.

    Returns
    -------
    main, lower, upper : ndarray, shape (Ncells,)
        Diagonal entries. lower[i] multiplies N_{i-1} in row i;
        upper[i] multiplies N_{i+1} in row i. lower[0] and upper[-1]
        are unused (zero).
    b : ndarray, shape (Ncells,)
        Right-hand side.
    """
    Ncells = grid.Ncells
    dx = grid.dx

    Ax_face = face_average(Ax_c)
    alpha_face, beta_face = sg_face_coefficients(Ax_face, dx, D_x)

    main = np.zeros(Ncells)
    lower = np.zeros(Ncells)
    upper = np.zeros(Ncells)
    b = np.zeros(Ncells)

    # Row 0: J_{3/2} - Jinj + lambda_0 * N_0 * dx = 0
    main[0] = lam[0] * dx + alpha_face[0]
    upper[0] = beta_face[0]
    b[0] = Jinj

    # Interior rows
    for i in range(1, Ncells - 1):
        lower[i] = -alpha_face[i - 1]
        main[i] = alpha_face[i] - beta_face[i - 1] + lam[i] * dx
        upper[i] = beta_face[i]

    # Last row: -J_{N-1/2} + lambda_{N-1} * N_{N-1} * dx = 0
    lower[Ncells - 1] = -alpha_face[Ncells - 2]
    main[Ncells - 1] = -beta_face[Ncells - 2] + lam[Ncells - 1] * dx

    return main, lower, upper, b


def assemble_steady_system(grid, Ax_c, D_x, lam, Jinj):
    """
    Assemble the tridiagonal steady FV system A*N = b, in banded form
    for scipy.linalg.solve_banded.

    Parameters
    ----------
    grid : Grid
    Ax_c : ndarray, shape (Ncells,)
    D_x : float
    lam : ndarray, shape (Ncells,)
    Jinj : float

    Returns
    -------
    ab : ndarray, shape (3, Ncells)
    b : ndarray, shape (Ncells,)
    """
    Ncells = grid.Ncells
    main, lower, upper, b = _build_diagonals(grid, Ax_c, D_x, lam, Jinj)

    ab = np.zeros((3, Ncells))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = main
    ab[2, :-1] = lower[1:]

    return ab, b


def assemble_steady_system_sparse(grid, Ax_c, D_x, lam, Jinj):
    """
    Assemble the same tridiagonal steady FV system A*N = b, as a sparse
    CSC matrix. Used by the time-dependent solver, where the matrix is
    reused (factored once) across many timesteps.

    Returns
    -------
    A : scipy.sparse.csc_matrix, shape (Ncells, Ncells)
    b : ndarray, shape (Ncells,)
    """
    Ncells = grid.Ncells
    main, lower, upper, b = _build_diagonals(grid, Ax_c, D_x, lam, Jinj)

    A = sp.diags(
        [lower[1:], main, upper[:-1]],
        offsets=[-1, 0, 1],
        format="csc",
    )

    return A, b


def solve_steady(grid, Ax_c, D_x, lam, Jinj):
    """
    Solve the steady FV system for N(x) at cell centers.

    Parameters
    ----------
    grid : Grid
    Ax_c : ndarray, shape (Ncells,)
    D_x : float
    lam : ndarray, shape (Ncells,)
    Jinj : float

    Returns
    -------
    ndarray, shape (Ncells,)
        Number density N(x) at cell centers.
    """
    ab, b = assemble_steady_system(grid, Ax_c, D_x, lam, Jinj)
    return solve_banded((1, 1), ab, b)