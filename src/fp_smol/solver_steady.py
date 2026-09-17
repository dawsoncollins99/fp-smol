"""
Steady-state finite-volume solver for the 1D Fokker-Planck equation in
x = r^2 space, using the Scharfetter-Gummel flux discretization.

Solves the steady conservation law

    dJ/dx + lambda(x) N(x) = 0

with an injection flux Jinj at the left boundary and zero flux at the
right boundary, giving a tridiagonal linear system A*N = b.
"""

import numpy as np
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


def assemble_steady_system(grid, Ax_c, D_x, lam, Jinj):
    """
    Assemble the tridiagonal steady FV system A*N = b, in banded form.

    Parameters
    ----------
    grid : Grid
        Grid object from fp_smol.grid.make_grid.
    Ax_c : ndarray, shape (Ncells,)
        Drift at cell centers, m^2/s.
    D_x : float
        Diffusivity, m^4/s.
    lam : ndarray, shape (Ncells,)
        Settling loss rate at cell centers, 1/s.
    Jinj : float
        Injection flux at the left boundary.

    Returns
    -------
    ab : ndarray, shape (3, Ncells)
        Banded-matrix representation for scipy.linalg.solve_banded.
    b : ndarray, shape (Ncells,)
        Right-hand side.
    """
    Ncells = grid.Ncells
    dx = grid.dx

    Ax_face = face_average(Ax_c)
    alpha_face, beta_face = sg_face_coefficients(Ax_face, dx, D_x)

    main = np.zeros(Ncells)
    lower = np.zeros(Ncells)  # lower[i] multiplies N_{i-1} in row i
    upper = np.zeros(Ncells)  # upper[i] multiplies N_{i+1} in row i
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

    # Banded storage for solve_banded: row 0 = superdiagonal (shifted),
    # row 1 = main diagonal, row 2 = subdiagonal (shifted)
    ab = np.zeros((3, Ncells))
    ab[0, 1:] = upper[:-1]
    ab[1, :] = main
    ab[2, :-1] = lower[1:]

    return ab, b


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