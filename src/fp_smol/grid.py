"""
Grid construction for the finite-volume Fokker-Planck solver in x = r^2 space.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    """
    Uniform finite-volume grid in x = r^2 space.

    Attributes
    ----------
    xf : ndarray, shape (N+1,)
        Cell face locations, m^2.
    xc : ndarray, shape (N,)
        Cell center locations, m^2.
    dx : float
        Uniform cell spacing, m^2.
    r_c : ndarray, shape (N,)
        Cell center radii, m (= sqrt(xc)).
    Ncells : int
        Number of cells.
    """

    xf: np.ndarray
    xc: np.ndarray
    dx: float
    r_c: np.ndarray
    Ncells: int


def make_grid(r_min, r_max, Ncells):
    """
    Build a uniform grid in x = r^2 space spanning [r_min^2, r_max^2].

    Parameters
    ----------
    r_min : float
        Minimum radius, meters. Use 0 for the non-Kohler (constant drift)
        case; use a small positive value (e.g. r_dry/2) for the Kohler
        case, to avoid the r=0 singularity in the Kohler equilibrium curve.
    r_max : float
        Maximum radius, meters.
    Ncells : int
        Number of finite-volume cells.

    Returns
    -------
    Grid
        Dataclass containing faces, centers, spacing, and center radii.
    """
    if r_min < 0:
        raise ValueError("r_min must be >= 0.")
    if r_max <= r_min:
        raise ValueError("r_max must be > r_min.")
    if Ncells < 1:
        raise ValueError("Ncells must be >= 1.")

    x_min = r_min**2
    x_max = r_max**2

    xf = np.linspace(x_min, x_max, Ncells + 1)
    xc = 0.5 * (xf[:-1] + xf[1:])
    dx = xf[1] - xf[0]
    r_c = np.sqrt(xc)

    return Grid(xf=xf, xc=xc, dx=dx, r_c=r_c, Ncells=Ncells)