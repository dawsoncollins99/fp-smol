"""
Time-dependent finite-volume solver for the 1D Fokker-Planck equation in
x = r^2 space, using backward Euler time-stepping.

The steady operator A*N = b (from solver_steady) gives the semi-discrete
system

    dN/dt = (b - A N) / dx,

which backward Euler advances as

    [I + (dt/dx) A] N^{n+1} = N^n + (dt/dx) b.

The matrix M = I + (dt/dx)*A is constant across timesteps (fixed dt), so
it is factored once (sparse LU) and reused at every step, matching the
MATLAB reference implementation's use of a cached `decomposition` object.
"""

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from fp_smol.solver_steady import assemble_steady_system_sparse


@dataclass
class TimeSteppingResult:
    """
    Results of a backward-Euler time integration.

    Attributes
    ----------
    time : ndarray, shape (nsteps+1,)
        Time points, including t=0.
    N_final : ndarray, shape (Ncells,)
        Solution at the final time.
    N_snapshots : dict[float, ndarray]
        Solution at each requested snapshot time actually reached.
    rel_L1 : ndarray, shape (nsteps+1,)
        Relative L1 error to the steady solution N_ss at each time point.
    N_ss : ndarray, shape (Ncells,)
        The reference steady-state solution used for comparison.
    """

    time: np.ndarray
    N_final: np.ndarray
    N_snapshots: dict
    rel_L1: np.ndarray
    N_ss: np.ndarray


def run_backward_euler(
    grid,
    Ax_c,
    D_x,
    lam,
    Jinj,
    dt,
    t_end,
    N0=None,
    snapshot_times=None,
):
    """
    Integrate the time-dependent FP equation forward with backward Euler.

    Parameters
    ----------
    grid : Grid
    Ax_c : ndarray, shape (Ncells,)
    D_x : float
    lam : ndarray, shape (Ncells,)
    Jinj : float
    dt : float
        Timestep, seconds.
    t_end : float
        Final time, seconds.
    N0 : ndarray, shape (Ncells,), optional
        Initial condition. Defaults to an empty chamber (N=0), matching
        the reference validation setup.
    snapshot_times : array_like, optional
        Times at which to save the full N(x) profile, in addition to the
        final state. Times outside [0, t_end] are ignored.

    Returns
    -------
    TimeSteppingResult
    """
    Ncells = grid.Ncells
    dx = grid.dx

    A, b = assemble_steady_system_sparse(grid, Ax_c, D_x, lam, Jinj)

    # Reference steady solution, for convergence comparison.
    N_ss = spla.spsolve(A, b)

    # M = I + (dt/dx) * A, factored once and reused every step.
    M = sp.eye(Ncells, format="csc") + (dt / dx) * A
    lu = spla.splu(M)

    source_increment = (dt / dx) * b

    nsteps = int(round(t_end / dt))
    time = np.arange(nsteps + 1) * dt

    N = np.zeros(Ncells) if N0 is None else np.asarray(N0, dtype=float).copy()

    rel_L1 = np.zeros(nsteps + 1)
    Nss_norm = np.sum(np.abs(N_ss))
    rel_L1[0] = np.sum(np.abs(N - N_ss)) / Nss_norm

    snapshot_times = np.asarray(snapshot_times if snapshot_times is not None else [])
    snapshot_times = snapshot_times[(snapshot_times >= 0) & (snapshot_times <= t_end)]
    snapshot_steps = np.round(snapshot_times / dt).astype(int)

    N_snapshots = {}
    if 0 in snapshot_steps:
        N_snapshots[0.0] = N.copy()

    for k in range(1, nsteps + 1):
        rhs = N + source_increment
        N = lu.solve(rhs)

        # Backward Euler + SG should preserve positivity; clip tiny
        # roundoff negatives only.
        tol = 1e-12 * max(np.max(N_ss), 1e-300)
        N = np.where((N < 0) & (N >= -tol), 0.0, N)

        rel_L1[k] = np.sum(np.abs(N - N_ss)) / Nss_norm

        if k in snapshot_steps:
            N_snapshots[k * dt] = N.copy()

    return TimeSteppingResult(
        time=time,
        N_final=N,
        N_snapshots=N_snapshots,
        rel_L1=rel_L1,
        N_ss=N_ss,
    )