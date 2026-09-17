import numpy as np
import pytest

from fp_smol.grid import make_grid
from fp_smol.physics import (
    xi1_func,
    kohler_params,
    drift_kohler,
    air_density,
    stokes_settling_coefficient,
)
from fp_smol.solver_steady import settling_loss_rate
from fp_smol.solver_time import run_backward_euler


def _time_v2_setup(D_x_um4=10.0, Ncells=2000):
    """Reproduce fp_FV_Kohler_physical_SG_time_v2.m's default setup."""
    um_per_m = 1e6

    T = 288.0
    p = 1000e2
    r_dry = 0.065e-6
    s = 0.0009
    h = 1.0
    Jinj = 1.0e8
    r_max = 30e-6
    r_min = r_dry / 2

    a_phys = 2 * xi1_func(T, p) / um_per_m**2
    D_x = D_x_um4 / um_per_m**4

    rho_air = air_density(T, p)
    C_stk = stokes_settling_coefficient(rho_air=rho_air)
    A_k, B_k = kohler_params(T, r_dry)

    grid = make_grid(r_min=r_min, r_max=r_max, Ncells=Ncells)
    Ax_c = drift_kohler(a_phys, s, grid.xc, A_k, B_k)
    lam = settling_loss_rate(grid.xc, C_stk, h)

    return grid, Ax_c, D_x, lam, Jinj


def test_converges_toward_steady_state():
    grid, Ax_c, D_x, lam, Jinj = _time_v2_setup(Ncells=1000)

    result = run_backward_euler(
        grid, Ax_c, D_x, lam, Jinj, dt=1.0, t_end=2000.0
    )

    # Error should shrink monotonically-ish and be tiny by the end.
    assert result.rel_L1[0] > result.rel_L1[-1]
    assert result.rel_L1[-1] < 1e-6


def test_error_decreases_over_time_checkpoints():
    grid, Ax_c, D_x, lam, Jinj = _time_v2_setup(Ncells=1000)

    result = run_backward_euler(
        grid, Ax_c, D_x, lam, Jinj, dt=1.0, t_end=2000.0,
        snapshot_times=[500, 1000, 2000],
    )

    N_500 = result.N_snapshots[500.0]
    N_1000 = result.N_snapshots[1000.0]
    N_2000 = result.N_snapshots[2000.0]
    N_ss = result.N_ss

    err_500 = np.sum(np.abs(N_500 - N_ss)) / np.sum(np.abs(N_ss))
    err_1000 = np.sum(np.abs(N_1000 - N_ss)) / np.sum(np.abs(N_ss))
    err_2000 = np.sum(np.abs(N_2000 - N_ss)) / np.sum(np.abs(N_ss))

    assert err_500 > err_1000 > err_2000
    assert err_2000 < 1e-5


def test_total_number_approaches_steady_total():
    grid, Ax_c, D_x, lam, Jinj = _time_v2_setup(Ncells=1000)

    result = run_backward_euler(
        grid, Ax_c, D_x, lam, Jinj, dt=1.0, t_end=2000.0
    )

    Ntot_final = np.sum(result.N_final) * grid.dx
    Ntot_ss = np.sum(result.N_ss) * grid.dx

    assert np.isclose(Ntot_final, Ntot_ss, rtol=1e-4)


def test_mean_radius_approaches_steady_value():
    grid, Ax_c, D_x, lam, Jinj = _time_v2_setup(Ncells=1000)

    result = run_backward_euler(
        grid, Ax_c, D_x, lam, Jinj, dt=1.0, t_end=2000.0
    )

    r_m_final = np.sum(grid.r_c * result.N_final) / np.sum(result.N_final)
    r_m_ss = np.sum(grid.r_c * result.N_ss) / np.sum(result.N_ss)

    assert np.isclose(r_m_final, r_m_ss, rtol=1e-4)


def test_zero_initial_condition_stays_nonnegative_throughout():
    grid, Ax_c, D_x, lam, Jinj = _time_v2_setup(Ncells=500)

    result = run_backward_euler(
        grid, Ax_c, D_x, lam, Jinj, dt=1.0, t_end=200.0,
        snapshot_times=[10, 50, 100, 200],
    )

    for N_snap in result.N_snapshots.values():
        assert np.all(N_snap >= -1e-8 * np.max(result.N_ss))