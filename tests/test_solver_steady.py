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
from fp_smol.solver_steady import (
    settling_loss_rate,
    assemble_steady_system,
    solve_steady,
)


def test_settling_loss_rate_scales_with_x():
    xc = np.array([1e-12, 2e-12, 4e-12])
    C_stk = 1.2e8
    h = 1.0
    lam = settling_loss_rate(xc, C_stk, h)
    assert np.allclose(lam, (C_stk / h) * xc)


def test_assemble_shapes():
    grid = make_grid(r_min=0.0, r_max=30e-6, Ncells=50)
    Ax_c = np.full(grid.Ncells, 1e-12)
    D_x = 1e-16
    lam = settling_loss_rate(grid.xc, 1.2e8, 1.0)
    ab, b = assemble_steady_system(grid, Ax_c, D_x, lam, Jinj=1e8)

    assert ab.shape == (3, grid.Ncells)
    assert b.shape == (grid.Ncells,)
    assert b[0] == 1e8
    assert np.all(b[1:] == 0.0)


def test_solution_is_positive_and_finite():
    grid = make_grid(r_min=0.0, r_max=30e-6, Ncells=200)
    Ax_c = np.full(grid.Ncells, 1e-12)
    D_x = 1e-16
    lam = settling_loss_rate(grid.xc, 1.2e8, 1.0)
    N = solve_steady(grid, Ax_c, D_x, lam, Jinj=1e8)

    assert np.all(np.isfinite(N))
    assert np.all(N >= -1e-6 * np.max(np.abs(N)))  # allow tiny roundoff


def test_mass_balance_injection_equals_total_settling_loss():
    # At steady state, everything injected must eventually settle out:
    # sum(lambda_i * N_i * dx) should equal Jinj.
    grid = make_grid(r_min=0.0, r_max=30e-6, Ncells=2000)
    Ax_c = np.full(grid.Ncells, 5e-13)
    D_x = 1e-16
    lam = settling_loss_rate(grid.xc, 1.2e8, 1.0)
    Jinj = 1e8

    N = solve_steady(grid, Ax_c, D_x, lam, Jinj)
    total_loss = np.sum(lam * N) * grid.dx

    assert np.isclose(total_loss, Jinj, rtol=1e-2)


# ---------------------------------------------------------------------
# Full integration test: reproduce the MATLAB benchmark plots exactly
# ---------------------------------------------------------------------

def _run_benchmark_case(D_x_um4):
    """Reproduce the exact setup from fp_FV_Kohler_physical_SG_benchmark.m"""
    um_per_m = 1e6

    T = 288.0
    p = 1000e2
    r_dry = 0.065e-6
    s = 0.0009
    h = 1.0
    Jinj = 1.0e8
    Ncells = 8000
    r_max = 30e-6
    r_min = r_dry / 2

    a_phys = 2 * xi1_func(T, p) / um_per_m**2  # m^2/s
    D_x = D_x_um4 / um_per_m**4  # m^4/s

    rho_air = air_density(T, p)
    C_stk = stokes_settling_coefficient(rho_air=rho_air)

    A_k, B_k = kohler_params(T, r_dry)

    grid = make_grid(r_min=r_min, r_max=r_max, Ncells=Ncells)
    Ax_c = drift_kohler(a_phys, s, grid.xc, A_k, B_k)
    lam = settling_loss_rate(grid.xc, C_stk, h)

    N = solve_steady(grid, Ax_c, D_x, lam, Jinj)

    r_m_um = np.sum(grid.r_c * N) / np.sum(N) * um_per_m
    return r_m_um


def test_benchmark_case_low_diffusivity():
    # Matches s0_09_Dx0_01_a206_rdry0_065_pr_r.pdf: r_m = 4.22 um
    r_m_um = _run_benchmark_case(D_x_um4=0.01)
    assert abs(r_m_um - 4.22) < 0.05


def test_benchmark_case_high_diffusivity():
    # Matches s0_09_Dx10_a206_rdry0_065_pr_r.pdf: r_m = 5.74 um
    r_m_um = _run_benchmark_case(D_x_um4=10.0)
    assert abs(r_m_um - 5.74) < 0.05


def test_higher_diffusivity_gives_broader_larger_mean_radius():
    # Physical sanity check independent of exact benchmark numbers:
    # more turbulent diffusion should broaden the DSD toward larger r_m.
    r_m_low = _run_benchmark_case(D_x_um4=0.01)
    r_m_high = _run_benchmark_case(D_x_um4=10.0)
    assert r_m_high > r_m_low