"""
Reproduce the two example steady-state droplet-radius PDF plots from the
original MATLAB benchmark script (fp_FV_Kohler_physical_SG_benchmark.m):

  1. s = 0.09%, D_x = 0.01 um^4/s, r_dry = 0.065 um  -> r_m ~= 4.22 um
  2. s = 0.09%, D_x = 10   um^4/s, r_dry = 0.065 um  -> r_m ~= 5.74 um

Saves each figure as a PNG in ../figures/.
"""

import numpy as np
import matplotlib.pyplot as plt

from fp_smol.grid import make_grid
from fp_smol.physics import (
    xi1_func,
    kohler_params,
    drift_kohler,
    air_density,
    stokes_settling_coefficient,
)
from fp_smol.solver_steady import settling_loss_rate, solve_steady

um_per_m = 1e6


def run_case(D_x_um4, s=0.0009, r_dry=0.065e-6, T=288.0, p=1000e2,
             h=1.0, Jinj=1.0e8, Ncells=8000, r_max=30e-6):
    """Run one steady-state case and return (r_um, pdf_r, r_m_um, a_phys_um2)."""
    r_min = r_dry / 2

    a_phys = 2 * xi1_func(T, p) / um_per_m**2  # m^2/s
    D_x = D_x_um4 / um_per_m**4                # m^4/s

    rho_air = air_density(T, p)
    C_stk = stokes_settling_coefficient(rho_air=rho_air)
    A_k, B_k = kohler_params(T, r_dry)

    grid = make_grid(r_min=r_min, r_max=r_max, Ncells=Ncells)
    Ax_c = drift_kohler(a_phys, s, grid.xc, A_k, B_k)
    lam = settling_loss_rate(grid.xc, C_stk, h)

    N = solve_steady(grid, Ax_c, D_x, lam, Jinj)

    # Convert to a properly normalized radius PDF: n(r) = 2 r N(x)
    r = grid.r_c
    n_r = 2.0 * r * N
    trapz = getattr(np, "trapezoid", None) or np.trapz
    Z = trapz(n_r, r)
    pdf_r = n_r / Z

    r_m_um = np.sum(r * N) / np.sum(N) * um_per_m

    return r * um_per_m, pdf_r, r_m_um, a_phys * 1e12


def make_plot(D_x_um4, s_pct, r_dry_um, expected_r_m, filename):
    r_um, pdf_r, r_m_um, a_phys_um2 = run_case(
        D_x_um4=D_x_um4, s=s_pct / 100.0, r_dry=r_dry_um * 1e-6
    )

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.plot(r_um, pdf_r, linewidth=2)
    ax.set_xlabel(r"$r$ ($\mu$m)")
    ax.set_ylabel(r"$p_r(r)$ (m$^{-1}$)")
    ax.set_title("pdf(r): FV + SG with Kohler drift (Python port)")
    ax.set_xlim(0, 15)
    ax.grid(True)

    txt = (
        f"s (%) = {s_pct:.3g}\n"
        f"D_x (um^4/s) = {D_x_um4:.3g}\n"
        f"a_phys (um^2/s) = {a_phys_um2:.3g}\n"
        f"r_m (um) = {r_m_um:.3g}\n"
        f"r_dry (um) = {r_dry_um:.3g}"
    )
    ax.text(
        0.62, 0.85, txt, transform=ax.transAxes, va="top", fontsize=11,
        bbox=dict(boxstyle="round", facecolor="0.95", edgecolor="0.5"),
    )

    fig.tight_layout()
    fig.savefig(filename, dpi=200)
    plt.close(fig)

    print(f"{filename}: computed r_m = {r_m_um:.4g} um "
          f"(MATLAB reference: {expected_r_m} um)")


if __name__ == "__main__":
    make_plot(D_x_um4=0.01, s_pct=0.09, r_dry_um=0.065,
              expected_r_m=4.22, filename="../figures/pdf_r_Dx0_01.png")
    make_plot(D_x_um4=10.0, s_pct=0.09, r_dry_um=0.065,
              expected_r_m=5.74, filename="../figures/pdf_r_Dx10.png")
