import numpy as np
import matplotlib.pyplot as plt

from fp_smol.grid import make_grid
from fp_smol.physics import (
    xi1_func, kohler_params, kohler_s_eq, kohler_critical, drift_kohler,
    air_density, stokes_settling_coefficient,
)
from fp_smol.solver_steady import settling_loss_rate
from fp_smol.solver_time import run_backward_euler

um_per_m = 1e6

# ---------------------------------------------------------------------
# 1. Kohler curve
# ---------------------------------------------------------------------
T = 288.0
r_dry = 0.065e-6
A_k, B_k = kohler_params(T, r_dry)
r_crit, s_crit = kohler_critical(A_k, B_k)

r_plot = np.linspace(0.05e-6, 2e-6, 2000)
s_eq = kohler_s_eq(r_plot, A_k, B_k)

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(r_plot * um_per_m, s_eq * 100, linewidth=2)
ax.axvline(r_crit * um_per_m, color="gray", linestyle="--", linewidth=1)
ax.axhline(s_crit * 100, color="gray", linestyle="--", linewidth=1)
ax.plot([r_crit * um_per_m], [s_crit * 100], "ro", zorder=5)
ax.annotate(
    f"critical point\n(r_crit={r_crit*um_per_m:.3f} um,\n s_crit={s_crit*100:.3f} %)",
    xy=(r_crit * um_per_m, s_crit * 100),
    xytext=(r_crit * um_per_m * 3, s_crit * 100 * 1.3),
    arrowprops=dict(arrowstyle="->"),
    fontsize=10,
)
ax.set_xlabel(r"Droplet radius $r$ ($\mu$m)")
ax.set_ylabel(r"Equilibrium supersaturation $s_{eq}$ (%)")
ax.set_title(f"Kohler curve for a dry aerosol radius of {r_dry*um_per_m:.3f} um")
ax.set_ylim(-0.5, s_crit * 100 * 1.6)
ax.grid(True)
fig.tight_layout()
fig.savefig("../figures/kohler_curve.png", dpi=200)
plt.close(fig)
print("Saved kohler_curve.png")

# ---------------------------------------------------------------------
# 2. Time-dependent convergence to steady state
# ---------------------------------------------------------------------
s = 0.0009
p = 1000e2
h = 1.0
Jinj = 1.0e8
Ncells = 2000
r_max = 30e-6
r_min = r_dry / 2
D_x_um4 = 10.0

a_phys = 2 * xi1_func(T, p) / um_per_m**2
D_x = D_x_um4 / um_per_m**4
rho_air = air_density(T, p)
C_stk = stokes_settling_coefficient(rho_air=rho_air)

grid = make_grid(r_min=r_min, r_max=r_max, Ncells=Ncells)
Ax_c = drift_kohler(a_phys, s, grid.xc, A_k, B_k)
lam = settling_loss_rate(grid.xc, C_stk, h)

result = run_backward_euler(grid, Ax_c, D_x, lam, Jinj, dt=1.0, t_end=2000.0)

fig, ax = plt.subplots(figsize=(7, 5))
ax.semilogy(result.time, result.rel_L1, linewidth=2)
ax.set_xlabel("time (s)")
ax.set_ylabel(r"relative $L_1$ error to steady solution")
ax.set_title("Backward-Euler convergence to the steady-state solution")
ax.grid(True, which="both")
fig.tight_layout()
fig.savefig("../figures/time_convergence.png", dpi=200)
plt.close(fig)
print("Saved time_convergence.png")

# ---------------------------------------------------------------------
# 3. PDF snapshots approaching steady state
# ---------------------------------------------------------------------
result2 = run_backward_euler(
    grid, Ax_c, D_x, lam, Jinj, dt=1.0, t_end=2000.0,
    snapshot_times=[50, 200, 500, 1000],
)

fig, ax = plt.subplots(figsize=(7, 5))
r_um = grid.r_c * um_per_m
for t_snap, N_snap in sorted(result2.N_snapshots.items()):
    n_r = 2.0 * grid.r_c * N_snap
    trapz = getattr(np, "trapezoid", None) or np.trapz
    Z = trapz(n_r, grid.r_c)
    if Z > 0:
        pdf_r = n_r / Z
        ax.plot(r_um, pdf_r, linewidth=1.5, label=f"t = {t_snap:.0f} s")

n_r_ss = 2.0 * grid.r_c * result2.N_ss
trapz = getattr(np, "trapezoid", None) or np.trapz
Z_ss = trapz(n_r_ss, grid.r_c)
pdf_r_ss = n_r_ss / Z_ss
ax.plot(r_um, pdf_r_ss, "k--", linewidth=2, label="steady state")

ax.set_xlabel(r"$r$ ($\mu$m)")
ax.set_ylabel(r"$p_r(r)$ (m$^{-1}$)")
ax.set_title("Radius PDF relaxing toward steady state (empty chamber start)")
ax.set_xlim(0, 15)
ax.legend()
ax.grid(True)
fig.tight_layout()
fig.savefig("../figures/time_pdf_snapshots.png", dpi=200)
plt.close(fig)
print("Saved time_pdf_snapshots.png")
