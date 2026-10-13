"""
Figures for the talk, docs/dx_estimation/talk/Dx_estimation_talk.tex.

Needs the Monte Carlo results: run run_monte_carlo.py first.
Usage (from the repo root): python examples/dx_estimation/figs_talk.py
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import core
from noisy import Model, Obs, poisson_loss_grad, descend, curvature_sd
from run_monte_carlo import load_result

OUT = Path(__file__).resolve().parents[2] / "docs" / "dx_estimation" / "talk" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "legend.fontsize": 8,
                     "axes.grid": True, "grid.alpha": 0.3, "mathtext.fontset": "dejavusans",
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE, RED = "#2a6fb0", "#e07b28", "#c0392b"
SEQ = ["#9ecae1", "#4292c6", "#08519c", "#08306b"]
a_growth, s_super = core.a, core.s          # dx/dt = a*s for large droplets (um^2/s)
m = Model(8000)
r = np.sqrt(m.x)


def save(fig, name):
    fig.savefig(OUT / name)
    plt.close(fig)


def r_axis(ax, ticks=(1, 5, 10, 15)):
    """Secondary top axis showing radius for a plot whose x-axis is x = r^2."""
    sec = ax.secondary_xaxis("top", functions=(lambda x: np.sqrt(np.maximum(x, 0)), lambda q: q ** 2))
    sec.set_xticks(ticks); sec.set_xlabel(r"radius $r$ ($\mu$m)", fontsize=9)
    return sec


# ---------------------------------------------------------------- random walks in x
rng = np.random.default_rng(1)
t = np.linspace(0, 150, 301); dt = t[1] - t[0]
fig, ax = plt.subplots(1, 2, figsize=(5.6, 2.6), sharey=True)
for axi, (D, title) in zip(ax, [(1.0, r"weak turbulence, $D_x=1$"), (10.0, r"strong turbulence, $D_x=10$")]):
    for j in range(25):
        w = np.cumsum(np.r_[0, rng.normal(0, np.sqrt(2 * D * dt), len(t) - 1)])
        axi.plot(t, 200 + a_growth * s_super * t + w, lw=0.6, alpha=0.8)
    axi.plot(t, 200 + a_growth * s_super * t, "k--", lw=1.4, label="steady $s$ only")
    axi.set(xlabel="time (s)", title=title)
ax[0].set_ylabel(r"$x=r^2$ ($\mu$m$^2$)"); ax[0].legend(loc="upper left", frameon=False)
fig.tight_layout(); save(fig, "d_walks.pdf")

# ---------------------------------------------------------------- distributions vs D_x
fig, ax = plt.subplots(figsize=(3.4, 2.7))
for D, c in zip([0.1, 1, 10, 100], SEQ):
    N, _ = m.solve(D); nr = 2 * r * N
    ax.plot(r, nr / np.trapezoid(nr, r), color=c, lw=1.8, label=f"{D:g}")
ax.set(xlim=(0, 22), xlabel=r"radius $r$ ($\mu$m)", ylabel=r"fraction per $\mu$m")
ax.legend(title=r"$D_x$ ($\mu$m$^4$/s)", title_fontsize=8, frameon=False)
fig.tight_layout(); save(fig, "d_dsd.pdf")

# ---------------------------------------------------------------- numerical diffusion
Dt = np.logspace(-4, 1, 200)
A_big = a_growth * s_super
fig, ax = plt.subplots(figsize=(3.6, 2.8))
for n, ls in [(8000, "-"), (32000, "-.")]:
    dx = (30 ** 2 - (core.r_dry / 2) ** 2) / n
    Pe = A_big * dx / Dt
    ax.loglog(Dt, Dt * (Pe / 2) / np.tanh(Pe / 2), ls, color=BLUE, lw=1.8, label=f"{n} cells")
    ax.axhline(A_big * dx / 2, color=BLUE, lw=0.6, ls=":")
ax.loglog(Dt, Dt, "k--", lw=1, label=r"$D_{\rm eff}=D_x$")
ax.set(xlabel=r"true $D_x$ ($\mu$m$^4$/s)", ylabel=r"effective $D_{\rm eff}$ ($\mu$m$^4$/s)", ylim=(1e-4, 15))
ax.text(1.2e-4, 0.0135, r"floor $|A|\Delta x/2$", fontsize=7, color=BLUE)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout(); save(fig, "d_numdiff.pdf")

# ---------------------------------------------------------------- expected counts mu and one sample c
N10, _ = m.solve(10.0)
xe = m.xf[::8]
ob = Obs(m, 8, 0.1); mu = ob.mu(N10)
c = np.random.default_rng(2026).poisson(mu)
fig, ax = plt.subplots(1, 2, figsize=(6.2, 2.75), sharey=True)
ax[0].stairs(mu, xe, fill=True, color=BLUE, alpha=0.85)
ax[0].set(title=r"expected counts $\mu_k$", xlabel=r"$x=r^2$ ($\mu$m$^2$)", ylabel="droplets per bin", xlim=(0, 250))
ax[1].stairs(c, xe, fill=True, color=ORANGE, alpha=0.85, label=r"one sample $c_k$")
ax[1].stairs(mu, xe, color="k", lw=0.9, label=r"$\mu_k$")
ax[1].set(title=r"one sample: $c_k \sim \mathrm{Poisson}(\mu_k)$", xlabel=r"$x=r^2$ ($\mu$m$^2$)", xlim=(0, 250))
ax[1].legend(frameon=False)
for axi in ax:
    r_axis(axi)
fig.tight_layout(); save(fig, "d_mu_c.pdf")
print(f"mu/c: total mu {mu.sum():.0f}, bins with mu>=1: {(mu >= 1).sum()}, max mu {mu.max():.1f}")

# ---------------------------------------------------------------- effect of sample size
rng = np.random.default_rng(7)
fig, ax = plt.subplots(1, 3, figsize=(6.4, 2.45))
for axi, V in zip(ax, [0.01, 0.1, 1.0]):
    mu_v = Obs(m, 8, V).mu(N10); c_v = rng.poisson(mu_v)
    axi.stairs(c_v, xe, fill=True, color=ORANGE, alpha=0.85)
    axi.stairs(mu_v, xe, color="k", lw=0.9)
    axi.set(xlim=(0, 250), xlabel=r"$x=r^2$ ($\mu$m$^2$)", title=f"$V={V:g}$ cm$^3$: {c_v.sum():,} droplets")
    axi.set_ylim(0, max(c_v.max(), mu_v.max()) * 1.08)
ax[0].set_ylabel("droplets per bin")
fig.tight_layout(w_pad=0.6); save(fig, "d_samples.pdf")

# ---------------------------------------------------------------- adjoint at a wrong guess (Poisson loss, exact expected counts)
D = 20.0
N20, lu = m.solve(D); mu20 = ob.mu(N20)
g = ob.V * m.dx * np.repeat(2 * (1 - mu / mu20), ob.m)      # data = exact expected counts at D_x = 10
psi = lu.solve(-g, trans="T")
fig, ax = plt.subplots(2, 1, figsize=(3.6, 3.3), sharex=True)
ax[0].plot(r, N10, color="k", lw=1.5, label="data ($D_x=10$)")
ax[0].plot(r, N20, color=ORANGE, lw=1.5, label="guess ($D_x=20$)")
ax[0].set(ylabel=r"$N$"); ax[0].legend(frameon=False, fontsize=7)
ax[1].plot(r, psi, color=BLUE, lw=1.6); ax[1].axhline(0, color="k", lw=0.6)
ax[1].set(xlim=(0, 20), xlabel=r"radius $r$ ($\mu$m)", ylabel=r"adjoint $\psi$")
fig.tight_layout(h_pad=0.3); save(fig, "d_adjoint_stack.pdf")

# ---------------------------------------------------------------- single-sample loss parabola
cnt = np.random.default_rng(3).poisson(mu).astype(float)
f = lambda ell: poisson_loss_grad(ob, ell, cnt)
ell, J, gr, it, st = descend(f, 0.0)
sd = curvature_sd(f, ell); Jpp = 2 / sd ** 2
ls_ = np.linspace(ell - 4 * sd, ell + 4 * sd, 41); Js = np.array([f(l)[0] for l in ls_])
fig, ax = plt.subplots(figsize=(4.2, 2.8))
ax.axvspan(np.exp(ell - sd), np.exp(ell + sd), color="#2ca02c", alpha=0.15, lw=0, label=r"estimate $\pm1$ sd")
ax.plot(np.exp(ls_), Js - J, "o", ms=3, color=BLUE, label="loss (minus minimum)")
ax.plot(np.exp(ls_), 0.5 * Jpp * (ls_ - ell) ** 2, "-", color=ORANGE, lw=1.2, label="parabola from curvature")
ax.axvline(10, color="k", ls="--", lw=0.8, label="true $D_x$")
ax.axhline(1, color="0.5", lw=0.6, ls=":")
ax.set(xlabel=r"$D_x$ ($\mu$m$^4$/s)", ylabel=r"$\mathcal{J}-\mathcal{J}_{\min}$", ylim=(-0.5, 25))
ax.legend(frameon=False, loc="upper center", fontsize=7, ncol=2)
fig.tight_layout(); save(fig, "d_single.pdf")
print(f"single sample: D_hat = {np.exp(ell):.2f}, sd = {sd:.3f}, interval [{np.exp(ell-sd):.2f}, {np.exp(ell+sd):.2f}]")

# ---------------------------------------------------------------- exact-data sweep (least-squares loss)
trues = [0.01, 0.1, 1, 10, 100]; starts = [1e-3, 1e-2, 1e-1, 1, 10, 100, 1e3]
cmap = plt.get_cmap("viridis")
fig, axes = plt.subplots(1, 5, figsize=(7.4, 2.3), sharey=True)
for axx, Dtrue in zip(axes, trues):
    No, _ = m.solve(Dtrue); ff = lambda l: core.loss_and_grad(m, l, No)
    for i, D0 in enumerate(starts):
        e, hh, _, _ = core.gradient_descent(ff, np.log(D0)); hh = np.array(hh)
        axx.plot(np.exp(hh[:, 0]), ".-", lw=0.9, ms=2, color=cmap(i / 6))
    axx.axhline(Dtrue, color="k", ls="--", lw=0.7)
    axx.set(yscale="log", title=f"true {Dtrue:g}", xlabel="iteration")
axes[0].set_ylabel(r"estimate ($\mu$m$^4$/s)")
fig.tight_layout(w_pad=0.3); save(fig, "d_sweep.pdf")

# ---------------------------------------------------------------- precision vs number of droplets (Monte Carlo)
fig, ax = plt.subplots(figsize=(3.5, 2.7))
for D, col in [(10.0, BLUE), (0.1, RED)]:
    res = [load_result(D, V, 0.0) for V in (0.01, 0.1, 1.0)]
    n = [r_["total"].mean() for r_ in res]
    ax.loglog(n, [1 / np.sqrt(r_["I"]) for r_ in res], "-", color=col, lw=1.2)
    ax.loglog(n, [r_["ell"].std(ddof=1) for r_ in res], "o", color=col, ms=5, label=f"$D_x={D:g}$")
ax.set(xlabel="droplets per sample", ylabel=r"relative error of $\hat D_x$")
ax.legend(frameon=False, loc="upper right")
fig.tight_layout(); save(fig, "d_scaling.pdf")

# ---------------------------------------------------------------- single-sample error bars (first 40 Monte Carlo samples)
res = load_result(10.0, 0.1, 0.0)
ell40, sd40 = res["ell"][:40], res["sd"][:40]
Dh, lo, hi = np.exp(ell40), np.exp(ell40 - sd40), np.exp(ell40 + sd40)
inside = (lo <= 10) & (hi >= 10)
fig, ax = plt.subplots(figsize=(6.2, 2.3))
idx = np.arange(40)
for mask, col, lab in [(inside, BLUE, "interval contains truth"), (~inside, RED, "interval misses truth")]:
    ax.errorbar(idx[mask], Dh[mask], yerr=[Dh[mask] - lo[mask], hi[mask] - Dh[mask]], fmt="o", ms=3.5,
                color=col, capsize=2, lw=1, label=lab)
ax.axhline(10, color="k", ls="--", lw=0.8)
ax.set(xlabel="sample", ylabel=r"$\hat D_x \pm 1$ sd ($\mu$m$^4$/s)", ylim=(8.3, 13.0))
ax.legend(frameon=False, ncol=2, loc="upper left")
fig.tight_layout(); save(fig, "d_errorbars.pdf")
print(f"error bars: {inside.sum()} of 40 contain the truth")

# ---------------------------------------------------------------- doubling D_x at small and large values
fig, ax = plt.subplots(2, 1, figsize=(3.6, 3.3), sharex=True)
for axi, D in zip(ax, [0.1, 10.0]):
    N1, _ = m.solve(D); N2, _ = m.solve(2 * D)
    n1, n2 = 2 * r * N1, 2 * r * N2
    axi.fill_between(r, n1, n2, color=RED, alpha=0.25, lw=0)
    axi.plot(r, n1, color="k", lw=1.4, label=f"$D_x={D:g}$")
    axi.plot(r, n2, color=ORANGE, lw=1.4, ls="--", label=f"$D_x={2*D:g}$")
    axi.legend(frameon=False, fontsize=7, loc="upper right"); axi.set_ylim(0, 5800)
    axi.set_ylabel(r"cm$^{-3}\mu$m$^{-1}$", fontsize=8)
    print(f"doubling D={D}: relative L1 change {np.trapezoid(abs(n2 - n1), r) / np.trapezoid(n1, r):.3f}")
ax[1].set(xlim=(0, 20), xlabel=r"radius $r$ ($\mu$m)")
fig.tight_layout(h_pad=0.3); save(fig, "d_doubling_stack.pdf")

# ---------------------------------------------------------------- predicted error vs detection limit (Fisher information)
def fisher_per_droplet(D, r_det, eps=1e-4):
    obs = Obs(m, 8, 1.0, r_det)
    N, _ = m.solve(D); Np, _ = m.solve(D * np.exp(eps)); Nm, _ = m.solve(D * np.exp(-eps))
    mu_ = obs.mu(N); dmu = (obs.mu(Np) - obs.mu(Nm)) / (2 * eps); o = obs.observed
    return np.sum(dmu[o] ** 2 / mu_[o]) / mu_[o].sum()

r_dets = np.arange(0.0, 5.01, 0.25)
Ds = [0.1, 1, 10, 100]
ipd = {D: np.array([fisher_per_droplet(D, rd) for rd in r_dets]) for D in Ds}
Ncount = 2000
fig, ax = plt.subplots(figsize=(3.6, 2.85))
ax.axvspan(1.5, 2.5, color="0.85", lw=0)
ax.text(2.0, 0.025, "PDI-like\ncutoff", ha="center", va="bottom", fontsize=7, color="0.3")
ax.axhspan(0.3, 1.0, color=RED, alpha=0.06, lw=0)
ax.text(4.9, 0.85, "prediction unreliable", ha="right", va="top", fontsize=7, color=RED)
for D, c_ in zip(Ds, SEQ):
    ax.semilogy(r_dets, 1 / np.sqrt(Ncount * ipd[D]), "-", color=c_, lw=1.8)
lab = {0.1: (0.6, 0.235), 1: (4.95, 0.205), 10: (4.95, 0.083), 100: (4.95, 0.040)}
for D, c_ in zip(Ds, SEQ):
    xt, yt = lab[D]
    ax.text(xt, yt, f"$D_x={D:g}$", color=c_, fontsize=8, ha="right" if D != 0.1 else "center", va="center")
ax.set(xlabel=r"detection limit $r_{\rm det}$ ($\mu$m)", ylabel=r"predicted relative error", ylim=(0.02, 1.0), xlim=(0, 5))
fig.tight_layout(); save(fig, "d_cutoff_pred.pdf")

i25 = int(np.argmin(abs(r_dets - 2.5)))
print("\nPredicted relative error with r_det = 2.5 um, by droplets counted (talk, Q2 table):")
for D in Ds:
    print(f"  D={D:6g}: " + "  ".join(f"{100 / np.sqrt(n_ * ipd[D][i25]):6.1f}%" for n_ in (200, 2000, 20000)))
