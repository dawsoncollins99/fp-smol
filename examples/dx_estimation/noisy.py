import numpy as np
from core import *

class Obs:
    """Bins of m consecutive cells; optional detection cutoff r_det (um): only bins entirely above it are observed."""
    def __init__(self, model, m=8, V=0.1, r_det=0.0):
        assert model.n % m == 0
        self.model, self.m, self.V = model, m, V
        self.K = model.n // m
        edges = model.xf[::m]                      # K+1 bin edges in x
        self.r_lo = np.sqrt(edges[:-1]); self.r_hi = np.sqrt(edges[1:])
        self.observed = edges[:-1] >= r_det**2     # bins entirely above the cutoff
    def mu(self, N):
        return self.V * self.model.dx * N.reshape(self.K, self.m).sum(axis=1)

def deviance(c, mu):
    mu = np.maximum(mu, 1e-300)
    t = np.where(c > 0, c * np.log(np.where(c > 0, c, 1.0) / mu), 0.0)
    return 2.0 * np.sum(t - c + mu)

def poisson_loss_grad(obs, ell, c):
    m = obs.model; D = np.exp(ell)
    N, lu = m.solve(D)
    mu = obs.mu(N)
    o = obs.observed
    J = deviance(c[o], mu[o])
    w = np.zeros(obs.K); w[o] = 2.0 * (1.0 - c[o] / np.maximum(mu[o], 1e-300))
    g = obs.V * m.dx * np.repeat(w, obs.m)
    psi = lu.solve(-g, trans="T")
    return J, D * (psi @ (m.dA_dD(D) @ N))

def descend(f, ell0, lo=np.log(1e-3), hi=np.log(1e3), max_move=1.0, tol=1e-7, max_iter=300, c=1e-4):
    ell = ell0; J, g = f(ell); step = 1.0; n = 0
    for _ in range(max_iter):
        if abs(g) < tol: break
        while True:
            move = -step * g
            if abs(move) > max_move: move = np.sign(move) * max_move
            trial = min(max(ell + move, lo), hi); move = trial - ell
            if move == 0:
                return ell, J, g, n, ("at bound" if ell in (lo, hi) else "converged")
            Jt, gt = f(trial)
            if Jt <= J + c * g * move: break
            step *= 0.5
            if step < 1e-14: return ell, J, g, n, "line search stalled"
        dg = gt - g
        step = move / dg if move * dg > 0 else 2 * step
        ell, J, g = trial, Jt, gt; n += 1
    return ell, J, g, n, "converged" if abs(g) < tol else "max iter"

def fisher(obs, D, e=1e-4):
    Np, _ = obs.model.solve(D * np.exp(e)); Nm, _ = obs.model.solve(D * np.exp(-e)); N, _ = obs.model.solve(D)
    mu = obs.mu(N); dmu = (obs.mu(Np) - obs.mu(Nm)) / (2 * e); o = obs.observed
    return np.sum(dmu[o]**2 / np.maximum(mu[o], 1e-300))

def curvature_sd(f, ell, h=1e-3):
    Jpp = (f(ell + h)[1] - f(ell - h)[1]) / (2 * h)
    return np.sqrt(2.0 / Jpp) if Jpp > 0 else np.nan

def monte_carlo(model, D_true, V, m=8, r_det=0.0, n_runs=200, seed=0, D0=1.0):
    obs = Obs(model, m, V, r_det)
    N_true, _ = model.solve(D_true); mu_true = obs.mu(N_true)
    rng = np.random.default_rng(seed)
    I = fisher(obs, D_true)
    out = []
    for k in range(n_runs):
        c = rng.poisson(mu_true).astype(float)
        f = lambda ell: poisson_loss_grad(obs, ell, c)
        ell, J, g, it, status = descend(f, np.log(D0))
        sd = curvature_sd(f, ell) if status == "converged" else np.nan
        out.append((ell, sd, J, it, status, c[obs.observed].sum()))
    return obs, I, out
