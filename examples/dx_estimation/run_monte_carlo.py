"""
Monte Carlo study for the noisy (Poisson-count) D_x estimation write-up.

For each configuration, draw 200 independent Poisson samples of the steady
size distribution, fit D_x to each with the adjoint gradient, and record the
estimate and its single-sample (curvature) error bar. Results are saved as
.npz files in ./results/ (not tracked by git) and are read by
figs_writeup_noisy.py and figs_talk.py.

Usage (from the repo root):
    python examples/dx_estimation/run_monte_carlo.py            # skip configs already saved
    python examples/dx_estimation/run_monte_carlo.py --force    # recompute everything

Takes roughly 6-10 minutes in total. Seeds match the ones used for the PDFs.
"""
import sys
import time
from pathlib import Path

import numpy as np

from noisy import Model, monte_carlo

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# (true D_x [um^4/s], sample volume V [cm^3], detection limit r_det [um], seed)
CONFIGS = [
    (10.0, 0.01, 0.0, 101), (10.0, 0.1, 0.0, 102), (10.0, 1.0, 0.0, 103),
    (0.1, 0.01, 0.0, 104), (0.1, 0.1, 0.0, 105), (0.1, 1.0, 0.0, 106),
    (10.0, 0.1, 1.0, 111), (10.0, 0.1, 2.0, 112), (10.0, 0.1, 3.0, 113),
    (0.1, 1.0, 1.0, 121), (0.1, 1.0, 2.0, 122), (0.1, 1.0, 3.0, 123),
]


def result_path(D, V, r_det):
    return RESULTS / f"res_D{float(D)}_V{float(V)}_r{float(r_det)}.npz"


def load_result(D, V, r_det):
    """Load one saved configuration as a dict of arrays."""
    p = result_path(D, V, r_det)
    if not p.exists():
        raise FileNotFoundError(f"{p} not found; run run_monte_carlo.py first.")
    with np.load(p) as f:
        return {k: f[k] for k in f.files}


def main(force=False, n_runs=200):
    RESULTS.mkdir(exist_ok=True)
    model = Model(8000)
    for D, V, r_det, seed in CONFIGS:
        path = result_path(D, V, r_det)
        if path.exists() and not force:
            print(f"skip   D={D} V={V} r_det={r_det} (already saved)")
            continue
        t0 = time.time()
        obs, I, out = monte_carlo(model, D, V, m=8, r_det=r_det, n_runs=n_runs, seed=seed)
        ell = np.array([o[0] for o in out])
        np.savez(path,
                 I=I, ell=ell, sd=np.array([o[1] for o in out]),
                 J=np.array([o[2] for o in out]), iters=np.array([o[3] for o in out]),
                 status=np.array([o[4] for o in out]), total=np.array([o[5] for o in out]),
                 K_obs=int(obs.observed.sum()))
        print(f"done   D={D} V={V} r_det={r_det}: predicted sd {1/np.sqrt(I):.4f}, "
              f"observed sd {ell.std(ddof=1):.4f}, bias {ell.mean() - np.log(D):+.4f} "
              f"({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main(force="--force" in sys.argv)
