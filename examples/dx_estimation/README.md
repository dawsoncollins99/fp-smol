# Estimating the diffusivity $D_x$ from droplet size distributions

Code for a series of synthetic parameter-estimation experiments: recover the
turbulent diffusivity $D_x$ of the steady Fokker–Planck model from droplet-size
data, using an adjoint gradient. The write-ups and talk built from this code are
in [`docs/dx_estimation/`](../../docs/dx_estimation/).

## Files

| File | What it does |
|---|---|
| `core.py` | Steady model in µm / s / cm⁻³ units: Köhler drift, Stokes settling, Scharfetter–Gummel finite-volume operator $A(D_x)$ and its derivative $\partial A/\partial D_x$, least-squares loss with adjoint gradient, gradient descent with line search. |
| `noisy.py` | Instrument-like data: bins of `m` model cells, sample volume `V`, detection limit `r_det`; Poisson-deviance loss and adjoint gradient; curvature (single-sample) error bars; Fisher information; Monte Carlo driver. |
| `run_monte_carlo.py` | Runs the 12 Monte Carlo configurations (200 noisy samples each) and saves them to `results/` (not tracked). |
| `figs_writeup_noise_free.py` | Figures for `docs/dx_estimation/writeup_noise_free/`. |
| `figs_writeup_noisy.py` | Figures for `docs/dx_estimation/writeup_noisy/` (needs `results/`). |
| `figs_talk.py` | Figures for `docs/dx_estimation/talk/` (needs `results/`). |
| `Dx_steady_estimation.ipynb` | Self-contained notebook for the noise-free experiment: many true values × many starting guesses. |

## Reproducing everything

From the repository root, with the project environment active (numpy, scipy, matplotlib):

```bash
python examples/dx_estimation/run_monte_carlo.py          # ~6-10 min; writes results/
python examples/dx_estimation/figs_writeup_noise_free.py  # ~10 s
python examples/dx_estimation/figs_writeup_noisy.py       # ~10 s
python examples/dx_estimation/figs_talk.py                # ~30 s
```

Each figure script writes PDFs into the `figures/` folder next to the matching
LaTeX file. Then build the documents (see `docs/dx_estimation/README.md`).

Random seeds are fixed, so the numbers and figures match the committed PDFs.

## Notes

- `core.py` is a standalone re-implementation of the steady solver in
  `src/fp_smol/` (same physics and Scharfetter–Gummel scheme, but in µm units and
  with $\partial A/\partial D_x$ for the adjoint). It reproduces the benchmark mean
  radii (4.22 µm at $D_x = 0.01$, 5.74 µm at $D_x = 10$ µm⁴/s). A later cleanup
  could build on `fp_smol` directly.
- The Bernoulli function is evaluated smoothly for all arguments (no hard cutoff),
  so the loss is differentiable and finite-difference gradient checks are clean.
- All experiments are "inverse crimes": the data are generated with the same model
  and grid that is fitted.
