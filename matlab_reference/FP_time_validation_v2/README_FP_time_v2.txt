FP time-dependent validation v2
================================

Purpose
-------
This is the first step toward FP-Smol. It contains NO coalescence yet.
It converts the existing steady Scharfetter-Gummel finite-volume solver
A*N=b into the time-dependent system

    dN/dt = (b - A*N)/dx

and advances it with backward Euler.

Primary validation
------------------
The default initial condition is N(x,0)=0. The code independently computes
the old steady solution N_ss=A\b and checks whether the time-dependent
solution approaches it.

For the default parameters, an independent numerical reproduction of the
discretization gave approximately:
  t = 500 s:  relative L1 error = 6.11e-2
  t = 1000 s: relative L1 error = 7.87e-5
  t = 2000 s: relative L1 error = 2e-14
  steady mean radius = 4.206 um
These values are useful sanity checks for the MATLAB run.

Figures
-------
1. Radius PDF at selected times, with the old steady FV PDF overlaid.
2. Relative L1 error between N(x,t) and N_ss.
3. Total number and mean radius versus time, with steady values shown.

Changes from the old steady code
--------------------------------
- Sparse assembly of the tridiagonal FV matrix.
- Backward-Euler time integration.
- Stable large-|Pe| evaluation of the SG Bernoulli coefficients.
- D_x is specified once in um^4/s and converted to SI, removing the prior
  mismatch between the calculation and plot annotation.

Required files
--------------
fp_FV_Kohler_physical_SG_time_v2.m
xi1_func.m
esat_Pa.m

Next step after validation
--------------------------
Add a conservative Smoluchowski collision/coalescence tendency to obtain
FP-Smol.
