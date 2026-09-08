import numpy as np
import pytest

from fp_smol.physics import (
    esat_Pa,
    xi1_func,
    kohler_params,
    kohler_s_eq,
    kohler_critical,
    drift_constant,
    drift_kohler,
)


def test_esat_at_freezing():
    # Known value: esat(0 C) ~= 611 Pa
    assert abs(esat_Pa(273.15) - 611.0) < 5.0


def test_esat_increases_with_temperature():
    assert esat_Pa(300.0) > esat_Pa(280.0)


def test_xi1_positive():
    # Standard chamber-like conditions from the benchmark script
    T, p = 288.0, 1000e2
    assert xi1_func(T, p) > 0


def test_kohler_params_positive():
    T = 288.0
    r_dry = 0.065e-6
    A_k, B_k = kohler_params(T, r_dry)
    assert A_k > 0
    assert B_k > 0


def test_kohler_s_eq_matches_benchmark_scale():
    # For small r_dry (0.065 um) at T=288K, s_eq should be on the order
    # of a few percent at small radii, decaying toward 0 for large r.
    T = 288.0
    r_dry = 0.065e-6
    A_k, B_k = kohler_params(T, r_dry)

    r_small = 0.1e-6
    r_large = 10e-6

    s_eq_small = kohler_s_eq(r_small, A_k, B_k)
    s_eq_large = kohler_s_eq(r_large, A_k, B_k)

    assert abs(s_eq_large) < abs(s_eq_small)


def test_kohler_critical_consistency():
    T = 288.0
    r_dry = 0.065e-6
    A_k, B_k = kohler_params(T, r_dry)
    r_crit, s_crit = kohler_critical(A_k, B_k)

    # s_eq should be at (or very near) its maximum at r_crit
    r_test = np.linspace(r_crit * 0.5, r_crit * 2, 2000)
    s_eq_test = kohler_s_eq(r_test, A_k, B_k)

    assert np.isclose(s_crit, s_eq_test.max(), rtol=1e-2)


def test_drift_constant_shape_and_value():
    x = np.linspace(1e-12, 1e-10, 50)
    a_phys = 1e-12
    s = 0.001
    Ax = drift_constant(a_phys, s, x)

    assert Ax.shape == x.shape
    assert np.allclose(Ax, a_phys * s)


def test_drift_kohler_reduces_to_ambient_s_far_from_critical():
    # Far from r_crit, s_eq -> 0, so drift -> a_phys * s
    T = 288.0
    r_dry = 0.065e-6
    A_k, B_k = kohler_params(T, r_dry)

    a_phys = 1e-12
    s = 0.001
    x_large = np.array([(50e-6) ** 2])  # large radius

    Ax = drift_kohler(a_phys, s, x_large, A_k, B_k)
    assert np.isclose(Ax[0], a_phys * s, rtol=1e-2)