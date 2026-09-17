import numpy as np
import pytest

from fp_smol.flux import bernoulli, sg_face_coefficients, face_average


def test_bernoulli_at_zero():
    # B(0) should be exactly 1 (limit of z/(e^z - 1))
    assert np.isclose(bernoulli(0.0), 1.0)


def test_bernoulli_matches_direct_formula_away_from_zero():
    z = np.array([0.5, 1.0, 2.0, -0.5, -1.0, -2.0])
    direct = z / (np.exp(z) - 1)
    assert np.allclose(bernoulli(z), direct, rtol=1e-10)


def test_bernoulli_taylor_region_matches_direct_formula():
    # Just inside the small-z branch: should still agree closely with
    # the direct formula (this catches Taylor-expansion bugs).
    z = np.array([1e-4, -1e-4, 5e-4, -5e-4])
    direct = z / (np.exp(z) - 1)
    assert np.allclose(bernoulli(z), direct, atol=1e-8)


def test_bernoulli_large_positive_limit():
    # B(z) -> 0 as z -> +inf
    assert bernoulli(100.0) == 0.0


def test_bernoulli_large_negative_limit():
    # B(z) -> -z as z -> -inf
    z = -100.0
    assert np.isclose(bernoulli(z), -z)


def test_bernoulli_symmetry_relation():
    # Identity: B(z) - B(-z) = -z  (follows from the definition)
    z = np.array([0.1, 1.0, 5.0, 20.0])
    assert np.allclose(bernoulli(z) - bernoulli(-z), -z, rtol=1e-8)


def test_sg_coefficients_zero_drift_reduces_to_pure_diffusion():
    # With Ax=0, Pe=0, so B(0)=B(-0)=1, giving alpha=D/dx, beta=-D/dx
    # (standard central-difference diffusion).
    dx = 1e-14
    D_x = 1e-16
    Ax_face = np.zeros(5)

    alpha, beta = sg_face_coefficients(Ax_face, dx, D_x)

    assert np.allclose(alpha, D_x / dx)
    assert np.allclose(beta, -D_x / dx)


def test_sg_coefficients_shapes():
    dx = 1e-14
    D_x = 1e-16
    Ax_face = np.linspace(-1e-12, 1e-12, 7)

    alpha, beta = sg_face_coefficients(Ax_face, dx, D_x)

    assert alpha.shape == Ax_face.shape
    assert beta.shape == Ax_face.shape


def test_face_average_simple_case():
    values = np.array([1.0, 3.0, 5.0, 7.0])
    avg = face_average(values)
    assert np.allclose(avg, [2.0, 4.0, 6.0])


def test_face_average_shape():
    values = np.linspace(0, 1, 10)
    avg = face_average(values)
    assert avg.shape == (9,)