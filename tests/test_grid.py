import numpy as np
import pytest

from fp_smol.grid import make_grid


def test_grid_shapes():
    g = make_grid(r_min=0.0, r_max=30e-6, Ncells=100)
    assert g.xf.shape == (101,)
    assert g.xc.shape == (100,)
    assert g.r_c.shape == (100,)
    assert g.Ncells == 100


def test_grid_endpoints():
    r_min, r_max = 0.0, 30e-6
    g = make_grid(r_min=r_min, r_max=r_max, Ncells=50)
    assert np.isclose(g.xf[0], r_min**2)
    assert np.isclose(g.xf[-1], r_max**2)


def test_grid_uniform_spacing():
    g = make_grid(r_min=0.0, r_max=30e-6, Ncells=200)
    diffs = np.diff(g.xf)
    assert np.allclose(diffs, g.dx)


def test_grid_centers_between_faces():
    g = make_grid(r_min=0.0, r_max=30e-6, Ncells=10)
    assert np.all(g.xc > g.xf[:-1])
    assert np.all(g.xc < g.xf[1:])


def test_grid_r_c_matches_sqrt_xc():
    g = make_grid(r_min=1e-9, r_max=30e-6, Ncells=10)
    assert np.allclose(g.r_c, np.sqrt(g.xc))


def test_grid_kohler_case_positive_r_min():
    r_dry = 0.065e-6
    r_min = r_dry / 2
    g = make_grid(r_min=r_min, r_max=30e-6, Ncells=8000)
    assert g.xf[0] == r_min**2
    assert g.r_c[0] > r_min  # first cell center is inside the domain


def test_invalid_r_min_negative():
    with pytest.raises(ValueError):
        make_grid(r_min=-1.0, r_max=1.0, Ncells=10)


def test_invalid_r_max_not_greater():
    with pytest.raises(ValueError):
        make_grid(r_min=1.0, r_max=1.0, Ncells=10)


def test_invalid_ncells():
    with pytest.raises(ValueError):
        make_grid(r_min=0.0, r_max=1.0, Ncells=0)