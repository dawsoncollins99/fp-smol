"""
Physics functions for the Fokker-Planck droplet-growth model.

Covers:
- Saturation vapor pressure (esat_Pa)
- Condensation growth-rate coefficient (xi1_func)
- Kohler curvature/solute parameters (kohler_params)
- Kohler equilibrium supersaturation and critical point
- Drift models in x = r^2 space (constant and Kohler)
- Stokes settling coefficient
"""

import numpy as np

# ---------------------------------------------------------------------
# Saturation vapor pressure
# ---------------------------------------------------------------------


def esat_Pa(T):
    """
    Saturation water vapor pressure over water.

    Parameters
    ----------
    T : float or array_like
        Temperature in Kelvin.

    Returns
    -------
    float or ndarray
        Saturation vapor pressure in Pa.
    """
    T = np.asarray(T, dtype=float)

    y = 100.0 * 10.0 ** (
        23.832241
        - 5.02808 * np.log10(T)
        - 1.3816e-7 * 10.0 ** (11.344 - 0.0303998 * T)
        + 8.1328e-3 * 10.0 ** (3.49149 - 1302.8844 / T)
        - 2949.076 / T
    )

    return y


# ---------------------------------------------------------------------
# Condensation growth-rate coefficient
# ---------------------------------------------------------------------


def xi1_func(T, press):
    """
    Condensation growth-rate coefficient xi1, where dr/dt = xi1*(S-1)/r.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    press : float
        Pressure in Pa.

    Returns
    -------
    float
        xi1 in um^2/s.
    """
    um_per_m = 1e6

    rhow = 1.0e3  # kg/m^3
    T0 = 273.15

    Lv = 2.5e6  # J/kg, latent heat
    Rv = 461.5  # J/(kg K), gas constant for water vapor

    # Thermal conductivity of air (Rogers & Yau, Table 7.1 linear fit)
    K = 7.7e-5 * (T - T0) + 0.02399

    # Diffusivity of water vapor, pressure-corrected
    D = 1.57e-7 * (T - 273.15) + 2.211e-5
    p0 = 1.0e5
    D = D * p0 / press

    denom = rhow * (
        (Lv / (Rv * T)) * Lv / (K * T) + Rv * T / (esat_Pa(T) * D)
    )

    return um_per_m**2 / denom  # um^2/s


# ---------------------------------------------------------------------
# Kohler theory: curvature (A_k) and solute (B_k) parameters
# ---------------------------------------------------------------------


def kohler_params(T, r_dry, rho_s=2160.0, M_s=58.5e-3, nu_s=2.0,
                   sigma_v=0.075, M_w=18e-3, rho_w=1000.0, R=8.314):
    """
    Kohler curvature and solute parameters for a single dry aerosol size.

    Defaults correspond to NaCl, matching the MATLAB benchmark script.

    Parameters
    ----------
    T : float
        Temperature in Kelvin.
    r_dry : float
        Dry aerosol radius in meters.
    rho_s : float
        Solute density, kg/m^3.
    M_s : float
        Solute molar mass, kg/mol.
    nu_s : float
        Van 't Hoff factor.
    sigma_v : float
        Surface tension of water, N/m.
    M_w : float
        Molar mass of water, kg/mol.
    rho_w : float
        Density of water, kg/m^3.
    R : float
        Universal gas constant, J/(mol K).

    Returns
    -------
    A_k : float
        Curvature parameter, meters.
    B_k : float
        Solute parameter, meters^3.
    """
    m_s = (4.0 / 3.0) * np.pi * r_dry**3 * rho_s  # kg solute per droplet

    A_k = 2 * sigma_v * M_w / (rho_w * R * T)               # m
    B_k = 3 * nu_s * m_s * M_w / (4 * np.pi * rho_w * M_s)  # m^3

    return A_k, B_k


def kohler_s_eq(r, A_k, B_k):
    """
    Kohler equilibrium supersaturation at radius r.

        s_eq(r) = A_k/r - B_k/r^3

    Parameters
    ----------
    r : float or array_like
        Droplet radius, meters.
    A_k, B_k : float
        Kohler curvature/solute parameters.

    Returns
    -------
    float or ndarray
        Equilibrium supersaturation (dimensionless).
    """
    r = np.asarray(r, dtype=float)
    return A_k / r - B_k / r**3


def kohler_critical(A_k, B_k):
    """
    Critical radius and supersaturation for Kohler activation.

    Returns
    -------
    r_crit : float
        Critical radius, meters.
    s_crit : float
        Critical supersaturation (dimensionless).
    """
    r_crit = np.sqrt(3 * B_k / A_k)
    s_crit = np.sqrt(4 * A_k**3 / (27 * B_k))
    return r_crit, s_crit


# ---------------------------------------------------------------------
# Drift models in x = r^2 space
# ---------------------------------------------------------------------


def drift_constant(a_phys, s, x):
    """
    Constant drift (Kohler off): A_x = a_phys * s.

    Parameters
    ----------
    a_phys : float
        Growth-rate coefficient, m^2/s.
    s : float
        Ambient supersaturation (dimensionless).
    x : array_like
        Grid points in x = r^2 space (only used for output shape).

    Returns
    -------
    ndarray
        Drift A_x(x), constant in x, m^2/s.
    """
    x = np.asarray(x, dtype=float)
    return a_phys * s * np.ones_like(x)


def drift_kohler(a_phys, s, x, A_k, B_k):
    """
    Kohler drift: A_x(x) = a_phys * (s - s_eq(r)), r = sqrt(x).

    Parameters
    ----------
    a_phys : float
        Growth-rate coefficient, m^2/s.
    s : float
        Ambient supersaturation (dimensionless).
    x : array_like
        Grid points in x = r^2 space, m^2.
    A_k, B_k : float
        Kohler curvature/solute parameters.

    Returns
    -------
    ndarray
        Drift A_x(x), m^2/s.
    """
    x = np.asarray(x, dtype=float)
    r = np.sqrt(x)
    s_eq = kohler_s_eq(r, A_k, B_k)
    return a_phys * (s - s_eq)


# ---------------------------------------------------------------------
# Settling
# ---------------------------------------------------------------------


def stokes_settling_coefficient(rho_w=1000.0, rho_air=None, g=9.81,
                                 mu_air=1.8e-5):
    """
    Stokes settling coefficient C_stk, where fall speed v_t = C_stk * x
    (x = r^2), so settling loss rate lambda(x) = (C_stk/h) * x.

    Parameters
    ----------
    rho_w : float
        Water density, kg/m^3.
    rho_air : float
        Air density, kg/m^3. Must be supplied (depends on T, p).
    g : float
        Gravitational acceleration, m/s^2.
    mu_air : float
        Dynamic viscosity of air, Pa*s.

    Returns
    -------
    float
        C_stk, units 1/(m s).
    """
    if rho_air is None:
        raise ValueError("rho_air must be provided (depends on T, p).")

    return 2 * (rho_w - rho_air) * g / (9 * mu_air)


def air_density(T, p, rgas=287.0):
    """Ideal-gas air density, kg/m^3."""
    return p / (rgas * T)