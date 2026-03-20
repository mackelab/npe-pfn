import astropy.units as u
import hypothesis
import numpy
import numpy as np
import numpy as np
import pickle
import torch

from galpy.orbit import Orbit
from galpy.potential import MWPotential2014, turn_physical_off, vcirc
from galpy.util import bovy_conversion, bovy_coords, save_pickles, bovy_plot
from .streams.coordinates import lb_to_phi12
from .streams.observation import (
    compute_obs_density as _compute_obs_density,
    compute_obs_density_no_interpolation as _compute_obs_density_no_interpolation,
)
from .streams.subhalos import simulate_subhalos_mwdm as _simulate_subhalos_mwdm
from scipy import integrate, interpolate
from scipy.integrate import quad
from torch.distributions.uniform import Uniform


def allocate_prior_stream_age():
    lower = torch.tensor(3).float().to(hypothesis.accelerator)
    upper = torch.tensor(7).float().to(hypothesis.accelerator)

    return Uniform(lower, upper)


def allocate_prior_wdm_mass():
    lower = torch.tensor(1).float().to(hypothesis.accelerator)
    upper = torch.tensor(50).float().to(hypothesis.accelerator)

    return Uniform(lower, upper)


def load_observed_gd1(path, phi, degree=1):
    data = np.genfromtxt(path, names=True)
    phi_max = max(phi) + 5  # For stability in fitting the splines
    phi_min = min(phi) - 5  # For stability in fitting the splines
    phi_data = data["phi1mid"]
    if phi_min < min(phi_data) or phi_max > max(phi_data):
        raise ValueError("Angles not supported by observation.")
    indices = (phi_data <= phi_max) & (phi_data >= phi_min)
    phi_data = phi_data[indices]
    linear_density = data["lindens"][indices]
    error = data["e_lindens"][indices]
    trend = np.polyfit(phi_data, linear_density, deg=degree)
    fitted = np.poly1d(trend)(phi_data)
    error /= fitted
    linear_density /= fitted
    # Fit a spline and extract the requested values
    l = np.array(linear_density)
    fit_density = interpolate.InterpolatedUnivariateSpline(phi_data, linear_density)
    fit_error = interpolate.InterpolatedUnivariateSpline(phi_data, error)
    linear_density = fit_density(phi)
    trend = np.polyfit(phi, linear_density, deg=degree)
    fitted = np.poly1d(trend)(phi)
    linear_density /= fitted
    error = fit_error(phi)
    error /= fitted

    return linear_density, error


h = 0.6774
ro = 8.0
vo = 220.0


def parse_times(times, age):
    if "sampling" in times:
        nsam = int(times.split("sampling")[0])
        return [
            float(ti) / bovy_conversion.time_in_Gyr(vo, ro)
            for ti in np.arange(1, nsam + 1) / (nsam + 1.0) * age
        ]
    return [float(ti) / bovy_conversion.time_in_Gyr(vo, ro) for ti in times.split(",")]


def parse_mass(mass):
    return [float(m) for m in mass.split(",")]


def nsubhalo(m):
    return 0.3 * (10.0**6.5 / m)


def rs(m, plummer=False, rsfac=1.0):
    if plummer:
        # print ('Plummer')
        return 1.62 * rsfac / ro * (m / 10.0**8.0) ** 0.5
    else:
        return 1.05 * rsfac / ro * (m / 10.0**8.0) ** 0.5


def alpha(m_wdm):
    return (0.048 / h) * (m_wdm) ** (-1.11)  # in Mpc , m_wdm in keV


def lambda_hm(m_wdm):
    nu = 1.12
    return 2 * numpy.pi * alpha(m_wdm) / (2 ** (nu / 5.0) - 1.0) ** (1 / (2 * nu))


def M_hm(m_wdm):
    Om_m = 0.3089
    rho_c = 1.27 * 10**11  # Msun/Mpc^3
    rho_bar = Om_m * rho_c
    return (4 * numpy.pi / 3) * rho_bar * (lambda_hm(m_wdm) / 2.0) ** 3


def Einasto(r):
    al = 0.678  # alpha_shape
    rm2 = 199  # kpc, see Erkal et al 1606.04946 for scaling to M^1/3
    return numpy.exp((-2.0 / al) * ((r / rm2) ** al - 1.0)) * 4 * numpy.pi * (r**2)


def dndM_cdm(M, c0kpc=2.02 * 10 ** (-13), mf_slope=-1.9):
    # c0kpc=2.02*10**(-13) #Msun^-1 kpc^-3 from Denis' paper
    m0 = 2.52 * 10**7  # Msun from Denis' paper
    return c0kpc * ((M / m0) ** mf_slope)


def fac(M, m_wdm):
    beta = -0.99
    gamma = 2.7
    return (1.0 + gamma * (M_hm(m_wdm) / M)) ** beta


def dndM_wdm(M, m_wdm, c0kpc=2.02 * 10 ** (-13), mf_slope=-1.9):
    return fac(M, m_wdm) * dndM_cdm(M, c0kpc=c0kpc, mf_slope=mf_slope)


def nsub_cdm(M1, M2, r=20.0, c0kpc=2.02 * 10 ** (-13), mf_slope=-1.9):
    # number density of subhalos in kpc^-3
    m1 = 10 ** (M1)
    m2 = 10 ** (M2)
    return (
        integrate.quad(dndM_cdm, m1, m2, args=(c0kpc, mf_slope))[0]
        * integrate.quad(Einasto, 0.0, r)[0]
        * (8.0**3.0)
        / (4 * numpy.pi * (r**3) / 3)
    )  # in Galpy units


def nsub_wdm(M1, M2, m_wdm, r=20.0, c0kpc=2.02 * 10 ** (-13), mf_slope=-1.9):
    m1 = 10 ** (M1)
    m2 = 10 ** (M2)
    return (
        integrate.quad(dndM_wdm, m1, m2, args=(m_wdm, c0kpc, mf_slope))[0]
        * integrate.quad(Einasto, 0.0, r)[0]
        * (8.0**3)
        / (4 * numpy.pi * (r**3) / 3)
    )  # in Galpy units


def simulate_subhalos_mwdm(
    sdf_pepper,
    m_wdm,
    mf_slope=-1.9,
    c0kpc=2.02 * 10 ** (-13),
    r=20.0,
    Xrs=5.0,
    sigma=120.0 / 220.0,
):
    return _simulate_subhalos_mwdm(
        sdf_pepper,
        m_wdm,
        mf_slope=mf_slope,
        c0kpc=c0kpc,
        r=r,
        Xrs=Xrs,
        sigma=sigma,
    )


def compute_obs_density_no_interpolation(phi1, apars, dens_apar):
    return _compute_obs_density_no_interpolation(phi1, apars, dens_apar)


def compute_obs_density(phi1, apars, dens_apar, Omega):
    return _compute_obs_density(phi1, apars, dens_apar, Omega)
