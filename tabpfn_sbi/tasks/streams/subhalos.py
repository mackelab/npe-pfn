import numpy as np

from scipy import integrate, interpolate
from scipy.integrate import quad

try:
    from galpy.util import bovy_conversion
except ImportError as error:
    bovy_conversion = None
    _STREAMS_SUBHALOS_IMPORT_ERROR = error
else:
    _STREAMS_SUBHALOS_IMPORT_ERROR = None


h = 0.6774
ro = 8.0
vo = 220.0


def _require_streams_subhalo_dependencies():
    if _STREAMS_SUBHALOS_IMPORT_ERROR is not None:
        raise ImportError(
            'Streams tasks require optional dependencies. Install with `pip install -e ".[hypothesis]"`.'
        ) from _STREAMS_SUBHALOS_IMPORT_ERROR


def rs(mass, plummer=False, rsfac=1.0):
    if plummer:
        return 1.62 * rsfac / ro * (mass / 10.0**8.0) ** 0.5
    return 1.05 * rsfac / ro * (mass / 10.0**8.0) ** 0.5


def alpha(m_wdm):
    return (0.048 / h) * (m_wdm) ** (-1.11)


def lambda_hm(m_wdm):
    nu = 1.12
    return 2 * np.pi * alpha(m_wdm) / (2 ** (nu / 5.0) - 1.0) ** (1 / (2 * nu))


def m_hm(m_wdm):
    om_m = 0.3089
    rho_c = 1.27 * 10**11
    rho_bar = om_m * rho_c
    return (4 * np.pi / 3) * rho_bar * (lambda_hm(m_wdm) / 2.0) ** 3


def einasto(radius):
    alpha_shape = 0.678
    rm2 = 199
    return (
        np.exp((-2.0 / alpha_shape) * ((radius / rm2) ** alpha_shape - 1.0))
        * 4
        * np.pi
        * (radius**2)
    )


def dndm_cdm(mass, c0kpc=2.02 * 10 ** (-13), mf_slope=-1.9):
    m0 = 2.52 * 10**7
    return c0kpc * ((mass / m0) ** mf_slope)


def fac(mass, m_wdm):
    beta = -0.99
    gamma = 2.7
    return (1.0 + gamma * (m_hm(m_wdm) / mass)) ** beta


def dndm_wdm(mass, m_wdm, c0kpc=2.02 * 10 ** (-13), mf_slope=-1.9):
    return fac(mass, m_wdm) * dndm_cdm(mass, c0kpc=c0kpc, mf_slope=mf_slope)


def nsub_wdm(m1, m2, m_wdm, radius=20.0, c0kpc=2.02 * 10 ** (-13), mf_slope=-1.9):
    mass1 = 10**m1
    mass2 = 10**m2
    return (
        integrate.quad(dndm_wdm, mass1, mass2, args=(m_wdm, c0kpc, mf_slope))[0]
        * integrate.quad(einasto, 0.0, radius)[0]
        * (8.0**3)
        / (4 * np.pi * (radius**3) / 3)
    )


def simulate_subhalos_mwdm(
    sdf_pepper,
    m_wdm,
    mf_slope=-1.9,
    c0kpc=2.02 * 10 ** (-13),
    r=20.0,
    Xrs=5.0,
    sigma=120.0 / 220.0,
):
    _require_streams_subhalo_dependencies()
    mass_bin_edges = [5.0, 6.0, 7.0, 8.0, 9.0]
    num_bins = len(mass_bin_edges) - 1
    nden_bin = np.empty(num_bins)
    rate_bin = np.empty(num_bins)
    for ll in range(num_bins):
        nden_bin[ll] = nsub_wdm(
            mass_bin_edges[ll],
            mass_bin_edges[ll + 1],
            m_wdm=m_wdm,
            radius=r,
            c0kpc=c0kpc,
            mf_slope=mf_slope,
        )
        mmid = 10 ** (0.5 * (mass_bin_edges[ll] + mass_bin_edges[ll + 1]))
        rate_bin[ll] = sdf_pepper.subhalo_encounters(
            sigma=sigma,
            nsubhalo=nden_bin[ll],
            bmax=Xrs * rs(mmid, plummer=True),
        )

    rate = np.sum(rate_bin)
    num_impacts = np.random.poisson(rate)
    norm = (
        1.0
        / quad(
            lambda mass: fac(mass, m_wdm) * (mass ** (mf_slope + 0.5)),
            10 ** mass_bin_edges[0],
            10 ** mass_bin_edges[num_bins],
        )[0]
    )

    def cdf(mass):
        return quad(
            lambda m: norm * fac(m, m_wdm) * (m ** (mf_slope + 0.5)),
            10 ** mass_bin_edges[0],
            mass,
        )[0]

    mm = np.linspace(mass_bin_edges[0], mass_bin_edges[num_bins], 10000)
    cdfl = [cdf(value) for value in 10**mm]
    icdf = interpolate.InterpolatedUnivariateSpline(cdfl, 10**mm, k=1)
    timpact_sub = np.array(sdf_pepper._uniq_timpact)[
        np.random.choice(
            len(sdf_pepper._uniq_timpact),
            size=num_impacts,
            p=sdf_pepper._ptimpact,
        )
    ]
    impact_angle_sub = np.array(
        [sdf_pepper._icdf_stream_len[ti](np.random.uniform()) for ti in timpact_sub]
    )

    sample_gm = lambda: icdf(np.random.uniform()) / bovy_conversion.mass_in_msol(vo, ro)
    gm_sub = np.array([sample_gm() for _ in impact_angle_sub])
    rs_sub = np.array([rs(gm * bovy_conversion.mass_in_msol(vo, ro)) for gm in gm_sub])
    impactb_sub = (
        (2.0 * np.random.uniform(size=len(impact_angle_sub)) - 1.0) * Xrs * rs_sub
    )
    subhalovel_sub = np.empty((len(impact_angle_sub), 3))
    for ii in range(len(timpact_sub)):
        subhalovel_sub[ii] = sdf_pepper._draw_impact_velocities(
            timpact_sub[ii], sigma, impact_angle_sub[ii], n=1
        )[0]
    if not sdf_pepper._gap_leading:
        impact_angle_sub *= -1.0

    return impact_angle_sub, impactb_sub, subhalovel_sub, timpact_sub, gm_sub, rs_sub
