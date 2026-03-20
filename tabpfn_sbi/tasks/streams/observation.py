import numpy as np

from scipy import interpolate


def compute_obs_density_no_interpolation(phi1, apars, dens_apar):
    apar_edge = []
    phi1_edge = []

    abw0 = apars[1] - apars[0]
    apar_edge.append(apars[0] - (abw0 / 2.0))

    phi1bw0 = phi1[1] - phi1[0]
    phi1_edge.append(phi1[0] - (phi1bw0 / 2.0))

    for ii in range(len(apars) - 1):
        abw = apars[ii + 1] - apars[ii]
        phi1bw = phi1[ii + 1] - phi1[ii]
        apar_edge.append(apars[ii] + abw / 2.0)
        phi1_edge.append(phi1[ii] + phi1bw / 2.0)

    abw_last = apars[len(apars) - 1] - apars[len(apars) - 2]
    apar_edge.append(apars[len(apars) - 1] + (abw_last / 2.0))

    phi1bw_last = phi1[len(phi1) - 1] - phi1[len(phi1) - 2]
    phi1_edge.append(phi1[len(phi1) - 1] + (phi1bw_last / 2.0))

    dapar_dphi1 = np.fabs(np.diff(apar_edge) / np.diff(phi1_edge))
    return dens_apar * dapar_dphi1


def compute_obs_density(phi1, apars, dens_apar, Omega):
    apar_edge = []
    phi1_edge = []

    abw0 = apars[1] - apars[0]
    apar_edge.append(apars[0] - (abw0 / 2.0))

    phi1bw0 = phi1[1] - phi1[0]
    phi1_edge.append(phi1[0] - (phi1bw0 / 2.0))

    for ii in range(len(apars) - 1):
        abw = apars[ii + 1] - apars[ii]
        phi1bw = phi1[ii + 1] - phi1[ii]
        apar_edge.append(apars[ii] + abw / 2.0)
        phi1_edge.append(phi1[ii] + phi1bw / 2.0)

    abw_last = apars[len(apars) - 1] - apars[len(apars) - 2]
    apar_edge.append(apars[len(apars) - 1] + (abw_last / 2.0))

    phi1bw_last = phi1[len(phi1) - 1] - phi1[len(phi1) - 2]
    phi1_edge.append(phi1[len(phi1) - 1] + (phi1bw_last / 2.0))

    dapar_dphi1 = np.fabs(np.diff(apar_edge) / np.diff(phi1_edge))
    ipdens_apar = interpolate.InterpolatedUnivariateSpline(apars, dens_apar)

    if phi1[1] < phi1[0]:
        ipphi1 = interpolate.InterpolatedUnivariateSpline(phi1[::-1], apars[::-1])
        ipdapar_dphi1 = interpolate.InterpolatedUnivariateSpline(
            phi1[::-1], dapar_dphi1[::-1]
        )
        dens_phi1 = interpolate.InterpolatedUnivariateSpline(
            phi1[::-1],
            ipdens_apar(ipphi1(phi1[::-1])) * ipdapar_dphi1(phi1[::-1]),
        )
    else:
        ipphi1 = interpolate.InterpolatedUnivariateSpline(phi1, apars)
        ipdapar_dphi1 = interpolate.InterpolatedUnivariateSpline(phi1, dapar_dphi1)
        dens_phi1 = interpolate.InterpolatedUnivariateSpline(
            phi1, ipdens_apar(ipphi1(phi1)) * ipdapar_dphi1(phi1)
        )

    return dens_phi1(phi1)
