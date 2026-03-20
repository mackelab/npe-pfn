import numpy as np

try:
    from galpy.util import bovy_coords
except ImportError as error:
    bovy_coords = None
    _STREAMS_COORDINATES_IMPORT_ERROR = error
else:
    _STREAMS_COORDINATES_IMPORT_ERROR = None


_TKOP = np.zeros((3, 3))
_TKOP[0, :] = [-0.4776303088, -0.1738432154, 0.8611897727]
_TKOP[1, :] = [0.510844589, -0.8524449229, 0.111245042]
_TKOP[2, :] = [0.7147776536, 0.4930681392, 0.4959603976]


def _require_streams_coordinates_dependencies():
    if _STREAMS_COORDINATES_IMPORT_ERROR is not None:
        raise ImportError(
            'Streams tasks require optional dependencies. Install with `pip install -e ".[hypothesis]"`.'
        ) from _STREAMS_COORDINATES_IMPORT_ERROR


if bovy_coords is not None:

    @bovy_coords.scalarDecorator
    @bovy_coords.degreeDecorator([0, 1], [0, 1])
    def lb_to_phi12(l, b, degree=False):
        radec = bovy_coords.lb_to_radec(l, b)
        ra = radec[:, 0]
        dec = radec[:, 1]
        xyz = np.array(
            [
                np.cos(dec) * np.cos(ra),
                np.cos(dec) * np.sin(ra),
                np.sin(dec),
            ]
        )
        phi_xyz = np.dot(_TKOP, xyz)
        phi2 = np.arcsin(phi_xyz[2])
        phi1 = np.arctan2(phi_xyz[1], phi_xyz[0])
        phi1[phi1 < 0.0] += 2.0 * np.pi
        return np.array([phi1, phi2]).T

    @bovy_coords.scalarDecorator
    @bovy_coords.degreeDecorator([0, 1], [0, 1])
    def phi12_to_lb(phi1, phi2, degree=False):
        phi_xyz = np.array(
            [
                np.cos(phi2) * np.cos(phi1),
                np.cos(phi2) * np.sin(phi1),
                np.sin(phi2),
            ]
        )
        eq_xyz = np.dot(_TKOP.T, phi_xyz)
        dec = np.arcsin(eq_xyz[2])
        ra = np.arctan2(eq_xyz[1], eq_xyz[0])
        ra[ra < 0.0] += 2.0 * np.pi
        return bovy_coords.radec_to_lb(ra, dec)

    @bovy_coords.scalarDecorator
    @bovy_coords.degreeDecorator([2, 3], [])
    def pmllpmbb_to_pmphi12(pmll, pmbb, l, b, degree=False):
        radec = bovy_coords.lb_to_radec(l, b)
        ra = radec[:, 0]
        dec = radec[:, 1]
        pmradec = bovy_coords.pmllpmbb_to_pmrapmdec(pmll, pmbb, l, b, degree=False)
        pmra = pmradec[:, 0]
        pmdec = pmradec[:, 1]
        phi12 = lb_to_phi12(l, b, degree=False)
        phi1 = phi12[:, 0]
        phi2 = phi12[:, 1]
        a = np.zeros((3, 3, len(ra)))
        a[0, 0] = np.cos(ra) * np.cos(dec)
        a[0, 1] = -np.sin(ra)
        a[0, 2] = -np.cos(ra) * np.sin(dec)
        a[1, 0] = np.sin(ra) * np.cos(dec)
        a[1, 1] = np.cos(ra)
        a[1, 2] = -np.sin(ra) * np.sin(dec)
        a[2, 0] = np.sin(dec)
        a[2, 1] = 0.0
        a[2, 2] = np.cos(dec)
        aphi_inv = np.zeros((3, 3, len(ra)))
        aphi_inv[0, 0] = np.cos(phi1) * np.cos(phi2)
        aphi_inv[0, 1] = np.cos(phi2) * np.sin(phi1)
        aphi_inv[0, 2] = np.sin(phi2)
        aphi_inv[1, 0] = -np.sin(phi1)
        aphi_inv[1, 1] = np.cos(phi1)
        aphi_inv[1, 2] = 0.0
        aphi_inv[2, 0] = -np.cos(phi1) * np.sin(phi2)
        aphi_inv[2, 1] = -np.sin(phi1) * np.sin(phi2)
        aphi_inv[2, 2] = np.cos(phi2)
        ta = np.dot(_TKOP, np.swapaxes(a, 0, 1))
        trans = np.zeros((2, 2, len(ra)))
        for ii in range(len(ra)):
            trans[:, :, ii] = np.dot(aphi_inv[:, :, ii], ta[:, :, ii])[1:, 1:]
        return (trans * np.array([[pmra, pmdec], [pmra, pmdec]])).sum(1).T

    @bovy_coords.scalarDecorator
    @bovy_coords.degreeDecorator([2, 3], [])
    def pmphi12_to_pmllpmbb(pmphi1, pmphi2, phi1, phi2, degree=False):
        lb = phi12_to_lb(phi1, phi2)
        radec = bovy_coords.lb_to_radec(lb[:, 0], lb[:, 1])
        ra = radec[:, 0]
        dec = radec[:, 1]
        a_inv = np.zeros((3, 3, len(ra)))
        a_inv[0, 0] = np.cos(ra) * np.cos(dec)
        a_inv[0, 1] = np.sin(ra) * np.cos(dec)
        a_inv[0, 2] = np.sin(dec)
        a_inv[1, 0] = -np.sin(ra)
        a_inv[1, 1] = np.cos(ra)
        a_inv[1, 2] = 0.0
        a_inv[2, 0] = -np.cos(ra) * np.sin(dec)
        a_inv[2, 1] = -np.sin(ra) * np.sin(dec)
        a_inv[2, 2] = np.cos(dec)
        aphi = np.zeros((3, 3, len(ra)))
        aphi[0, 0] = np.cos(phi1) * np.cos(phi2)
        aphi[0, 1] = -np.sin(phi1)
        aphi[0, 2] = -np.cos(phi1) * np.sin(phi2)
        aphi[1, 0] = np.sin(phi1) * np.cos(phi2)
        aphi[1, 1] = np.cos(phi1)
        aphi[1, 2] = -np.sin(phi1) * np.sin(phi2)
        aphi[2, 0] = np.sin(phi2)
        aphi[2, 1] = 0.0
        aphi[2, 2] = np.cos(phi2)
        taphi = np.dot(_TKOP.T, np.swapaxes(aphi, 0, 1))
        trans = np.zeros((2, 2, len(ra)))
        for ii in range(len(ra)):
            trans[:, :, ii] = np.dot(a_inv[:, :, ii], taphi[:, :, ii])[1:, 1:]
        pmradec = (trans * np.array([[pmphi1, pmphi2], [pmphi1, pmphi2]])).sum(1).T
        pmra = pmradec[:, 0]
        pmdec = pmradec[:, 1]
        return bovy_coords.pmrapmdec_to_pmllpmbb(pmra, pmdec, ra, dec)

    def convert_track_lb_to_phi12(track):
        phi12 = lb_to_phi12(track[:, 0], track[:, 1], degree=True)
        phi12[phi12[:, 0] > 180.0, 0] -= 360.0
        pmphi12 = pmllpmbb_to_pmphi12(
            track[:, 4],
            track[:, 5],
            track[:, 0],
            track[:, 1],
            degree=True,
        )
        out = np.empty_like(track)
        out[:, :2] = phi12
        out[:, 2] = track[:, 2]
        out[:, 3] = track[:, 3]
        out[:, 4:] = pmphi12
        return out

    def phi12_to_lb_6d(phi1, phi2, dist, pmphi1, pmphi2, vlos):
        l, b = phi12_to_lb(phi1, phi2, degree=True)
        pmll, pmbb = pmphi12_to_pmllpmbb(pmphi1, pmphi2, phi1, phi2, degree=True)
        return [l, b, dist, pmll, pmbb, vlos]
else:

    def lb_to_phi12(l, b, degree=False):
        _require_streams_coordinates_dependencies()

    def phi12_to_lb(phi1, phi2, degree=False):
        _require_streams_coordinates_dependencies()

    def pmllpmbb_to_pmphi12(pmll, pmbb, l, b, degree=False):
        _require_streams_coordinates_dependencies()

    def pmphi12_to_pmllpmbb(pmphi1, pmphi2, phi1, phi2, degree=False):
        _require_streams_coordinates_dependencies()

    def convert_track_lb_to_phi12(track):
        _require_streams_coordinates_dependencies()

    def phi12_to_lb_6d(phi1, phi2, dist, pmphi1, pmphi2, vlos):
        _require_streams_coordinates_dependencies()
