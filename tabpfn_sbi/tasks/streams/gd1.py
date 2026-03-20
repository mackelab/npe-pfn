try:
    from galpy.actionAngle import actionAngleIsochroneApprox
    from galpy.df import streamdf, streamgapdf
    from galpy.orbit import Orbit
    from galpy.potential import MWPotential2014
    from galpy.util import bovy_conversion

    from ..streampepperdf import streampepperdf
except ImportError as error:
    actionAngleIsochroneApprox = None
    streamdf = None
    streamgapdf = None
    Orbit = None
    MWPotential2014 = None
    bovy_conversion = None
    streampepperdf = None
    _STREAMS_GD1_IMPORT_ERROR = error
else:
    _STREAMS_GD1_IMPORT_ERROR = None
from .coordinates import phi12_to_lb_6d


R0, V0 = 8.0, 220.0


def _require_streams_gd1_dependencies():
    if _STREAMS_GD1_IMPORT_ERROR is not None:
        raise ImportError(
            'Streams tasks require optional dependencies. Install with `pip install -e ".[hypothesis]"`.'
        ) from _STREAMS_GD1_IMPORT_ERROR


def setup_gd1model(
    leading=True,
    pot=None,
    timpact=None,
    hernquist=True,
    new_orb_lb=None,
    isob=0.8,
    age=9.0,
    sigv=0.5,
    singleImpact=False,
    length_factor=1.0,
    **kwargs,
):
    _require_streams_gd1_dependencies()
    if pot is None:
        pot = MWPotential2014
    aai = actionAngleIsochroneApprox(pot=pot, b=isob)
    if new_orb_lb is None:
        obs = Orbit(
            phi12_to_lb_6d(0, -0.82, 10.1, -8.5, -2.15, -257.0),
            lb=True,
            solarmotion=[-11.1, 24.0, 7.25],
            ro=8.0,
            vo=220.0,
        )
    else:
        obs = Orbit(
            new_orb_lb,
            lb=True,
            solarmotion=[-11.1, 24.0, 7.25],
            ro=8.0,
            vo=220.0,
        )

    if timpact is None:
        sdf = streamdf(
            sigv / 220.0,
            progenitor=obs,
            pot=pot,
            aA=aai,
            leading=leading,
            nTrackChunks=11,
            vsun=[-11.1, 244.0, 7.25],
            tdisrupt=age / bovy_conversion.time_in_Gyr(V0, R0),
            vo=V0,
            ro=R0,
        )
    elif singleImpact:
        sdf = streamgapdf(
            sigv / 220.0,
            progenitor=obs,
            pot=pot,
            aA=aai,
            leading=leading,
            nTrackChunks=11,
            vsun=[-11.1, 244.0, 7.25],
            tdisrupt=age / bovy_conversion.time_in_Gyr(V0, R0),
            vo=V0,
            ro=R0,
            timpact=timpact,
            spline_order=3,
            hernquist=hernquist,
            **kwargs,
        )
    else:
        sdf = streampepperdf(
            sigv / 220.0,
            progenitor=obs,
            pot=pot,
            aA=aai,
            leading=leading,
            nTrackChunks=101,
            vsun=[-11.1, 244.0, 7.25],
            tdisrupt=age / bovy_conversion.time_in_Gyr(V0, R0),
            vo=V0,
            ro=R0,
            timpact=timpact,
            spline_order=1,
            hernquist=hernquist,
            length_factor=length_factor,
        )
    sdf.turn_physical_off()
    return sdf
