import glob
import pickle

import numpy as np

try:
    from galpy.util import bovy_conversion
    from hypothesis.simulation import Simulator as BaseSimulator
except ImportError as error:
    bovy_conversion = None

    class BaseSimulator:  # type: ignore[no-redef]
        pass

    _STREAMS_SIMULATOR_IMPORT_ERROR = error
else:
    _STREAMS_SIMULATOR_IMPORT_ERROR = None

from .coordinates import lb_to_phi12
from .gd1 import setup_gd1model
from .observation import compute_obs_density
from .subhalos import simulate_subhalos_mwdm


def _require_streams_simulator_dependencies():
    if _STREAMS_SIMULATOR_IMPORT_ERROR is not None:
        raise ImportError(
            'Streams tasks require optional dependencies. Install with `pip install -e ".[hypothesis]"`.'
        ) from _STREAMS_SIMULATOR_IMPORT_ERROR


class GD1StreamSimulator(BaseSimulator):
    def __init__(self, hernquist_profile=True, max_subhalo_impacts=64):
        _require_streams_simulator_dependencies()
        super().__init__()
        self.hernquist_profile = hernquist_profile
        self.isob = 0.45
        self.chunk_size = 64
        self.length_factor = float(1)
        self.max_subhalo_impacts = int(max_subhalo_impacts)
        self.new_orb_lb = [
            188.04928416766532,
            51.848594007807456,
            7.559027173643999,
            12.260258757214746,
            -5.140630283489461,
            7.162732847549563,
        ]

    def _compute_impact_times(self, age):
        impact_times = []
        time_in_gyr = bovy_conversion.time_in_Gyr(vo=float(220), ro=float(8))
        for time in (
            np.arange(1, self.max_subhalo_impacts + 1) / self.max_subhalo_impacts
        ):
            impact_times.append(time / time_in_gyr)
        return impact_times

    def _simulate_stream(self, age):
        sigv_age = (0.3 * 3.2) / age
        impacts = self._compute_impact_times(age)
        stream_smooth_trailing = setup_gd1model(
            age=age,
            isob=self.isob,
            leading=False,
            new_orb_lb=self.new_orb_lb,
            sigv=sigv_age,
        )
        stream_trailing = setup_gd1model(
            age=age,
            hernquist=self.hernquist_profile,
            isob=self.isob,
            leading=False,
            length_factor=self.length_factor,
            new_orb_lb=self.new_orb_lb,
            sigv=sigv_age,
            timpact=impacts,
        )
        return None, stream_smooth_trailing, None, stream_trailing

    def forward(self, inputs):
        outputs = []
        for input in inputs:
            success = False
            while not success:
                try:
                    outputs.append(self._simulate_stream(input.item()))
                    success = True
                except Exception:
                    pass
        return outputs


class WDMSubhaloSimulator(BaseSimulator):
    def __init__(
        self, streams, resolution=0.01, record_impacts=False, allow_no_impacts=True
    ):
        _require_streams_simulator_dependencies()
        super().__init__()
        self.record_impacts = record_impacts
        self.allow_no_impacts = allow_no_impacts
        self.Xrs = float(5)
        self.ravg = float(20)
        self.streams = streams
        self.apars = np.arange(0.01, 1.0, resolution)
        self.dens_unp_leading = None
        self.omega_unp_leading = None
        self.dens_unp_trailing = None
        self.omega_unp_trailing = None
        self._precompute_smooth_stream_properties()

    def _precompute_smooth_stream_properties(self):
        self._precompute_smooth_stream_trailing()

    def _precompute_smooth_stream_leading(self):
        stream_smooth = self.streams[0]
        self.dens_unp_leading = [stream_smooth._density_par(a) for a in self.apars]
        self.omega_unp_leading = [
            stream_smooth.meanOmega(a, oned=True) for a in self.apars
        ]

    def _precompute_smooth_stream_trailing(self):
        stream_smooth = self.streams[1]
        self.dens_unp_trailing = [stream_smooth._density_par(a) for a in self.apars]
        self.omega_unp_trailing = [
            stream_smooth.meanOmega(a, oned=True) for a in self.apars
        ]

    def _simulate_observation(self, wdm_mass, leading):
        if leading:
            return self._simulate_observation_leading(wdm_mass)
        return self._simulate_observation_trailing(wdm_mass)

    def _simulate_observation_leading(self, wdm_mass):
        success = False
        output = None
        stream = self.streams[2]
        dens_unp = self.dens_unp_leading
        omega_unp = self.omega_unp_leading
        while not success:
            try:
                output = self._simulate(wdm_mass, stream, dens_unp, omega_unp)
                success = True
            except Exception as error:
                print(error)
        return output

    def _simulate_observation_trailing(self, wdm_mass):
        success = False
        output = None
        stream = self.streams[3]
        dens_unp = self.dens_unp_trailing
        omega_unp = self.omega_unp_trailing
        while not success:
            try:
                output = self._simulate(wdm_mass, stream, dens_unp, omega_unp)
                success = True
            except Exception as error:
                print(error)
        return output

    def _simulate(self, wdm_mass, stream, dens_unp, omega_unp):
        outputs = simulate_subhalos_mwdm(
            stream, m_wdm=wdm_mass, r=self.ravg, Xrs=self.Xrs
        )
        impact_angles = outputs[0]
        impactbs = outputs[1]
        subhalovels = outputs[2]
        timpacts = outputs[3]
        gms = outputs[4]
        rss = outputs[5]
        num_impacts = len(gms)
        has_impacts = num_impacts > 0
        if not self.allow_no_impacts and not has_impacts:
            raise ValueError("Only observations with impacts are allowed!")
        if has_impacts:
            stream.set_impacts(
                impactb=impactbs,
                subhalovel=subhalovels,
                impact_angle=impact_angles,
                timpact=timpacts,
                rs=rss,
                GM=gms,
            )
            dens_omega = np.array(
                [stream._densityAndOmega_par_approx(a) for a in self.apars]
            ).T
            mT = stream.meanTrack(self.apars, _mO=dens_omega[1], coord="lb")
            phi = lb_to_phi12(mT[0], mT[1], degree=True)[:, 0]
            phi[phi > 180] -= 360
            density = compute_obs_density(phi, self.apars, dens_omega[0], dens_omega[1])
        else:
            mT = stream.meanTrack(self.apars, _mO=omega_unp, coord="lb")
            phi = lb_to_phi12(mT[0], mT[1], degree=True)[:, 0]
            phi[phi > 180] -= 360
            density = compute_obs_density(phi, self.apars, dens_unp, omega_unp)
        if np.isnan(density).sum() > 0:
            raise ValueError("nan values have been computed.")
        if self.record_impacts:
            return num_impacts, phi, density
        return phi, density

    def _simulate_observations(self, wdm_mass):
        return self._simulate_observation(wdm_mass, leading=False)

    def forward(self, inputs):
        outputs = []
        inputs = inputs.view(-1, 1)
        for input in inputs:
            observation = self._simulate_observations(input.item())
            outputs.append(observation)
        return outputs


class PresimulatedStreamsWDMSubhaloSimulator(WDMSubhaloSimulator):
    def __init__(self, datadir, stream_model_index, resolution=0.01):
        streams = self._load_streams(datadir, stream_model_index)
        super().__init__(streams, resolution=resolution)

    def _load_streams(self, datadir, index):
        stream_models_path_query = datadir + "/stream-models/streams-*"
        directories = glob.glob(stream_models_path_query)
        directories.sort()
        stream_directories = directories
        stream_blocks = len(directories)
        streams_per_block = len(np.load(stream_directories[0] + "/inputs.npy"))
        block_index = int(index / streams_per_block)
        stream_index_in_block = index % streams_per_block
        stream_directory = stream_directories[block_index]
        path = stream_directory + "/outputs.pickle"
        with open(path, "rb") as file_handle:
            streams = pickle.load(file_handle)[stream_index_in_block]
        return streams
