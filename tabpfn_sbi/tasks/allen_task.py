import logging

import numpy as np
import torch

import tabpfn_sbi.tasks.allen.allen_utils as allen_utils
from tabpfn_sbi.tasks.allen.HodgkinHuxley import HodgkinHuxley
from tabpfn_sbi.tasks.allen.HodgkinHuxleyStatsMoments import HodgkinHuxleyStatsMoments

from .base import InferenceTask

log = logging.getLogger(__name__)


class AllenTask(InferenceTask):
    def __init__(
        self,
        observation_id=1,
        obs_type="real",
        prior_log=False,
        cache_dir=".",
    ):
        if obs_type not in ["real", "synthetic"]:
            raise ValueError(
                f"obs_type must be either 'real' or 'synthetic', got {obs_type}"
            )
        if observation_id != 1:
            log.warning(
                f"observation_id is {observation_id}, but we usually use Allen observation_id=1 to init the simulator."
            )

        self.observation_id = observation_id
        self.cache_dir = cache_dir
        self.obs_type = obs_type

        ephys_cell, sweep_number, A_soma = allen_utils.get_allen_task_parameters(
            self.observation_id
        )
        obs = allen_utils.allen_obs_data(
            ephys_cell=ephys_cell,
            sweep_number=sweep_number,
            A_soma=A_soma,
            dir_cache=cache_dir,
        )

        self.JUNCTION_POTENTIAL = -14
        obs["data"] = obs["data"] + self.JUNCTION_POTENTIAL
        I = obs["I"]
        dt = obs["dt"]
        t_on = obs["t_on"]
        t_off = obs["t_off"]

        self.sim = HodgkinHuxley(
            I,
            dt,
            V0=obs["data"][0],
            cython=True,
            prior_log=prior_log,
        )

        self.N_XCORR = 0
        self.N_MOM = 4
        self.N_SUMMARY = 7

        self.stats = HodgkinHuxleyStatsMoments(
            t_on=t_on,
            t_off=t_off,
            n_xcorr=self.N_XCORR,
            n_mom=self.N_MOM,
            n_summary=self.N_SUMMARY,
        )

        def simulator(thetas):
            seeds = np.random.randint(0, 2**32, size=thetas.shape[0])  # set globally
            r = [
                self.sim.gen_single(theta, seed=seed)
                for theta, seed in zip(thetas, seeds)
            ]
            ss = self.stats.calc(r)
            return torch.from_numpy(ss.astype(np.float32))

        prior = allen_utils.prior(prior_log=prior_log)

        super().__init__(prior=prior, simulator=simulator)

    def get_observation(self, idx: int, device: str = "cpu"):
        if self.obs_type == "real":
            obs = self.get_real_observation(idx)
            obs_stats = allen_utils.allen_obs_stats(  # handles differing t_on, t_off
                data=obs,
                n_xcorr=self.N_XCORR,
                n_mom=self.N_MOM,
                n_summary=self.N_SUMMARY,
            )
        elif self.obs_type == "synthetic":
            obs = self.get_synthetic_observation(idx)
            obs_stats = self.stats.calc([obs])
        return torch.from_numpy(obs_stats.astype(np.float32)).to(device)

    def get_real_observation(self, idx: int):
        ephys_cell, sweep_number, A_soma = allen_utils.get_allen_task_parameters(idx)
        obs = allen_utils.allen_obs_data(
            ephys_cell=ephys_cell,
            sweep_number=sweep_number,
            A_soma=A_soma,
            dir_cache=self.cache_dir,
        )
        obs["data"] = obs["data"] + self.JUNCTION_POTENTIAL
        return obs

    def get_synthetic_observation(self, idx: int):
        obs = allen_utils.synth_obs_data(idx, dir_cache=self.cache_dir)
        # no need to add JUNCTION_POTENTIAL, synthetic obs were generated based on a simulator where this was already applied
        return obs

    def get_gt_theta_synthetic(self, idx: int):
        theta = allen_utils.synth_obs_theta(idx)
        return torch.from_numpy(theta)

    # setting a default batch size for allen to avoid memory issues
    def get_simulator(self, batch_size=10000, device="cpu"):
        return super().get_simulator(batch_size, device)
