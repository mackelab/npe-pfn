import os
import pickle

import numpy as np
import torch
from sbi.utils import BoxUniform

from tabpfn_sbi.tasks.allen.HodgkinHuxleyStatsMoments import HodgkinHuxleyStatsMoments


def get_allen_task_parameters(task_idx):
    assert task_idx in range(1, 11), "task idx must be between 1 and 10"

    list_cells_AllenDB = [
        (518290966, 57, 0.0234 / 126),
        (509881736, 39, 0.0153 / 184),
        (566517779, 46, 0.0195 / 198),
        (567399060, 38, 0.0259 / 161),
        (569469018, 44, 0.033 / 403),
        (532571720, 42, 0.0139 / 127),
        (555060623, 34, 0.0294 / 320),
        (534524026, 29, 0.027 / 209),
        (532355382, 33, 0.0199 / 230),
        (526950199, 37, 0.0186 / 218),
    ]
    ephys_cell, sweep_number, A_soma = list_cells_AllenDB[task_idx - 1]
    return ephys_cell, sweep_number, A_soma


def allen_obs_data(ephys_cell, sweep_number, A_soma, dir_cache="."):
    """Data for x_o. Cell from AllenDB
    Parameters
    ----------
    ephys_cell : int
        Cell identity from AllenDB
    sweep_number : int
        Stimulus identity for cell ephys_cell from AllenDB
    """
    # TODO only supports local loading of files via path
    real_data_path = os.path.join(
        dir_cache,
        f"allen_support_files/ephys_cell_{ephys_cell}_sweep_number_{sweep_number}.pkl",
    )

    # not sure whats up with this pickle encoding
    def pickle_load(file):
        """Loads data from file."""
        f = open(file, "rb")
        data = pickle.load(f, encoding="latin1")
        f.close()
        return data

    real_data_obs, I_real_data, dt, t_on, t_off = pickle_load(real_data_path)

    duration = 1450.0
    t = np.arange(0, duration, dt)

    # external current
    I = I_real_data / A_soma  # muA/cm2

    # return real_data_obs, I_obs
    return {
        "data": real_data_obs.reshape(-1),
        "time": t,
        "dt": dt,
        "I": I.reshape(-1),
        "t_on": t_on,
        "t_off": t_off,
    }


def allen_obs_stats(
    data,
    n_xcorr=5,
    n_mom=5,
    n_summary=13,
):
    """Summary stats for x_o. Cell from AllenDB
    Parameters
    ----------
    ephys_cell : int
        Cell identity from AllenDB
    sweep_number : int
        Stimulus identity for cell ephys_cell from AllenDB
    """

    t_on = data["t_on"]
    t_off = data["t_off"]

    s = HodgkinHuxleyStatsMoments(
        t_on, t_off, n_xcorr=n_xcorr, n_mom=n_mom, n_summary=n_summary
    )
    return s.calc([data])


def synth_obs_data(idx: int, dir_cache="."):
    assert idx in range(1, 11), "task idx must be between 1 and 10"

    synth_data_path = os.path.join(
        dir_cache,
        f"allen_support_files/synthetic_obs_{idx}.pkl",
    )

    with open(synth_data_path, "rb") as f:
        synth_data = pickle.load(f)

    return synth_data


def synth_obs_theta(idx: int):
    assert idx in range(1, 11), "task idx must be between 1 and 10"

    # could move this to a file
    gt_thetas = np.array(
        [
            [
                8.932692,
                2.006126,
                0.028068965,
                0.03327207,
                2792.28,
                65.309746,
                0.03135364,
                86.49111,
            ],
            [
                7.4164505,
                0.82499933,
                0.016678464,
                0.045744028,
                59.59537,
                63.6171,
                0.07078508,
                78.66387,
            ],
            [
                10.32953,
                2.4506733,
                0.05042587,
                0.03049464,
                2833.1992,
                65.26796,
                0.09762452,
                72.68365,
            ],
            [
                9.053179,
                2.2230043,
                0.025084043,
                0.06314755,
                2984.3423,
                65.21115,
                0.035672132,
                88.9565,
            ],
            [
                10.788687,
                2.4330573,
                0.0018535767,
                0.12134603,
                525.6326,
                62.318,
                0.026455127,
                68.965355,
            ],
            [
                15.966566,
                2.8125353,
                0.00016230323,
                0.034970816,
                998.69965,
                54.86428,
                0.12653835,
                96.33733,
            ],
            [
                6.649622,
                1.9998106,
                0.046611875,
                0.0019918683,
                2863.3604,
                62.2985,
                0.038908705,
                75.2866,
            ],
            [
                7.978747,
                1.5670553,
                0.010259892,
                0.08008519,
                99.82189,
                66.32357,
                0.043602005,
                64.09779,
            ],
            [
                7.7967424,
                1.6968954,
                0.0127043985,
                0.07608426,
                2704.6282,
                63.813427,
                0.034412667,
                79.90711,
            ],
            [
                8.847322,
                1.6688964,
                0.05155613,
                0.0054907375,
                1174.22,
                67.95606,
                0.04107448,
                78.74205,
            ],
        ],
        dtype=np.float32,
    )

    return gt_thetas[idx - 1]


def prior(prior_log=False):
    """Prior"""
    range_lower = param_transform(
        prior_log, np.array([0.5, 1e-4, 1e-4, 1e-4, 50.0, 40.0, 1e-4, 35.0])
    )
    range_upper = param_transform(
        prior_log, np.array([80.0, 15.0, 0.6, 0.6, 3000.0, 90.0, 0.15, 100.0])
    )

    prior_min = torch.tensor(range_lower.astype(np.float32))
    prior_max = torch.tensor(range_upper.astype(np.float32))
    return BoxUniform(prior_min, prior_max)


def param_transform(prior_log, x):
    if prior_log:
        return np.log(x)
    else:
        return x


def param_invtransform(prior_log, x):
    if prior_log:
        return np.exp(x)
    else:
        return x
