import os
from pathlib import Path
from functools import partial

# XLA disable preallocating memory
# Less efficient but more stable in combination with PyTorch
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import torch
import numpy as np

from tabpfn_sbi.tasks.base import InferenceTask

try:
    import jax
    import jax.numpy as jnp
    import jaxley as jx
    import pandas as pd
    from jax import config
    from jaxley_models.pyloric.model import PyloricNetwork
    from pyloric.interface import create_prior, summary_stats
except ImportError as error:
    jax = None
    jnp = None
    jx = None
    pd = None
    config = None
    PyloricNetwork = None
    create_prior = None
    summary_stats = None
    _PYLORIC_IMPORT_ERROR = error
else:
    config.update("jax_enable_x64", True)
    config.update("jax_platform_name", "gpu")
    _PYLORIC_IMPORT_ERROR = None

# import multiprocessing as mp


PYLORIC_FILES_DIR = Path(__file__).resolve().parent / "files" / "pyloric"


def _require_pyloric_dependencies():
    if _PYLORIC_IMPORT_ERROR is not None:
        raise ImportError(
            'Pyloric tasks require optional dependencies. Install with `pip install -e ".[pyloric]"`.'
        ) from _PYLORIC_IMPORT_ERROR


def convert_to_params_list(params):
    _require_pyloric_dependencies()
    # Convert units
    N = len(params)
    params = params.copy()
    for neuron in ["AB/PD", "LP", "PY"]:
        params[neuron] = params[neuron] * 1e-3  # mS
    params["Synapses"] = jnp.exp(params["Synapses"].values) * 1e3  # mS -> uS

    params_jaxley = []
    for group_key, syn_or_channel_name in params.keys():
        if group_key != "Synapses":
            name = f"{syn_or_channel_name}_g{syn_or_channel_name}"
            value = params[group_key, syn_or_channel_name]
            if N > 1:
                value = jnp.array(value).reshape((N, 1))
            else:
                value = jnp.array(value)
            params_jaxley.append({name: value})

    types = ["glut", "chol", "glut", "chol", "glut", "glut", "glut"]

    for i, val in enumerate(params["Synapses"].values.T):
        synapse_type = types[i]
        if "glut" in synapse_type:
            if N > 1:
                val = jnp.array(val).reshape((N, 1))
            else:
                val = jnp.array(val)
            params_jaxley.append({f"GlutamatergicSynapse_gS": val})
        else:
            if N > 1:
                val = jnp.array(val).reshape((N, 1))
            else:
                val = jnp.array(val)
            params_jaxley.append({f"CholinergicSynapse_gS": val})
    return params_jaxley


def build_network():
    _require_pyloric_dependencies()
    net = PyloricNetwork()
    net.record("v")

    params = create_prior().sample((1,)).iloc[0]

    for group_key, syn_or_channel_name in params.keys():
        if group_key != "Synapses":
            neuron = net.select(net.groups[group_key.lower().replace("/", "_")])
            value = params[group_key, syn_or_channel_name].item()

            neuron.make_trainable(
                f"{syn_or_channel_name}_g{syn_or_channel_name}", value
            )
            # neuron.set(f"{syn_or_channel_name}_g{syn_or_channel_name}", value)

    for i, val in enumerate(params["Synapses"].values):
        synapse = net.select(edges=i)

        if "glut" in synapse.edges["type"].item().lower():
            # synapse.set(f"GlutamatergicSynapse_gS", val)
            synapse.make_trainable(f"GlutamatergicSynapse_gS", val.item())
            # synapse.set(f"GlutamatergicSynapse_gS", val)
        else:
            # synapse.set(f"CholinergicSynapse_gS", val)
            synapse.make_trainable(f"CholinergicSynapse_gS", val.item())
            # synapse.set(f"CholinergicSynapse_gS", val)
    net.to_jax()
    return net


def build_simulator(net):
    _require_pyloric_dependencies()

    def simulator(params_list):
        v = jx.integrate(net, params=params_list, t_max=11_000.0, delta_t=0.025)
        return v

    return simulator


def convert_to_result_dict(v):
    v = np.array(v[:, :-1])
    return {"voltage": v, "dt": 0.025, "t_max": 11000}


def results_to_ss_array(v, num_cores=1, plateau_durations=True):
    _require_pyloric_dependencies()
    # More cores will crash kernel
    # Summary stats are actually quite complicated so lets do that outside of jax
    if v.ndim == 2:
        return np.array(
            summary_stats(
                convert_to_result_dict(v),
                stats_customization={"plateau_durations": plateau_durations},
            )
        )
    else:
        ss_fn = partial(
            summary_stats, stats_customization={"plateau_durations": plateau_durations}
        )
        v = np.array(v)
        vs = [convert_to_result_dict(v_) for v_ in v]
        # NOTE: MP fork default is not compatible with multithreaded applications like jax
        # with mp.get_context("spawn").Pool(num_cores) as pool:
        #     ss_list = pool.map(ss_fn, vs)
        # MP has wierd interactions with jax and doesnt really speed up the process
        ss_list = list(map(ss_fn, vs))
        return np.array(ss_list).squeeze(1)


valid_x_mean = torch.tensor(
    [
        1.5957e03,
        4.0928e02,
        4.3499e02,
        3.6788e02,
        2.3040e-01,
        2.7659e-01,
        2.4521e-01,
        4.4092e-01,
        6.2142e-01,
        7.2396e02,
        2.6796e02,
        3.1295e02,
        -1.5332e02,
        2.2246e-01,
        -7.5185e-02,
    ]
)

valid_x_std = torch.tensor(
    [
        1.0121e03,
        7.1285e02,
        6.5246e02,
        5.1249e02,
        3.0380e-01,
        3.6969e-01,
        2.8238e-01,
        1.9260e-01,
        2.0908e-01,
        8.0660e02,
        6.0093e02,
        3.8383e02,
        6.9402e02,
        1.6771e-01,
        3.5354e-01,
    ]
)


class PyloricTask(InferenceTask):
    def __init__(self, nan_to_num=-10, plateau_durations=True):
        _require_pyloric_dependencies()
        self.plateau_durations = plateau_durations
        self.nan_to_num = nan_to_num
        self._prior = create_prior()
        self._columns = self._prior.sample((1,)).columns
        thetas = np.load(PYLORIC_FILES_DIR / "pyloric_thetas.npy")
        ss_stats = np.load(PYLORIC_FILES_DIR / "pyloric_ss_stats.npy")
        self._initial_thetas = thetas
        self._inital_xs = ss_stats
        if self.plateau_durations:
            ss_stats = torch.tensor(ss_stats)
        else:
            ss_stats = torch.tensor(ss_stats[:, :15])
        valid_min = ss_stats[(ss_stats != -1.0).all(-1)].min(0).values
        self.nan_val = torch.tensor(valid_min - nan_to_num, dtype=torch.float32)

    def get_valid_x_mean_std(self):
        return valid_x_mean, valid_x_std

    def get_initial_dataset(self, num_simulations=None):
        thetas = torch.tensor(self._initial_thetas, dtype=torch.float32)
        ss_stats = torch.tensor(self._inital_xs, dtype=torch.float32)
        if num_simulations is not None:
            thetas = thetas[:num_simulations]
            ss_stats = ss_stats[:num_simulations]
        if self.plateau_durations:
            ss_stats = torch.where(ss_stats == -1.0, self.nan_val, ss_stats)
            return thetas, ss_stats
        else:
            ss_stats = ss_stats[:, :15]
            ss_stats = torch.where(ss_stats == -1.0, self.nan_val, ss_stats)
            return thetas, ss_stats

    def get_is_valid_fn(self):
        def is_valid(x):
            return (x != self.nan_val).all(-1)

        return is_valid

    def get_observation(self, num_observation):
        if num_observation == -1:
            _require_pyloric_dependencies()
            # Experimental data
            assert self.plateau_durations is False
            df = pd.read_csv(PYLORIC_FILES_DIR / "observation.csv")
            return torch.tensor(np.array(df), dtype=torch.float32)
        else:
            obs = torch.load(
                PYLORIC_FILES_DIR / "pyloric_observations.pt", weights_only=False
            )[num_observation]
            if self.plateau_durations:
                return obs
            else:
                return obs[:15]

    def get_true_parameters(self, num_observation):
        return torch.load(
            PYLORIC_FILES_DIR / "pyloric_true_params.pt", weights_only=False
        )[num_observation]

    def get_prior_dist(self, device="cpu"):
        _require_pyloric_dependencies()
        prior = create_prior()
        return prior.numerical_prior

    def get_simulator(
        self, type="summary_stats", device="cpu", voltage_noise=0.5, device_idx=0
    ):
        _require_pyloric_dependencies()
        net = build_network()
        simulator = build_simulator(net)

        _simulator = jax.jit(
            jax.vmap(simulator), device=jax.devices(device)[device_idx]
        )

        def simulator_fn(params):
            params = np.array(params)
            params = pd.DataFrame(params, columns=self._columns)
            params_list = convert_to_params_list(params)
            v = _simulator(params_list)
            if voltage_noise is not None:
                v = np.array(v)
                v += np.random.normal(0, voltage_noise, v.shape)
            if type == "summary_stats":
                ss_array = results_to_ss_array(
                    v, plateau_durations=self.plateau_durations
                )
                ss_array = np.nan_to_num(ss_array, nan=-1.0)
                ss_stats = torch.tensor(ss_array, dtype=torch.float32)
                ss_stats = torch.where(ss_stats == -1.0, self.nan_val, ss_stats)
                return ss_stats
            else:
                return torch.tensor(v, dtype=torch.float32)

        return simulator_fn
