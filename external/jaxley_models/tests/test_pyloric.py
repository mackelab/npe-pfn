# Test data `data/pyloric{seed}_T{temp}.csv` was generated using the following code:
#
# from pyloric import simulate, create_prior
# import torch
# import numpy as np

# def simulate_pyloric_network(seed=0, temperature=283.0, t_max=500.0):
#     torch.manual_seed(seed)
#     prior = create_prior()
#     params = prior.sample((1,)).loc[0]
#     simulation_output = simulate(params, t_max=t_max, noise_std=0.0, temperature=temperature)

#     t = np.arange(0, simulation_output["t_max"], simulation_output["dt"])
#     v = simulation_output["voltage"]
#     return params.to_dict(), t, v


# for seed, temp in zip([0, 1, 0], [283.0, 283.0, 273.0]):
#     params, t, v = simulate_pyloric_network(seed=seed, temperature=temp)
#     np.savez(f"pyloric{seed}_T{temp:.0f}.npz", t=t, v=v, params=params)

# install pyloric using:
# git clone https://github.com/mackelab/pyloric
# pip install -e pyloric

from jax import config

config.update("jax_enable_x64", True)
config.update("jax_platform_name", "cpu")

import jax.numpy as jnp
import jaxley as jx
import numpy as np
import pandas as pd
import pytest

from jaxley_models.pyloric.model import PyloricNetwork


@pytest.mark.parametrize("temp", [283.0, 283.0, 273.0])
@pytest.mark.parametrize("seed", [0, 1, 0])
def test_pyloric(temp, seed):
    data = np.load(f"data/pyloric{seed}_T{temp:.0f}.npz", allow_pickle=True)
    t_true = data["t"]
    v_true = data["v"]
    params = pd.Series(data["params"].item())

    net = PyloricNetwork()
    net.record("v")

    for neuron in ["AB/PD", "LP", "PY"]:
        params[neuron] = params[neuron] * 1e-3  # mS
    params["Synapses"] = jnp.exp(params["Synapses"].values) * 1e3  # mS -> uS

    for group_key, syn_or_channel_name in params.keys():
        if group_key != "Synapses":
            neuron = net.select(net.groups[group_key.lower().replace("/", "_")])
            value = params[group_key, syn_or_channel_name].item()
            neuron.set(f"{syn_or_channel_name}_g{syn_or_channel_name}", value)

    for i, val in enumerate(params["Synapses"].values):
        synapse = net.select(edges=i)
        if "glut" in synapse.edges["type"].item().lower():
            synapse.set(f"GlutamatergicSynapse_gS", val)
        else:
            synapse.set(f"CholinergicSynapse_gS", val)

    ts = jnp.arange(0, 500.0, 0.025)
    v = jx.integrate(net, t_max=500.0, delta_t=0.025)

    # small numerical differences lead to drift in spikes and large errors, errors < 1
    # is already a good match
    assert jnp.mean(jnp.abs(v_true - v[:, :-1])) < 1
