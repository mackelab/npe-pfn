# %%
import os
import pickle
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sbi import analysis as analysis

from tabpfn_sbi.tasks.allen_task import AllenTask

# %%
# TODO deal with paths
run_result_dir = "/mnt/qb/work/macke/mwe824/tabpfn_sbi/results/allen_synth"
csv_path = os.path.join(run_result_dir, "summary.csv")
support_dir = "/mnt/qb/work/macke/mwe824"

df = pd.read_csv(csv_path)

print(df)


# %%
method_name = "tsnpe"
task_name = "allen"
num_simulations = 10000
seed = 0
obs_type = "synthetic"  # NOTE: IMPORTANT to set this correctly in accordance with the result dir

num_observations = 10

n_rows = num_observations
n_cols = 10  # num of samples (including the observation)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 20))

allen_task = AllenTask(cache_dir=support_dir, obs_type=obs_type)
prior = allen_task.get_prior_dist()
simulator = allen_task.get_simulator()


for i in range(num_observations):
    print(f"Processing observation {i + 1}")

    query_conditions = [
        f"method == '{method_name}'",
        f"task == '{task_name}'",
        f"num_simulations == {num_simulations}",
        f"seed == {seed}",
        f"observation_ids == '[{i + 1}]'",
    ]
    query_string = " and ".join(query_conditions)

    filtered_df = df.query(query_string)
    assert len(filtered_df) == 1
    row = filtered_df.iloc[0]

    obs = allen_task.get_observation(
        int(row["observation_ids"].strip("[]")), device="cpu"
    )
    model_id = row["model_id"]
    with open(os.path.join(run_result_dir, f"models/model_{model_id}.pkl"), "rb") as f:
        posterior = pickle.load(f)

    print(type(posterior))

    start = perf_counter()
    posterior_samples = posterior.sample((9,), x=obs)
    print(f"Sampling took {perf_counter() - start:.2f}s")

    seeds = np.arange(posterior_samples.shape[0])
    raw_spike_trains = [
        allen_task.sim.gen_single(theta, seed=seed)
        for theta, seed in zip(posterior_samples, seeds)
    ]

    if obs_type == "real":
        raw_obs = allen_task.get_real_observation(i + 1)
    elif obs_type == "synthetic":
        raw_obs = allen_task.get_synthetic_observation(i + 1)
    else:
        raise ValueError()

    raw_spike_trains = [raw_obs] + raw_spike_trains

    for idx, sim in enumerate(raw_spike_trains):
        ax = axes[i, idx]
        if idx == 0:
            ax.plot(sim["data"], c="C3")
        else:
            ax.plot(sim["data"])
        ax.set_ylim((-110, 60))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_visible(False)


plt.tight_layout()
plt.show()
