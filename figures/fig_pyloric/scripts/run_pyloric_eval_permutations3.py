from functools import partial
from tabpfn_sbi.tasks.pyloric import PyloricTask
from tabpfn_sbi.methods.tabpfn_restricted_prior import TabPFNRestrictedPrior
from tabpfn_sbi.methods.tabpfn_sbi import FilteredTabPFNSBI
from tabpfn_sbi.methods.tabpfn_support_posterior import PosteriorSupport
import torch
import os
import wandb
import numpy as np
from tabpfn.model.multi_head_attention import set_flex_attention

set_flex_attention(False)


seed = 3
torch.manual_seed(seed)
np.random.seed(seed)
torch.cuda.manual_seed_all(seed)


task = PyloricTask(plateau_durations=False)
prior = task.get_prior_dist()

simulator = task.get_simulator(device="cuda")

thetas_start, xs_start = task.get_initial_dataset()
valid_fn = task.get_is_valid_fn()

is_valid_start = valid_fn(xs_start)

xs_std = xs_start[is_valid_start].std(axis=0)


def standardized_dist(x, x_train):
    return torch.mean(((x - x_train) / xs_std).pow(2).sum(axis=1))


def filter_valid_best(x, theta_train, x_train, N):
    is_valid = valid_fn(x_train)
    theta_valid = theta_train[is_valid]
    x_valid = x_train[is_valid]

    std_x_valid = x_valid.std(axis=0)

    dist_x = ((x - x_valid) / std_x_valid).pow(2).sum(axis=1)
    idx = dist_x.argsort()[:N]

    return theta_valid[idx], x_valid[idx]


# Real observation
x_o = task.get_observation(-1).cpu()
x_mean, x_std = task.get_valid_x_mean_std()


def energy_scoring_rule(X, x):
    beta = 1
    X_z = (X - x_mean) / x_std
    x_z = (x - x_mean) / x_std
    term_1 = (
        2
        / X.shape[0]
        * torch.sum(torch.sqrt(torch.sum(torch.square(X_z - x_z), axis=-1)))
    )
    term_2 = (
        1
        / (X.shape[0] * (X.shape[0] - 1))
        * torch.sum(
            torch.sqrt(
                torch.sum(torch.square(X_z[:, None, :] - X_z[None, :, :]), axis=-1)
            )
        )
    )
    return term_1 - term_2


energy_scores_list = []
valid_rates_list = []
for i in range(10):
    posterior = torch.load(
        f"../../../notebooks/posteriors3/pyloric_posterior44.pt", weights_only=False
    )
    permutation = torch.randperm(31)
    posterior.append_simulations(
        posterior._theta_train[..., permutation], posterior._x_train
    )
    posterior.prior = torch.distributions.Independent(
        torch.distributions.Uniform(
            posterior.prior.base_dist.low[..., permutation],
            posterior.prior.base_dist.high[..., permutation],
        ),
        1,
    )
    samples = posterior.sample((1_000,), x=x_o, max_iter_rejection=2)
    # Invert permutation
    permutation = torch.argsort(permutation)
    samples = samples[..., permutation]
    xs_pred = simulator(samples)
    is_valid = valid_fn(xs_pred)
    valid_rate = torch.sum(is_valid) / xs_pred.shape[0]
    xs_pred_valid = xs_pred[is_valid]
    energy_scores = energy_scoring_rule(xs_pred_valid, x_o)
    print(energy_scores, valid_rate)
    energy_scores_list.append(energy_scores)
    valid_rates_list.append(valid_rate)

print(energy_scores_list)
print(valid_rates_list)

import pickle

with open("energy_scores_list_3.pkl", "wb") as f:
    pickle.dump(energy_scores_list, f)
with open("valid_rates_list_3.pkl", "wb") as f:
    pickle.dump(valid_rates_list, f)
