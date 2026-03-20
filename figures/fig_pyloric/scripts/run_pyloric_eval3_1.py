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


seed = 0
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)

if not os.path.exists("posteriors_results3"):
    os.mkdir("posteriors_results3")


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


# Eval every 5 posterior
for i in [20]:
    posterior = torch.load(f"posteriors3/pyloric_posterior{i}.pt", weights_only=False)
    samples = posterior.sample((10_000,), x=x_o, max_iter_rejection=2)
    xs_pred = []
    for j in range(10):
        samples_batch = samples[j * 1000 : (j + 1) * 1000]
        xs_pred_batch = simulator(samples_batch)
        xs_pred.append(xs_pred_batch)
    xs_pred = torch.cat(xs_pred, dim=0)
    is_valid = valid_fn(xs_pred)
    valid_rate = torch.sum(is_valid) / xs_pred.shape[0]
    xs_pred_valid = xs_pred[is_valid]
    energy_scores = energy_scoring_rule(xs_pred_valid, x_o)

    print(f"Valid rate: {valid_rate}")
    print(f"Energy score: {energy_scores}")

    torch.save(energy_scores, f"posteriors_results3/pyloric_energy_scores{i}.pt")
    torch.save(valid_rate, f"posteriors_results3/pyloric_valid_rate{i}.pt")
    torch.save(xs_pred, f"posteriors_results3/pyloric_xs_pred{i}.pt")
    torch.save(samples, f"posteriors_results3/pyloric_samples{i}.pt")
