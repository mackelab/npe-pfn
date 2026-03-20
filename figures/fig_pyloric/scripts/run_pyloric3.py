from functools import partial
import gc
from tabpfn_sbi.tasks.pyloric import PyloricTask
from tabpfn_sbi.methods.tabpfn_restricted_prior import TabPFNRestrictedPrior
from tabpfn_sbi.methods.tabpfn_sbi import FilteredTabPFNSBI
from tabpfn_sbi.methods.tabpfn_support_posterior import PosteriorSupport
import torch
import os
import wandb
import numpy as np
import jax
from tabpfn.model.multi_head_attention import set_flex_attention

set_flex_attention(False)

seed = 0
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed_all(0)

# Initialize wandb
wandb.init(
    project="pyloric_task_experimental_10k", settings=wandb.Settings(init_timeout=120)
)

if not os.path.exists("posteriors3"):
    os.mkdir("posteriors3")

if not os.path.exists("restricted_priors3"):
    os.mkdir("restricted_priors3")

task = PyloricTask(plateau_durations=False)
prior = task.get_prior_dist()

simulator = task.get_simulator(device="cuda")
# Here we start with only 10k initial simulations
thetas_start, xs_start = task.get_initial_dataset(num_simulations=5_000)
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


restricted_prior = TabPFNRestrictedPrior(prior, acceptance_threshold=0.3)
restricted_prior.append_simulations(thetas_start, is_valid_start)

posterior = FilteredTabPFNSBI(
    prior=prior,
    filter_type=filter_valid_best,
)
posterior.append_simulations(thetas_start, xs_start)

# Real observation
x_o = task.get_observation(-1).cpu()

theta_per_round = []
x_per_round = []

# Do not get stuck in rejection loop
posterior.sample = partial(posterior.sample, max_iter_rejection=1)
for round_num in range(45):
    print("TSN round:", round_num)
    posterior_support = PosteriorSupport(
        restricted_prior,
        posterior,
        x_o,
        sampling_method="sir",
        oversample_sir=10,
        allowed_false_negatives=0.001,
    )
    try:
        theta_i, ess = posterior_support.sample((1000,), return_ess=True)
    except:
        torch.cuda.empty_cache()
        theta_i, ess = posterior_support.sample((1000,), return_ess=True)
    theta_i = theta_i.cpu()
    xs_i = simulator(theta_i).cpu()

    # Handle NaN values in theta_i and xs_i
    theta_nan_mask = torch.isnan(theta_i).any(dim=1)
    xs_nan_mask = torch.isnan(xs_i).any(dim=1)
    nan_mask = theta_nan_mask | xs_nan_mask

    if nan_mask.any():
        print(
            f"Found {nan_mask.sum().item()} NaN values, replacing with non-NaN values from previous iterations"
        )
        # Get non-NaN values from previous iterations
        valid_thetas = torch.cat(theta_per_round, dim=0)
        valid_xs = torch.cat(x_per_round, dim=0)

        # Replace NaN values with random valid values from previous iterations
        num_nan = nan_mask.sum().item()
        if num_nan > 0 and len(valid_thetas) > 0:
            random_indices = torch.randint(0, len(valid_thetas), (num_nan,))
            theta_i[nan_mask] = valid_thetas[random_indices]
            xs_i[nan_mask] = valid_xs[random_indices]

    print("ESS:", ess.mean())
    theta_per_round.append(theta_i)
    x_per_round.append(xs_i)
    y_valid = valid_fn(xs_i).cpu()

    valid_dist_to_xo = standardized_dist(xs_i[y_valid], x_o)
    print("Valid prob:", torch.tensor(y_valid, dtype=torch.float32).mean())
    print("Valid dist to xo:", valid_dist_to_xo)

    # Log metrics to wandb
    wandb.log(
        {
            "round": round_num,
            "ESS": ess.mean().item(),
            "Valid prob": torch.tensor(y_valid, dtype=torch.float32).mean().item(),
            "Valid dist to xo": valid_dist_to_xo.item(),
        }
    )

    thetas_overrounds = torch.cat(theta_per_round, dim=0)
    xs_overrounds = torch.cat(x_per_round, dim=0)

    theta_cat = torch.cat((thetas_start, thetas_overrounds), dim=0)
    xs_cat = torch.cat((xs_start, xs_overrounds), dim=0)
    restricted_prior.append_simulations(theta_i, y_valid)
    posterior.append_simulations(theta_cat, xs_cat)

    # Clear GPU memory and caches
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()  # Ensure GPU operations are complete
    # jax.clear_caches()  # Clear all JAX caches
    # del theta_i, xs_i  # Explicitly delete tensors we no longer need
    # gc.collect()  # Run garbage collection

    torch.save(posterior, f"posteriors3/pyloric_posterior{round_num}.pt")
    torch.save(
        restricted_prior, f"restricted_priors3/pyloric_restricted_prior{round_num}.pt"
    )

# Final eval
samples = posterior.sample((1000,))
xs_pred = simulator(samples)
valid = valid_fn(xs_pred)
valid_dist_to_xo = standardized_dist(xs_pred[valid], x_o)
valid_prob = torch.tensor(valid, dtype=torch.float32).mean()
wandb.log(
    {
        "Final valid prob": valid_prob.item(),
        "Final dist to xo": valid_dist_to_xo.item(),
    }
)

# Finish wandb run
wandb.finish()
