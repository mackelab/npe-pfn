import time

import torch

from tabpfn_sbi.tasks.base import Task


def compute_standardized_distance(cfg, model, task: Task, logger):
    observations_ids = cfg.eval.observation_ids
    num_posterior_samples = cfg.eval.num_eval_samples
    num_prior_samples = cfg.eval.num_prior_samples

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    test_thetas = prior.sample((num_prior_samples,))
    test_xs = simulator(test_thetas)

    feature_means = torch.mean(test_xs, dim=0, keepdim=True)
    feature_stds = torch.std(test_xs, dim=0, keepdim=True)

    eps = 1e-10
    feature_stds = torch.where(
        feature_stds < eps, torch.ones_like(feature_stds), feature_stds
    )

    dists = []
    sampling_times = 0.0

    for idx in observations_ids:
        x_o = task.get_observation(idx)
        start_time = time.time()
        samples_model = model.sample((num_posterior_samples,), x_o)
        end_time = time.time()
        sampling_times += end_time - start_time

        pos_pred = simulator(samples_model)
        std_pos_pred = (pos_pred - feature_means) / feature_stds
        std_obs = (x_o - feature_means) / feature_stds

        std_dist = torch.mean(torch.norm(std_pos_pred - std_obs, dim=1))

        dists.append(float(std_dist))

        logger.info(f"Standardized distance for observation {idx}: {std_dist}")

    return {"standardized_distance": dists}, sampling_times
