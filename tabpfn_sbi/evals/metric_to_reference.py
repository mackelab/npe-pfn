import time
from typing import List

import torch
from sbi.utils.metrics import c2st

from tabpfn_sbi.tasks.base import Task

# Any other meaningful metrics


def sliced_wasserstein_distance(samples_true, samples_model, num_projections=1000):
    """Compute the sliced Wasserstein distance between two sets of samples."""
    projections = torch.randn(
        num_projections, samples_true.shape[-1], device=samples_true.device
    )
    projections = projections / torch.norm(projections, dim=-1, keepdim=True)
    projections = projections.t()
    samples_true_proj = torch.matmul(samples_true, projections)
    samples_model_proj = torch.matmul(samples_model, projections)

    samples_true_proj = torch.sort(samples_true_proj, dim=0).values
    samples_model_proj = torch.sort(samples_model_proj, dim=0).values

    swd = torch.mean(torch.abs(samples_true_proj - samples_model_proj))
    return swd


METRICS = {"c2st": c2st, "swd": sliced_wasserstein_distance}


def compute_metric2reference(cfg, metrics: List[str], model, task: Task, logger):
    observations_ids = cfg.eval.observation_ids
    num_eval_samples = cfg.eval.num_eval_samples

    metric_values = {metric: [] for metric in metrics}
    sampling_times = 0.0

    for metric in metrics:
        metric_fn = METRICS[metric]
        for idx in observations_ids:
            # Fetch reference samples and observation
            x_o = task.get_observation(idx)
            samples_true = task.get_reference_posterior_samples(idx)
            assert samples_true.shape[0] >= num_eval_samples
            samples_true = samples_true[:num_eval_samples]
            # Sample from the model
            start_time = time.time()
            samples_model = model.sample((num_eval_samples,), x_o)
            end_time = time.time()
            sampling_times += end_time - start_time
            samples_model = torch.nan_to_num(samples_model)
            metric_value = metric_fn(samples_true, samples_model)
            metric_values[metric].append(float(metric_value))

            logger.info(f"Metric {metric} for observation {idx}: {metric_value}")

    sampling_times /= len(observations_ids) * len(metrics)
    return metric_values, sampling_times
