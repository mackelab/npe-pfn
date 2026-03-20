import time

import torch

from tabpfn_sbi.tasks.base import Task


def compute_average_log_prob(cfg, model, task: Task, logger):
    observations_ids = cfg.eval.observation_ids

    log_probs = []
    sampling_times = 0.0

    for idx in observations_ids:
        x_o = task.get_observation(idx)
        samples_true = task.get_reference_posterior_samples(idx)
        start_time = time.time()
        log_prob = model.log_prob(samples_true, x_o)
        end_time = time.time()
        sampling_times += end_time - start_time

        average_log_prob = torch.mean(log_prob)

        log_probs.append(float(average_log_prob))

        logger.info(
            f"Average log prob for observation {idx}: {float(average_log_prob)}"
        )

    return {"log_prob": log_probs}, sampling_times
