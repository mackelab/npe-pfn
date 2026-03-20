import torch

from tabpfn_sbi.tasks.base import Task


def compute_nll(cfg, model, task: Task, logger):
    num_eval_samples = cfg.eval.num_eval_samples

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    test_thetas = prior.sample((num_eval_samples,))
    test_xs = simulator(test_thetas)
    # Better if both have log_prob_batched
    nlls = []
    for theta, x in zip(test_thetas, test_xs):
        log_prob = model.log_prob(theta, x)
        nll = -log_prob
        nlls.append(nll)
    nll = torch.mean(torch.tensor(nlls))
    logger.info(f"NLL: {nll}")
    return {"nll": [float(nll)]}, None
