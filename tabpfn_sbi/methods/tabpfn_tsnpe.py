import logging
from typing import Mapping, Optional

import torch
from sbi import inference as inference
from tabpfn.model.multi_head_attention import set_flex_attention

from tabpfn_sbi.methods.tabpfn_sbi import FilteredTabPFNSBI
from tabpfn_sbi.methods.tabpfn_support_posterior import PosteriorSupport

log = logging.getLogger(__name__)


def run_ts_tabpfn_interface(cfg, task):
    set_flex_attention(cfg.method.use_flex_attention)

    eval_observations = cfg.eval.observation_ids
    if len(eval_observations) > 1:
        raise ValueError("Sequential methods only support one observation at a time")

    observation_id = eval_observations[0]

    return run_ts_tabpfn(
        task=task,
        num_simulations=cfg.task.num_simulations,
        observation_id=observation_id,
        num_rounds=cfg.method.sequential.num_rounds,
        proposal_batch_size=cfg.method.sequential.proposal_batch_size,
        simulation_batch_size=cfg.method.sequential.simulation_batch_size,
        num_samples_to_estimate_support=cfg.method.sequential.num_samples_to_estimate_support,
        allowed_false_negatives=cfg.method.sequential.allowed_false_negatives,
        context_size=cfg.method.sequential.context_size,
        log_prob_mode=cfg.method.sequential.log_prob_mode,
        sampling_method=cfg.method.sequential.sampling_method,
        max_iter_rejection=cfg.method.sequential.max_iter_rejection,
        oversample_sir=cfg.method.sequential.oversample_sir,
        filtering=cfg.method.sequential.filtering,
        regressor_init_kwargs=cfg.method.regressor_init_kwargs,
        classifier_init_kwargs=cfg.method.classifier_init_kwargs,
    )


def run_ts_tabpfn(
    task,
    num_simulations: int,
    observation_id: Optional[int] = None,
    observation: Optional[torch.Tensor] = None,
    num_rounds: int = 10,
    proposal_batch_size: int = 1000,
    simulation_batch_size: int = 1000,
    num_samples_to_estimate_support: int = 10_000,
    allowed_false_negatives: float = 0.0001,
    context_size: int = 10_000,
    log_prob_mode: str = "ratio_based",
    sampling_method: str = "rejection",
    max_iter_rejection: int = 1000,
    oversample_sir: int = 100,
    filtering: str = "no_filtering",
    regressor_init_kwargs: Mapping = {},
    classifier_init_kwargs: Mapping = {},
):
    """Runs (S)NPE from `sbi`
    Args:
        task: Task instance
        num_simulations: Simulation budget
        observation_id: ID of the observation to load, alternative to `observation`
        observation: Observation, alternative to `observation_id`
        num_rounds: Number of rounds
        proposal_batch_size: Batch size for proposal sampling
        simulation_batch_size: Batch size for simulator
        num_samples_to_estimate_support: Number of samples to estimate support
        allowed_false_negatives: Allowed false negatives, epsilon
        use_constrained_prior: Use constrained prior (only possible currently if prior is uniform, could be generalized)
        constrained_prior_quantile: Constrained prior quantile
        log_prob_mode: Log probability mode, ["ratio_based", "autoregressive"]
        filtering: Filtering method, ["no_filtering", "latest_filtering", "random_filtering", "standardized_euclidean"]
    Returns:
        Posterior from the last round
    """

    # not sure if this observation inferace makes sense for us
    assert not (observation_id is None and observation is None)
    assert not (observation_id is not None and observation is not None)

    if num_rounds == 1:
        log.info(f"Running NPE")
        num_simulations_per_round = num_simulations
    else:
        log.info(f"Running SNPE")
        num_simulations_per_round = num_simulations // num_rounds

    log.info(f"Number of simulations per round: {num_simulations_per_round}")

    if simulation_batch_size > num_simulations_per_round:
        simulation_batch_size = num_simulations_per_round
        log.warning("Reduced simulation_batch_size to num_simulation_per_round")

    prior = task.get_prior_dist()
    if observation is None:
        observation = task.get_observation(observation_id)

    simulator = task.get_simulator()

    tabpfn_posterior = FilteredTabPFNSBI(
        prior=prior,
        regressor_init_kwargs=regressor_init_kwargs,
        classifier_init_kwargs=classifier_init_kwargs,
        filter_type=filtering,
        filter_context_size=context_size,
    )
    proposal = prior

    theta_per_round = []
    x_per_round = []
    for round_num in range(num_rounds):
        log.info(f"Round {round_num + 1}/{num_rounds}")
        # TODO the proposal in here never gets the batch size for proposal.sample, currently default 10k
        # Same for other sampling arguments like progress bar etc.
        # The progress bar of the simulator is hard coded.
        log.info("Drawing from proposal and simulating!")
        theta, x = inference.simulate_for_sbi(
            simulator,
            proposal,
            num_simulations=num_simulations_per_round,
            simulation_batch_size=simulation_batch_size,
        )

        theta_per_round.append(theta)  # append to the end
        x_per_round.append(x)

        theta_cat = torch.cat(theta_per_round, dim=0)
        x_cat = torch.cat(x_per_round, dim=0)

        log.info("Appending simulations and initializing restricted proposal!")
        posterior = tabpfn_posterior.append_simulations(theta_cat, x_cat)

        if round_num == num_rounds - 1:
            break

        posterior_support = PosteriorSupport(
            prior,
            posterior,
            obs=observation,
            num_samples_to_estimate_support=num_samples_to_estimate_support,
            batch_size_for_estimate_support=proposal_batch_size,
            allowed_false_negatives=allowed_false_negatives,
            sampling_method=sampling_method,
            max_iter_rejection=max_iter_rejection,
            oversample_sir=oversample_sir,
            log_prob_kwargs={"mode": log_prob_mode},
        )
        proposal = posterior_support

    return posterior
