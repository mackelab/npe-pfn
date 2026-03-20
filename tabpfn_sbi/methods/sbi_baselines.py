import logging
import random
import time
from functools import partial

import numpy as np
import torch
import torch.multiprocessing as mp
from sbi.inference import NPSE, SNLE, SNPE, SNRE
from sbi.inference.posteriors import EnsemblePosterior
from sbi.neural_nets import classifier_nn, likelihood_nn, posterior_nn
from sbi.neural_nets.embedding_nets import CNNEmbedding, FCEmbedding
from sbi.utils import RestrictedPrior, get_density_thresholder
from tqdm import tqdm

from tabpfn_sbi.methods.neural_summary_stats import run_pretraining

log = logging.getLogger(__name__)


# TODO: clean up whole file


def get_inf_method(name: str):
    if name == "nle" or name == "snle":
        return SNLE
    elif name == "nre" or name == "snre":
        return SNRE
    elif (
        name == "npe"
        or name == "snpe"
        or name == "tsnpe"
        or name == "npe_infomax"
        or name == "npe_ensemble"
        or name == "npe_sweeper"
    ):
        return SNPE
    elif name == "nse" or name == "snse":
        return NPSE
    else:
        raise ValueError(f"Unknown method {name}")


def get_embedding_net(name: str, input_shape, **params):
    if name == "mlp":
        assert len(input_shape) == 1, "MLP only supports 1D input"
        return FCEmbedding(input_shape[0], **params)
    elif name == "cnn":
        print(input_shape)
        return CNNEmbedding(input_shape[1:], **params)
    elif name == "none":
        return torch.nn.Identity()
    else:
        raise ValueError(f"Unknown embedding net {name}")


def get_estimator(name: str, name_estimator: str, embedding_net, **params):
    if name == "nle" or name == "snle":
        estimator = likelihood_nn(name_estimator, embedding_net=embedding_net, **params)
    elif name == "nre" or name == "snre":
        estimator = classifier_nn(
            name_estimator, embedding_net_x=embedding_net, **params
        )
    elif (
        name == "npe"
        or name == "snpe"
        or name == "tsnpe"
        or name == "npe_infomax"
        or name == "npe_ensemble"
        or name == "npe_sweeper"
    ):
        estimator = posterior_nn(name_estimator, embedding_net=embedding_net, **params)
    else:
        raise ValueError(f"Unknown method {name}")

    return estimator


def run_basic_sbi_interface(cfg, task):
    name = cfg.method.name
    method = get_inf_method(name)

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    thetas = prior.sample((cfg.task.num_simulations,))
    xs = simulator(thetas)

    posterior = train_basic_sbi_interface(cfg, name, method, prior, thetas, xs)

    return posterior


def train_basic_sbi_interface(cfg, name, method, prior, thetas, xs, seed=None):
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        log.info(f"Set random seed to {seed}")

    # Set up embedding net
    input_shape = xs.shape[1:]
    embedding_net = get_embedding_net(
        cfg.method.embedding_net.name, input_shape, **cfg.method.embedding_net.params
    )

    # If pretraining is enabled, train the embedding net
    if hasattr(cfg.method, "pretraining"):
        embedding_net = run_pretraining(cfg, embedding_net, thetas, xs)

    # Set up estimator i.e. likelihood, classifier or posterior neural networks
    estimator = get_estimator(
        name, cfg.method.estimator.name, embedding_net, **cfg.method.estimator.params
    )

    # Set up neural inference
    inf = method(
        prior,
        estimator,
        device=cfg.method.device,
        show_progress_bars=cfg.method.show_progress_bars,
    )

    # Perform training
    inf = inf.append_simulations(thetas, xs)
    _ = inf.train(**cfg.method.train)

    # Build posterior
    posterior = inf.build_posterior(**cfg.method.sampler.params)

    posterior.sample = partial(
        posterior.sample, show_progress_bars=cfg.method.show_progress_bars
    )

    return posterior


def run_sweeper_sbi_interface(cfg, task):
    name = cfg.method.name
    method = get_inf_method(name)

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    thetas = prior.sample((cfg.task.num_simulations,))
    xs = simulator(thetas)

    posterior = train_sweeper_sbi_interface(cfg, name, method, prior, thetas, xs)

    return posterior


def train_sweeper_sbi_interface(cfg, name, method, prior, thetas, xs, seed=None):
    # Set random seed if provided
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        log.info(f"Set random seed to {seed}")

    # Set up embedding net
    input_shape = xs.shape[1:]
    assert cfg.method.embedding_net.name == "none"
    embedding_net = get_embedding_net(
        cfg.method.embedding_net.name, input_shape, **cfg.method.embedding_net.params
    )

    # Sweep plan
    best_val_loss = float("inf")
    best_estimator_name = None
    best_estimator_params = None
    best_train_params = None

    num_simulations = thetas.shape[0]
    assert num_simulations == cfg.task.num_simulations  # sanity

    max_iterations = cfg.method.max_sweep_iterations
    max_time_seconds = cfg.method.max_sweep_time_seconds
    max_gradient_steps = cfg.method.max_gradient_steps

    start_time = time.time()
    iteration = 0

    while iteration < max_iterations and (time.time() - start_time) < max_time_seconds:
        log.info(f"Sweep iteration {iteration + 1}")
        if iteration > 0:
            estimator_name = random.choice(["nsf", "maf"])
            estimator_params = {
                "z_score_theta": "independent",
                "z_score_x": "independent",
                "hidden_features": random.choice([25, 50, 100, 200]),
                "num_transforms": random.choice([3, 5, 7, 9]),
                "num_bins": 10,  # only relevant for nsf, will be ignored for other estimators
            }
            train_params = {
                "training_batch_size": random.choice([50, 100, 200, 500, 1000]),
                "learning_rate": random.choice(
                    [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
                ),
                "validation_fraction": 0.1,
                "max_num_epochs": None,  # will be set later via num simulations and batch size
                "clip_max_norm": 5.0,
            }
            # assumes training_batch_size divides num_simulations
            steps_per_epoch = max(
                num_simulations // train_params["training_batch_size"], 1
            )
            train_params["max_num_epochs"] = max_gradient_steps // steps_per_epoch
        else:
            # first iteration uses default params
            # also this means default is trained to the end
            estimator_name = cfg.method.estimator.name
            estimator_params = cfg.method.estimator.params
            train_params = cfg.method.train

        estimator = get_estimator(
            name,
            estimator_name,
            embedding_net,
            **estimator_params,
        )

        inf = method(
            prior,
            estimator,
            device=cfg.method.device,
            show_progress_bars=cfg.method.show_progress_bars,
        )

        # Perform training
        inf = inf.append_simulations(thetas, xs)

        training_start = time.time()
        _ = inf.train(**train_params)
        training_time = time.time() - training_start
        log.info(f"Training time: {training_time:.2f} seconds")

        val_loss = inf._summary["best_validation_loss"][0]
        log.info(f"Validation loss: {val_loss}")

        epochs_trained = inf._summary["epochs_trained"][0]
        log.info(f"Epochs trained: {epochs_trained}")

        # Update best params if needed
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_estimator_name = estimator_name
            best_estimator_params = estimator_params
            best_train_params = train_params

        iteration += 1

    elapsed_time = time.time() - start_time
    log.info(f"Completed {iteration} iterations in {elapsed_time:.2f} seconds")

    # TRAIN WITH BEST PARAMS
    # set to a high value for final training
    best_train_params["max_num_epochs"] = 100_000_000
    log.info(f"Best validation loss: {best_val_loss}")
    log.info(
        f"Best estimator: {best_estimator_name} with params: {best_estimator_params} and train params: {best_train_params}"
    )

    estimator = get_estimator(
        name,
        best_estimator_name,
        embedding_net,
        **best_estimator_params,
    )

    inf = method(
        prior,
        estimator,
        device=cfg.method.device,
        show_progress_bars=cfg.method.show_progress_bars,
    )

    # Perform training
    inf = inf.append_simulations(thetas, xs)
    _ = inf.train(**best_train_params)

    # Build posterior
    posterior = inf.build_posterior(**cfg.method.sampler.params)
    posterior.sample = partial(
        posterior.sample, show_progress_bars=cfg.method.show_progress_bars
    )

    return posterior


def run_npe_ensemble_sbi_interface(cfg, task):
    name = cfg.method.name
    method = get_inf_method(name)

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    thetas = prior.sample((cfg.task.num_simulations,))
    xs = simulator(thetas)

    # Generate random seeds for each estimator
    np.random.seed(cfg.method.seed if hasattr(cfg.method, "seed") else 42)
    seeds = np.random.randint(0, 1000000, size=cfg.method.num_estimators).tolist()
    log.info(f"Generated random seeds for estimators: {seeds}")

    # Create a partial function with fixed arguments
    train_fn = partial(train_basic_sbi_interface, cfg, "npe", method, prior, thetas, xs)

    # Check if multiprocessing should be used
    use_multiprocessing = getattr(cfg.method, "use_multiprocessing", False)

    if use_multiprocessing:
        # Use multiprocessing to train SNPEs in parallel
        # NOTE: This kinda only makes sense if you have a lot of CPU cores.
        num_processes = min(cfg.method.num_estimators, mp.cpu_count())
        log.info(
            f"Training {cfg.method.num_estimators} SNPEs using {num_processes} processes"
        )

        with mp.Pool(processes=num_processes) as pool:
            # Pass seeds as arguments to each process
            posteriors = list(
                tqdm(
                    pool.starmap(train_fn, [(seed,) for seed in seeds]),
                    total=cfg.method.num_estimators,
                    desc="Training SNPEs",
                )
            )
    else:
        # Use sequential processing
        log.info(f"Training {cfg.method.num_estimators} SNPEs sequentially")
        posteriors = []
        for seed in tqdm(seeds, desc="Training SNPEs"):
            posterior = train_fn(seed)
            posteriors.append(posterior)

    # Create ensemble posterior
    ensemble_posterior = EnsemblePosterior(posteriors)

    ensemble_posterior.sample = partial(
        ensemble_posterior.sample, show_progress_bars=cfg.method.show_progress_bars
    )

    return ensemble_posterior


def run_seq_sbi_interface(cfg, task):
    eval_observations = cfg.eval.observation_ids
    if len(eval_observations) > 1:
        raise ValueError("Sequential methods only support one observation at a time")

    observation_id = eval_observations[0]
    observation = task.get_observation(observation_id)

    name = cfg.method.name
    method = get_inf_method(name)

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    # set up embedding net
    input_shape = simulator(prior.sample((1,))).shape[1:]
    embedding_net = get_embedding_net(
        cfg.method.embedding_net.name, input_shape, **cfg.method.embedding_net.params
    )

    # Set up estimator i.e. likelihood, classifier or posterior neural networks
    estimator = get_estimator(
        name,
        cfg.method.estimator.name,
        embedding_net,
        **cfg.method.estimator.params,
    )

    # Set up neural inference
    inf = method(
        prior,
        estimator,
        device=cfg.method.device,
        show_progress_bars=cfg.method.show_progress_bars,
    )
    # initialize proposal
    proposal = prior

    # sequential rounds
    num_simulations_per_round = cfg.task.num_simulations // cfg.method.num_rounds
    for round_num in range(cfg.method.num_rounds):
        log.info(f"Round {round_num + 1}/{cfg.method.num_rounds}")
        # sample
        thetas = proposal.sample((num_simulations_per_round,))
        xs = simulator(thetas)
        # perform training
        kwargs = {"proposal": proposal} if name == "snpe" else {}
        inf = inf.append_simulations(thetas, xs, **kwargs)
        _ = inf.train(**cfg.method.train)
        # build posterior
        posterior = inf.build_posterior(**cfg.method.sampler.params)
        posterior.set_default_x(observation)
        # update proposal
        if name == "tsnpe":
            accept_reject_fn = get_density_thresholder(
                posterior,
                quantile=cfg.method.truncation.density_threshold_quantile,
            )
            proposal = RestrictedPrior(
                prior,
                accept_reject_fn=accept_reject_fn,
                sample_with=cfg.method.truncation.sample_with,
                posterior=posterior,
            )
        else:
            proposal = posterior

    posterior.sample = partial(
        posterior.sample, show_progress_bars=cfg.method.show_progress_bars
    )

    return posterior
