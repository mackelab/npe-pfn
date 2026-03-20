import logging
import os
import random
import socket
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from tabpfn_sbi.evals.eval_calibration import compute_calibration
from tabpfn_sbi.evals.eval_log_prob import compute_average_log_prob
from tabpfn_sbi.evals.eval_nll import compute_nll
from tabpfn_sbi.evals.eval_standardized_distance import compute_standardized_distance
from tabpfn_sbi.evals.metric_to_reference import compute_metric2reference
from tabpfn_sbi.methods import get_method
from tabpfn_sbi.tasks import get_task
from tabpfn_sbi.utils.data_utils import (
    generate_globally_unique_model_id,
    init_dir,
    load_model,
    save_model,
    save_summary,
)

# Backends


logo = """
 ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄   ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄        ▄  ▄▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄  ▄▄▄▄▄▄▄▄▄▄▄
▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░░▌      ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌▐░░░░░░░░░░░▌
 ▀▀▀▀█░█▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░▌░▌     ▐░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀█░▌▀▀▀▀█░█▀▀▀▀
     ▐░▌     ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░▌▐░▌    ▐░▌▐░▌          ▐░▌       ▐░▌    ▐░▌
     ▐░▌     ▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░▌ ▐░▌   ▐░▌▐░█▄▄▄▄▄▄▄▄▄ ▐░█▄▄▄▄▄▄▄█░▌    ▐░▌
     ▐░▌     ▐░░░░░░░░░░░▌▐░░░░░░░░░░▌ ▐░░░░░░░░░░░▌▐░░░░░░░░░░░▌▐░▌  ▐░▌  ▐░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌     ▐░▌
     ▐░▌     ▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀▀▀ ▐░█▀▀▀▀▀▀▀▀▀ ▐░▌   ▐░▌ ▐░▌ ▀▀▀▀▀▀▀▀▀█░▌▐░█▀▀▀▀▀▀▀█░▌    ▐░▌
     ▐░▌     ▐░▌       ▐░▌▐░▌       ▐░▌▐░▌          ▐░▌          ▐░▌    ▐░▌▐░▌          ▐░▌▐░▌       ▐░▌    ▐░▌
     ▐░▌     ▐░▌       ▐░▌▐░█▄▄▄▄▄▄▄█░▌▐░▌          ▐░▌          ▐░▌     ▐░▐░▌ ▄▄▄▄▄▄▄▄▄█░▌▐░█▄▄▄▄▄▄▄█░▌▄▄▄▄█░█▄▄▄▄
     ▐░▌     ▐░▌       ▐░▌▐░░░░░░░░░░▌ ▐░▌          ▐░▌          ▐░▌      ▐░░▌▐░░░░░░░░░░░▌▐░░░░░░░░░░▌▐░░░░░░░░░░░▌
      ▀       ▀         ▀  ▀▀▀▀▀▀▀▀▀▀   ▀            ▀            ▀        ▀▀  ▀▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀  ▀▀▀▀▀▀▀▀▀▀▀
"""


def main():
    """Main script function"""
    print(logo)
    try:
        _main()
    except Exception as e:
        import traceback

        print("Exception occurred:", e)
        traceback.print_exc()
        raise


@hydra.main(config_path="../../conf", config_name="config.yaml", version_base=None)
def _main(cfg: DictConfig):
    """Evaluate score based inference"""
    log = logging.getLogger(__name__)
    log.info(OmegaConf.to_yaml(cfg))

    output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    # Go back to the folder named "cfg.name"
    output_super_dir = os.path.dirname(output_dir)
    while os.path.basename(output_super_dir) != cfg.name:
        output_super_dir = os.path.dirname(output_super_dir)

    init_dir(output_super_dir)

    log.info(f"Working directory : {os.getcwd()}")
    log.info(f"Output directory  : {output_dir}")
    log.info("Output super directory: {}".format(output_super_dir))
    log.info(f"Hostname: {socket.gethostname()}")
    log.info(f"Torch devices: {torch.cuda.device_count()}")

    seed = cfg.seed
    set_seed(seed)

    # Generate 3 seeds for training_data, training_model, and evaluation
    training_data_seed = random.randint(0, 2**32 - 1)
    training_model_seed = random.randint(0, 2**32 - 1)
    evaluation_seed = random.randint(0, 2**32 - 1)

    # If model id is specified, no training!

    # Initialize task
    log.info(f"Task: {cfg.task.name}")
    set_seed(training_data_seed)
    task_name = cfg.task.name
    task = get_task(task_name, **dict(cfg.task.params))
    num_simulations = cfg.task.num_simulations

    # NOTE SBIBM can only simulate on CPU (custom tasks might want to do it on gpu)
    # Load model if model_id is specified
    set_seed(training_model_seed)
    if cfg.model_id is not None:
        model_id = cfg.model_id
        log.info(f"Loading model {model_id}")
        model = load_model(output_super_dir, model_id)
        training_time = None

        new_cfg = model.cfg
        # Update with current evaluation config
        new_cfg.eval = cfg.eval
        new_cfg.save_summary = cfg.save_summary
        new_cfg.save_model = cfg.save_model
        new_cfg.model_id = cfg.model_id
        new_cfg.method.sampler = cfg.method.sampler

        cfg = new_cfg

        # Update task if necessary
        task_name = cfg.task.name
        task = get_task(task_name)
    else:
        # Run method
        log.info(f"Method: {cfg.method.name}")
        log.info(f"Fitting method...")
        method = cfg.method.name
        runner = get_method(method)

        start_time = time.time()
        model = runner(cfg, task)
        end_time = time.time()
        training_time = end_time - start_time
        log.info(f"Training time: {training_time}")

    # Evaluate model
    log.info(f"Evaluating model")
    set_seed(evaluation_seed)
    eval_methods = cfg.eval.name

    metric_values = {}
    sampling_times = {}
    if eval_methods[0] is None:
        metric_values = {s: None for s in eval_methods}
        sampling_times = {s: None for s in eval_methods}
    else:
        for eval_method in eval_methods:
            if eval_method == "sbc" or eval_method == "tarp":
                metric_value, sampling_time = compute_calibration(
                    cfg, [eval_method], model, task, log
                )
            elif eval_method == "std_dist":
                metric_value, sampling_time = compute_standardized_distance(
                    cfg, model, task, log
                )
            elif eval_method == "log_prob":
                metric_value, sampling_time = compute_average_log_prob(
                    cfg, model, task, log
                )
            elif eval_method == "nll":
                metric_value, sampling_time = compute_nll(cfg, model, task, log)
            else:
                metric_value, sampling_time = compute_metric2reference(
                    cfg, [eval_method], model, task, log
                )
            metric_values[eval_method] = metric_value[eval_method]
            sampling_times[eval_method] = sampling_time

    for eval_method, metric_value in metric_values.items():
        log.info(f"Metric {eval_method}: {metric_value}")

    log.info(f"Sampling times: {sampling_times}")

    # Saving results if necessary
    is_save_model = cfg.save_model
    is_save_summary = cfg.save_summary

    if is_save_model and (cfg.model_id is None):
        model_id = generate_globally_unique_model_id()
        log.info(f"Saving model with id {model_id}")
        try:
            save_model(model, output_super_dir, model_id)
        except Exception as e:
            log.error(f"Error saving model: {e}")
            # Print traceback
            exc_type, exc_value, exc_traceback = sys.exc_info()
            import traceback

            traceback.print_exception(
                exc_type, exc_value, exc_traceback, limit=2, file=sys.stdout
            )

    if is_save_summary:
        for metric in metric_values:
            log.info("Saving summary")
            saved_summary = False
            while not saved_summary:
                # If file is blocked by another process, it might raise an exception
                try:
                    save_summary(
                        output_super_dir,
                        method=method,
                        estimator=cfg.method.estimator.name,
                        embedding_net=cfg.method.embedding_net.name,
                        task=task_name,
                        num_simulations=num_simulations,
                        model_id=model_id,
                        observation_ids=cfg.eval.observation_ids,
                        metric=metric,
                        value=metric_values[metric],
                        seed=seed,
                        time_train=training_time,
                        time_eval=sampling_times[metric],
                        cfg=cfg,
                    )
                    saved_summary = True
                except Exception as e:
                    log.error(f"Error saving summary: {e}")

    # If you use a sweeper this output will be used to "evaluate" the run i.e. as
    # the objective function e.g. Bayesian optimization.
    # For multiple metrics, return the mean of all (non-None) metric values.
    metric_value_list = [m if m is not None else 0 for m in metric_values.values()]
    metric_value_list = [m[0] if isinstance(m, list) else m for m in metric_value_list]
    return sum(metric_value_list) / len(metric_value_list) if metric_value_list else 0


def set_seed(seed: int):
    """This methods just sets the seed."""
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
