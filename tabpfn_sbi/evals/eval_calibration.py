import numpy as np
import torch
from sbi.diagnostics import check_sbc, run_sbc, run_tarp
from sklearn.ensemble import RandomForestRegressor

from tabpfn_sbi.methods.tabpfn_sbi import TabPFNSBI


def compute_calibration(cfg, method, model, task, logger):
    calibration_methods = method
    num_sbc_samples = cfg.eval.num_sbc_samples
    num_posterior_samples = cfg.eval.num_posterior_samples

    prior = task.get_prior_dist()
    simulator = task.get_simulator()

    # Test simulations
    test_thetas = prior.sample((num_sbc_samples,))
    test_xs = simulator(test_thetas)

    results = {}
    for method in calibration_methods:
        if method == "sbc":
            ranks, dap_samples = run_sbc(
                test_thetas,
                test_xs,
                model,
                reduce_fns=model.log_prob,
                num_posterior_samples=num_posterior_samples,
                use_batched_sampling=not isinstance(model, TabPFNSBI),
            )

            check_stats = check_sbc(
                ranks,
                test_thetas,
                dap_samples,
                num_posterior_samples=num_posterior_samples,
            )
            pval = check_stats["ks_pvals"].numpy()
            logger.info(f"kolmogorov-smirnov p-values {pval}")

            results[method] = [int(x) for x in ranks.squeeze().numpy()]
        elif method == "tarp":
            # Fit linear regression model
            test_x_flat = test_xs.view(test_xs.shape[0], -1)
            reg = RandomForestRegressor(
                n_estimators=100, max_depth=10, random_state=42
            ).fit(test_x_flat, test_thetas)
            reference = reg.predict(test_x_flat)

            # Scale references to have the same mean and std as thetas
            z_score_ref = (
                reference - reference.mean(0, keepdims=True)
            ) / reference.std(0, keepdims=True)
            z_score_ref = torch.tensor(z_score_ref)

            ecp, alpha = run_tarp(
                test_thetas,
                test_xs,
                model,
                references=z_score_ref,
                num_posterior_samples=num_posterior_samples,
                use_batched_sampling=not isinstance(model, TabPFNSBI),
                z_score_theta=True,
            )
            # atc, ks_pval = check_tarp(ecp, alpha)
            test_stat_ks = torch.max(torch.abs(ecp - alpha))
            ks_pval = np.exp(-2 * test_stat_ks**2 * num_sbc_samples)
            logger.info(f"kolmogorov-smirnov p-values {ks_pval}")
            results[method] = (
                [float(a) for a in np.array(alpha)],
                [float(e) for e in np.array(ecp)],
            )

        else:
            raise NotImplementedError(f"Calibration method {method} not implemented")

    return results, None
