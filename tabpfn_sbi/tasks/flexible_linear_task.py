from typing import Optional

import numpy as np
import pymc as pm
import torch
from omegaconf import DictConfig
from torch.distributions import Distribution

from .base import InferenceTask


class FlexibleLinearTask(InferenceTask):
    """Task for flexible linear regression with configurable distributions."""

    def __init__(
        self,
        dim: int,
        feature_dist: str,
        noise_dist: str,
        feature_scale: float,
        noise_scale: float,
        mcmc_cfg: DictConfig,
        *args,
        **kwargs,
    ) -> None:
        """Flexible linear task with configurable distributions

        Args:
            dim (int): Number of dimensions (input and output)
            feature_dist (str): Feature distribution
            noise_dist (str): Noise distribution
            feature_scale (float): Feature scale
            noise_scale (float): Noise scale
        """
        self.n_output_dim = dim
        self.n_input_dim = dim
        self.feature_dist = feature_dist
        self.noise_dist = noise_dist
        self.prior_dist = feature_dist
        self.feature_scale = feature_scale
        self.noise_scale = noise_scale
        self.prior_scale = feature_scale
        self.mcmc_cfg = mcmc_cfg
        simulator = self.get_simulator()
        prior = self.get_prior()

        # generate 10 parameters, observations and reference posterior samples
        thetas, xs = self.sample(10)
        reference_posterior_samples = self.compute_reference_posterior_samples(xs)
        # make sure these are subscriptable
        self.thetas = thetas
        self.xs = xs
        self.reference_posterior_samples = reference_posterior_samples

        super().__init__(prior, simulator)

    def get_true_parameter(self, idx: int, device: str = "cpu") -> torch.Tensor:
        """Get the true parameter for the given index."""
        return self.thetas[idx - 1]

    def get_observation(self, idx: int, device: str = "cpu") -> torch.Tensor:
        """Get the observation for the given index."""
        return self.xs[idx - 1]

    def get_reference_posterior_samples(self, idx: int, device: str = "cpu"):
        """Get reference posterior samples for the given index."""
        return self.reference_posterior_samples[idx - 1]

    def get_simulator(self):
        """Return the simulator function"""

        def simulator(theta):
            return self.sample(theta.shape[0], theta=theta)[1]

        return simulator

    def get_prior(self) -> Distribution:
        """Return a prior distribution based on the configured feature distribution"""
        # Create a prior matching the feature distribution used for true_theta
        if self.prior_dist == "normal":
            return torch.distributions.Normal(
                loc=torch.zeros(self.n_input_dim),
                scale=torch.ones(self.n_input_dim) * self.prior_scale,
            )
        elif self.prior_dist == "studentt":
            return torch.distributions.StudentT(
                df=torch.ones(self.n_input_dim) * 5,
                loc=torch.zeros(self.n_input_dim),
                scale=torch.ones(self.n_input_dim) * self.prior_scale,
            )
        elif self.prior_dist == "laplace":
            return torch.distributions.Laplace(
                loc=torch.zeros(self.n_input_dim),
                scale=torch.ones(self.n_input_dim) * self.prior_scale,
            )
        elif self.prior_dist == "cauchy":
            # PyTorch doesn't have a built-in Cauchy, but it's equivalent to StudentT with df=1
            return torch.distributions.StudentT(
                df=torch.ones(self.n_input_dim),
                loc=torch.zeros(self.n_input_dim),
                scale=torch.ones(self.n_input_dim) * self.prior_scale,
            )
        elif self.prior_dist == "logistic":
            # For logistic distribution
            class LogisticDistribution(torch.distributions.Distribution):
                def __init__(self, loc, scale):
                    super().__init__(validate_args=False)
                    self.loc = loc
                    self.scale = scale
                    batch_shape = loc.size()
                    event_shape = torch.Size([])
                    super(LogisticDistribution, self).__init__(batch_shape, event_shape)

                def sample(self, sample_shape=torch.Size()):
                    shape = self._extended_shape(sample_shape)
                    u = torch.rand(shape, device=self.loc.device)
                    return self.loc - self.scale * torch.log(1.0 / u - 1.0)

                def log_prob(self, value):
                    z = (value - self.loc) / self.scale
                    return (
                        -z
                        - 2 * torch.nn.functional.softplus(-z)
                        - torch.log(self.scale)
                    )

            return LogisticDistribution(
                loc=torch.zeros(self.n_input_dim),
                scale=torch.ones(self.n_input_dim) * self.prior_scale,
            )
        elif self.prior_dist == "logitnormal":
            # Create a transformed distribution: LogitNormal
            base_distribution = torch.distributions.Normal(
                loc=torch.zeros(self.n_input_dim),
                scale=torch.ones(self.n_input_dim) * self.prior_scale,
            )
            transforms = [torch.distributions.SigmoidTransform()]
            return torch.distributions.TransformedDistribution(
                base_distribution, transforms
            )
        else:
            raise ValueError(f"Prior distribution {self.prior_dist} not supported")

    def sample(self, n_samples: int, theta: Optional[torch.Tensor] = None):
        true_theta = theta
        # convert x to numpy array
        if true_theta is not None:
            true_theta = true_theta.numpy()

        if true_theta is None:
            size = (n_samples, self.n_input_dim)
            if self.feature_dist == "normal":
                true_theta = np.random.normal(0, self.feature_scale, size=size)
            elif self.feature_dist == "studentt":
                true_theta = np.random.standard_t(5, size=size) * self.feature_scale
            elif self.feature_dist == "laplace":
                true_theta = np.random.laplace(0, self.feature_scale, size=size)
            elif self.feature_dist == "cauchy":
                true_theta = np.random.standard_cauchy(size=size) * self.feature_scale
            elif self.feature_dist == "logistic":
                true_theta = np.random.logistic(0, self.feature_scale, size=size)
            elif self.feature_dist == "logitnormal":
                # For LogitNormal: generate normal values and apply invlogit transformation
                normal_values = np.random.normal(0, self.feature_scale, size=size)
                true_theta = 1.0 / (1.0 + np.exp(-normal_values))  # invlogit function
            else:
                raise ValueError(
                    f"Feature distribution {self.feature_dist} not supported"
                )

        # Generate observed data based on the likelihood distribution
        # Vectorized implementation - no for loop needed
        size = (n_samples, self.n_input_dim)
        if self.noise_dist == "normal":
            noise = np.random.normal(0, self.noise_scale, size=size)
            x_data = true_theta + noise
        elif self.noise_dist == "studentt":
            noise = np.random.standard_t(5, size=size) * self.noise_scale
            x_data = true_theta + noise
        elif self.noise_dist == "laplace":
            noise = np.random.laplace(0, self.noise_scale, size=size)
            x_data = true_theta + noise
        elif self.noise_dist == "cauchy":
            noise = np.random.standard_cauchy(size=size) * self.noise_scale
            x_data = true_theta + noise
        elif self.noise_dist == "logistic":
            noise = np.random.logistic(0, self.noise_scale, size=size)
            x_data = true_theta + noise
        elif self.noise_dist == "logitnormal":
            # For LogitNormal, generate normal noise and apply invlogit to (theta + noise)
            normal_noise = np.random.normal(0, self.noise_scale, size=size)
            # Add noise and transform with inverse logit function
            x_data = 1.0 / (1.0 + np.exp(-(true_theta + normal_noise)))
        else:
            raise ValueError(f"Noise distribution {self.noise_dist} not supported")

        # convert to torch tensor
        x_data = torch.from_numpy(x_data).float()
        true_theta = torch.from_numpy(true_theta).float()

        return true_theta, x_data

    def compute_reference_posterior_samples(self, xs: torch.Tensor) -> torch.Tensor:
        """Compute the reference posterior for the given observations."""
        n_samples = xs.shape[0]

        # Instead of using a nested function, process sequentially
        all_samples = []
        for i in range(n_samples):
            all_samples.append(self._compute_single_posterior(xs[i].reshape(1, -1)))

        return torch.stack(all_samples)

    def _compute_single_posterior(self, observation: torch.Tensor) -> torch.Tensor:
        """Compute the reference posterior for a single observation."""
        with pm.Model() as model:
            theta = create_pymc_distribution(
                dist_name=self.prior_dist,
                name="theta",
                dim=self.n_input_dim,
                sigma=self.feature_scale,
            )
            x_obs = create_pymc_distribution(
                dist_name=self.noise_dist,
                name="x",
                dim=self.n_output_dim,
                sigma=self.noise_scale,
                mu=theta,
                observed=observation,
            )
            # perform sampling
            trace = pm.sample(
                self.mcmc_cfg.n_samples_mcmc // self.mcmc_cfg.chains,
                tune=self.mcmc_cfg.n_tune,
                return_inferencedata=True,
                chains=self.mcmc_cfg.chains,
                cores=self.mcmc_cfg.cores,
                progressbar=False,
            )

        # Determine variable names based on trace content
        var_names = []
        if "theta" in trace.posterior:
            var_names = ["theta"]
        else:
            # Look for individual theta_i variables
            var_names = [
                var for var in trace.posterior.data_vars if var.startswith("theta_")
            ]

        # Extract posterior samples
        if "theta" in trace.posterior:
            # For multivariate distribution
            samples = trace.posterior.theta.values.reshape(
                -1, self.n_input_dim
            )  # Flatten chains and draws
        else:
            # Handle individual theta_i variables
            samples_list = []
            for var in var_names:
                samples_list.append(
                    trace.posterior[var].values.flatten()
                )  # Flatten chains and draws
            samples = np.column_stack(samples_list)  # Stack as columns

        # Convert to torch tensor
        samples = torch.from_numpy(samples).float()
        # Only keep the first n_posterior_samples
        samples = samples[: self.mcmc_cfg.n_samples_mcmc]
        return samples


def create_pymc_distribution(dist_name, name, dim, sigma, mu=None, observed=None):
    """Create PyMC distribution based on provided name and parameters."""
    # Default mu is zeros if not provided
    if mu is None:
        mu = np.zeros(dim)

    if dist_name == "normal":
        return pm.MvNormal(name, mu=mu, cov=np.eye(dim) * sigma**2, observed=observed)
    elif dist_name == "studentt":
        return pm.StudentT(name, nu=5, mu=mu, sigma=sigma, observed=observed)
    elif dist_name == "laplace":
        return pm.Laplace(name, mu=mu, b=sigma, observed=observed)
    elif dist_name == "cauchy":
        return pm.Cauchy(name, alpha=mu, beta=sigma, observed=observed)
    elif dist_name == "logistic":
        return pm.Logistic(name, mu=mu, s=sigma, observed=observed)
    elif dist_name == "logitnormal":
        return pm.LogitNormal(name, mu=mu, sigma=sigma, observed=observed)
    else:
        raise ValueError(f"Distribution {dist_name} not supported")
