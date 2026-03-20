import torch
import torch.distributions as D

from .base import InferenceTask


class MisspecifiedTask(InferenceTask):
    """Task for inference in a Gaussian model with misspecified prior or likelihood."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_true_parameter(self, idx: int, device: str = "cpu") -> torch.Tensor:
        """Get the true parameter for a given index.

        Args:
            idx (int): Index of the parameter
            device (str): Device to use

        Returns:
            torch.Tensor: True parameter
        """
        torch.manual_seed(idx)
        return self.prior.sample((1,)).to(device)

    def get_observation(self, idx: int, device: str = "cpu") -> torch.Tensor:
        """Get the observation for a given index.

        Args:
            idx (int): Index of the observation
            device (str): Device to use

        Returns:
            torch.Tensor: Observation
        """
        theta_o = self.get_true_parameter(idx, device)
        return self.ground_truth.sample_data(theta_o)

    def get_reference_posterior_samples(self, idx: int, device: str = "cpu"):
        """Get samples from the reference posterior.

        Args:
            idx (int): Index of the observation
            device (str): Device to use

        Returns:
            torch.Tensor: Posterior samples
        """
        torch.manual_seed(idx)
        x_o = self.get_observation(idx, device)
        samples = self.ground_truth.get_reference_posterior(x_o).sample((10_000,))
        return samples.reshape(10_000, self.dim)


class LikelihoodMisspecifiedTask(MisspecifiedTask):
    """Task for inference in a Gaussian model with misspecified likelihood."""

    def __init__(
        self,
        dim: int = 2,
        tau_m: float = 2.0,  # Misspecified likelihood variance
        lambda_val: float = 0.5,  # Mixture weight
    ):
        """Initialize the Gaussian misspecified likelihood task.

        Args:
            dim (int): Dimensionality of the parameter space
            tau_m (float): Variance factor for the misspecified likelihood
            lambda_val (float): Mixture weight
        """
        self.dim = dim
        self.tau_m = tau_m
        self.lambda_val = lambda_val

        # Setup ground truth model
        self.ground_truth = GroundTruthModel(dim=dim)

        # define prior
        self.prior = self.ground_truth.prior_dist

        def simulator(thetas):
            # thetas shape: (batch_size, dim)
            batch_size = thetas.shape[0]

            # Generate a batch of Bernoulli samples - decide which distribution to use for each sample
            is_beta = torch.bernoulli(
                torch.tensor(self.lambda_val, dtype=torch.float32).expand(batch_size)
            )

            # Initialize result tensor with the same shape as thetas
            result = torch.zeros_like(thetas)

            # Process beta samples
            beta_mask = is_beta == 1
            beta_samples = D.Beta(torch.tensor(2.0), torch.tensor(5.0)).sample(
                (beta_mask.sum(), self.dim)
            )
            result[beta_mask] = beta_samples

            # Process normal samples
            # For samples where is_beta=0, we sample from N(theta, tau_m * I)
            normal_mask = ~beta_mask
            normal_dist = D.MultivariateNormal(
                loc=thetas[normal_mask],
                covariance_matrix=self.tau_m * torch.eye(self.dim),
            )
            result[normal_mask] = normal_dist.sample()

            return result

        super().__init__(prior=self.prior, simulator=simulator)


class PriorMisspecifiedTask(MisspecifiedTask):
    """Task for inference in a Gaussian model with misspecified prior."""

    def __init__(
        self,
        dim: int = 2,
        mu_m: float = 2.5,  # Misspecified prior mean
        tau_m: float = 2.0,  # Misspecified prior variance
    ):
        """Initialize the Gaussian misspecified prior task.

        Args:
            dim (int): Dimensionality of the parameter space
            mu_m (float): Mean value for the misspecified prior
            tau_m (float): Variance factor for the misspecified prior
            device (str, optional): Device to use. Defaults to None.
            cache_dir (str, optional): Directory for caching. Defaults to ".".
        """
        self.dim = dim
        self.mu_m = mu_m
        self.tau_m = tau_m

        # Setup ground truth model
        self.ground_truth = GroundTruthModel(dim=dim)

        # Define prior (misspecified)
        prior = D.MultivariateNormal(
            self.mu_m * torch.ones(self.dim), self.tau_m * torch.eye(self.dim)
        )

        # Define log likelihood function
        def loglikelihood_fn(thetas):
            thetas = thetas.reshape(-1, self.dim)
            batch_shape = thetas.shape[:-1]

            def likelihood(theta):
                return D.MultivariateNormal(theta, self.ground_truth.sigma_likelihood)

            return D.Independent(likelihood(thetas), 1)

        super().__init__(prior=prior, loglikelihood_fn=loglikelihood_fn)


class GroundTruthModel:
    """Ground truth model with standard Gaussian prior and identity covariance likelihood."""

    def __init__(self, dim=2, device=None):
        self.dim = dim
        self.mu_prior = torch.ones(dim)  # Prior mean is set to 1
        self.sigma_prior = torch.eye(dim)  # Prior covariance is identity
        self.sigma_likelihood = torch.eye(dim)  # Likelihood covariance is identity

        # Create distributions
        self.prior_dist = D.MultivariateNormal(self.mu_prior, self.sigma_prior)

    def sample_prior(self, num_samples=1):
        """Sample from N(mu_prior, sigma_prior)."""
        return self.prior_dist.sample((num_samples,))

    def sample_data(self, parameters):
        """Generate one observation from N(parameters, sigma_likelihood) for each parameter vector."""
        parameters = parameters.view(-1, self.dim)
        # Generate one sample for each parameter vector
        data = torch.stack(
            [
                D.MultivariateNormal(param, self.sigma_likelihood).sample()
                for param in parameters
            ]
        )
        # Shape will be (num_parameters, dim)
        return data

    def get_reference_posterior(self, observations):
        """Get the reference posterior for a given set of observations."""
        mean = 0.5 * (observations + self.mu_prior)
        cov = self.sigma_likelihood / 2
        return D.MultivariateNormal(mean, cov)
