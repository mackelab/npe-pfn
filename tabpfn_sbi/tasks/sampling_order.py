import torch

from .base import InferenceTask


class SamplingOrderTask(InferenceTask):
    def __init__(self, *args, **kwargs):
        self.n_input_dim = kwargs.get("n_input_dim", None)
        self.n_output_dim = kwargs.get("n_output_dim", None)

        self.permutation = kwargs.get("permutation", None)
        all_permutations = self.get_all_permutations()
        assert self.permutation is None or self.permutation in all_permutations, (
            "Permutation must be a permutation of range(n_output_dim)"
        )

        # Sample and store task data (observations)
        self.xs, self.ys = self.sample(20)

        # Sample and store separate prior data
        n_prior_samples = 50_000
        self.prior_xs, self.prior_ys = self.sample(n_prior_samples)

        # Define prior that randomly samples from stored prior data
        class StoredPrior:
            def __init__(self, prior_ys):
                self.prior_ys = prior_ys
                self.n_samples = len(prior_ys)
                # Each stored sample has equal probability: 1/n_samples
                self.log_prob_per_sample = -torch.log(
                    torch.tensor(self.n_samples, dtype=torch.float)
                )

            def sample(self, sample_shape=(1,)):
                n_samples = (
                    sample_shape[0] if isinstance(sample_shape, tuple) else sample_shape
                )
                idx = torch.randint(0, len(self.prior_ys), (n_samples,))
                return self.prior_ys[idx]

            def log_prob(self, value):
                """This is super ineffective but I don't know how to do it better now"""
                # Handle batch dimensions
                if value.ndim == 1:
                    value = value.unsqueeze(0)

                # For each value, check if it exists in prior_ys (with tolerance)
                batch_size = value.shape[0]
                log_probs = torch.full(
                    (batch_size,), float("-inf"), device=value.device
                )

                # Using a small tolerance for floating point comparison
                for i in range(batch_size):
                    # Calculate distance to all stored prior samples
                    distances = torch.norm(self.prior_ys - value[i].unsqueeze(0), dim=1)
                    # Check if any distance is very small (match found)
                    if torch.any(distances < 0.5):
                        log_probs[i] = self.log_prob_per_sample

                return log_probs

        self.prior = StoredPrior(self.prior_ys)

        # Define simulator that finds matching y and returns corresponding x
        def simulator(ys):
            # ys shape: (batch_size, n_output_dim)
            batch_size = ys.shape[0]
            xs_out = torch.zeros((batch_size, self.n_input_dim))

            # For each input y, find the closest matching y in our prior data
            for i, y in enumerate(ys):
                # Calculate distances to all stored prior ys
                distances = torch.norm(self.prior_ys - y.unsqueeze(0), dim=1)
                # Get index of closest match
                closest_idx = torch.argmin(distances)
                # check that distance of closest match is not too large
                assert distances[closest_idx] < 1e-6, (
                    f"Distance of closest match is too large {distances[closest_idx]}"
                )
                # Return corresponding x from prior data
                xs_out[i] = self.prior_xs[closest_idx]

            return xs_out

        self.simulator = simulator

        super().__init__(prior=self.prior, simulator=self.simulator)

    def get_all_permutations(self):
        """
        Returns all possible permutations of range(n_output_dim)
        Returns:
            List of permutations, each permutation is a list of integers
        """
        import itertools

        if self.n_output_dim is None:
            raise ValueError("n_output_dim must be set by child class")
        return list(itertools.permutations(range(self.n_output_dim)))

    def permute_y(self, y, permutation):
        """
        Permute the output y according to the given permutation
        Args:
            y: tensor of shape (n_samples, n_output_dim)
            permutation: list of integers representing the permutation
        Returns:
            permuted y tensor of shape (n_samples, n_output_dim)
        """
        return y[:, permutation]

    def get_true_parameter(self, idx: int, device: str = "cpu") -> torch.Tensor:
        """Get the true parameter y for a given index."""
        torch.manual_seed(idx)
        return self.ys[idx]

    def get_observation(self, idx: int, device: str = "cpu") -> torch.Tensor:
        """Get the observation x for a given index."""
        return self.xs[idx]

    def _sample_features(self, n_samples: int):
        return torch.randn(n_samples, self.n_input_dim)

    def sample(self, n_samples, x=None, permutation=None):
        """Sample from the task."""
        pass

    def get_reference_posterior_samples(self, idx: int, device: str = "cpu"):
        """Get samples from the reference posterior."""
        torch.manual_seed(idx)
        x_o = self.get_observation(idx, device)
        # repeat observation 10_000 times
        n_samples = 10_000
        x_o = x_o.repeat(n_samples, 1)
        # sample from reference posterior
        samples = self.sample(n_samples, x_o)[1]
        return samples.reshape(n_samples, self.n_output_dim)


class OrderSimpleNonlinearTask(SamplingOrderTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """ simple nonlinear task with
        x = [x1, x2] ~ N(0, 1)
        y1 ~ N(x1, 1)
        y2|y1 ~ N(sin(y1 + x2), 1)
        y3|y1, y2 ~ N(y2^2 + y1, 1)
        y4|y1, y2, y3 ~ N(y1 * y2 + y3, 1)
        """
        self.n_input_dim = 2
        self.n_output_dim = 4

    def sample(self, n_samples, x=None, permutation=None):
        if x is None:
            x = self._sample_features(n_samples)

        y1 = x[:, 0] + torch.randn(n_samples)
        y2 = torch.sin(y1 + x[:, 1]) + torch.randn(n_samples)
        y3 = y2**2 + y1 + torch.randn(n_samples)
        y4 = y1 * y2 + y3 + torch.randn(n_samples)
        y = torch.stack([y1, y2, y3, y4], dim=-1)

        if permutation is not None:
            y = self.permute_y(y, permutation)

        return x, y


class OrderMixedDistTask(SamplingOrderTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        """
        mixed dist task with
        x = [x1, x2] ~ U(-2, 2)
        y1 ~ Gamma(shape=1 + abs(x1), scale=1)
        y2|y1 ~ Uniform(0, y1*2 + abs(x2))
        y3|y1, y2 ~ Beta(alpha=1 + y1, beta=2 + y2)
        """
        self.n_input_dim = 2
        self.n_output_dim = 3

    def sample(self, n_samples, x=None, permutation=None):
        if x is None:
            x = self._sample_features(n_samples)

        y1 = torch.distributions.gamma.Gamma(1 + torch.abs(x[:, 0]), 1).sample()
        y2 = torch.rand(n_samples) * (y1 * 2 + torch.abs(x[:, 1]))
        y3 = torch.distributions.beta.Beta(1 + y1, 2 + y2).sample()
        y = torch.stack([y1, y2, y3], dim=-1)

        if permutation is not None:
            y = self.permute_y(y, permutation)

        return x, y
