from typing import Any, Optional

import torch
from sbibm.tasks import get_task as get_task_sbibm

from .base import InferenceTask, Simulator


# General permutation task - permutes parameters
class PermutedTask(InferenceTask):
    def __init__(self, task_name: str, permutation: list[int]):
        self.task_name = task_name
        self.permutation = permutation

        self.task = get_task_sbibm(task_name)

        # Verify that the permutation is valid
        param_dim = self._get_param_dim()
        if len(self.permutation) != param_dim:
            raise ValueError(
                f"Permutation length ({len(self.permutation)}) must match "
                f"the number of parameters ({param_dim})"
            )

        # Initialize parent class with the base task's prior and simulator
        super().__init__(
            prior=self._permuted_prior(),
            simulator=self._permuted_simulator,
        )

    def _get_param_dim(self):
        """Get the dimension of parameters from the task."""
        if hasattr(self.task, "dim_parameters"):
            return self.task.dim_parameters
        elif hasattr(self.task, "output_dim"):
            return self.task.output_dim
        else:
            raise AttributeError(
                f"Task {self.task_name} has no attribute dim_parameters or output_dim"
            )

    def _permute(self, params):
        """Permute parameters according to the permutation."""
        if params.ndim == 1:
            return params[self.permutation]
        return params[:, self.permutation]

    def _unpermute(self, params):
        """Unpermute parameters according to the inverse permutation."""
        inv_perm = torch.zeros(len(self.permutation), dtype=torch.long)
        for i, p in enumerate(self.permutation):
            inv_perm[p] = i

        if params.ndim == 1:
            return params[inv_perm]
        return params[:, inv_perm]

    def _permuted_simulator(self, thetas):
        """Run simulator with unpermuted parameters."""
        thetas_unpermuted = self._unpermute(thetas)
        return self.task.get_simulator()(thetas_unpermuted)

    def _permuted_prior(self) -> Any:
        """Get the original prior with permutation applied to samples."""
        original_prior = self.task.get_prior_dist()

        # adapt sample method
        original_sample = original_prior.sample

        def permuted_sample(num_samples=torch.Size()):
            samples = original_sample(num_samples)
            return self._permute(samples)

        original_prior.sample = permuted_sample

        # adapt log_prob method
        original_log_prob = original_prior.log_prob

        def permuted_log_prob(value):
            return original_log_prob(self._unpermute(value))

        original_prior.log_prob = permuted_log_prob

        return original_prior

    def get_reference_posterior_samples(
        self, idx: int, device: str = "cpu"
    ) -> torch.Tensor:
        """Get reference posterior samples with permutation applied."""
        samples = self.task.get_reference_posterior_samples(idx)

        # Permute the samples
        permuted_samples = self._permute(samples)
        return permuted_samples.to(device)

    def get_true_parameter(self, idx: int, device: str = "cpu") -> torch.Tensor:
        """Get true parameter with permutation applied."""
        param = self.task.get_true_parameters(idx)

        # Permute parameter
        permuted_param = self._permute(param)
        return permuted_param.to(device)

    def get_simulator(
        self, batch_size: Optional[int] = None, device: str = "cpu"
    ) -> Simulator:
        """Return a simulator that handles permuted parameters."""
        return Simulator(
            simulator=self._permuted_simulator, device=device, batch_size=batch_size
        )

    def get_observation(self, idx: int, device: str = "cpu") -> torch.Tensor:
        """Get observation with permutation applied."""
        observation = self.task.get_observation(idx)
        return observation
