import os
from typing import Callable, Tuple

import torch
from pyro.distributions import ConditionalDistribution
from torch import Tensor, nn
from torchdiffeq import odeint

from .base import InferenceTask
from .utils.distributions import MCMCDistribution
from .utils.mcmc import MCMC
from .utils.mcmc_kernels import (
    AdaptiveMultivariateGaussianKernel,
    KernelScheduler,
    LearnableIndependentKernel,
)


class LotkaVolterraODE(nn.Module):
    def __init__(self, x0: float = 1.0, x1: float = 1.0) -> None:
        """Implements a Lotka Volterra ODE

        Args:
            x0 (float, optional): Initial population of prey. Defaults to 1.0.
            x1 (float, optional): Initial population of predator. Defaults to 1.0.
            clamp_max_params (float, optional): Max clamp of parameters for numerical stability. Defaults to 10.0.
        """
        super().__init__()
        self.batch_size = 1
        self.register_buffer("alpha", torch.ones(1))
        self.register_buffer("beta", torch.ones(1))
        self.register_buffer("gamma", torch.ones(1))
        self.register_buffer("delta", torch.ones(1))
        self.register_buffer("x0", torch.as_tensor(x0))
        self.register_buffer("x1", torch.as_tensor(x1))

    @torch.jit.export  # type: ignore
    def set_theta(self, theta: Tensor) -> None:
        # This preserves stability of the ode.
        theta = theta.reshape(-1, 4)
        params = theta.split(1, dim=-1)
        self.batch_size = theta.shape[0]
        self.alpha = params[0]
        self.beta = params[1]
        self.gamma = params[2]
        self.delta = params[3]

    @torch.jit.export  # type: ignore
    def get_initial_state(self) -> Tuple[Tensor, Tensor]:
        """Returns the initial state"""
        return self.x0.clone().repeat(self.batch_size, 1), self.x1.clone().repeat(  # type: ignore
            self.batch_size, 1
        )

    def forward(self, t: float, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        prey, predator = state
        dprey = self.alpha * prey - self.beta * prey * predator
        dpredator = self.delta * prey * predator - self.gamma * predator
        return dprey, dpredator


class LotkaVolterraPosterior(ConditionalDistribution):
    def __init__(self, prior, potential_fn) -> None:
        super().__init__()
        self.prior = prior
        self.potential_fn = potential_fn

        k1 = AdaptiveMultivariateGaussianKernel()
        k2 = LearnableIndependentKernel()

        self.k = KernelScheduler(
            [k2, k1, k2, k1, k2, k1, k2], [50, 100, 150, 200, 225, 250, 300]
        )

    def condition(self, context: Tensor):
        shape = context.shape
        # proposal = SIRDistribution(self.prior, self.potential_fn, context=context, K=10)
        mcmc = MCMC(
            self.k,
            self.potential_fn,
            self.prior,
            context=context,
            thinning=5,
            warmup_steps=300,
            num_chains=500,
            device=context.device,
        )

        return MCMCDistribution(mcmc)


class LotkaVolterraTask(InferenceTask):
    def __init__(
        self,
        t_max: float = 40.0,
        time_points_observed: int = 150,
        observation_noise: float = 0.05,
        prior_scale: float = 0.5,
        prior_mean: float = 0.0,
        flatten: bool = False,
    ) -> None:
        """Lotka volterra inference task

        Args:
            t_max (int, optional): Time to simulate. Defaults to 20.
            time_points_observed (int, optional): Timepoints observed for inference. Defaults to 50.
            observation_noise (float, optional): Observation noise. Defaults to 0.05.
            initial_noise (float, optional): Initial condition noise. Defaults to 0.0.
            prior_scale (float, optional): Prior scale. Defaults to 0.2.
            prior_mean (float, optional): Prior mean. Defaults to 0.0.
            odeint_kwargs (dict, optional): Kwargs for odesolver. Defaults to {}.
        """
        # This might cause problem with SLURM
        base_path = os.path.dirname(os.path.abspath(__file__))
        from diskcache import Cache

        self.cache_params = Cache(os.path.join(base_path, "files/lv_params"))
        self.cache_obs = Cache(os.path.join(base_path, "files/lv_obs"))
        self.cache_posterior = Cache(os.path.join(base_path, "files/lv_posterior"))

        prior = torch.distributions.Independent(
            torch.distributions.Normal(
                torch.ones(4) * prior_mean, torch.ones(4) * prior_scale
            ),
            1,
        )

        self.flatten = flatten
        self.prior_scale = prior_scale
        self.prior_mean = prior_mean
        self.ode = torch.jit.script(LotkaVolterraODE())  # type: ignore
        self.t_max = t_max
        self.time_points_observed = time_points_observed

        def likelihood_fn(theta):
            batch_shape = theta.shape[:-1]
            theta = theta.sigmoid()
            obs_noise = observation_noise * torch.ones(1, device=theta.device)
            t = torch.linspace(
                0, self.t_max, self.time_points_observed, device=theta.device
            )
            self.ode.set_theta(theta.exp())
            try:  # type: ignore
                sol = odeint(
                    self.ode,
                    self.ode.get_initial_state(),
                    t,
                    atol=1e-6,
                    rtol=1e-6,  # type: ignore
                )
            except:
                sol = odeint(
                    self.ode,
                    self.ode.get_initial_state(),
                    t,
                    method="euler",
                    options={"step_size": 1e-3},  # type: ignore
                )
            if self.flatten:
                x = torch.vstack(sol).squeeze(-1).transpose(0, 1)  # type: ignore
                mask = torch.isfinite(x)
                x[~mask] = 0.0
                x = x.reshape(*batch_shape, -1)
                return torch.distributions.Independent(
                    torch.distributions.Normal(x, obs_noise), 1
                )
            else:
                # Will be of shape (batch_size, time_points_observed, 2)
                x = (
                    torch.stack(sol, axis=-1)
                    .squeeze(-2)
                    .transpose(0, 1)
                    .transpose(1, 2)
                )  # type: ignore
                mask = torch.isfinite(x)
                x[~mask] = 0.0
                return torch.distributions.Independent(
                    torch.distributions.Normal(x, obs_noise), 2
                )

        super().__init__(prior, likelihood_fn, None)

        self.input_dim = time_points_observed * 2
        self.output_dim = 4
        self.event_shape = torch.Size((time_points_observed, 2))

    def get_loglikelihood_fn(self, device: str = "cpu") -> Callable:
        self.ode.to(device)  # type: ignore
        return super().get_loglikelihood_fn(device)

    def get_true_posterior(self, device: str = "cpu"):
        return LotkaVolterraPosterior(
            self.get_prior_dist(device), self.get_potential_fn(device)
        )

    def get_true_parameter(self, idx: int, device: str = "cpu") -> Tensor:
        @self.cache_params.memoize()
        def sample_memorized(idx):
            torch.manual_seed(idx)
            return self.prior.sample((1,)).to(device)

        return sample_memorized(idx)

    def get_observation(self, idx: int, device: str = "cpu") -> Tensor:
        """Returns the true observed data.

        Args:
            idx (int): Index of the observed data.
            device (str, optional): Device. Defaults to "cpu".

        Returns:
            Tensor: Observed data
        """

        simulator = self.get_simulator(device=device)

        @self.cache_obs.memoize()
        def simulate_memorized(idx):
            torch.manual_seed(idx)
            theta_o = self.get_true_parameter(idx, device)
            theta_o = theta_o.to(device)
            return simulator(theta_o)

        x = simulate_memorized(idx)
        if self.flatten:
            x = x.reshape(1, -1)
        else:
            x = x.reshape(1, 2, -1)
        return x

    def get_reference_posterior_samples(self, idx: int, device: str = "cpu"):
        posterior = self.get_true_posterior(device)
        x_o = self.get_observation(idx, device)  # .squeeze(0)
        q = posterior.condition(x_o)

        @self.cache_posterior.memoize()
        def sample_memorized(idx):
            torch.manual_seed(idx)
            return q.sample((10_000,)).squeeze(1)

        return sample_memorized(idx)
