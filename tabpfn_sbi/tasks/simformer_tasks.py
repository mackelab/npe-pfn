try:
    import jax
    import jax.numpy as jnp
    import blackjax
    import os

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    from functools import partial
except ImportError:
    pass

import torch
from torch import Tensor
import numpy as np
from tabpfn_sbi.tasks.base import InferenceTask


class PriorWrapper(torch.distributions.Distribution):
    def __init__(self, prior):
        self.prior = prior

    def sample(self, sample_shape=torch.Size(), key=None):
        if key is None:
            key = jax.random.key(torch.randint(0, 2**32 - 1, ()).item())
        numel = torch.Size(sample_shape).numel()
        keys = jax.random.split(key, numel)
        theta = jax.jit(jax.vmap(self.prior))(keys)
        return torch.from_numpy(np.array(theta))

    def log_prob(self, theta):
        return torch.zeros(theta.shape[:-1])


class SimulatorWrapper(torch.nn.Module):
    def __init__(self, simulator):
        super().__init__()
        self.simulator = simulator
        self.device = "cpu"

    def to(self, device: str = "cpu"):
        self.device = device
        return self

    def forward(self, theta, key=None):
        if key is None:
            key = jax.random.key(torch.randint(0, 2**32 - 1, ()).item())
        theta = theta.cpu().numpy()
        theta_shape = theta.shape
        theta = jnp.array(theta)
        theta = theta.reshape(-1, theta.shape[-1])
        keys = jax.random.split(key, theta.shape[0])
        x = jax.jit(jax.vmap(self.simulator))(keys, theta)
        x = np.array(x)
        x = torch.from_numpy(x)
        x = x.reshape(theta_shape[:-1] + (-1,))
        x = x.to(self.device)
        return x


class NonlinearGaussianTreeTask(InferenceTask):
    def __init__(self):
        prior = nonlinear_gaussian_tree_task.prior
        simulator = nonlinear_gaussian_tree_task.simulator
        log_potential_fn = nonlinear_gaussian_tree_task.log_potential_fn
        prior_wrapped = PriorWrapper(prior)
        mcmc = build_mcmc_sampler(prior, log_potential_fn)
        self.sampler = jax.jit(jax.vmap(mcmc, in_axes=(0, None)))

        base_path = os.path.dirname(os.path.abspath(__file__))
        from diskcache import Cache

        self.cache_params = Cache(
            os.path.join(base_path, "files/nonlinear_gaussian_tree_params")
        )
        self.cache_obs = Cache(
            os.path.join(base_path, "files/nonlinear_gaussian_tree_obs")
        )
        self.cache_posterior = Cache(
            os.path.join(base_path, "files/nonlinear_gaussian_tree_posterior")
        )

        super().__init__(prior_wrapped)
        self.simulator = SimulatorWrapper(simulator)

    def get_prior(self):
        return self.prior

    def get_simulator(self, device: str = "cpu"):
        self.simulator.to(device)
        return self.simulator

    def get_true_parameter(self, idx: int, device: str = "cpu") -> Tensor:
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

        return simulate_memorized(idx)

    def get_reference_posterior_samples(self, idx: int, device: str = "cpu"):
        @self.cache_posterior.memoize()
        def sample_memorized(idx):
            key = jax.random.key(idx)
            keys = jax.random.split(key, 10_000)
            x_o = self.get_observation(idx, device)
            x_o = x_o.cpu().numpy()
            x_o = jnp.array(x_o).squeeze(0)
            theta_samples = self.sampler(keys, x_o)
            theta_samples = np.array(theta_samples)
            theta_samples = torch.from_numpy(theta_samples)
            theta_samples = theta_samples.reshape(10_000, -1).to(device)
            return theta_samples

        return sample_memorized(idx)


class NonlinearMarkovChainTask(InferenceTask):
    def __init__(self, length=10):
        nonlinear_marcov_chain.length = length
        prior = nonlinear_marcov_chain.prior
        simulator = nonlinear_marcov_chain.simulator
        log_potential_fn = nonlinear_marcov_chain.log_potential_fn
        prior_wrapped = PriorWrapper(prior)
        mcmc = build_mcmc_sampler(prior, log_potential_fn)
        self.sampler = jax.jit(jax.vmap(mcmc, in_axes=(0, None)))
        self.length = length

        base_path = os.path.dirname(os.path.abspath(__file__))
        from diskcache import Cache

        self.cache_params = Cache(
            os.path.join(
                base_path,
                "files/nonlinear_markov_chain_params_length_{}".format(length),
            )
        )
        self.cache_obs = Cache(
            os.path.join(
                base_path, "files/nonlinear_markov_chain_obs_length_{}".format(length)
            )
        )
        self.cache_posterior = Cache(
            os.path.join(
                base_path,
                "files/nonlinear_markov_chain_posterior_length_{}".format(length),
            )
        )

        super().__init__(prior_wrapped)
        self.simulator = SimulatorWrapper(simulator)

    def get_prior(self):
        return self.prior

    def get_simulator(self, device: str = "cpu"):
        self.simulator.to(device)
        return self.simulator

    def get_true_parameter(self, idx: int, device: str = "cpu") -> Tensor:
        def sample_memorized(idx):
            torch.manual_seed(idx)
            return self.prior.sample((1,)).to(device)

        return sample_memorized(idx)

    def get_observation(self, idx: int, device: str = "cpu") -> Tensor:
        simulator = self.get_simulator(device=device)

        @self.cache_obs.memoize()
        def simulate_memorized(idx):
            torch.manual_seed(idx)
            theta_o = self.get_true_parameter(idx, device)
            theta_o = theta_o.to(device)
            return simulator(theta_o)

        return simulate_memorized(idx)

    def get_reference_posterior_samples(self, idx: int, device: str = "cpu"):
        @self.cache_posterior.memoize()
        def sample_memorized(idx):
            key = jax.random.key(idx)
            keys = jax.random.split(key, 10_000)
            x_o = self.get_observation(idx, device)
            x_o = x_o.cpu().numpy()
            x_o = jnp.array(x_o).squeeze(0)
            theta_samples = self.sampler(keys, x_o)
            theta_samples = np.array(theta_samples)
            theta_samples = torch.from_numpy(theta_samples)
            theta_samples = theta_samples.reshape(10_000, -1).to(device)
            return theta_samples

        return sample_memorized(idx)


class LongNonlinearMarkovChainTask(NonlinearMarkovChainTask):
    def __init__(self):
        super().__init__(length=50)
        prior = nonlinear_marcov_chain.prior
        log_potential_fn = nonlinear_marcov_chain.log_potential_fn
        mcmc = build_mcmc_sampler(prior, log_potential_fn, num_mcmc_steps=30_000)
        self.sampler = jax.jit(jax.vmap(mcmc, in_axes=(0, None)))


def build_mcmc_sampler(prior, log_potential_fn, num_mcmc_steps=10_000):
    @jax.jit
    def mcmc(key, x_o, init_sample=50, num_mcmc_steps=num_mcmc_steps):
        key_init, key_sir, key_mcmc = jax.random.split(key, 3)
        log_density_fn = partial(log_potential_fn, x=x_o)
        theta = jax.vmap(prior)(jax.random.split(key_init, init_sample))
        theta_logpdf = jax.vmap(log_density_fn)(theta)
        theta_init = jax.random.choice(key_sir, theta, p=jax.nn.softmax(theta_logpdf))
        kernel = blackjax.hmc(
            log_density_fn,
            num_integration_steps=10,
            step_size=0.1,
            inverse_mass_matrix=jnp.eye(theta.shape[-1]),
        )
        state = kernel.init(theta_init)

        def step(state, key):
            new_state, info = kernel.step(key, state)
            # jax.debug.print("accepted: {accepted}", accepted=info.acceptance_rate.mean())
            return new_state, None

        state, _ = jax.lax.scan(step, state, jax.random.split(key_mcmc, num_mcmc_steps))
        return state.position

    return mcmc


class nonlinear_gaussian_tree_task:
    """
    Nonlinear Gaussian Tree Task.

    This function defines a probabilistic model that represents a nonlinear Gaussian tree task.
    It generates random variables theta1, theta2, theta3, x1, x2, x3, and x4, and returns their names,
    a joint sampler, and a potential function.

    Returns:
        var_names (list): List of variable names.
        joint_sampler (callable): Joint sampler function.
        potential_fn (callable): Potential function.
    """

    @staticmethod
    def prior(key):
        key1, key2, key3 = jax.random.split(key, 3)
        theta1 = jax.random.normal(key1, (1,))
        theta2 = jax.random.normal(key2, (1,)) + theta1
        theta3 = jax.random.normal(key3, (1,)) + theta1
        return jnp.concatenate([theta1, theta2, theta3])

    @staticmethod
    def simulator(key, theta):
        theta1, theta2, theta3 = theta
        key1, key2, key3, key4 = jax.random.split(key, 4)
        z1 = jnp.sin(theta2) ** 2
        z2 = 0.1 * theta2**2
        z3 = 0.1 * theta3**2
        z4 = jnp.cos(theta3) ** 2

        x1 = z1 + jax.random.normal(key1, (1,)) * 0.2
        x2 = z2 + jax.random.normal(key2, (1,)) * 0.2
        x3 = z3 + jax.random.normal(key3, (1,)) * 0.6
        x4 = z4 + jax.random.normal(key4, (1,)) * 0.1
        return jnp.concatenate([x1, x2, x3, x4])

    @staticmethod
    def log_potential_fn(theta, x):
        theta1, theta2, theta3 = theta
        logpdf1 = jax.scipy.stats.norm.logpdf(theta1, 0, 1)
        logpdf2 = jax.scipy.stats.norm.logpdf(theta2, theta1, 1)
        logpdf3 = jax.scipy.stats.norm.logpdf(theta3, theta1, 1)
        logpdf4 = jax.scipy.stats.norm.logpdf(x[0], jnp.sin(theta2) ** 2, 0.2)
        logpdf5 = jax.scipy.stats.norm.logpdf(x[1], 0.1 * theta2**2, 0.2)
        logpdf6 = jax.scipy.stats.norm.logpdf(x[2], 0.1 * theta3**2, 0.6)
        logpdf7 = jax.scipy.stats.norm.logpdf(x[3], jnp.cos(theta3) ** 2, 0.1)
        return jnp.sum(
            logpdf1 + logpdf2 + logpdf3 + logpdf4 + logpdf5 + logpdf6 + logpdf7
        )


class nonlinear_marcov_chain:
    """
    Nonlinear Markov Chain Task.

    This function defines a probabilistic model that represents a nonlinear Markov chain task.
    It generates a chain of latent variables theta0, theta1, ..., theta9 and observations x0, x1, ..., x9,
    where each theta_i depends on the previous theta, and each x_i is a nonlinear function of theta_i.
    Returns prior, simulator, and log_potential_fn, similar to nonlinear_gaussian_tree_task.

    Returns:
        prior (callable): Prior sampler function.
        simulator (callable): Simulator function.
        log_potential_fn (callable): Log potential function.
    """

    length = 10

    @classmethod
    def prior(cls, key):
        keys = jax.random.split(key, cls.length)
        theta = []
        theta0 = jax.random.normal(keys[0], (1,))
        theta.append(theta0)  # Do not squeeze
        for i in range(1, cls.length):
            theta_i = jax.random.normal(keys[i], (1,)) + theta[i - 1]
            theta.append(theta_i)
        return jnp.concatenate(theta, axis=0)

    @classmethod
    def simulator(cls, key, theta):
        # theta: shape (10,)
        keys = jax.random.split(key, cls.length)
        x = []
        for i in range(cls.length):
            xi = jnp.sin(theta[i]) + jax.random.normal(keys[i], (1,)) * 0.5
            x.append(xi)
        return jnp.concatenate(x, axis=0)

    @classmethod
    def log_potential_fn(cls, theta, x):
        # theta: shape (10,)
        # x: shape (10,)
        logpdfs = []
        # prior for theta0
        logpdfs.append(jax.scipy.stats.norm.logpdf(theta[0], 0, 1))
        # transition priors for theta1..theta9
        for i in range(1, cls.length):
            logpdfs.append(jax.scipy.stats.norm.logpdf(theta[i], theta[i - 1], 1))
        # likelihoods for x0..x9
        for i in range(cls.length):
            logpdfs.append(jax.scipy.stats.norm.logpdf(x[i], jnp.sin(theta[i]), 0.5))
        return jnp.sum(jnp.stack(logpdfs))
