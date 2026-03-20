import inspect
import pickle
from copy import deepcopy
from typing import Callable, Optional

import torch
from sbibm.tasks.task import Task
from torch import Tensor
from torch.distributions import Distribution


def eval_function_batched_sequential(func, *args, batch_size=1000, dim=0, device="cpu"):
    # Simuators that do not support batching
    batched_x = list(
        zip(*[torch.split(x, batch_size, dim=dim) for x in args if x is not None])
    )

    ys = []
    for arg in batched_x:
        ys.append(func(*[a.to(device) for a in arg]))
    ys = torch.vstack(ys)

    return ys


class Simulator:
    """Base class for all simulaotrs"""

    def __init__(
        self, simulator: Callable, device: str = "cpu", batch_size: Optional[int] = None
    ) -> None:
        """An standard simulator

        Args:
            simulator (Callable): Simulator function
            device (str, optional): Device to simulate on. Defaults to "cpu".
            batch_size (Optional[int], optional): Batching, required if not enough memory. Defaults to None.
        """
        self.simulator = simulator
        self.device = device
        self.batch_size = batch_size

    def __call__(self, thetas: Tensor) -> Tensor:
        """Performs simulation."""
        thetas = thetas.to(self.device)
        if self.batch_size is None:
            return self.simulator(thetas)
        else:
            if self.device == "cpu":
                return eval_function_batched_sequential(
                    self.simulator,
                    thetas,
                    batch_size=self.batch_size,
                    dim=0,
                    device=self.device,
                )
            else:
                return eval_function_batched_sequential(
                    self.simulator,
                    thetas,
                    batch_size=self.batch_size,
                    dim=0,
                    device=self.device,
                )


class InferenceTask(Task):
    """Classical inverse problems. We are interested in an unknown parameter theta given some observed data x_o"""

    def __init__(
        self,
        prior: Distribution,
        loglikelihood_fn: Optional[Callable] = None,
        simulator: Optional[Callable] = None,
    ):
        """Inferece task base class.

        Args:
            prior (Distribution): Prior distribution over the paramters.
            likelihood_fn (Optional[Callable], optional): Likelihood function. Defaults to None.
            simulator (Optional[Callable], optional): Simulator function. Defaults to None.
        """
        self.simulator = simulator
        self.loglikelihood_fn = loglikelihood_fn
        self.prior = prior

    def get_observation(self, idx: int, device: str = "cpu") -> Tensor:
        """Returns the observed data.

        Args:
            idx (int): Index of the observed data.
            device (str, optional): Device. Defaults to "cpu".

        Returns:
            Tensor: Observed data
        """
        raise NotImplementedError("Not implemented/tractable :(")

    def get_true_observation(self, idx: int, device: str = "cpu") -> Tensor:
        """Returns the true observed data.

        Args:
            idx (int): Index of the observed data.
            device (str, optional): Device. Defaults to "cpu".

        Returns:
            Tensor: Observed data
        """
        raise NotImplementedError("Not implemented/tractable :(")

    def get_reference_posterior(self, idx: int, device: str = "cpu"):
        """Returns the reference posterior distribution if available.

        Args:
            idx (int): Index of the observed data
            device (str, optional): Sets the device of the object. Defaults to "cpu".

        Raises:
            NotImplementedError: If not implemented or intractable.
        """
        raise NotImplementedError("Not implemented/tractable :(")

    def get_true_posterior(self, idx: int, device: str = "cpu"):
        """Returns the true posterior distribution if available.

        Args:
            device (str, optional): Sets the device of the object. Defaults to "cpu".

        Raises:
            NotImplementedError: If not implemented or intractable.
        """
        raise NotImplementedError("Not implemented/tractable :(")

    def get_loglikelihood_fn(self, device: str = "cpu") -> Callable:
        """Return the loglikelihood function.

        Raises:
            NotImplementedError: If intractable

        Returns:
            Callable: Function returning a distribution.
        """
        if self.loglikelihood_fn is not None:
            return self.loglikelihood_fn
        else:
            raise NotImplementedError("Pass it during initialization...")

    def get_potential_fn(self, device: str = "cpu") -> Callable:
        """Return a potential function i.e. the unormalized posterior distirbution.

        Args:
            device (str, optional): Device. Defaults to "cpu".

        Returns:
            Callable: Function that gets parameter and data and computes the log posterior potential.
        """
        likelihood = self.get_loglikelihood_fn(device=device)
        prior = self.get_prior_dist(device=device)

        def potential_fn(x, theta):
            x = x.to(device)
            theta = theta.to(device)
            likelihood_fn = likelihood(theta)
            ll = likelihood_fn.log_prob(x)
            lp = prior.log_prob(theta)
            while ll.dim() < lp.dim():
                ll = ll.unsqueeze(-1)
            while lp.dim() < ll.dim():
                lp = lp.unsqueeze(-1)
            l = ll + lp
            return l.squeeze()

        return potential_fn

    def get_simulator(
        self, batch_size: Optional[int] = None, device: str = "cpu"
    ) -> Simulator:
        """Return the simulator function

        Args:
            batch_size (Optional[int], optional): Batch size. Defaults to None.
            device (str, optional): Device. Defaults to "cpu".

        Raises:
            NotImplementedError: If not implemented

        Returns:
            Simulator: An simulator which produces data given parameters.
        """

        if self.simulator is not None:
            simulator = Simulator(self.simulator, device=device, batch_size=batch_size)
            return simulator
        elif self.loglikelihood_fn is not None:
            ll = self.get_loglikelihood_fn(device=device)

            def likelihood_based_sim(theta):
                return ll(theta).sample()  # type: ignore

            simulator = Simulator(
                likelihood_based_sim, device=device, batch_size=batch_size
            )
            return simulator
        else:
            raise NotImplementedError("Either specifiy a potential_fn or a simulator")

    def get_prior(self, device: str = "cpu") -> Distribution:
        """Returns the prior distribution as a function to match SBIBM API

        Args:
            device (str, optional): Device. Defaults to "cpu".

        Returns:
            Distributions: Prior
        """
        _prior = self.get_prior_dist(device=device)

        def prior(num_samples=1):
            return _prior.sample((num_samples,))

        return prior

    def get_prior_dist(self, device: str = "cpu") -> Distribution:
        """Returns the prior distribution.

        Args:
            device (str, optional): Device. Defaults to "cpu".

        Returns:
            Distributions: Prior
        """
        if device == "cpu":
            return self.prior
        else:
            prior = deepcopy(self.prior)
            if hasattr(prior, "base_dist"):
                for key, val in prior.base_dist.__dict__.items():  # type: ignore
                    if isinstance(val, torch.Tensor):
                        prior.base_dist.__dict__[key] = val.to(device)  # type: ignore
            else:
                for key, val in self.prior.__dict__.items():
                    if isinstance(val, torch.Tensor):
                        prior.__dict__[key] = val.to(device)
            return prior

    @staticmethod
    def _is_picklable(obj):
        try:
            pickle.dumps(obj)
        except:
            return False
        return True

    def __getstate__(self):
        args = deepcopy(self.__dict__)
        for key, arg in args.items():
            if not InferenceTask._is_picklable(arg):
                args[key] = None

        return args

    def __setstate__(self, d):
        args = inspect.getargs(self.__init__.__code__).args[1:]
        init_args = {}
        for arg in args:
            init_args[arg] = d[arg]
        self.__init__(**init_args)
