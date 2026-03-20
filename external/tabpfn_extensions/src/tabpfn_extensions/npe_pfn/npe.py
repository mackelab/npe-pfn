from __future__ import annotations

import random
from pathlib import Path
from typing import Optional, Self, Sequence, Union, Literal

import numpy as np
import torch
from sklearn.base import check_is_fitted

from tabpfn.base import initialize_tabpfn_model
from tabpfn.model.bar_distribution import FullSupportBarDistribution
from tabpfn.utils import infer_device_and_type, infer_random_state

from torch import Tensor


class TabPFNNPE:
    """TabPFN model for neural posterior estimation.

    This model combines allows to approximate multivariate condition density p(y|x).
    Given that y is a vector of parameters, the model will estimate the multivariate
    posterior distribution of y given x.

    This class allows to fit a TabPFN model to the input data and estimate the posterior
    distribution of the target.


    Parameters:


    Examples:
    ```python title="Example"

    ```
    """

    _y_train_mean: Tensor | None
    _y_train_std: Tensor | None
    _x_train_mean: Tensor | None
    _x_train_std: Tensor | None
    _x_train: Tensor | None
    _y_train: Tensor | None

    def __init__(
        self,
        *,
        model_path: str | Path | Literal["auto"] = "auto",
        device: str | torch.device | Literal["auto"] = "auto",
        fit_mode: Literal[
            "low_memory",
            "fit_preprocessors",
            "fit_with_cache",
        ] = "fit_preprocessors",
        random_state: int | np.random.RandomState | np.random.Generator | None = 0,
    ) -> None:
        super().__init__()
        self.model_path = model_path
        self.fit_mode = fit_mode
        self.device = device
        self.random_state = random_state
        self._model = None
        self._config = None
        self._bardist = None
        self._y_train_mean = None
        self._y_train_std = None
        self._x_train_mean = None
        self._x_train_std = None
        self._x_train = None
        self._y_train = None

    def fit(self, x: Tensor, y: Tensor) -> Self:
        """Fit the model.

        Args:
            x: Batch of inputs. B x D1
            y: Batch of outputs. B x D2

        Returns:
            self
        """
        static_seed, rng = infer_random_state(self.random_state)

        # Load the model and config
        # TODO Make this bardist bounded if needed
        self._model, self._config, self._bardist = initialize_tabpfn_model(
            model_path=self.model_path,
            which="regressor",
            fit_mode=self.fit_mode,
            static_seed=static_seed,
        )

        # Determine device and precision
        self.device = infer_device_and_type(self.device)

        # Standard check for inputs
        assert x.shape[0] == y.shape[0], "The number of samples in x and y must match."
        assert x.ndim == 2, "x must be a 2D tensor."

        # TODO Probably one should also add a few preprocessors for X
        self._x_train_mean = torch.mean(x, axis=0, keepdims=True).to(self.device)
        self._x_train_std = torch.std(x, axis=0, keepdims=True).to(self.device) + 1e-10

        self._x_train = (x.to(self.device) - self._x_train_mean) / self._x_train_std

        # Standardize thetas
        self._y_train_mean = torch.mean(y, axis=0, keepdims=True).to(self.device)
        self._y_train_std = torch.std(y, axis=0, keepdims=True).to(self.device) + 1e-10

        self._y_train = (y.to(self.device) - self._y_train_mean) / self._y_train_std

        self._model.to(self.device)
        self._bardist.to(self.device)
        self._model.eval()

        return self

    def sample(self, x: Tensor) -> Tensor:
        """Sample from the posterior distribution.

        Args:
            xs: The input data. B x D

        Returns:
            torch.Tensor: The samples from the posterior distribution.
        """
        # TODO: Maybe make this like sbi i.e. .sample will only sample from on posterior
        # multiple times (will repeat it internally here)
        # TODO: For sampling from a batch of xs, then sample_batched should be used.
        # Standard input checks
        x = x.unsqueeze(0) if x.ndim == 1 else x
        assert x.ndim == 2, "x must be a 2D tensor."
        assert x.shape[1] == self._x_train.shape[1], (
            "The number of features in x must match the training data."
        )

        # Standardize the input
        x = x.to(self.device)
        x = (x - self._x_train_mean) / self._x_train_std

        ys = []

        n, d1 = self._x_train.shape
        x_train = self._x_train.to(self.device).reshape(n, d1, 1)
        y_train = self._y_train.to(self.device)
        x = x.to(self.device).reshape(x.shape[0], d1, 1)

        # Sample the posterior distribution autoregressively
        for i in range(self._y_train.shape[1]):
            with torch.no_grad():
                y_i_train = y_train[:, i]
                logits = self._model(train_x=x_train, train_y=y_i_train, test_x=x)
                y_i = self._bardist.sample(logits[:, 0, :]).to(self.device)
                ys.append(y_i.unsqueeze(-1))
                x_train = torch.cat(
                    [x_train, y_i_train.unsqueeze(-1).unsqueeze(-1)], dim=-2
                )
                x = torch.cat([x, y_i.unsqueeze(-1).unsqueeze(-1)], dim=-2)

        ys = torch.cat(ys, dim=1)
        ys = ys.to(self.device)
        ys = ys * self._y_train_std + self._y_train_mean
        return ys

    def log_prob(self, x: Tensor, y: Tensor) -> Tensor:
        """Compute the log probabilities of the y given x.

        Args:
            x: Batch of inputs. B x D1
            y: Batch of outputs. B x D2

        Returns:
            Tensor: The log probabilities of the y given x.
        """
        # Standard input checks
        # TODO: Relax this analogously to sample, sample_batched
        assert x.ndim == 2, "x must be a 2D tensor."
        assert x.shape[1] == self._x_train.shape[1], (
            "The number of features in x must match the training data."
        )
        assert y.ndim == 2, "y must be a 2D tensor."
        assert y.shape[1] == self._y_train.shape[1], (
            "The number of features in y must match the training data."
        )
        assert x.shape[0] == y.shape[0], "The number of samples in x and y must match."

        # Standardize the input
        x = x.to(self.device)
        y = y.to(self.device)
        x = (x - self._x_train_mean) / self._x_train_std
        y = (y - self._y_train_mean) / self._y_train_std
        # the BarDist

        log_probs = []

        n, d1 = self._x_train.shape
        x_train = self._x_train.to(self.device).reshape(n, d1, 1)
        y_train = self._y_train.to(self.device)
        x = x.to(self.device).reshape(x.shape[0], d1, 1)

        # Sample the posterior distribution autoregressively
        for i in range(self._y_train.shape[1]):
            with torch.no_grad():
                y_i_train = y_train[:, i]
                logits = self._model(train_x=x_train, train_y=y_i_train, test_x=x)
                y_i = y[:, i]
                log_prob_i = -self._bardist(logits[:, 0, :], y_i).to(self.device)
                x_train = torch.cat(
                    [x_train, y_i_train.unsqueeze(-1).unsqueeze(-1)], dim=-2
                )
                x = torch.cat([x, y_i.unsqueeze(-1).unsqueeze(-1)], dim=-2)
                log_probs.append(log_prob_i.unsqueeze(-1))

        log_probs = torch.cat(log_probs, dim=-1)
        log_probs = log_probs.to(self.device) + torch.log(self._y_train_std)

        return log_probs.sum(dim=-1)
