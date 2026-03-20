#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

from __future__ import annotations

import copy
import numpy as np
import random
import torch
from sklearn.base import BaseEstimator
from tqdm import tqdm
from typing import Optional

from .. import utils_todo
from tabpfn_extensions import TabPFNRegressor, TabPFNClassifier
from tabpfn_extensions.utils import USE_TABPFN_LOCAL


class TabPFNUnsupervisedModel(BaseEstimator):
    """TabPFN experiments model for imputation, outlier detection, and synthetic data generation.

    This model combines a TabPFNClassifier for categorical features and a TabPFNRegressor for
    numerical features to perform various experiments learning tasks on tabular data.

    Parameters:
        tabpfn_clf : TabPFNClassifier, optional
            TabPFNClassifier instance for handling categorical features. If not provided, the model
            assumes that there are no categorical features in the data.

        tabpfn_reg : TabPFNRegressor, optional
            TabPFNRegressor instance for handling numerical features. If not provided, the model
            assumes that there are no numerical features in the data.

    Attributes:
        categorical_features : list
            List of indices of categorical features in the input data.

    Examples:
    ```python title="Example"
    >>> tabpfn_clf = TabPFNClassifier()
    >>> tabpfn_reg = TabPFNRegressor()
    >>> model = TabPFNUnsupervisedModel(tabpfn_clf, tabpfn_reg)
    >>>
    >>> X = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    >>> model.fit(X)
    >>>
    >>> X_imputed = model.impute(X)
    >>> X_outliers = model.outliers(X)
    >>> X_synthetic = model.generate_synthetic_data(n_samples=100)
    ```
    """

    def _more_tags(self):
        return {"allow_nan": True}

    def __init__(
        self,
        tabpfn_clf: Optional[TabPFNClassifier] = None,
        tabpfn_reg: Optional[TabPFNRegressor] = None,
    ) -> None:
        """Initialize the TabPFNUnsupervisedModel.

        Args:
            tabpfn_clf : TabPFNClassifier, optional
                TabPFNClassifier instance for handling categorical features. If not provided, the model
                assumes that there are no categorical features in the data.

            tabpfn_reg : TabPFNRegressor, optional
                TabPFNRegressor instance for handling numerical features. If not provided, the model
                assumes that there are no numerical features in the data.

        Raises:
            AssertionError
                If both tabpfn_clf and tabpfn_reg are None.
        """
        assert (
            tabpfn_clf is not None or tabpfn_reg is not None
        ), "You cannot set both `tabpfn_clf` and `tabpfn_reg` to None. You can set one to None, if your table exclusively consists of categoricals/numericals."

        self.tabpfn_clf = tabpfn_clf
        self.tabpfn_reg = tabpfn_reg
        self.estimators = [self.tabpfn_clf, self.tabpfn_reg]

        self.categorical_features = []

    def set_categorical_features(self, categorical_features):
        self.categorical_features = categorical_features
        for estimator in self.estimators:
            try:
                estimator.set_categorical_features(categorical_features)
            except:
                pass

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """Fit the model to the input data.

        Args:
            X : array-like of shape (n_samples, n_features)
                Input data to fit the model.

            y : array-like of shape (n_samples,), optional
                Target values.

        Returns:
            self : TabPFNUnsupervisedModel
                Fitted model.
        """
        self.X_ = copy.deepcopy(X)
        self.y = copy.deepcopy(y)
        self.categorical_features = utils_todo.infer_categorical_features(
            X, self.categorical_features
        )

    def init_model_and_get_model_config(self):
        for estimator in self.estimators:
            estimator.init_model_and_get_model_config()

    def impute_(
        self,
        X: torch.tensor,
        t: float = 0.000000001,
        n_permutations: int = 10,
        condition_on_all_features: bool = True,
    ) -> torch.tensor:
        """
        Impute missing values (np.nan) in X by sampling all cells independently from the trained models

        :param X: Input data of the shape (num_examples, num_features) with missing values encoded as np.nan
        :param t: Temperature for sampling from the imputation distribution, lower values are more deterministic
        :return: Imputed data, with missing values replaced
        """

        n_features = X.shape[1]
        all_features = list(range(n_features))

        X_fit = self.X_
        impute_X = copy.deepcopy(X)

        for i in tqdm(range(len(all_features))):
            column_idx = all_features[i]

            if not condition_on_all_features:
                conditional_idx = all_features[:i] if i > 0 else []
            else:
                conditional_idx = list(set(range(X.shape[1])) - {column_idx})

            y_predict = impute_X[:, column_idx]

            if torch.isnan(y_predict).sum() == 0:
                continue

            X_where_y_is_nan = impute_X[torch.isnan(y_predict)]
            X_where_y_is_nan = X_where_y_is_nan.reshape(-1, impute_X.shape[1])

            densities = []
            for perm in efficient_random_permutation(conditional_idx, n_permutations):
                perm = perm + (column_idx,)
                _, pred = self.impute_single_permutation_(
                    X_where_y_is_nan, perm, t, condition_on_all_features
                )
                densities.append(pred)

            if not self.use_classifier_(column_idx, X_fit[:, column_idx]):
                pred_merged = densities[0][
                    "criterion"
                ].average_bar_distributions_into_this(
                    [d["criterion"] for d in densities],
                    [torch.tensor(d["logits"]) for d in densities],
                )
                pred_sampled = densities[0]["criterion"].sample(pred_merged, t=t)
            else:
                pred = torch.stack([d for d in densities]).mean(dim=0)
                pred_sampled = (
                    torch.distributions.Categorical(probs=pred).sample().float()
                )

            impute_X[torch.isnan(y_predict), column_idx] = pred_sampled

        return impute_X

    def impute_single_permutation_(
        self,
        X: torch.tensor,
        feature_permutation: list[int] | tuple[int],
        t: float = 0.000000001,
        condition_on_all_features: bool = True,
    ) -> torch.tensor:
        """
        Impute missing values (np.nan) in X by sampling all cells independently from the trained models

        :param X: Input data of the shape (num_examples, num_features) with missing values encoded as np.nan
        :param t: Temperature for sampling from the imputation distribution, lower values are more deterministic
        :return: Imputed data, with missing values replaced
        """
        X_fit = self.X_
        impute_X = copy.deepcopy(X)

        for i in range(len(feature_permutation)):
            column_idx = feature_permutation[i]

            if not condition_on_all_features:
                conditional_idx = feature_permutation[:i] if i > 0 else []
            else:
                conditional_idx = list(set(range(X.shape[1])) - {column_idx})

            y_predict = impute_X[:, column_idx]

            if torch.isnan(y_predict).sum() == 0:
                continue

            X_where_y_is_nan = impute_X[torch.isnan(y_predict)]
            X_where_y_is_nan = X_where_y_is_nan.reshape(-1, impute_X.shape[1])

            model, X_predict, _ = self.density_(
                X_where_y_is_nan, X_fit, conditional_idx, column_idx
            )

            pred, pred_sampled = self.sample_from_model_prediction_(
                column_idx, X_fit, model, X_predict, t
            )

            impute_X[torch.isnan(y_predict), column_idx] = pred_sampled

        return impute_X, pred

    def sample_from_model_prediction_(self, column_idx, X_fit, model, X_predict, t):
        if not self.use_classifier_(column_idx, X_fit[:, column_idx]):
            pred = model.predict(X_predict.numpy(), output_type="full")
            pred_sampled = pred["criterion"].sample(torch.tensor(pred["logits"]), t=t)
        else:
            pred = model.predict_proba(X_predict.numpy())
            pred_sampled = (
                torch.distributions.Categorical(probs=torch.tensor(pred))
                .sample()
                .float()
            )

        return pred, pred_sampled

    def use_classifier_(self, column_idx, y):
        return (
            column_idx in self.categorical_features
            and len(np.unique(y)) < 10  # TODO: self.tabpfn_clf.max_num_classes_
        )

    def density_(
        self,
        X_predict: torch.tensor,
        X_fit: torch.tensor,
        conditional_idx: list[int],
        column_idx: int,
    ) -> torch.tensor:
        # self.init_model_and_get_model_config()

        if len(conditional_idx) > 0:
            # If not the first feature, use all previous features
            mask = torch.zeros_like(X_fit).bool()
            mask[:, conditional_idx] = True
            X_fit, y_fit = X_fit[mask], X_fit[:, column_idx]
            X_fit = X_fit.reshape(mask.shape[0], -1)

            mask = torch.zeros_like(X_predict).bool()
            mask[:, conditional_idx] = True
            X_predict, y_predict = X_predict[mask], X_predict[:, column_idx]
            X_predict = X_predict.reshape(mask.shape[0], -1)
        else:
            # If the first feature, use a zero feature as input
            # Because of preprocessing, we can't use a zero feature, so we use a random feature
            X_fit, y_fit = torch.randn_like(X_fit[:, 0:1]), X_fit[:, 0]
            X_predict, y_predict = torch.randn_like(X_predict[:, 0:1]), X_predict[:, 0]

        model = (
            self.tabpfn_clf
            if self.use_classifier_(column_idx, y_fit)
            else self.tabpfn_reg
        )

        model.fit(X_fit.numpy(), y_fit.numpy())

        return model, X_predict, y_predict

    def impute(
        self, X: torch.tensor, t: float = 0.000000001, n_permutations: int = 10
    ) -> torch.tensor:
        """Impute missing values in the input data.

        Args:
            X : torch.Tensor of shape (n_samples, n_features)
                Input data with missing values encoded as np.nan.

            t : float, default=0.000000001
                Temperature for sampling from the imputation distribution. Lower values result in
                more deterministic imputations.

        Returns:
            torch.Tensor of shape (n_samples, n_features)
                Imputed data with missing values replaced.
        """
        return self.impute_(
            X, t, condition_on_all_features=True, n_permutations=n_permutations
        )

    def outliers_single_permutation_(
        self, X: torch.tensor, feature_permutation: list[int] | tuple[int]
    ) -> torch.tensor:
        log_p = torch.zeros_like(
            X[:, 0]
        )  # Start with a log probability of 0 (log(1) = 0)

        for i, column_idx in enumerate(feature_permutation):
            model, X_predict, y_predict = self.density_(
                X, self.X_, feature_permutation[:i], column_idx
            )
            if self.use_classifier_(column_idx, y_predict):
                pred = model.predict_proba(X_predict.numpy())[y_predict]
            else:
                pred = model.predict(X_predict.numpy(), output_type="full")
                pred = pred["criterion"].pdf(pred["logits"], torch.tensor(y_predict))

            # Convert probabilities to log probabilities
            log_pred = torch.log(pred)

            # Add log probabilities instead of multiplying probabilities
            log_p = log_p + log_pred

        return log_p, torch.exp(log_p)

    def outliers_pdf(self, X: torch.tensor, n_permutations: int = 10) -> torch.tensor:
        X_store = copy.deepcopy(self.X_)
        mask = torch.ones_like(X_store).bool()
        mask[self.categorical_features] = False
        self.X_ = self.X_[mask]
        mask = torch.ones_like(X).bool()
        mask[self.categorical_features] = False
        X = X[mask]

        pdf = self.outliers(X, n_permutations=n_permutations)
        self.X_ = X_store
        return pdf

    def outliers_pmf(self, X: torch.tensor, n_permutations: int = 10) -> torch.tensor:
        X_store = copy.deepcopy(self.X_)
        mask = torch.zeros_like(X_store).bool()
        mask[self.categorical_features] = True
        self.X_ = self.X_[mask]
        mask = torch.zeros_like(X).bool()
        mask[self.categorical_features] = True
        X = X[mask]

        pmf = self.outliers(X, n_permutations=n_permutations)
        self.X_ = X_store
        return pmf

    def outliers(self, X: torch.tensor, n_permutations: int = 10) -> torch.tensor:
        """
        Preferred implementation for outliers, where we calculate the sample probability for each sample in X by
        multiplying the probabilities of each feature according to chain rule of probability. The first feature is
        estimated by using a zero feature as input.

        Args
            X: Samples to calculate the sample probability for, shape (n_samples, n_features)

        Returns:
            Sample unnormalized probability for each sample in X, shape (n_samples,)
        """
        # self.init_model_and_get_model_config()

        n_features = X.shape[1]
        all_features = list(range(n_features))

        densities = []
        for perm in efficient_random_permutation(all_features, n_permutations):
            perm_density_log, perm_density = self.outliers_single_permutation_(
                X, feature_permutation=perm
            )
            densities.append(perm_density)

        # Average the densities across all permutations
        return torch.stack(densities).mean(dim=0)

    def generate_synthetic_data(self, n_samples=100, t=1.0, n_permutations=3):
        """
        Generate synthetic data using the trained models. Uses imputation method to generate synthetic data, passed with
        a matrix of nans. Samples are generated feature by feature in one pass, so samples are not dependent on each
        other per feature.

        Args:
            n_samples : int, default=100
                Number of synthetic samples to generate.

            t : float, default=1.0
                Temperature for sampling from the imputation distribution. Lower values result in
                more deterministic samples.

        Returns:
            torch.Tensor of shape (n_samples, n_features)
                Generated synthetic data.

        Raises:
            AssertionError
                If the model is not fitted.
        """
        # TODO: Test what happens if we generate one feature at a time, with train data only for that featur
        #  and previous ones, like outliers
        assert hasattr(
            self, "X_"
        ), "You need to fit the model before generating synthetic data"

        X = torch.zeros(n_samples, self.X_.shape[1]) * np.nan
        return self.impute_(
            X, t=t, condition_on_all_features=False, n_permutations=n_permutations
        )

    def get_embeddings_(self, X: torch.tensor) -> torch.tensor:
        model = self.tabpfn_reg
        model.fit(
            self.X_,
            self.y
            if self.y is not None
            else (torch.zeros_like(self.X_[:, 0])),  # Must contain more than one class
        )  # Fit the data for random labels
        embs = model.get_embeddings(X, additional_y=None)
        return embs.reshape(X.shape[0], -1)

    def get_embeddings(self, X: torch.tensor, per_column: bool = False) -> torch.tensor:
        """
        Get the transformer embeddings for the test data X.

        Args:
            X:

        Returns:
            torch.Tensor of shape (n_samples, embedding_dim)
        """
        if per_column:
            return self.get_embeddings_per_column(X)
        else:
            return self.get_embeddings_(X)

    def get_embeddings_per_column(self, X: torch.tensor) -> torch.tensor:
        """
        Alternative implementation for get_embeddings, where we get the embeddings for each column as a label
         separately and concatenate the results. This alternative way needs more passes but might be more accurate
        """
        embs = []
        for column_idx in range(0, X.shape[1]):
            mask = torch.zeros_like(self.X_).bool()
            mask[:, column_idx] = True
            X_train, y_train = (
                self.X_[~(mask)].reshape(self.X_.shape[0], -1),
                self.X_[mask],
            )

            X_pred, y_pred = X[~(mask)].reshape(X.shape[0], -1), X[mask]

            model = (
                self.tabpfn_clf
                if column_idx in self.categorical_features
                else self.tabpfn_reg
            )
            model.fit(X_train, y_train)
            embs += [model.get_embeddings(X_pred, additional_y=None)]

        return torch.cat(embs, 1).reshape(embs[0].shape[0], -1)


def efficient_random_permutation(indices, n_permutations=10):
    perms = []
    n_iter = 0
    while len(perms) < n_permutations and n_iter < n_permutations * 10:
        perm = efficient_random_permutation_(indices)
        if perm not in perms:
            perms += [perm]
        n_iter += 1

    return perms


def efficient_random_permutation_(indices):
    """
    Generate a single random permutation from a very large space.

    :param n: The size of the permutation (number of elements)
    :return: A list representing a random permutation of numbers from 0 to n-1
    """
    # Create a list of numbers from 0 to n-1
    permutation = indices

    # Shuffle the list in-place
    for i in range(len(indices) - 1, 0, -1):
        # Pick a random index from 0 to i
        j = random.randint(0, i)
        # Swap elements at i and j
        permutation[i], permutation[j] = permutation[j], permutation[i]

    return tuple(permutation)
