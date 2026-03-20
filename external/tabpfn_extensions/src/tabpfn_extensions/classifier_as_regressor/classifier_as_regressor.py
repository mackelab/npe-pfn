#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import numpy as np
import torch
from sklearn.base import RegressorMixin
from sklearn.preprocessing import LabelEncoder

from tabpfn_extensions import TabPFNRegressor
from tabpfn.model.bar_distribution import BarDistribution


class ClassifierAsRegressor(RegressorMixin):
    """Wrapper class to use a classifier as a regressor.

    This class takes a classifier estimator and converts it into a regressor by
    encoding the target labels and treating the regression problem as a
    classification task.

    Parameters:
        estimator : object
            Classifier estimator to be used as a regressor.

    Attributes:
        label_encoder_ : LabelEncoder
            Label encoder used to transform target regression labels to classes.
        y_train_ : array-like of shape (n_samples,)
            Transformed target labels used for training.
        categorical_features : list
            List of categorical feature indices.

    Examples:
    ```python title="Example"
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.model_selection import train_test_split
    >>> from tabpfn_extensions import ManyClassClassifier, TabPFNClassifier, ClassifierAsRegressor
    >>> x, y = load_diabetes(return_X_y=True)
    >>> x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    >>> clf = TabPFNClassifier()
    >>> clf = ManyClassClassifier(clf, n_estimators=10, alphabet_size=clf.max_num_classes_)
    >>> reg = ClassifierAsRegressor(clf)
    >>> reg.fit(x_train, y_train)
    >>> y_pred = reg.predict(x_test)
    ```
    """

    def __init__(self, estimator):
        assert hasattr(
            estimator, "predict_proba"
        ), "The estimator must have a predict_proba method to be used as a regressor."
        self.estimator = estimator

    def fit(self, X, y):
        """Fit the classifier as a regressor.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Training data.
            y : array-like of shape (n_samples,)
                Target labels.

        Returns:
            self : object
                Fitted estimator.
        """
        self.label_encoder_ = LabelEncoder()
        y_transformed = self.label_encoder_.fit_transform(y)
        self.y_train_ = y_transformed

        return self.estimator.fit(X, y_transformed)

    def _get_bar_dist(self):
        """Get the bar distribution for the target labels.

        Returns:
            BarDistribution
                Bar distribution for the target labels.
        """
        unique_y = self.label_encoder_.classes_
        bucket_widths = unique_y[1:] - unique_y[:-1]

        borders = (
            unique_y - (np.array([bucket_widths[0]] + bucket_widths.tolist()) / 2)
        ).tolist() + [unique_y[-1] + bucket_widths[-1] / 2]
        return BarDistribution(borders=torch.tensor(borders).float())

    def predict(self, X):
        """Predict the target values for the input data.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Input data.

        Returns:
            y_pred : array-like of shape (n_samples,)
                Predicted target values.
        """
        return self.predict_full(X)["mean"]

    def set_categorical_features(self, categorical_features):
        """Set the categorical feature indices.

        Parameters:
            categorical_features : list
                List of categorical feature indices.
        """
        self.categorical_features = categorical_features

    def get_optimization_mode(self):
        """Get the optimization mode for the regressor.

        Returns:
            str
                Optimization mode ("mean").
        """
        return "mean"

    @staticmethod
    def probabilities_to_logits_multiclass(probabilities, eps=1e-6):
        """Convert probabilities to logits for a multi-class problem.

        Parameters:
            probabilities : array-like of shape (n_samples, n_classes)
                Input probabilities for each class.
            eps : float, default=1e-6
                Small value to avoid division by zero or taking logarithm of zero.

        Returns:
            logits : array-like of shape (n_samples, n_classes)
                Output logits for each class.
        """
        probabilities = np.clip(probabilities, eps, 1 - eps)
        logits = np.log(probabilities) - np.log(
            1 - probabilities.sum(axis=1, keepdims=True) + eps
        )
        return logits

    def predict_full(self, X):
        """Predict the full set of output values for the input data.

        Parameters:
            X : array-like of shape (n_samples, n_features)
                Input data.

        Returns:
            dict
                Dictionary containing the predicted output values, including:
                - "mean": Predicted mean values.
                - "median": Predicted median values.
                - "mode": Predicted mode values.
                - "logits": Predicted logits.
                - "buckets": Predicted bucket probabilities.
                - "quantile_{q:.2f}": Predicted quantile values for each quantile q.
        """
        class_probs = self.estimator.predict_proba(X)
        class_logits = (
            torch.tensor(
                ClassifierAsRegressor.probabilities_to_logits_multiclass(class_probs)
            )
            .float()
            .unsqueeze(0)
        )
        bar_dist = self._get_bar_dist()
        r = TabPFNRegressor._post_process_predict_full(
            prediction=class_logits, criterion=bar_dist
        )

        return r
