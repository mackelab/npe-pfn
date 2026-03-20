#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import math
import numpy as np
from sklearn.multiclass import OutputCodeClassifier
from sklearn.utils import check_random_state
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import _num_samples, check_is_fitted


class ManyClassClassifier(OutputCodeClassifier):
    """Output-Code multiclass strategy with deciary codebook.

    This class extends the original OutputCodeClassifier to support n-ary codebooks (with n=alphabet_size),
    allowing for handling more classes.

    Parameters:
        estimator : estimator object
            An estimator object implementing :term:`fit` and one of
            :term:`decision_function` or :term:`predict_proba`. The base classifier
            should be able to handle up to `alphabet_size` classes.

        random_state : int, RandomState instance, default=None
            The generator used to initialize the codebook.
            Pass an int for reproducible output across multiple function calls.
            See :term:`Glossary <random_state>`.

    Attributes:
        estimators_ : list of `int(n_classes * code_size)` estimators
            Estimators used for predictions.

        classes_ : ndarray of shape (n_classes,)
            Array containing labels.

        code_book_ : ndarray of shape (n_classes, `len(estimators_)`)
            Deciary array containing the code of each class.

    Examples:
    ```python title="Example"
    >>> from sklearn.datasets import load_iris
    >>> from tabpfn.scripts.estimator import ManyClassClassifier, TabPFNClassifier
    >>> from sklearn.model_selection import train_test_split
    >>> x, y = load_iris(return_X_y=True)
    >>> x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)
    >>> clf = TabPFNClassifier()
    >>> clf = ManyClassClassifier(clf, alphabet_size=clf.max_num_classes_)
    >>> clf.fit(x_train, y_train)
    >>> clf.predict(x_test)
    ```
    """

    def __init__(
        self,
        estimator,
        *,
        alphabet_size=None,
        n_estimators=None,
        n_estimators_redundancy=4,
        random_state=None,
    ):
        self.estimator = estimator
        self.random_state = random_state
        self.alphabet_size = alphabet_size
        self.n_estimators = n_estimators
        self.n_estimators_redundancy = n_estimators_redundancy

    def get_alphabet_size(self):
        if self.alphabet_size is None:
            return self.estimator.max_num_classes_
        return self.alphabet_size

    def get_n_estimators(self, n_classes):
        if self.n_estimators is None:
            return (
                math.ceil(1 + math.log(n_classes, self.get_alphabet_size()))
                * 2
                * self.n_estimators_redundancy
            )
        return self.n_estimators

    def _generate_codebook(self, n_classes, n_estimators, alphabet_size):
        """Generate an efficient codebook using the provided alphabet size.

        This function generates a codebook where for each codeword at most `alphabet_size - 1` classes
        are mapped to unique classes, and the remaining classes are mapped to the "rest" class. The codebook
        is generated to provide for each class a unique codeword with maximum reduncancy between codewords.
        Greedy optimization is used to find the optimal codewords.

        Parameters:
            n_classes : int
                Number of classes.
            n_estimators : int
                Number of estimators.
            alphabet_size : int, default=10
                Size of the alphabet for the codebook. The first `alphabet_size - 1` classes
                will be mapped to unique classes, and the remaining classes will be mapped
                to the "rest" class.

        Returns:
            codebook : ndarray of shape (n_estimators, n_classes)
                Efficient n-ary codebook.
        """
        n_classes_in_codebook = min(n_classes, alphabet_size - 1)
        n_classes_remaining = n_classes - n_classes_in_codebook

        # Initialize the codebook with zeros
        codebook = np.zeros((n_estimators, n_classes), dtype=int)

        def generate_codeword():
            choices = list(range(0, n_classes_in_codebook)) + [
                n_classes_in_codebook for _ in range(n_classes_remaining)
            ]
            return np.random.permutation(
                choices,
            )

        # Generate the first codeword randomly
        codebook[0] = generate_codeword()

        # Generate the remaining codewords
        for i in range(1, n_estimators):
            # Initialize the current codeword with random values
            current_codeword = generate_codeword()
            max_distance, n_min_distance_min, max_distance_codeword = (
                0,
                1000,
                current_codeword,
            )

            # Iteratively improve the current codeword
            for _ in range(1000):  # Number of iterations can be adjusted
                # Compute the Hamming distances between the current codeword and all previous codewords
                distances = np.sum(
                    (codebook[:i] == n_classes_in_codebook)
                    != (current_codeword == n_classes_in_codebook),
                    axis=0,
                )

                # Find the minimum Hamming distance
                min_distance = np.min(distances, axis=0)

                n_min_distance = np.sum(distances == min_distance)

                # print(min_distance, n_min_distance)

                if min_distance > max_distance or (
                    min_distance == max_distance and n_min_distance < n_min_distance_min
                ):
                    max_distance_codeword = current_codeword
                    max_distance = min_distance
                    n_min_distance_min = n_min_distance

                current_codeword = generate_codeword()

            # Assign the optimized codeword to the codebook
            codebook[i] = max_distance_codeword

        return codebook

    def fit(self, X, y, **fit_params):
        """Fit underlying estimators.

        Parameters:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Data.

            y : array-like of shape (n_samples,)
                Multi-class targets.

            **fit_params : dict
                Parameters passed to the ``estimator.fit`` method of each
                sub-estimator.

        Returns:
            self : object
                Returns a fitted instance of self.
        """
        y = self._validate_data(X="no_validation", y=y)

        random_state = check_random_state(self.random_state)
        check_classification_targets(y)

        self.classes_ = np.unique(y)
        n_classes = self.classes_.shape[0]
        if n_classes == 0:
            raise ValueError(
                "DeciaryOutputCodeClassifier can not be fit when no class is present."
            )

        self.no_mapping_needed_ = False
        if n_classes <= self.get_alphabet_size():
            self.no_mapping_needed_ = True
            return self.estimator.fit(X, y, **fit_params)

        # Generate efficient deciary codebook
        self.code_book_ = self._generate_codebook(
            n_classes,
            self.get_n_estimators(n_classes),
            alphabet_size=self.get_alphabet_size(),
        )

        print(
            f"Using codebook with {self.code_book_.shape[0]} estimators and {self.code_book_.shape[1]} classes"
        )

        self.classes_index_ = {c: i for i, c in enumerate(self.classes_)}

        self.Y_train = np.array(
            [self.code_book_[:, y[i]] for i in range(_num_samples(y))],
            dtype=int,
        )
        self.X_train = X

        # print('CLF DIST IN', list(zip(np.unique(y), (np.unique(y, return_counts=True)[1] / y.shape[0]).tolist())))

        return self

    def predict_proba(self, X):
        """Predict probabilities using the underlying estimators.

        Parameters:
            X : {array-like, sparse matrix} of shape (n_samples, n_features)
                Data.

        Returns:
            p : ndarray of shape (n_samples, n_classes)
                Returns the probability of the samples for each class in the model,
                where classes are ordered as they are in `self.classes_`.
        """
        check_is_fitted(self)

        if self.no_mapping_needed_:
            return self.estimator.predict_proba(X)

        Y = np.array(
            [
                _fit_and_predict_proba(
                    self.estimator, self.X_train, self.Y_train[:, i], X
                )
                for i in range(self.code_book_.shape[0])
            ],
            dtype=np.float64,
        )

        # Y shape is (n_estimators, n_samples, n_classes_per_code)
        n_estimators, n_samples, n_classes_per_code = Y.shape
        rest_class = np.max(self.code_book_)

        # Compute the weighted probabilities for each class
        probabilities = np.zeros((n_samples, self.classes_.shape[0]))

        for i in range(n_estimators):
            for j in range(self.classes_.shape[0]):
                if self.code_book_[i, j] != rest_class:
                    j_remapped = self.code_book_[i, j]
                    probabilities[:, j] += Y[
                        i, :, j_remapped
                    ]  # / (1 - Y[i, :, rest_class])
                    # print(j, Y[i, :, j_remapped])

        assert not (
            (self.code_book_ != rest_class).sum(0) == 0
        ).any(), f"Some classes are not mapped to any codeword. {self.code_book_} {self.classes_} {((self.code_book_ != rest_class).sum(0) == 0)}"

        # Normalize the weighted probabilities to get the final class probabilities
        probabilities /= (self.code_book_ != rest_class).sum(0)
        probabilities /= probabilities.sum(axis=1, keepdims=True)

        return probabilities

    def set_categorical_features(self, categorical_features):
        self.categorical_features = categorical_features


def _fit_and_predict_proba(estimator, X_train, Y_train, X):
    """Predict probabilities for deciary values using a single estimator."""
    estimator.fit(X_train, Y_train)

    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(X)
    else:
        raise AttributeError("Base estimator doesn't have `predict_proba` method.")
