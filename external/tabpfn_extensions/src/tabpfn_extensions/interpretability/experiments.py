#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0


class FeatureSelectionExperiment(Experiment):
    """
    This class is used to run experiments on generating synthetic data.
    """

    name = "FeatureSelectionExperiment"

    def plot(self, **kwargs):
        pass

    def run(self, tabpfn, **kwargs):
        """

        :param tabpfn:
        :param kwargs:
            indices: list of indices from X features to use
        :return:
        """
        from tabpfn.scripts.estimator.interpretability import feature_selection

        results = []

        assert kwargs.get("dataset") is not None, "Dataset must be provided"
        dataset = copy.deepcopy(kwargs.get("dataset"))

        sfs = feature_selection(tabpfn, dataset.x, dataset.y, n_features_to_select=1)

        self.plot()


class FeatureRetainer(BaseEstimator, ClassifierMixin):
    def __init__(self, estimator, retained_indices):
        self.estimator = estimator
        self.retained_indices = retained_indices

    def fit(self, X, y=None):
        self.estimator.fit(X, y)
        return self

    def predict(self, X):
        X_retained = self.transform(X)
        return self.estimator.predict(X_retained)

    def predict_proba(self, X):
        X_retained = self.transform(X)
        return self.estimator.predict_proba(X_retained)

    def transform(self, X):
        X_retained = X.clone()
        removed_indices = [
            i for i in range(X.shape[1]) if i not in self.retained_indices
        ]
        X_retained[:, removed_indices] = np.inf
        return X_retained

    def score(self, X, y):
        X_retained = self.transform(X)
        return self.estimator.score(X_retained, y)

    def _more_tags(self):
        return {"allow_nan": True, "allow_inf": True}


class FeatureSelectionInPredictExperiment(Experiment):
    """
    This class is used to run experiments on generating synthetic data.
    """

    # TODO: Also without selection, check if can use info from train

    name = "FeatureSelectionInPredictExperiment"

    def plot(self, **kwargs):
        pass

    def run(self, tabpfn, **kwargs):
        """

        :param tabpfn:
        :param kwargs:
            indices: list of indices from X features to use
        :return:
        """
        CV_FOLDS = 5
        from sklearn.model_selection import cross_val_score

        results = []

        assert kwargs.get("dataset") is not None, "Dataset must be provided"
        dataset = copy.deepcopy(kwargs.get("dataset"))

        print(cross_val_score(tabpfn, dataset.x, dataset.y, cv=CV_FOLDS))

        retained_indices = [0, 1, 2]
        feature_retainer = FeatureRetainer(
            estimator=tabpfn, retained_indices=retained_indices
        )

        print(cross_val_score(feature_retainer, dataset.x, dataset.y, cv=CV_FOLDS))
        print(
            cross_val_score(
                tabpfn, dataset.x[:, retained_indices], dataset.y, cv=CV_FOLDS
            )
        )

        self.plot()
