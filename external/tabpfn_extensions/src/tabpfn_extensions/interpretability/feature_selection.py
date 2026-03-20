#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import cross_val_score


def feature_selection(
    estimator, X, y, n_features_to_select=3, feature_names=None, **kwargs
):
    if hasattr(estimator, "fit_at_predict_time"):
        if estimator.fit_at_predict_time:
            print(
                "WARNING: We recommend to set fit_at_predict_time to False for SHAP values to "
                "be calculated, this will significantly speed up calculation."
            )

    if hasattr(estimator, "show_progress"):
        show_progress_ = estimator.show_progress
        estimator.show_progress = False
        try:
            return _feature_selection(
                estimator, X, y, n_features_to_select, feature_names, **kwargs
            )
        finally:
            estimator.show_progress = show_progress_
    else:
        return _feature_selection(
            estimator, X, y, n_features_to_select, feature_names, **kwargs
        )


def _feature_selection(
    estimator, X, y, n_features_to_select=3, feature_names=None, **kwargs
):
    # TODO: Try https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/#sequentialfeatureselector
    # TODO: Could use more feature in training, but only keep fewer in test
    # TODO: the fit function is somehow still called; We need to to change the feature selection
    #  method so it sets feature to none in test; Could be done with a wrapper that sets the feature subsets in fit
    #  and predict
    CV_FOLDS = 5

    score_all = cross_val_score(estimator, X, y, cv=CV_FOLDS)

    # TODO: Feature selection is done without CV, i.e. final CV scores might be biased (too good)
    sfs = SequentialFeatureSelector(
        estimator, n_features_to_select=n_features_to_select, direction="forward"
    )
    sfs.fit(X, y)
    sfs.get_support()
    X_transformed = sfs.transform(X)

    score_selected = cross_val_score(estimator, X_transformed, y, cv=CV_FOLDS)

    print(f"Score with all features: {score_all.mean()} +/- {score_all.std()}")
    print(
        f"Score with selected features: {score_selected.mean()} +/- {score_selected.std()}"
    )

    if feature_names is not None:
        print(
            "Features selected by forward sequential selection: "
            f"{np.array(feature_names)[sfs.get_support()].tolist()}"
        )

    return sfs
