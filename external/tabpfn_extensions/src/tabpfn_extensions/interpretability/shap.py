#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from multiprocessing import Pool

import shap
from ..utils import is_tabpfn


def calculate_shap_subset(args):
    X_subset, background, model, feature_idx = args
    explainer = shap.PermutationExplainer(model, background)
    return explainer.shap_values(X_subset)[:, feature_idx]


def parallel_permutation_shap(model, X, background=None, n_jobs=-1):
    # If no background data provided, use X
    if background is None:
        background = X

    # Split data into chunks for parallel processing
    n_features = X.shape[1]
    feature_indices = list(range(n_features))

    # Prepare arguments for parallel processing
    args_list = [(X, background, model, idx) for idx in feature_indices]

    # Run parallel computation
    with Pool(processes=n_jobs) as pool:
        shap_values_per_feature = pool.map(calculate_shap_subset, args_list)

    # Combine results
    shap_matrix = np.column_stack(shap_values_per_feature)

    return shap_matrix


def plot_shap(shap_values: np.ndarray):
    """
    Plots the shap values for the given test data. It will plot aggregated shap values for each feature, as well
    as per sample shap values. Additionally, if multiple samples are provided, it will plot the 3 most important
    interactions with the most important feature.

    Parameters:
        shap_values:
    """
    import shap

    if len(shap_values.shape) == 3:
        print("Computing shap values for the first class (index 0).")
        shap_values = shap_values[:, :, 0]

    shap.plots.bar(shap_values=shap_values, show=False)
    plt.title("Aggregate feature importances across the test examples")
    plt.show()
    shap.summary_plot(shap_values=shap_values, show=False)
    # plot the distribution of importances for each feature over all samples
    plt.title(
        "Feature importances for each feature for each test example (a dot is one feature for one example)"
    )
    plt.show()

    most_important = shap_values.abs.mean(0).values.argsort()[-1]
    print(
        f'Now we analyze the strongest feature interactions of the most important feature, namely the feature "{most_important}".'
    )
    if len(shap_values) > 1:
        plot_shap_feature(shap_values, most_important)


def plot_shap_feature(shap_values_, feature_name, n_plots=1):
    import shap

    # we can use shap.approximate_interactions to guess which features
    # may interact with age
    inds = shap.utils.potential_interactions(
        shap_values_[:, feature_name], shap_values_
    )

    # make plots colored by each of the top three possible interacting features
    for i in range(n_plots):
        shap.plots.scatter(
            shap_values_[:, feature_name],
            color=shap_values_[:, inds[i]],
            show=False,
        )
        plt.title(
            f"Feature {feature_name} with a color coding representing the value of ({inds[i]})"
        )


def get_shap_values(estimator, test_x, attribute_names=None, **kwargs) -> np.ndarray:
    """
    Computes SHAP (SHapley Additive exPlanations) values for the model's predictions on the given input features.

    Parameters:
        test_x (Union[pd.DataFrame, np.ndarray]): The input features to compute SHAP values for.
        kwargs (dict): Additional keyword arguments to pass to the SHAP explainer.

    Returns:
        np.ndarray: The computed SHAP values.
    """

    if isinstance(test_x, torch.Tensor):
        test_x = test_x.cpu().numpy()

    if isinstance(test_x, np.ndarray):
        test_x = pd.DataFrame(test_x)
        if attribute_names is not None:
            test_x.rename(
                columns={c: n for c, n in enumerate(attribute_names.tolist())},
                inplace=True,
            )
        else:
            test_x.rename(columns={c: str(c) for c in test_x.columns}, inplace=True)

    if hasattr(estimator, "predict_function_for_shap"):
        predict_function_for_shap = estimator.predict_function_for_shap
    else:
        predict_function_for_shap = (
            "predict_proba" if hasattr(estimator, "predict_proba") else "predict"
        )

    if hasattr(estimator, "fit_at_predict_time"):
        if not estimator.fit_at_predict_time:
            print(
                "WARNING: We recommend to set fit_at_predict_time to True for SHAP values to "
                "be calculated, this will significantly speed up calculation."
            )

    def get_shap():
        if is_tabpfn(estimator):
            explainer = get_tabpfn_explainer(
                estimator, test_x, predict_function_for_shap, **kwargs
            )
        else:
            explainer = get_default_explainer(
                estimator, test_x, predict_function_for_shap, **kwargs
            )
        return explainer(test_x)

    if hasattr(estimator, "show_progress"):
        show_progress_ = estimator.show_progress
        estimator.show_progress = False
        try:
            shap_values = get_shap()
        finally:
            estimator.show_progress = show_progress_
    else:
        shap_values = get_shap()

    return shap_values


def get_tabpfn_explainer(
    estimator, test_x, predict_function_for_shap="predict", **kwargs
):
    import shap

    return shap.Explainer(
        getattr(estimator, predict_function_for_shap),
        np.ones(test_x.iloc[0:1, :].shape) * float("nan"),
        **kwargs,
    )


def get_default_explainer(
    estimator, test_x, predict_function_for_shap="predict", **kwargs
):
    import shap

    background = shap.maskers.Independent(test_x, max_samples=1000)

    return shap.Explainer(
        getattr(estimator, predict_function_for_shap),
        test_x,
        **kwargs,
    )
