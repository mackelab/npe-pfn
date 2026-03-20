#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import numpy as np
from typing import Dict, List, Literal, Optional, Tuple


def infer_categorical_features(X: np.ndarray, categorical_features) -> List[int]:
    """
    Infer the categorical features from the input data.
    We take `self.categorical_features` as the initial list of categorical features.


    Parameters:
        X (ndarray): The input data.

    Returns:
        Tuple[int, ...]: The indices of the categorical features.
    """
    max_unique_values_as_categorical_feature = 10
    min_unique_values_as_numerical_feature = 10

    _categorical_features = []
    for i in range(X.shape[-1]):
        # Filter categorical features, with too many unique values
        if i in categorical_features and (
            len(np.unique(X[:, i])) <= max_unique_values_as_categorical_feature
        ):
            _categorical_features += [i]

        # Filter non-categorical features, with few unique values
        elif (
            i not in categorical_features
            and len(np.unique(X[:, i])) < min_unique_values_as_numerical_feature
            and X.shape[0] > 100
        ):
            categorical_features += [i]

    return _categorical_features
