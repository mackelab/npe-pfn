#  Copyright (c) Prior Labs GmbH 2025.
#  Licensed under the Apache License, Version 2.0

import os

import os
from typing import Any, Type, Tuple, Protocol


class TabPFNEstimator(Protocol):
    def fit(self, X: Any, y: Any) -> Any:
        ...

    def predict(self, X: Any) -> Any:
        ...


def is_tabpfn(estimator: Any) -> bool:
    """Check if an estimator is a TabPFN model."""
    try:
        return any(
            [
                "TabPFN" in str(estimator.__class__),
                "TabPFN" in str(estimator.__class__.__bases__),
                any("TabPFN" in str(b) for b in estimator.__class__.__bases__),
                "tabpfn.base_model.TabPFNBaseModel" in str(estimator.__class__.mro()),
            ]
        )
    except (AttributeError, TypeError):
        return False


from typing import Tuple, Type
import os
USE_TABPFN_LOCAL = os.getenv("USE_TABPFN_LOCAL", "true").lower() == "true"

def get_tabpfn_models() -> Tuple[Type, Type, Type]:
    """Get TabPFN models with fallback between local and client versions."""
    if USE_TABPFN_LOCAL:
        try:
            from tabpfn import TabPFNClassifier, TabPFNRegressor
            from tabpfn.preprocessing import PreprocessorConfig

            return TabPFNClassifier, TabPFNRegressor, PreprocessorConfig
        except ImportError:
            pass

    try:
        from tabpfn_client import (
            TabPFNClassifier as ClientTabPFNClassifier,
            TabPFNRegressor as ClientTabPFNRegressor,
        )
        from tabpfn_client.estimator import (
            PreprocessorConfig as ClientPreprocessorConfig,
        )

        # Wrapper classes to add device parameter
        class TabPFNClassifier(ClientTabPFNClassifier):
            def __init__(self, *args, device=None, **kwargs):
                super().__init__(*args, **kwargs)
                # Ignoring the device parameter for now

        class TabPFNRegressor(ClientTabPFNRegressor):
            def __init__(self, *args, device=None, **kwargs):
                super().__init__(*args, **kwargs)
                # Ignoring the device parameter for now

        return TabPFNClassifier, TabPFNRegressor, ClientPreprocessorConfig

    except ImportError:
        raise ImportError(
            "Neither local TabPFN nor TabPFN client could be imported. Install with:\n"
            "pip install tabpfn\n"
            "or\n"
            "pip install tabpfn-client"
        )


TabPFNClassifier, TabPFNRegressor, PreprocessorConfig = get_tabpfn_models()
