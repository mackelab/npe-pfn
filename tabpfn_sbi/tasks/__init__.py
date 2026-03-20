from sbibm.tasks import get_task as get_task_sbibm

from .allen_task import AllenTask
from .gaussian_misspecified import LikelihoodMisspecifiedTask, PriorMisspecifiedTask
from .hh_task import HHTask
from .lv_task import LotkaVolterraTask
from .sir import SIRTask
from .spatial_sir import SpatialSIRTask
from .flexible_linear_task import FlexibleLinearTask
from .simformer_tasks import (
    NonlinearGaussianTreeTask,
    NonlinearMarkovChainTask,
    LongNonlinearMarkovChainTask,
)
from .uncond_tasks import (
    TASK_NAMES_TWO_DIM_UNCOND,
    TASK_NAMES_UCI,
    UncondDensityEstimationTask,
)
from .sampling_order import OrderSimpleNonlinearTask, OrderMixedDistTask
from .permuted_task import PermutedTask
from .hypothesis_tasks import (
    WeinbergTask,
    DeathTask,
    MG1Task,
    BiomolecularDockingTask,
    StreamsTask,
)

TASK_NAMES_SBIBM = [
    "lotka_volterra",
    "bernoulli_glm",
    "bernoulli_glm_raw",
    "gaussian_linear",
    "gaussian_linear_uniform",
    "gaussian_mixture",
    "slcp",
    "gaussian_nonlinear",
    "slcp_distractors",
    "sir",
    "two_moons",
]


def get_task(name, **task_kwargs):
    if name in TASK_NAMES_SBIBM:
        return get_task_sbibm(name)
    elif name == "lotka_volterra_long":
        return LotkaVolterraTask(**task_kwargs)
    elif name == "sir_long":
        return SIRTask(**task_kwargs)
    elif (
        name == "spatial_sir"
        or name == "spatial_sir_large"
        or name == "spatial_sir_medium"
    ):
        return SpatialSIRTask(**task_kwargs)
    elif name == "hh":
        return HHTask(**task_kwargs)
    elif name == "allen":
        return AllenTask(**task_kwargs)
    elif name == "flexible_linear":
        return FlexibleLinearTask(**task_kwargs)
    elif name == "pyloric":
        try:
            from tabpfn_sbi.tasks.pyloric import PyloricTask

            return PyloricTask(**task_kwargs)
        except ImportError:
            raise ImportError(
                "Pyloric task not available, please install the required dependencies as specified in README."
            )
    elif name == "misspecified_prior":
        return PriorMisspecifiedTask(**task_kwargs)
    elif name == "misspecified_likelihood":
        return LikelihoodMisspecifiedTask(**task_kwargs)
    elif name in TASK_NAMES_TWO_DIM_UNCOND or name in TASK_NAMES_UCI:
        return UncondDensityEstimationTask(name, **task_kwargs)
    elif name == "simple_nonlinear":
        return OrderSimpleNonlinearTask(**task_kwargs)
    elif name == "mixed_dist":
        return OrderMixedDistTask(**task_kwargs)
    elif name == "permuted_task":
        return PermutedTask(**task_kwargs)
    elif name == "weinberg":
        return WeinbergTask(**task_kwargs)
    elif name == "death":
        return DeathTask(**task_kwargs)
    elif name == "mg1":
        return MG1Task(**task_kwargs)
    elif name == "streams":
        return StreamsTask(**task_kwargs)
    elif name == "biomolecular_docking":
        return BiomolecularDockingTask(**task_kwargs)
    elif name == "nonlinear_gaussian_tree":
        return NonlinearGaussianTreeTask(**task_kwargs)
    elif name == "nonlinear_marcov_chain":
        return NonlinearMarkovChainTask(**task_kwargs)
    elif name == "nonlinear_marcov_chain_long":
        return LongNonlinearMarkovChainTask(**task_kwargs)
    else:
        raise ValueError(f"Task {name} not found.")
