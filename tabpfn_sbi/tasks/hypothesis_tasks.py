import multiprocessing
import pickle
from pathlib import Path

import numpy as np
import torch

from .base import InferenceTask

try:
    from hypothesis.benchmark.biomoleculardocking import (
        Prior as BiomolecularDockingPrior,
        Simulator as BiomolecularDockingSimulator,
    )
    from hypothesis.benchmark.death import (
        Prior as DeathPrior,
        Simulator as DeathSimulator,
    )
    from hypothesis.benchmark.mg1 import Prior as MG1Prior, MG1Simulator
    from hypothesis.benchmark.weinberg import (
        Prior as WeinbergPrior,
        Simulator as WeinbergSimulator,
    )
except ImportError as error:
    BiomolecularDockingPrior = None
    BiomolecularDockingSimulator = None
    DeathPrior = None
    DeathSimulator = None
    MG1Prior = None
    MG1Simulator = None
    WeinbergPrior = None
    WeinbergSimulator = None
    _HYPOTHESIS_IMPORT_ERROR = error
else:
    _HYPOTHESIS_IMPORT_ERROR = None

try:
    from tabpfn_sbi.tasks.streams.simulator import WDMSubhaloSimulator
    from tabpfn_sbi.tasks.util import allocate_prior_wdm_mass as PriorMass
except ImportError as error:
    WDMSubhaloSimulator = None
    PriorMass = None
    _STREAMS_IMPORT_ERROR = error
else:
    _STREAMS_IMPORT_ERROR = None


DEFAULT_STREAMS_FOLDER = Path(__file__).resolve().parent / "files" / "streams"


def _require_hypothesis_dependencies():
    if _HYPOTHESIS_IMPORT_ERROR is not None:
        raise ImportError(
            'Hypothesis benchmark tasks require optional dependencies. Install with `pip install -e ".[hypothesis]"`.'
        ) from _HYPOTHESIS_IMPORT_ERROR


def _require_streams_dependencies():
    if _STREAMS_IMPORT_ERROR is not None:
        raise ImportError(
            'Streams tasks require optional dependencies. Install with `pip install -e ".[hypothesis]"`.'
        ) from _STREAMS_IMPORT_ERROR


def _resolve_stream_file_paths(stream_folder=None):
    stream_path = (
        Path(stream_folder) if stream_folder is not None else DEFAULT_STREAMS_FOLDER
    )
    stream_path = stream_path.expanduser()
    if not stream_path.exists():
        raise FileNotFoundError(
            f"StreamsTask requires presimulated GD-1 files in `{stream_path}`. Generate them with `python -m tabpfn_sbi.tasks.streams.generate_streams`."
        )

    file_paths = sorted(
        (path for path in stream_path.iterdir() if path.suffix == ".pkl"),
        key=lambda path: int(path.stem.rsplit("_", 1)[-1]),
    )
    if not file_paths:
        raise FileNotFoundError(
            f"No presimulated GD-1 files were found in `{stream_path}`. Generate them with `python -m tabpfn_sbi.tasks.streams.generate_streams`."
        )

    return stream_path, file_paths


def wrap_as_torch_distribution(prior):
    class TorchDistribution(torch.distributions.Distribution):
        def __init__(self, prior):
            self.prior = prior

        def sample(self, sample_shape=torch.Size()):
            # Fixing some wierd behavours
            if sample_shape == torch.Size():
                sample_shape = (1,)
            return self.prior.sample(sample_shape).reshape(*sample_shape, -1)

        def log_prob(self, value):
            return torch.zeros(value.shape[:-1])

    return TorchDistribution(prior)


class WeinbergTask(InferenceTask):
    def __init__(self):
        _require_hypothesis_dependencies()
        # There 0d uniform is not compatible
        super().__init__(
            torch.distributions.Independent(
                torch.distributions.Uniform(torch.tensor([0.5]), torch.tensor([1.5])), 1
            )
        )
        self.simulator = WeinbergSimulator()

    def get_prior(self):
        return self.prior

    def get_simulator(self):
        return self.simulator


class DeathTask(InferenceTask):
    def __init__(self):
        _require_hypothesis_dependencies()
        super().__init__(wrap_as_torch_distribution(DeathPrior()))
        self.simulator = DeathSimulator()

    def get_prior(self):
        return self.prior

    def get_simulator(self):
        return self.simulator


class MG1Task(InferenceTask):
    def __init__(self):
        _require_hypothesis_dependencies()
        super().__init__(wrap_as_torch_distribution(MG1Prior()))
        self.simulator = MG1Simulator()
        self.simulator.num_steps = self.simulator.steps
        self.simulator.num_percentiles = self.simulator.percentiles

    def get_prior(self):
        return self.prior

    def get_simulator(self):
        return self.simulator


class BiomolecularDockingTask(InferenceTask):
    def __init__(self):
        _require_hypothesis_dependencies()
        super().__init__(wrap_as_torch_distribution(BiomolecularDockingPrior()))

        sim = BiomolecularDockingSimulator()

        def simulator(theta):
            xs = []

            for i in range(theta.shape[0]):
                x = sim.simulate(theta[i], sim.default_experimental_design)
                xs.append(x.flatten())
            return torch.stack(xs, axis=0)

        self.simulator = simulator

    def get_prior(self):
        return self.prior

    def get_simulator(self):
        return self.simulator


class SubhaloSimulator:
    def __init__(self, stream_folder=None, ages=None, device="cpu", num_workers=50):
        _require_streams_dependencies()
        if ages is None:
            raise ValueError(
                "StreamsTask requires cached stream ages to build the simulator."
            )
        self.stream_folder, self.files = _resolve_stream_file_paths(stream_folder)
        self.ages_to_idx = {float(age): i for i, age in enumerate(ages)}
        self.num_workers = num_workers
        self.device = device

    def simulate(self, age, mass):
        stream_idx = self.ages_to_idx[age]
        with open(self.files[stream_idx], "rb") as file_handle:
            stream = pickle.load(file_handle)["stream"]
        simulator = WDMSubhaloSimulator(stream, record_impacts=True)
        count, x, y = simulator(torch.tensor(mass).view(1, 1))[0]
        xs = torch.concatenate(
            [
                torch.tensor(count).view(1),
                torch.tensor(x).view(-1),
                torch.tensor(y).view(-1),
            ],
            axis=-1,
        )
        return xs

    def __call__(self, theta):
        ages, masses = theta[..., 0], theta[..., 1]
        ages = [float(age) for age in ages]
        masses = [float(mass) for mass in masses]
        # Use multiprocessing to simulate
        with multiprocessing.Pool(self.num_workers) as pool:
            xs = pool.starmap(self.simulate, zip(ages, masses))
        return torch.stack(xs, axis=0).float().to(self.device)


class SubhaloPrior(torch.distributions.Distribution):
    def __init__(self, stream_folder=None, device="cpu"):
        _require_streams_dependencies()
        self.stream_folder, self.files = _resolve_stream_file_paths(stream_folder)
        ages = []
        for file_path in self.files:
            with open(file_path, "rb") as file_handle:
                age = pickle.load(file_handle)["age"]
            ages.append(float(age))
        self.ages = torch.tensor(ages).to(device)
        self.mass_prior = PriorMass()
        self.device = device

    def sample_age(self, num_samples: int):
        idx = torch.randint(0, len(self.ages), (num_samples,))
        return self.ages[idx][..., None]

    def sample_mass(self, num_samples: int):
        return self.mass_prior.sample((num_samples,))

    def sample(self, sample_shape: torch.Size = torch.Size()):
        numel = torch.Size(sample_shape).numel()
        ages = self.sample_age(numel)
        masses = self.sample_mass(numel)
        ages = ages.reshape(-1, 1)
        masses = masses.reshape(-1, 1)

        ages = ages.to(self.device)
        masses = masses.to(self.device)

        return (
            torch.concatenate([ages, masses], axis=-1)
            .reshape(sample_shape + (2,))
            .float()
        )

    def to(self, device):
        self.device = device
        return self

    def log_prob(self, value):
        value = value.to(self.device)
        return torch.zeros(value.shape[:-1])


class StreamsTask(InferenceTask):
    def __init__(self, stream_folder=None, device="cpu"):
        _require_streams_dependencies()
        super().__init__(SubhaloPrior(stream_folder, device))
        self.simulator = SubhaloSimulator(stream_folder, self.prior.ages, device)

    def get_prior(self):
        return self.prior

    def get_simulator(self):
        return self.simulator
