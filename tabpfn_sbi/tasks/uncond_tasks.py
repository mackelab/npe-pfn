# partly adapted from https://github.com/rtqichen/ffjord/blob/master/lib/toy_data.py
import os
import warnings

import matplotlib.image as mpimg
import numpy as np
import sklearn
import sklearn.datasets
import torch
from sklearn.utils import shuffle as util_shuffle
from torch import Tensor
from torch.distributions import Distribution

TASK_NAMES_TWO_DIM_UNCOND = [
    "checkerboard",
    "rings",
    "swissroll",
    "2spirals",
    "pinwheel",
    "8gaussians",
    "attempto",
]

TASK_NAMES_UCI = [
    "gas",
    "power",
    "hepmass",
    "miniboone",
]


# NOTE: Somewhat hacky, this is a dataset as a prior. Will always return the same samples and 0 log_prob.
class DatasetPrior(Distribution):
    def __init__(self, dataset: Tensor):
        self.dataset = dataset

    # Careful with batched sampling! should be "prior.sample((cfg.task.num_simulations,))"
    def sample(self, shape):
        if shape[0] > self.dataset.shape[0]:
            warnings.warn(
                f"Requested {shape[0]} samples, but only {self.dataset.shape[0]} available. Returning all."
            )
        return self.dataset[: shape[0]]

    def log_prob(self, theta):
        warnings.warn("Log probability of prior is not known. Returning zeros.")
        return torch.zeros(theta.shape[0])


class UncondDensityEstimationTask:
    def __init__(self, task_name, data_dir=None):
        if task_name in TASK_NAMES_TWO_DIM_UNCOND:
            self.train_dataset = inf_train_gen(task_name, batch_size=1_000_000)
            self.test_dataset = inf_train_gen(task_name, batch_size=10_000, seed=0)
        elif task_name in TASK_NAMES_UCI:
            self.test_dataset, self.train_dataset = get_uci_samples(
                task_name, data_dir=data_dir
            )
        else:
            raise ValueError(f"Invalid task name {task_name}")

    def get_simulator(self):
        class ConstantSimulator:
            def __call__(self, theta):
                return torch.zeros(theta.shape[0], 1)

        return ConstantSimulator()

    def get_prior_dist(self):
        return DatasetPrior(self.train_dataset)

    def get_observation(self, idx=None):
        return torch.zeros(1, 1)

    def get_reference_posterior_samples(self, idx=None):
        return self.test_dataset


def inf_train_gen(simulator, batch_size=10000, seed=None):
    if seed is None:
        seed = np.random.randint(0, 2**32)  # from previous globally set seed
    rng = np.random.default_rng(seed)

    if simulator == "swissroll":
        data = sklearn.datasets.make_swiss_roll(n_samples=batch_size, noise=1.0)[0]
        data = data.astype("float32")[:, [0, 2]]
        data /= 5
        random_samples = data

    elif simulator == "rings":
        n_samples4 = n_samples3 = n_samples2 = batch_size // 4
        n_samples1 = batch_size - n_samples4 - n_samples3 - n_samples2

        # so as not to have the first point = last point, we set endpoint=False
        linspace4 = np.linspace(0, 2 * np.pi, n_samples4, endpoint=False)
        linspace3 = np.linspace(0, 2 * np.pi, n_samples3, endpoint=False)
        linspace2 = np.linspace(0, 2 * np.pi, n_samples2, endpoint=False)
        linspace1 = np.linspace(0, 2 * np.pi, n_samples1, endpoint=False)

        circ4_x = np.cos(linspace4)
        circ4_y = np.sin(linspace4)
        circ3_x = np.cos(linspace4) * 0.75
        circ3_y = np.sin(linspace3) * 0.75
        circ2_x = np.cos(linspace2) * 0.5
        circ2_y = np.sin(linspace2) * 0.5
        circ1_x = np.cos(linspace1) * 0.25
        circ1_y = np.sin(linspace1) * 0.25

        X = (
            np.vstack(
                [
                    np.hstack([circ4_x, circ3_x, circ2_x, circ1_x]),
                    np.hstack([circ4_y, circ3_y, circ2_y, circ1_y]),
                ]
            ).T
            * 3.0
        )
        X = util_shuffle(X, random_state=seed)

        # Add noise
        X = X + rng.normal(scale=0.08, size=X.shape)

        random_samples = X.astype("float32")

    elif simulator == "8gaussians":
        scale = 4.0
        centers = [
            (1, 0),
            (-1, 0),
            (0, 1),
            (0, -1),
            (1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), 1.0 / np.sqrt(2)),
            (-1.0 / np.sqrt(2), -1.0 / np.sqrt(2)),
        ]
        centers = [(scale * x, scale * y) for x, y in centers]
        dataset = []
        for i in range(batch_size):
            point = rng.standard_normal(2) * 0.5
            idx = rng.integers(8)
            center = centers[idx]
            point[0] += center[0]
            point[1] += center[1]
            dataset.append(point)
        dataset = np.array(dataset, dtype="float32")
        dataset /= 1.414
        random_samples = dataset

    elif simulator == "pinwheel":
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = batch_size // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)

        features = rng.standard_normal((num_classes * num_per_class, 2)) * np.array(
            [radial_std, tangential_std]
        )
        features[:, 0] += 1.0
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack(
            [np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)]
        )
        rotations = np.reshape(rotations.T, (-1, 2, 2))

        random_samples = 2 * rng.permutation(
            np.einsum("ti,tij->tj", features, rotations)
        )

    elif simulator == "2spirals":
        n = np.sqrt(np.random.rand(batch_size // 2, 1)) * 540 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        d1y = np.sin(n) * n + np.random.rand(batch_size // 2, 1) * 0.5
        x = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y)))) / 3
        x += np.random.randn(*x.shape) * 0.1
        random_samples = x

    elif simulator == "checkerboard":
        x1 = np.random.rand(batch_size) * 4 - 2
        x2_ = np.random.rand(batch_size) - np.random.randint(0, 2, batch_size) * 2
        x2 = x2_ + (np.floor(x1) % 2)
        random_samples = np.concatenate([x1[:, None], x2[:, None]], 1) * 2

    elif simulator == "attempto":
        base_path = os.path.dirname(os.path.abspath(__file__))
        img = mpimg.imread(os.path.join(base_path, "files/density_images/ada_gnu.jpg"))
        # https://commons.wikimedia.org/wiki/File:MGLA-adabyron.png GNU Free Documentation License
        gray_image = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        gray_image = np.max(gray_image) - gray_image

        y_dim, x_dim = gray_image.shape
        x = np.arange(y_dim)
        y = np.arange(y_dim)
        x, y = np.meshgrid(x, y)
        pos = np.dstack((x, y)).reshape(-1, 2)
        pos = pos - np.array([y_dim / 2, x_dim / 2])
        pos = pos / np.array([y_dim / 8, x_dim / 8])

        gray_image_reshape = gray_image.reshape(-1)
        draw_samples = np.random.choice(
            np.arange((len(pos))),
            batch_size,
            p=gray_image_reshape / np.sum(gray_image_reshape),
            replace=True,
        )

        random_samples = pos[draw_samples]
        # noise_scale = 0.05
        noise_scale = 0.001
        random_samples[:, 0] += np.random.randn(batch_size) * noise_scale
        random_samples[:, 1] *= -1
        random_samples[:, 1] += np.random.randn(batch_size) * noise_scale

    else:
        raise ValueError(f"Invalid data type {simulator}")

    return torch.from_numpy(random_samples.astype("float32"))


def get_uci_samples(dataset_name, data_dir=""):
    uci = np.load(os.path.join(data_dir, "uci-datasets", f"{dataset_name.upper()}.npy"))
    uci = uci[np.random.permutation(uci.shape[0])]
    uci = torch.from_numpy(uci.astype(np.float32))

    print(uci.shape)

    match dataset_name:
        case "gas":
            split_uci = 105_206
        case "power":
            split_uci = 204_928
        case "hepmass":
            split_uci = 174_987
        case "miniboone":
            split_uci = 3_648
        case _:
            raise ValueError(f"Invalid dataset name {dataset_name}")

    return uci[:split_uci], uci[split_uci:]
