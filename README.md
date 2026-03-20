# Reproducing "Effortless, Simulation-Efficient Bayesian Inference using Tabular Foundation Models"

This repository contains the code, experiments, task assets, and figure-generation workflows used to reproduce the results from the paper "Effortless, Simulation-Efficient Bayesian Inference using Tabular Foundation Models".

Paper: <https://arxiv.org/abs/2504.17660>

## Installation

Install with `uv` (recommended):

```bash
uv sync
source .venv/bin/activate
```

Or install it with pip (requires Python 3.11):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e external/tabpfn -e external/tabpfn_extensions -e external/sbibm
pip install -e .
```

For extra julia dependencies of sbibm, you need to run the following:

```bash
bash install_diffeqtorch.sh
```

NOTE: This can take up to 10 minutes to install all julia dependencies.

Some tasks require optional dependencies:

- `hypothesis` and `streams` tasks:

```bash
pip install -e ".[hypothesis]"
```

  with `uv`:

```bash
uv sync --extra hypothesis
```

  the `streams` task additionally needs presimulated GD-1 stream files:

```bash
python -m tabpfn_sbi.tasks.streams.generate_streams
```

  this writes the required pickles to `tabpfn_sbi/tasks/files/streams/`.

- `pyloric` task:

```bash
pip install -e ".[pyloric]"
```

  with `uv`:

```bash
uv sync --extra pyloric
```

- if that fails because of local Cython/wheel-building issues for `pyloric`, install it manually as a fallback:

```bash
cd ..
git clone https://github.com/mackelab/pyloric.git
cd pyloric
pip install .
```

- both extras together:

```bash
pip install -e ".[hypothesis,pyloric]"
```

  with `uv`:

```bash
uv sync --extra hypothesis --extra pyloric
```

If you want to use CUDA in JAX for the `pyloric` task, install:

```bash
pip install jax[cuda12]
```

## Benchmarking

After the install via pip, a command line tool is available to run the benchmarking.

To test if the installation was successful, run:

```bash
tabpfnbm --help
```

This will show you all currently available commands. Which are also listed in `/conf`.

To run the benchmarking, run:

```bash
tabpfnbm
```

This will run the benchmarking on the default settings. To change the settings, you can either change the default settings in `/conf` or pass the settings as arguments to the command line tool. For example:

```bash
tabpfnbm method=tabpfn eval=swd
```

will run the benchmarking with the tabpfn method and the sliced Wasserstein distance as evaluation metric.

All results will be saved in the `/results` folder with the "name" of the current run (which by default is "benchmark").

### Multirun

To run multiple runs of the benchmarking, you can use the `multirun` command. For example:

```bash
tabpfnbm -m method=tabpfn,npe task.num_simulation=1000,10_000,100_000
```

which will lauch all combinations of values as jobs based on the configure launcher. This
by default is "slurm", which requires a slurm cluster to be available. You can change it to "local" to run the jobs locally. But this is not recommended for large sweeps.

To configure what resources are required by each job there are a few options available in the `conf/partition` file. For example:

```bash
tabpfnbm -m method=tabpfn partition=h100
```

will run the benchmarking with the tabpfn method on the "h100" partition i.e. on a GPU.

### Experiments

To save an experiment, you can put your configurations in the `/conf/experiment` folder. This
can for example looks like

```yaml
# @package _global_
name: example

hydra:
  mode: MULTIRUN
  sweeper:
    params:
      seed: 0,1,2,3,4
      eval: c2st
      method: npe
      task: gaussian_linear, two_moons
      task.num_simulations: 1000, 10_000, 100_000

  run:
    dir: results/${name}
  sweep:
    dir: results/${name}
    subdir: ${hydra.job.override_dirname}


defaults:
  - _self_
  - override /partition: cpu2
```

Which will run NPE on 5 different seeds on two different tasks each for 1k,10k and 100k sims with jobs that will only
allocated CPUs.

To run this experiment, you can use the `experiment` command. For example:

```bash
tabpfnbm +experiment=example
```

note that you require the `+`, as this is not part of default arguments, but an additional argument to override the default arguments.

### Sweepers

To define a sweep, you can put your configurations in the `/conf/sweeper` folder. There are a few available i.e. `tpe` which is a kind
of bayesian optimization, `grid` which is a grid search and `random` which is a random search.

The sweep uses as objective function the output of the main script (which is the average metrics by default).

See <https://hydra.cc/docs/plugins/optuna_sweeper/> for more information on how to configure the sweepers.

## Reproducing Paper Figures

The recommended entry point for reproducing paper results is the `figures/` folder.

- Each figure subfolder has its own `README.md` describing:
  - which result files or cached assets it reads
  - which `conf/experiment/*.yaml` configs need to be run
  - any extra task-specific setup such as optional dependencies or cached stream files
- Start by opening the relevant figure folder, for example `figures/fig2/README.md` or `figures/fig_newtasks/README.md`.
- Then run the listed experiments with:

```bash
tabpfnbm +experiment=<experiment_name>
```

- Most experiment configs write fresh outputs to `results/<name>/`, while many figure notebooks also read committed summaries from `main_results/`.
- Some figures additionally depend on local helper scripts inside the figure folder, or task-side generated assets such as `tabpfn_sbi/tasks/files/streams/` for the streams benchmark.

In short: pick the figure you want to reproduce, read the `README.md` in that figure folder, run the listed experiments, and then use the notebook or scripts in the same folder to rebuild the final panel.

## Citation

If you use this repository, please cite the paper:

```bibtex
@article{vetter2025effortless,
  title={Effortless, Simulation-Efficient Bayesian Inference using Tabular Foundation Models},
  author={Vetter, Julius and Gloeckler, Manuel and Gedon, Daniel and Macke, Jakob H.},
  journal={arXiv preprint arXiv:2504.17660},
  year={2025},
  doi={10.48550/arXiv.2504.17660}
}
```

## Development

Currently, the following strucutre is intended:

- tabpfn_sbi: The main package
- tabpfn_sbi/benchmarking: The benchmarking scripts (mainly the command line tool)
- tabpfn_sbi/evals: The evaluation metrics/scripts (e.g. swd, c2st) or more
- tabpfn_sbi/methods: The methods and how to run them (e.g. tabpfn, npe) or more
- tabpfn_sbi/tasks:
  - The tasks which provide the simulator and prior.
  - Optionally, the tasks can also provide reference posteriors for evaluation.
- tabpfn_sbi/utils: Utility functions and classes
