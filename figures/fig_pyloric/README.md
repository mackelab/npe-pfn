# Pyloric Figure

## Preview

![Pyloric Figure](fig_pyloric.svg)

## Results Sources

- `figures/fig_pyloric/posteriors_results3/`
- `figures/fig_pyloric/trace_data_845_082_0044.npz`
- `figures/fig_pyloric/scripts/`

## Experiments

No direct `conf/experiment/*.yaml` entry is required for this folder.

## Notes

- This figure is driven by the local scripts in `figures/fig_pyloric/scripts/` together with the cached posterior artifacts in this folder.
- Use `figures/fig_pyloric/scripts/run_pyloric3.py` and the `figures/fig_pyloric/scripts/run_pyloric_eval3_*.py` / `figures/fig_pyloric/scripts/run_pyloric_eval_permutations*.py` helpers as the primary entry points for reproducing the pyloric panels.
- There is currently no matching `conf/experiment/*.yaml` entry for the pyloric figure pipeline.
