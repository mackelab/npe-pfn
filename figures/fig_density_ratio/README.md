# Density Ratio Figure

## Preview

![Density Ratio Figure](fig_density_ratio.svg)

## Results Sources

- `main_results/density_ratio/summary.csv`
- `main_results/density_ratio/summary_timing.csv`
- `main_results/density_ratio/logprobs_*.pt`

## Experiments

- `conf/experiment/sbibm_core.yaml`
- `conf/experiment/sbibm_core_tabpfn.yaml`

## Notes

- The notebook in this folder reads precomputed density-ratio artifacts from `main_results/density_ratio`.
- The compared tasks are Gaussian linear and two moons, which are covered by the core SBIBM benchmark experiments.
