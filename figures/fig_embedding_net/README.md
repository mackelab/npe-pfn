# Embedding Network Figure

## Preview

![Embedding Network Figure](fig_embedding.svg)

## Results Sources

- `main_results/spatial_sir_sbibm/summary.csv`
- `results/sbibm_spatialsir/`
- `results/sbibm_lv_long/`

## Experiments

- `conf/experiment/lv_long_npe.yaml`
- `conf/experiment/lv_long_tabpfn.yaml`
- `conf/experiment/lv_long_tabpfn_raw.yaml`
- `conf/experiment/spatial_sir_baselines.yaml`
- `conf/experiment/spatial_sir_baselines_nll.yaml`
- `conf/experiment/spatial_sir_tabpfn.yaml`
- `conf/experiment/spatial_sir_tabpfn_raw.yaml`
- `conf/experiment/spatial_sir_tabpfn_nll.yaml`
- `conf/experiment/spatial_sir_tabpfn_raw_nll.yaml`

## Notes

- The embedding-network panels are assembled from long Lotka-Volterra runs and spatial-SIR runs.
- The notebook in this folder reads `results/sbibm_lv_long` and `results/sbibm_spatialsir`; the committed `main_results` mirror currently only covers the spatial-SIR summary.
