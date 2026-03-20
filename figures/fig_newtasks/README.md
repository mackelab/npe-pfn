# New Tasks Figure

## Preview

No inline preview asset is currently checked in for this folder.

- `rebutal_table_nll.md`
- `rebutal_table_nll_small.md`
- `rebutal_table_c2st.md`

## Results Sources

- `main_results/sbibm_newtasks/summary.csv`
- `results/sbibm_newtasks/summary.csv`
- `tabpfn_sbi/tasks/files/streams/`

## Experiments

- `conf/experiment/hypothesis_bm.yaml`
- `conf/experiment/hypothesis_bm_tabpfn.yaml`
- `conf/experiment/streams_bm.yaml`
- `conf/experiment/streams_bm_tabpfn.yaml`
- `conf/experiment/streams_bm_long.yaml`
- `conf/experiment/streams_bm_tabpfn_long.yaml`
- `conf/experiment/simformer_bm.yaml`
- `conf/experiment/simformer_bm_tabpfn.yaml`
- `conf/experiment/simformer_bm_c2st.yaml`
- `conf/experiment/simformer_bm_tabpfn_c2st.yaml`

## Notes

- The notebooks in this folder read `main_results/sbibm_newtasks/summary.csv` for the checked-in tables and can also work from `results/sbibm_newtasks/summary.csv` for fresh reruns.
- This folder covers the hypothesis benchmarks (`weinberg`, `mg1`, `biomolecular_docking`), the streams task, and the simformer tasks grouped into the new-tasks benchmark.
- Install the optional dependencies before rerunning the hypothesis or streams parts: `pip install -e ".[hypothesis]"` or `uv sync --extra hypothesis`.
- The streams task also requires presimulated GD-1 stream pickles under `tabpfn_sbi/tasks/files/streams/`; generate them with `python -m tabpfn_sbi.tasks.streams.generate_streams` or the SLURM wrapper `tabpfn_sbi/tasks/streams/generate_streams.sh`.
- The legacy helper script `figures/fig_newtasks/scripts/simulate_streams.py` now forwards to the task-side generator so the generated files line up with `tabpfn_sbi.tasks.hypothesis_tasks.StreamsTask`.
