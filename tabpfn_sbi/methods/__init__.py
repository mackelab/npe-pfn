from tabpfn_sbi.methods.sbi_baselines import (
    run_basic_sbi_interface,
    run_npe_ensemble_sbi_interface,
    run_seq_sbi_interface,
    run_sweeper_sbi_interface,
)
from tabpfn_sbi.methods.tabpfn_sbi import run_basic_tabpfn_interface
from tabpfn_sbi.methods.tabpfn_tsnpe import run_ts_tabpfn_interface

SBI_names = ["nle", "nre", "npe", "nse", "npe_infomax"]
SEQ_SBI_names = ["tsnpe", "snle", "snpe", "snre"]
TABPFN_names = [
    "tabpfn",
    "filtered_tabpfn",
    "uncond_tabpfn",
    "tabpfn_infomax",
    "filtered_tabpfn_infomax",
    "filtered_tabpfn_binary",
]
SEQ_TABPFN_names = ["ts_tabpfn"]


def get_method(name: str):
    if name in SBI_names:
        return run_basic_sbi_interface
    elif name == "npe_ensemble":
        return run_npe_ensemble_sbi_interface
    elif name == "npe_sweeper":
        return run_sweeper_sbi_interface
    elif name in SEQ_SBI_names:
        return run_seq_sbi_interface
    elif name in TABPFN_names:
        return run_basic_tabpfn_interface
    elif name in SEQ_TABPFN_names:
        return run_ts_tabpfn_interface
    else:
        raise ValueError(f"Unknown method {name}")
