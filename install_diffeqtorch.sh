#!/bin/bash
conda activate tabpfn
echo "Installing Julia"
# Interactive version
# curl -fsSL https://install.julialang.org | sh # This is interactive! Choose location
curl -fsSL https://install.julialang.org | sh -s -- --yes
echo "Source bashrc"
source ~/.bashrc
echo "Installing Julia 1.5.1"
juliaup add 1.5.1
juliaup default 1.5.1
echo "Installing diffeqtorch"
conda activate tabpfn
pip install julia
pip install diffeqtorch
export JULIA_SYSIMAGE_DIFFEQTORCH="$HOME/.julia_sysimage_diffeqtorch.so"
python -c "from diffeqtorch.install import install_and_test; install_and_test()"
