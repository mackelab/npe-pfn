#!/bin/bash
#SBATCH --job-name=simulate_streams
#SBATCH --partition=cpu-ferranti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=simulate_streams_%A_%a.out
#SBATCH --error=simulate_streams_%A_%a.err
#SBATCH --array=0-60

# Load your Python environment if needed
# module load python/3.9

START_INDEX=$((1222 + 4 * SLURM_ARRAY_TASK_ID))
NUM_WORKERS=${SLURM_CPUS_PER_TASK:-4}
python -m tabpfn_sbi.tasks.streams.generate_streams --start="$START_INDEX" --num-workers="$NUM_WORKERS"
