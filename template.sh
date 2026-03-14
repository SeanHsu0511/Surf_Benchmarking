#!/bin/bash
###### 1. Slurm directives ######
#SBATCH --job-name=darcy_1B
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:v100:1
#SBATCH --array=0-19                  # 20  Jobs
#SBATCH --cpus-per-task=8
#SBATCH --time=7-00:00:00             # D-HH:MM:SS
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

###### 2. Pre‑run housekeeping ######
mkdir -p logs                          # create log dir if missing

# Print hostname and show GPU info
echo "Job launched on $(hostname) at $(date)"
nvidia-smi || echo "No GPUs visible"

###### 3. Module environment ######
module purge                           # start from a clean slate
module load gcc cuda                   # compiler & CUDA toolkit
module load python3                    # site‑provided Python (optional)

###### 4. Python virtual‑env activation ######
CONDA_PATH="$HOME/storage/miniconda3/etc/profile.d/conda.sh"

if [ -f "$CONDA_PATH" ]; then
    source "$CONDA_PATH"
    echo "Successfully sourced conda.sh"
else
    echo "ERROR: conda.sh not found at $CONDA_PATH" >&2
    exit 1
fi

conda activate bench

echo "Currently using Python from: $(which python)"

###### 5. Sanity check ######
python - <<'PY'
import jax, platform
print("Python   :", platform.python_version())
print("JAX      :", jax.__version__)
print("Devices  :", jax.devices())
print("CUDA OK? :", jax.devices()[0].device_kind)
PY

###### 6. Your workload ######
python generate_pool.py --job_id $SLURM_ARRAY_TASK_ID