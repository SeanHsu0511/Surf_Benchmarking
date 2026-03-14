#!/bin/bash
###### 1. Slurm directives (edit as needed) ######
#SBATCH --job-name=generate_darcy      # short label "TEST" visible in `squeue`
#SBATCH --partition=gh                 # queue / partition the system. Use gh-dev for testing and gh for longer scripts
#SBATCH --account=NAIRR240304          #  project code: I assume all are using the same one (NAIRR240304)
#SBATCH --nodes=1                      # number of physical nodes
#SBATCH --ntasks-per-node=1            # MPI tasks per node (1 if pure Python)
#SBATCH --cpus-per-task=4              # OpenMP / num_threads
#SBATCH --time=1:00:00                # HH:MM:SS wall‑clock limit
#SBATCH --output=logs/%x.%j.out        # stdout goes here (%x=job‑name, %j=job‑ID)
#SBATCH --error=logs/%x.%j.err         # stderr goes here

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
VENV=$SCRATCH/vista_pytorch_venv       # adjust path to your venv
if [[ -d $VENV ]]; then
  source "$VENV/bin/activate"
  export LD_LIBRARY_PATH="$VENV/lib:$LD_LIBRARY_PATH"
  echo "Activated venv at $VENV"
else
  echo "ERROR: venv not found at $VENV" >&2
  exit 1
fi

###### 5. Sanity check ######
python - <<'PY'
import torch, platform, os
print("Python :", platform.python_version())
print("Torch  :", torch.__version__)
print("CUDA OK:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device :", torch.cuda.get_device_name(0))
PY

###### 6. Your workload ######
python generate_pool.py  