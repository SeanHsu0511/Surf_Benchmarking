import argparse
import os
import h5py
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jit
from tqdm import tqdm
from GRF_jax import GRF_jax
from solve_gwf_jax import v_solve_gwf_jax




# --- Settings ---
S = 128
ALPHA = 2.0
TAU = 3.0
TOTAL_SAMPLES = 100000  
BATCH_SIZE = 128
SAMPLES_PER_FILE = 1000000  
FILES_PER_JOB = 50          
F_SOURCE = jnp.ones((S, S))

@jit
def get_binary_a_jax(U):
    return jnp.where(U >= 0, 12.0, 3.0)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--job_id', type=int, default=0, help="Slurm array task ID")
    args = parser.parse_args()

    output_dir = "data_pool_128"
    os.makedirs(output_dir, exist_ok=True)

    master_key = jax.random.PRNGKey(888 + args.job_id * 1000000)
    
    
    # Prepare constant F matrix batch
    F_batch = jnp.tile(F_SOURCE[jnp.newaxis, :, :], (BATCH_SIZE, 1, 1))

    # 1. Initialize HDF5 file
    for file_idx in range(FILES_PER_JOB):

        file_tag = f"part_{args.job_id:03d}_{file_idx:02d}"
        output_path = os.path.join(output_dir, f"{file_tag}.h5")
        
        print(f"\n>>> Starting File {file_idx+1}/{FILES_PER_JOB}: {output_path}")

        with h5py.File(output_path, 'w') as f:
            dset_a = f.create_dataset('a', shape=(0, S, S), maxshape=(None, S, S), 
                                      dtype='float32', chunks=(BATCH_SIZE, S, S), compression="gzip", compression_opts=4)
            dset_u = f.create_dataset('u', shape=(0, S, S), maxshape=(None, S, S), 
                                      dtype='float32', chunks=(BATCH_SIZE, S, S), compression="gzip", compression_opts=4)

            pbar = tqdm(range(0, SAMPLES_PER_FILE, BATCH_SIZE), desc=f"Job {args.job_id} [{file_tag}]")

            for _ in pbar:
                master_key, *subkeys = jax.random.split(master_key, BATCH_SIZE + 1)
                subkeys = jnp.array(subkeys)

                U_batch = vmap(lambda k: GRF_jax(ALPHA, TAU, S, k))(subkeys)
                a_batch = get_binary_a_jax(U_batch)
                u_batch = v_solve_gwf_jax(a_batch, F_batch, S)
                
                new_count = dset_a.shape[0] + BATCH_SIZE
                dset_a.resize(new_count, axis=0)
                dset_u.resize(new_count, axis=0)

                dset_a[-BATCH_SIZE:] = np.array(a_batch, dtype=np.float32)
                dset_u[-BATCH_SIZE:] = np.array(u_batch, dtype=np.float32)

        print(f"Finished {output_path}")

if __name__ == "__main__":
    main()