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
F_SOURCE = jnp.ones((S, S))
OUTPUT_PATH = 'data_pool_128x128.h5'

@jit
def get_binary_a_jax(U):
    return jnp.where(U >= 0, 12.0, 3.0)

def main():
    master_key = jax.random.PRNGKey(888)
    
    # Prepare constant F matrix batch
    F_batch = jnp.tile(F_SOURCE[jnp.newaxis, :, :], (BATCH_SIZE, 1, 1))

    # 1. Initialize HDF5 file
    with h5py.File(OUTPUT_PATH, 'w') as f:
        dset_a = f.create_dataset('a', shape=(0, S, S), maxshape=(None, S, S), 
                                  dtype='float32', chunks=(BATCH_SIZE, S, S), compression="gzip")
        dset_u = f.create_dataset('u', shape=(0, S, S), maxshape=(None, S, S), 
                                  dtype='float32', chunks=(BATCH_SIZE, S, S), compression="gzip")

        pbar = tqdm(range(0, TOTAL_SAMPLES, BATCH_SIZE), desc="Generating HDF5 Data Pool")

        for i in pbar:
            current_batch_size = min(BATCH_SIZE, TOTAL_SAMPLES - i)
            if current_batch_size != BATCH_SIZE:
                F_batch = jnp.tile(F_SOURCE[jnp.newaxis, :, :], (current_batch_size, 1, 1))

            master_key, *subkeys = jax.random.split(master_key, current_batch_size + 1)
            subkeys = jnp.array(subkeys)

            # --- GPU Computation Section ---
            U_batch = vmap(lambda k: GRF_jax(ALPHA, TAU, S, k))(subkeys)
            a_batch = get_binary_a_jax(U_batch)
            u_batch = v_solve_gwf_jax(a_batch, F_batch, S)
            
            a_np = np.array(a_batch, dtype=np.float32)
            u_np = np.array(u_batch, dtype=np.float32)

            new_count = dset_a.shape[0] + current_batch_size
            dset_a.resize(new_count, axis=0)
            dset_u.resize(new_count, axis=0)

            dset_a[-current_batch_size:] = a_np
            dset_u[-current_batch_size:] = u_np

    print(f"\nTask Complete! Data pool saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()