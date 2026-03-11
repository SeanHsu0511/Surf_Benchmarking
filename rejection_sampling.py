import os
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from solve_gwf_jax import solve_gwf_jax
from GRF_jax import GRF_jax
import h5py
from tqdm import tqdm

# --- Settings ---
S = 128
ALPHA = 2.0
TAU = 3.0
THRESHOLD = 4e-4
GT_SEED = 3        
POOL_PATH = 'data_pool_128x128.h5'
F_SOURCE = jnp.ones((S, S))

# --- Observation Mode Settings ---
# Options: 'fully_observed_low_res' or 'sparse_random'
OBS_MODE = 'sparse_random' 
NUM_RANDOM_SENSORS = 64 
U_RESOLUTION = 16 

def get_random_sensor_indices(s, num_sensors, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.choice(s * s, size=num_sensors, replace=False)
    x = indices // s
    y = indices % s
    return x, y

def apply_u_low_res(u_field, res, s=128):
    if res == s: return u_field
    factor = s // res
    if u_field.ndim == 3:
        n = u_field.shape[0]
        low_res = u_field.reshape(n, res, factor, res, factor).mean(axis=(2, 4))
        return np.repeat(np.repeat(low_res, factor, axis=-1), factor, axis=-2)
    else:
        low_res = u_field.reshape(res, factor, res, factor).mean(axis=(1, 3))
        return np.repeat(np.repeat(low_res, factor, axis=-1), factor, axis=-2)

def main():
    save_dir = f'results_seed_{GT_SEED}_{OBS_MODE}'
    os.makedirs(save_dir, exist_ok=True)

    # 1. 生成 Ground Truth
    print(f"Generating GT for Seed {GT_SEED} (Mode: {OBS_MODE})...")
    ref_key = jax.random.PRNGKey(GT_SEED)
    u_ref_grf = GRF_jax(ALPHA, TAU, S, ref_key)
    a_ref = np.array(jnp.where(u_ref_grf >= 0, 12.0, 3.0)) 
    u_ref_full = np.array(solve_gwf_jax(a_ref, np.array(F_SOURCE), S))

    # Define baseline observations and reference plots (subplot count depends on mode)
    n_ref_cols = 2 if OBS_MODE == 'sparse_random' else 3
    fig_ref, ax_ref = plt.subplots(1, n_ref_cols, figsize=(5 * n_ref_cols, 4))
    
    ax_ref[0].imshow(a_ref, cmap='coolwarm', origin='lower')
    ax_ref[0].set_title("Reference a(x)")
    
    if OBS_MODE == 'sparse_random':
        obs_x, obs_y = get_random_sensor_indices(S, NUM_RANDOM_SENSORS, seed=99)
        ref_obs = u_ref_full[obs_x, obs_y]
        ax_ref[1].imshow(u_ref_full, cmap='jet', origin='lower')
        ax_ref[1].scatter(obs_y, obs_x, c='red', s=15, marker='x', label='Sensors')
        ax_ref[1].set_title(f"Ref u(x) w/ Sensors (N={NUM_RANDOM_SENSORS})")
    else:
        ref_obs = apply_u_low_res(u_ref_full, U_RESOLUTION)
        ax_ref[1].imshow(ref_obs, cmap='jet', origin='lower')
        ax_ref[1].set_title(f"Ref u(x) Low-Res ({U_RESOLUTION}x{U_RESOLUTION})")
        ax_ref[2].imshow(u_ref_full, cmap='jet', origin='lower')
        ax_ref[2].set_title("Ref u(x) Full-Res")

    plt.tight_layout()
    plt.savefig(f'{save_dir}/0_reference_and_observation.png')
    plt.close()

    # Define baseline observation values
    if OBS_MODE == 'sparse_random':
        obs_x, obs_y = get_random_sensor_indices(S, NUM_RANDOM_SENSORS, seed=99)
        ref_obs = u_ref_full[obs_x, obs_y]
    else:
        ref_obs = apply_u_low_res(u_ref_full, U_RESOLUTION)

    # 2. Load data pool and perform "Rotation-Augmented" filtering
    accepted_info = [] 
    accepted_sparse_maes = [] 
    

    print(f"Opening data pool and starting rotation-augmented filtering ...")
    with h5py.File(POOL_PATH, 'r') as f:
        pool_a = f['a']
        pool_u = f['u']
        num_samples = pool_u.shape[0]
        step = 5000 
        
        for start in tqdm(range(0, num_samples, step), desc="Filtering"):
            end = min(start + step, num_samples)
            a_chunk = pool_a[start:end]
            u_chunk = pool_u[start:end]
            
            # Try four rotations (0, 90, 180, 270)
            for k in [0, 1, 2, 3]:
                # Rotate a counter-clockwise by k*90 degrees
                a_rot = np.rot90(a_chunk, k=k, axes=(1, 2))
                # Rotate u clockwise by k*90 degrees (Opposite direction due to Solver's transpose property)
                u_rot = np.rot90(u_chunk, k=-k, axes=(1, 2))

                

                if OBS_MODE == 'sparse_random':
                    sampled_obs = u_rot[:, obs_x, obs_y]
                    chunk_maes = np.mean(np.abs(sampled_obs - ref_obs), axis=1)
                else:
                    u_rot_low = apply_u_low_res(u_rot, U_RESOLUTION)
                    chunk_maes = np.mean(np.abs(u_rot_low - ref_obs), axis=(1, 2))
                
                mask = chunk_maes <= THRESHOLD
                rel_idx = np.where(mask)[0]
                for r in rel_idx:
                    accepted_info.append((a_rot[r], u_rot[r]))
                    accepted_sparse_maes.append(chunk_maes[r])

    total_acc = len(accepted_info)
    print(f"Filtering complete! Found {total_acc} samples (including rotation augmentation).")

    # 3. Save and Plot
    if total_acc > 0:
        final_a = np.array([item[0] for item in accepted_info])
        final_u = np.array([item[1] for item in accepted_info])

        np.savez_compressed(
            f'{save_dir}/accepted_data.npz',
            a_ref=a_ref, u_ref=u_ref_full,
            a=final_a, u=final_u
        )
        
        for i in range(min(total_acc, 10)):
            sparse_mae = accepted_sparse_maes[i]
            full_mae = np.mean(np.abs(final_u[i] - u_ref_full))
            
            n_cols = 2 if OBS_MODE == 'sparse_random' else 3
            fig, ax = plt.subplots(1, n_cols, figsize=(5.5 * n_cols, 4))
            
            ax[0].imshow(final_a[i], cmap='coolwarm', origin='lower')
            ax[0].set_title(f"Accepted a #{i+1}")
            
            if OBS_MODE == 'sparse_random':
                ax[1].imshow(final_u[i], cmap='jet', origin='lower')
                ax[1].scatter(obs_y, obs_x, c='red', s=12, marker='o', edgecolors='white')
                ax[1].set_title("u(x) with Sensors")
            else:
                ax[1].imshow(apply_u_low_res(final_u[i], U_RESOLUTION), cmap='jet', origin='lower')
                ax[1].set_title(f"Obs Low-Res")
                ax[2].imshow(final_u[i], cmap='jet', origin='lower')
                ax[2].set_title("Full-Res u(x)")
            
            plt.suptitle(f"Sample #{i+1} | Sparse MAE: {sparse_mae:.2e} | Full MAE: {full_mae:.2e}", 
                         fontsize=12, y=0.98)
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.savefig(f'{save_dir}/comparison_{i+1}.png')
            plt.close()
        print(f"Results and plots saved to {save_dir}")
    else:
        print("No samples found matching the threshold.。")

if __name__ == "__main__":
    main()






