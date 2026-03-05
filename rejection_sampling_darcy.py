import numpy as np
import matplotlib.pyplot as plt
import os
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from GRF import GRF
from solve_gwf import solve_gwf

# --- parameter setting ---
S = 128
ALPHA = 2
TAU = 3
N_TOTAL_TRIES = 1000000
THRESHOLD = 4e-4
OBS_GRID_SIZE = 16  # 16x16 grid
F_SOURCE = np.ones((S, S))

def get_binary_a(alpha, tau, s, rng):
    """
    Generates a binary field based on Gaussian Random Field (GRF).
    """
    norm_a = GRF(alpha, tau, s, rng=rng) 
    thresh_a = np.zeros((s, s))
    thresh_a[norm_a >= 0] = 12.0
    thresh_a[norm_a < 0] = 3.0
    return thresh_a

def get_sparse_indices(s, grid_size):
    """Returns coordinates for a uniform 16x16 observation grid."""
    idx = np.linspace(0, s-1, grid_size, dtype=int)
    xv, yv = np.meshgrid(idx, idx)
    return xv.flatten(), yv.flatten()

def single_rejection_task(sample_idx, alpha, tau, s, f, ref_obs_values, obs_idx, u_ref_full):
    """
    Worker task: Uses a unique seed per sample to ensure independent realizations.
    """
    rng = np.random.default_rng(seed=sample_idx)

    a_sampled = get_binary_a(alpha, tau, s, rng=rng)
    u_sampled = solve_gwf(a_sampled, f)
    
    obs_x, obs_y = obs_idx
    sampled_obs_values = u_sampled[obs_x, obs_y]
    
    sparse_mae = np.mean(np.abs(sampled_obs_values - ref_obs_values))
    accepted = sparse_mae <= THRESHOLD
    
    if accepted:
        full_field_abs_sum = np.sum(np.abs(u_sampled - u_ref_full))
        return True, sparse_mae, full_field_abs_sum, a_sampled, u_sampled
    else:
        return False, sparse_mae, None, None, None

def main():
    output_dir = 'rejection_sampling_results_binary'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/accepted_samples', exist_ok=True)

    # Initialize log file
    log_file_path = os.path.join(output_dir, 'sampling_log.txt')
    with open(log_file_path, 'w') as f_log:
        f_log.write("Sample_Index, MAE, Status\n")

    # 1. Generate Reference Ground Truth
    print("Generating binary reference truth...")
    rng_ref = np.random.default_rng(seed=0)
    a_ref = get_binary_a(ALPHA, TAU, S, rng=rng_ref)
    u_ref = solve_gwf(a_ref, F_SOURCE)
    
    obs_x, obs_y = get_sparse_indices(S, OBS_GRID_SIZE)
    ref_obs_values = u_ref[obs_x, obs_y]

    # 2. Plot Reference Truth and Observation points
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    im0 = ax[0].imshow(a_ref, cmap='coolwarm')
    ax[0].set_title("Reference Binary a(x)")
    plt.colorbar(im0, ax=ax[0])
    
    im1 = ax[1].imshow(u_ref, cmap='jet')
    ax[1].scatter(obs_y, obs_x, c='red', s=15, marker='x', label='Obs Points (16x16)')
    ax[1].set_title("Reference u(x) with Sparse Obs")
    ax[1].legend()
    plt.colorbar(im1, ax=ax[1])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/0_reference_truth_with_obs.png')
    plt.close()
    print(f"Reference plots saved to {output_dir}/0_reference_truth_with_obs.png")

    u_ref_full = u_ref 
    running_full_error_sum = 0.0  
    total_pixels = S * S          # 128 * 128 = 16384

    # 3. Parallel Sampling
    num_cpus = multiprocessing.cpu_count()
    print(f"Starting parallel sampling (Binary Mode) using {num_cpus} cores...")
    
    total_accepted = 0
    acceptance_history = []
    batch_size = 1000

    with ProcessPoolExecutor(max_workers=num_cpus) as executor:
        for batch_start in range(0, N_TOTAL_TRIES, batch_size):
            futures = [executor.submit(single_rejection_task, batch_start + i + 1, ALPHA, TAU, S, F_SOURCE, ref_obs_values, (obs_x, obs_y), u_ref_full) 
                       for i in range(batch_size)]
            
            batch_log_data = []
            for i, future in enumerate(futures):
                sample_idx = batch_start + i + 1
                is_ok, s_mae, f_abs_sum, a_acc, u_acc = future.result()
                
                status = "Accepted" if is_ok else "Rejected"
                batch_log_data.append(f"{sample_idx}, {s_mae:.8e}, {status}\n")
                
                if is_ok:
                    total_accepted += 1
                    running_full_error_sum += f_abs_sum
                    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
                    ax[0].imshow(a_acc, cmap='coolwarm')
                    ax[0].set_title(f"Accepted a (MAE: {s_mae:.2e})")
                    ax[1].imshow(u_acc, cmap='jet')
                    ax[1].set_title(f"Accepted u (Sample #{total_accepted})")
                    plt.savefig(f'{output_dir}/accepted_samples/take_{total_accepted}.png')
                    plt.close()

            
            with open(log_file_path, 'a') as f_log:
                f_log.writelines(batch_log_data)

            
            current_tries = batch_start + batch_size
            current_rate = (total_accepted / current_tries) * 100
            acceptance_history.append(current_rate)

            # Calculate overall full-field MAE (MAE over all pixels and all accepted cases)
            if total_accepted > 0:
                # Average error = Total error / (Number of samples * Pixels per sample)
                overall_full_field_mae = running_full_error_sum / (total_accepted * total_pixels)
                eval_str = f" | Full-Field MAE: {overall_full_field_mae:.8e}"
            else:
                eval_str = " | Full-Field MAE: N/A"
            
            if current_tries % 1000 == 0:
                print(f"tries: {current_tries:7d} | accepted: {total_accepted:5d} | current acceptance rate: {current_rate:.5f}%{eval_str}")

    # 4. Plot statistics charts
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(1, len(acceptance_history)+1) * batch_size, acceptance_history, color='tab:blue')
    plt.title("Acceptance Rate Convergence")
    plt.xlabel("Total Tries")
    plt.ylabel("Acceptance Rate (%)")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(f'{output_dir}/rejection_statistics.png')
    
    print(f"\n" + "="*40)
    print(f"Simulation Complete")
    print(f"Total Tries: {N_TOTAL_TRIES}")
    print(f"Total Accepted: {total_accepted}")
    print(f"Final Acceptance Rate: {current_rate:.6f}%")
    print(f"Log saved to: {log_file_path}")
    print("="*40)

if __name__ == "__main__":
    main()