import os
import numpy as np
import jax
import jax.numpy as jnp
from jax import vmap, jit
import matplotlib.pyplot as plt
from tqdm import tqdm

from GRF_jax import GRF_jax
from solve_gwf_jax import solve_gwf_jax, v_solve_gwf_jax

# --- settings ---
S = 128
ALPHA = 2.0
TAU = 3.0
N_TOTAL_TRIES = 1000000
BATCH_SIZE = 128        
THRESHOLD = 2e-4
OBS_GRID_SIZE = 8       
OBS_PROPORTION = 0.8    
RESOLUTION = 128 # 新增 hyperparameter
F_SOURCE = jnp.ones((S, S))

def get_sparse_indices_jax(s, grid_size, proportion=0.8):
    margin = (1.0 - proportion) / 2.0
    start = int(s * margin)
    end = int(s * (1.0 - margin))
    idx = jnp.linspace(start, end - 1, grid_size, dtype=jnp.int32)
    xv, yv = jnp.meshgrid(idx, idx, indexing='ij')
    return xv.flatten(), yv.flatten()

@jit
def get_binary_a_jax(U):
    return jnp.where(U >= 0, 12.0, 3.0)

def apply_resolution_limit(a_field, res):
    """
    將 S*S 的場降級為 res*res 的解析度，再放回 S*S。
    例如: S=128, res=64, 則每個 2*2 的區域數值會相同。
    """
    s = a_field.shape[-1] # 處理 (S, S) 或 (Batch, S, S)
    factor = s // res
    
    # 1. 下採樣：計算平均值 (這裏用簡單的 reshape + mean)
    # 假設 a_field 形狀為 (..., S, S)
    reshaped = a_field.reshape(a_field.shape[:-2] + (res, factor, res, factor))
    low_res = reshaped.mean(axis=(-3, -1))
    
    # 2. 上採樣：利用 repeat 放大回原尺寸
    # 這樣 solve_gwf 拿到的輸入依然是 (S, S)，但特徵是低解析度的
    high_res_blocked = jnp.repeat(jnp.repeat(low_res, factor, axis=-1), factor, axis=-2)
    
    return high_res_blocked

def main():
    output_dir = 'rejection_sampling_jax_binary'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/accepted_samples', exist_ok=True)

    # 1. 生成 Reference Ground Truth
    print("正在生成 Reference Truth...")
    ref_key = jax.random.PRNGKey(3)
    u_ref_grf = GRF_jax(ALPHA, TAU, S, ref_key)
    a_ref_raw = get_binary_a_jax(u_ref_grf)
    a_ref = apply_resolution_limit(a_ref_raw, RESOLUTION) # 套用解析度限制
    u_ref = jnp.array(solve_gwf_jax(np.array(a_ref), np.array(F_SOURCE), S))

    obs_x, obs_y = get_sparse_indices_jax(S, OBS_GRID_SIZE, proportion=OBS_PROPORTION)
    ref_obs_values = u_ref[obs_x, obs_y]

    # 繪製 Reference 圖 (略，同前)
    fig, ax = plt.subplots(1, 2, figsize=(12, 5))
    ax[0].imshow(a_ref, cmap='coolwarm', origin='lower')
    ax[0].set_title("Reference a(x)")
    im1 = ax[1].imshow(u_ref, cmap='jet', origin='lower')
    ax[1].scatter(obs_y, obs_x, c='red', s=20, marker='x', label='Obs Points')
    ax[1].set_title(f"Reference u(x) ({OBS_PROPORTION*100:.0f}% center)")
    ax[1].legend()
    plt.colorbar(im1, ax=ax[1])
    plt.savefig(f'{output_dir}/0_reference_truth.png')
    plt.close()

    # 2. 初始化統計
    total_accepted = 0
    accepted_a = []
    accepted_u = []
    master_key = jax.random.PRNGKey(123)

    # 3. 使用 tqdm 監控進度
    # total 設定為總批次數
    pbar = tqdm(range(0, N_TOTAL_TRIES, BATCH_SIZE), desc="抽樣進度", unit="batch")

    # 在 main() 裡面，迴圈開始前準備好 F_batch
    F_batch = jnp.tile(F_SOURCE[jnp.newaxis, :, :], (BATCH_SIZE, 1, 1))

    for batch_idx in pbar:
        master_key, *subkeys = jax.random.split(master_key, BATCH_SIZE + 1)
        subkeys = jnp.array(subkeys)

        # 1. 全 GPU 生成與求解
        U_batch = vmap(lambda k: GRF_jax(ALPHA, TAU, S, k))(subkeys)
        a_batch_raw = get_binary_a_jax(U_batch)
        # 這裡使用 vmap 或直接對 batch 維度處理
        a_batch = apply_resolution_limit(a_batch_raw, RESOLUTION)
        u_batch = v_solve_gwf_jax(a_batch, F_batch, S) 

        # 2. 判定 (仍在 GPU)
        sampled_obs = u_batch[:, obs_x, obs_y]
        sparse_maes = jnp.mean(jnp.abs(sampled_obs - ref_obs_values), axis=1)
        accepted_mask = sparse_maes <= THRESHOLD
        
        # 3. 處理被接受的樣本
        accepted_indices = np.where(accepted_mask)[0]
        for idx in accepted_indices:
            total_accepted += 1
            
            # --- 修正處：定義 a_acc 與 u_acc ---
            a_acc = a_batch[idx]
            u_acc = u_batch[idx]
            s_mae = sparse_maes[idx]
            full_mae = jnp.mean(jnp.abs(u_acc - u_ref))
            
            # 轉換為 numpy 供儲存與繪圖
            a_acc_np = np.array(a_acc)
            u_acc_np = np.array(u_acc)
            
            accepted_a.append(a_acc_np)
            accepted_u.append(u_acc_np)
            
            # 繪圖 (前 100 張)
            if total_accepted <= 100:
                plt.figure(figsize=(10, 4))
                plt.subplot(121)
                plt.imshow(a_acc_np, cmap='coolwarm', origin='lower')
                plt.title(f"Accepted a #{total_accepted}")
                plt.subplot(122)
                plt.imshow(u_acc_np, cmap='jet', origin='lower')
                plt.title(f"Sparse MAE: {s_mae:.2e}\nFull-MAE: {full_mae:.2e}")
                plt.savefig(f'{output_dir}/accepted_samples/sample_{total_accepted}.png')
                plt.close()

        # 更新進度條右側的資訊
        current_tries = batch_idx + BATCH_SIZE
        acc_rate = (total_accepted / current_tries) * 100
        pbar.set_postfix({
            "Accepted": total_accepted,
            "Rate": f"{acc_rate:.4f}%"
        })

    # 4. 最終儲存
    if accepted_a:
        # 將 list 轉換為 numpy array
        final_a_samples = np.array(accepted_a)
        final_u_samples = np.array(accepted_u)
        
        save_path = f'{output_dir}/accepted_data.npz'
        np.savez_compressed(
            save_path, 
            a_ref=np.array(a_ref),   # 儲存 GT 係數場
            u_ref=np.array(u_ref),   # 儲存 GT 解場
            a=final_a_samples,       # 儲存所有被接受的 a
            u=final_u_samples        # 儲存所有被接受的 u
        )
        
        print(f"\n--- 任務完成 ---")
        print(f"成功儲存 {total_accepted} 組樣本至 {save_path}")
        print(f"包含 GT 數據: a_ref, u_ref")
        print(f"包含樣本數據: a (shape: {final_a_samples.shape}), u (shape: {final_u_samples.shape})")

if __name__ == "__main__":
    main()