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
TOTAL_SAMPLES = 100000  # 預計生成的樣本總數
BATCH_SIZE = 128
F_SOURCE = jnp.ones((S, S))
OUTPUT_PATH = 'data_pool_128x128.h5'

@jit
def get_binary_a_jax(U):
    return jnp.where(U >= 0, 12.0, 3.0)

def main():
    master_key = jax.random.PRNGKey(888)
    
    # 準備恆定的 F 矩陣
    F_batch = jnp.tile(F_SOURCE[jnp.newaxis, :, :], (BATCH_SIZE, 1, 1))

    # 1. 建立 HDF5 檔案
    with h5py.File(OUTPUT_PATH, 'w') as f:
        # 建立可擴展的 Dataset
        # shape=(0, S, S) 代表初始長度為 0
        # maxshape=(None, S, S) 代表第一個維度可以無限增長
        # chunks=True 啟用分塊存儲，對讀取效能與壓縮非常重要
        dset_a = f.create_dataset('a', shape=(0, S, S), maxshape=(None, S, S), 
                                  dtype='float32', chunks=(BATCH_SIZE, S, S), compression="gzip")
        dset_u = f.create_dataset('u', shape=(0, S, S), maxshape=(None, S, S), 
                                  dtype='float32', chunks=(BATCH_SIZE, S, S), compression="gzip")

        pbar = tqdm(range(0, TOTAL_SAMPLES, BATCH_SIZE), desc="HDF5 數據池生成中")

        for i in pbar:
            # 確保不會生成超過 TOTAL_SAMPLES
            current_batch_size = min(BATCH_SIZE, TOTAL_SAMPLES - i)
            if current_batch_size != BATCH_SIZE:
                F_batch = jnp.tile(F_SOURCE[jnp.newaxis, :, :], (current_batch_size, 1, 1))

            master_key, *subkeys = jax.random.split(master_key, current_batch_size + 1)
            subkeys = jnp.array(subkeys)

            # --- GPU 運算部分 ---
            U_batch = vmap(lambda k: GRF_jax(ALPHA, TAU, S, k))(subkeys)
            a_batch = get_binary_a_jax(U_batch)
            u_batch = v_solve_gwf_jax(a_batch, F_batch, S)
            
            # 將結果轉為 NumPy (釋放 VRAM 並準備寫入 RAM/Disk)
            a_np = np.array(a_batch, dtype=np.float32)
            u_np = np.array(u_batch, dtype=np.float32)

            # --- HDF5 增量寫入部分 ---
            # 1. 擴展 Dataset 大小
            new_count = dset_a.shape[0] + current_batch_size
            dset_a.resize(new_count, axis=0)
            dset_u.resize(new_count, axis=0)

            # 2. 寫入最新的數據到末尾
            dset_a[-current_batch_size:] = a_np
            dset_u[-current_batch_size:] = u_np

            # 這裡不需要手動空 list，因為 a_np, u_np 在下一次迴圈會被覆蓋
            # Python 的垃圾回收機制會自動處理

    print(f"\n任務完成！數據池已儲存至 {OUTPUT_PATH}")

if __name__ == "__main__":
    main()