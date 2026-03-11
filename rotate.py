import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from solve_gwf_jax import solve_gwf_jax
from GRF_jax import GRF_jax

# --- Settings ---
S = 128
ALPHA = 2.0
TAU = 3.0
F_SOURCE = jnp.ones((S, S))

def verify_rotational_symmetry():
    # 1. 生成原始的 a 和解出原始的 u
    print("正在生成原始場與基準解...")
    seed = 42
    ref_key = jax.random.PRNGKey(seed)
    u_grf = GRF_jax(ALPHA, TAU, S, ref_key)
    a_orig = np.array(jnp.where(u_grf >= 0, 12.0, 3.0))
    u_orig = np.array(solve_gwf_jax(a_orig, np.array(F_SOURCE), S))

    angles = [90, 180, 270]
    fig, axes = plt.subplots(len(angles), 3, figsize=(15, 12))

    print(f"{'Angle':>10} | {'MAE (Rotated u vs Solve(Rotated a))':>40}")
    print("-" * 55)

    for i, angle in enumerate(angles):
        k = angle // 90  # np.rot90 的參數
        
        # 方法 A: 先旋轉 a，再丟進 solver 求解
        a_rotated = np.rot90(a_orig, k=k)
        u_from_rotated_a = np.array(solve_gwf_jax(a_rotated, np.array(F_SOURCE), S))
        
        # 方法 B: 先求解，再把解 u 旋轉
        u_rotated_after_solve = np.rot90(u_orig, k=-k)
        
        # 計算誤差
        mae = np.mean(np.abs(u_from_rotated_a - u_rotated_after_solve))
        print(f"{angle:>10}° | {mae:>40.4e}")

        # 視覺化
        # 第一欄: 旋轉後的 a
        axes[i, 0].imshow(a_rotated, cmap='coolwarm', origin='lower')
        axes[i, 0].set_title(f"Rotated a ({angle}°)")
        
        # 第二欄: 方法 A 的結果
        axes[i, 1].imshow(u_from_rotated_a, cmap='jet', origin='lower')
        axes[i, 1].set_title(f"Solve(Rotated a)")
        
        # 第三欄: 誤差圖 (Difference Map)
        diff = np.abs(u_from_rotated_a - u_rotated_after_solve)
        im = axes[i, 2].imshow(diff, cmap='inferno', origin='lower')
        axes[i, 2].set_title(f"Abs Diff (MAE: {mae:.2e})")
        plt.colorbar(im, ax=axes[i, 2])

    plt.tight_layout()
    plt.savefig('rotational_symmetry_test.png')
    plt.show()

if __name__ == "__main__":
    verify_rotational_symmetry()