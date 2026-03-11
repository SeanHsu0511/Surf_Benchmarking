import jax
import jax.numpy as jnp
from jax.scipy.sparse.linalg import cg
from functools import partial
import numpy as np
from scipy.interpolate import RectBivariateSpline

# 強制開啟雙精度
from jax import config
config.update("jax_enable_x64", True)

# --- 1. 先定義矩陣生成函式 ---
def get_interp_matrix(K, to_node=True):
    coords_center = np.linspace(1/(2*K), (2*K-1)/(2*K), K)
    coords_node = np.linspace(0, 1, K)
    I = np.eye(K)
    M = np.zeros((K, K))
    old = coords_center if to_node else coords_node
    new = coords_node if to_node else coords_center
    for i in range(K):
        z = np.tile(I[:, i:i+1], (1, 2))
        sp = RectBivariateSpline(old, np.array([0, 1]), z, kx=3, ky=1)
        M[:, i] = sp(new, np.array([0.5])).flatten()
    return jnp.array(M)

# --- 2. 執行預計算 (這部分要在 solver 被呼叫前完成) ---
S_FIXED = 128
M_TNODE_J = get_interp_matrix(S_FIXED, to_node=True)   # Center -> Node
M_TNODE_T = M_TNODE_J.T                                # 預計算轉置以加速
M_TCNTR_J = get_interp_matrix(S_FIXED, to_node=False)  # Node -> Center
M_TCNTR_T = M_TCNTR_J.T                                # 預計算轉置

# --- 3. 定義 Solver ---
@partial(jax.jit, static_argnums=(2,))
def solve_gwf_jax(coef_raw, F_raw, K):
    n = K - 2
    dx_inv_sq = (K - 1) ** 2

    # 使用預計算好的 JAX 矩陣，避免重複執行 CPU 邏輯
    coef_v = M_TNODE_J @ coef_raw @ M_TNODE_T
    coef_v = jnp.clip(coef_v, 3.0, 12.0)
    F_v = M_TNODE_J @ F_raw @ M_TNODE_T
    
    f_vec = F_v[1:-1, 1:-1].flatten(order='F') 

    # 算子係數與位移邏輯 (維持你確認過的 e-17 物理邏輯)
    cs = (coef_v[:-2, 1:-1] + coef_v[1:-1, 1:-1]) / 2.0
    cn = (coef_v[2:, 1:-1] + coef_v[1:-1, 1:-1]) / 2.0
    cw = (coef_v[1:-1, :-2] + coef_v[1:-1, 1:-1]) / 2.0
    ce = (coef_v[1:-1, 2:] + coef_v[1:-1, 1:-1]) / 2.0
    diag = (cw + ce + cs + cn) * dx_inv_sq

    def laplacian_op(p_flat):
        p = p_flat.reshape((n, n), order='F')
        p_s = jnp.vstack([jnp.zeros((1, n)), p[:-1, :]])
        p_n = jnp.vstack([p[1:, :], jnp.zeros((1, n))])
        p_w = jnp.hstack([jnp.zeros((n, 1)), p[:, :-1]])
        p_e = jnp.hstack([p[:, 1:], jnp.zeros((n, 1))])
        val = diag * p - (cs*p_s + cn*p_n + cw*p_w + ce*p_e) * dx_inv_sq
        return val.flatten(order='F')

    p_vec, _ = cg(laplacian_op, f_vec, tol=1e-14, maxiter=10000)
    p_internal = p_vec.reshape((n, n), order='F')
    P_node = jnp.pad(p_internal, ((1, 1), (1, 1)), mode='constant')
    
    # 逆插值回到格心，對齊 MATLAB
    P_final = M_TCNTR_J @ P_node @ M_TCNTR_T
    
    return P_final.T


# 定義向量化版本：in_axes=(0, 0, None) 代表 a 和 F 有 batch 維度，S 是常數
v_solve_gwf_jax = jax.vmap(solve_gwf_jax, in_axes=(0, 0, None))