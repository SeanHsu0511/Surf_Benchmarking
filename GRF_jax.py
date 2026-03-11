import jax
import jax.numpy as jnp
from jax.scipy.fft import idctn
from functools import partial  # 引入這個

# 使用 partial 預先設定好 jit 的參數，這是 JAX 官方最推薦的穩健寫法
@partial(jax.jit, static_argnums=(2,))
def GRF_jax(alpha, tau, s, key):
    """
    JAX 版本的 Gaussian Random Field 生成器
    """
    # 1. 隨機數生成
    xi = jax.random.normal(key, (s, s))
    
    # 2. 建立網格
    k = jnp.arange(s)
    k1, k2 = jnp.meshgrid(k, k)
    
    # 3. 計算譜系系數
    coef = (tau**(alpha - 1)) * (jnp.pi**2 * (k1**2 + k2**2) + tau**2)**(-alpha / 2)
    
    # 4. 頻域相乘
    L = s * coef * xi
    
    # 5. 強制零頻項為 0 (DC component)
    L = L.at[0, 0].set(0.0)
    
    # 6. 離散餘弦逆變換
    U = idctn(L, type=2, norm='ortho')
    
    return U