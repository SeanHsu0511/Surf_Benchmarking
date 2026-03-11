import jax
import jax.numpy as jnp
from jax.scipy.fft import idctn
from functools import partial  

@partial(jax.jit, static_argnums=(2,))
def GRF_jax(alpha, tau, s, key):
    """
    Gaussian Random Field generator in jax version
    """
    # 1. Random number generation
    xi = jax.random.normal(key, (s, s))
    
    # 2. Grid generation
    k = jnp.arange(s)
    k1, k2 = jnp.meshgrid(k, k)
    
    # 3. Compute spectral coefficients
    coef = (tau**(alpha - 1)) * (jnp.pi**2 * (k1**2 + k2**2) + tau**2)**(-alpha / 2)
    
    # 4. Frequency domain multiplication
    L = s * coef * xi
    
    # 5. Force DC component (zero frequency) to 0.0
    L = L.at[0, 0].set(0.0)
    
    # 6. Inverse Discrete Cosine Transform (IDCT)
    U = idctn(L, type=2, norm='ortho')
    
    return U