import numpy as np
from scipy.fft import idctn 

def GRF(alpha, tau, s, rng=None):
    if rng is None:
        rng = np.random.default_rng()
    xi = rng.standard_normal((s, s))
    
    k = np.arange(s)
    k1, k2 = np.meshgrid(k, k)
    
    coef = (tau**(alpha - 1)) * (np.pi**2 * (k1**2 + k2**2) + tau**2)**(-alpha / 2)
    
    L = s * coef * xi
    L[0, 0] = 0 
    
    U = idctn(L, type=2, norm='ortho')
    
    return U