import numpy as np
from scipy.interpolate import RectBivariateSpline
import scipy.sparse as sp              
from scipy.sparse.linalg import spsolve 

def solve_gwf(coef, F):
    K = len(coef)
    
    coords1 = np.linspace(1/(2*K), (2*K-1)/(2*K), K)
    coords2 = np.linspace(0, 1, K)
    
    sp_coef = RectBivariateSpline(coords1, coords1, coef, kx=3, ky=3) 
    coef_v = np.clip(sp_coef(coords2, coords2), 3.0, 12.0)
    
    sp_F = RectBivariateSpline(coords1, coords1, F, kx=3, ky=3)
    F_v = sp_F(coords2, coords2)
    
    n = K - 2
    N = n * n
    
    rows, cols, data = [], [], []
    for j in range(1, K-1):
        for i in range(1, K-1):
            idx = (i-1) + (j-1)*n
            c_curr = coef_v[i, j]
            
            cw = (coef_v[i, j-1] + c_curr) / 2
            ce = (coef_v[i, j+1] + c_curr) / 2
            cs = (coef_v[i-1, j] + c_curr) / 2
            cn = (coef_v[i+1, j] + c_curr) / 2
            
            
            rows.append(idx); cols.append(idx); data.append(cw + ce + cs + cn)
            
            
            if j > 1:   rows.append(idx); cols.append(idx-n); data.append(-cw)
            if j < K-2: rows.append(idx); cols.append(idx+n); data.append(-ce)
            if i > 1:   rows.append(idx); cols.append(idx-1); data.append(-cs)
            if i < K-2: rows.append(idx); cols.append(idx+1); data.append(-cn)


    A = sp.coo_matrix((data, (rows, cols)), shape=(N, N)).tocsc()
    A = A * (K-1)**2
    
    f_vec = F_v[1:K-1, 1:K-1].flatten(order='F')
    
    p_vec = spsolve(A, f_vec)
    p_internal = p_vec.reshape((n, n), order='F')
    

    P_v = np.zeros((K, K))
    P_v[1:K-1, 1:K-1] = p_internal
    
    sp_P = RectBivariateSpline(coords2, coords2, P_v, kx=3, ky=3)
    P_final = sp_P(coords1, coords1).T
    
    return P_final