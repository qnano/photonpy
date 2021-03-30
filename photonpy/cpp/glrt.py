
"""
Generalized likelihood-ratio test to detect spots vs background
"""
from .context import Context
import ctypes as ct
from .estimator import Estimator

import numpy as np
from scipy.stats import norm

def CreateEstimator(psf, ctx:Context):
    _GLRT_CreatePSF = ctx.smlm.lib.GLRT_CreatePSF
    _GLRT_CreatePSF.argtypes = [
            ct.c_void_p,
            ct.c_void_p
            ]
    _GLRT_CreatePSF.restype = ct.c_void_p

    inst = _GLRT_CreatePSF(psf.inst, ctx.inst if ctx else None)
    psf = Estimator(ctx, inst)
    return psf

def ComputeLogPFA(diag):
    ll_on = diag[:,0]
    ll_off = diag[:,1]
    
    #float Tg = 2.0f*(h1_ll - h0_ll);	// Test statistic (T_g)
	#float pfa = 2.0f * cdf(-sqrt(Tg)); // False positive probability (P_fa)

    Tg = 2*(ll_on-ll_off)
    Tg = np.maximum(0,Tg)
    logpfa = 2 * norm.logcdf(-np.sqrt(Tg))

    return logpfa
