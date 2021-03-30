"""
Test for SIMFLUX estimators with generic PSFs
"""
import numpy as np
import matplotlib.pyplot as plt

import photonpy.smlm.util as su
from photonpy.cpp.context import Context
import photonpy.cpp.gaussian as gaussian

from photonpy.cpp.simflux import SIMFLUX

mod = np.array([
     [0, 1.8, 0, 0.95, 0, 1/6],
     [1.9, 0, 0, 0.95, 0, 1/6],
     [0, 1.8, 0, 0.95, 2*np.pi/3, 1/6],
     [1.9, 0, 0, 0.95, 2*np.pi/3, 1/6],
     [0, 1.8, 0, 0.95, 4*np.pi/3, 1/6],
     [1.9, 0, 0, 0.95, 4*np.pi/3, 1/6]
])


def plot_traces(traces, theta, axis=0):
    plt.figure()
    for k in range(len(traces)):
        plt.plot(np.arange(len(traces[k])),traces[k][:,axis])
        plt.plot([len(traces[k])],[theta[k,axis]],'o')
        
with Context(debugMode=False) as ctx:
    g = gaussian.Gaussian(ctx)

    roisize=10
    sf_theta=[[roisize//2, roisize//2, 0, 10000, 5]]
    sf_theta=np.repeat(sf_theta,20,0)
    
    psf = g.CreatePSF_XYZIBg(roisize, gaussian.Gauss3D_Calibration(), cuda=True)

    spot_mod = mod.reshape(len(mod)*6)[None].repeat(len(sf_theta),0)

    s = SIMFLUX(ctx)
    sum_estim = s.CreateEstimator(psf, len(mod), False)
    sf_estim = s.CreateEstimator(psf, len(mod), True)

    ev = sf_estim.ExpectedValue(sf_theta, constants=spot_mod)  
    sf_smp = np.random.poisson(ev)
    su.imshow_hstack(sf_smp[0])
    
    sf_result,diag,sf_traces = sf_estim.Estimate(sf_smp,constants=spot_mod)
    sum_result,diag,sum_traces = sum_estim.Estimate(sf_smp)

    deriv, mu = sf_estim.Derivatives(sf_theta,constants=spot_mod)
    for k in range(deriv.shape[1]):
         plt.figure()
         su.imshow_hstack(deriv[0,k])
         plt.title(f"Simflux psf derivative {sf_estim.param_names[k]}")

    crlb,chisq = sf_estim.CRLBAndChiSquare(sf_theta, sf_smp,constants=spot_mod)
    
    plot_traces(sf_traces, sf_result, 2)
    