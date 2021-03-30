import numpy as np
import matplotlib.pyplot as plt

import photonpy.smlm.util as su
from photonpy.cpp.context import Context
import photonpy.cpp.gaussian as gaussian

from photonpy.cpp.simflux import SIMFLUX

mod = np.array([
           [0, 1.8, 0,  0.95, 0, 1/6],
           [1.9, 0, 0,0.95, 0, 1/6],
           [0, 1.8, 0,  0.95, 2*np.pi/3, 1/6],
           [1.9, 0, 0,0.95, 2*np.pi/3, 1/6],
           [0, 1.8,0,   0.95, 4*np.pi/3, 1/6],
           [1.9, 0, 0,0.95, 4*np.pi/3, 1/6]
          ])

with Context() as ctx:
    g = gaussian.GaussianPSFMethods(ctx)

    sigma=1.5
    roisize=10
    theta=[[roisize//2, roisize//2, 1000, 1]]
        
    s = SIMFLUX(ctx)
    
    psf = g.CreatePSF_XYIBg(roisize, sigma, True)
    sf_psf = s.CreateEstimator_Gauss2D(sigma, len(mod), roisize, len(mod), True)
    smp = sf_psf.GenerateSample(theta, constants=mod)
    estim,diag,traces = sf_psf.Estimate(smp,constants=mod)
    
    IBg = np.reshape(diag, (len(mod),4))
    print(f"Intensities (unmodulated): {IBg[:,0]}")
    
    ev = sf_psf.ExpectedValue(theta,constants=mod)
    crlb = sf_psf.CRLB(theta,constants=mod)
    print(f"Simflux CRLB: { crlb}")
    smp = np.random.poisson(ev)
    su.imshow_hstack(smp[0])

    deriv, mu = sf_psf.Derivatives(theta,constants=mod)
    for k in range(deriv.shape[1]):
         plt.figure()
         su.imshow_hstack(deriv[0,k])
         plt.title(f"Simflux psf derivative {sf_psf.param_names[k]}")

    crlb2,chisq = sf_psf.CRLBAndChiSquare(theta,smp,constants=mod)

    estim = psf.Estimate(smp.sum(1))[0]
    g_crlb = psf.CRLB(estim)
    
    print(f"Regular 2D Gaussian CRLB: {g_crlb}")
    
    estimates, perFrameFits, traces = sf_psf.Estimate(smp, constants=mod)
    
    IBg = perFrameFits[:,:2]
    IBg_crlb = perFrameFits[:,2:]
    
    