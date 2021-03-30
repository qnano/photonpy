import matplotlib.pyplot as plt
import numpy as np
import time

import photonpy.smlm.util as su
from photonpy.cpp.context import Context
import photonpy.cpp.gaussian as gaussian
from photonpy.cpp.calib import sCMOS_Calib

import photonpy.cpp.com as _com
import photonpy.cpp.phasor as _phasor

with Context() as ctx:
    g = gaussian.Gaussian(ctx)

    sigma=1.6
    roisize=7

    psf = g.CreatePSF_XYIBg(roisize, sigma, False)
    theta = [4,4,10000,10]

    plt.figure()
    img = psf.GenerateSample([theta])
    plt.figure()
    plt.set_cmap('inferno')
    plt.imshow(img[0])
    
    _phasor.Localize(img[0])
    
    com = _com.CreateEstimator(roisize,ctx)
    phasor = _phasor.CreateEstimator(roisize,ctx)
    
    com_estim = com.Estimate(img)[0]
    print(f"COM: {com_estim}")
    
    phasor_estim = phasor.Estimate(img)[0]
    print(f"Phasor: {phasor_estim}")
    