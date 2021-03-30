#!pip install photonpy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from photonpy.cpp.gaussian import GaussianPSFMethods
from photonpy.cpp.context import Context

import os
os.makedirs('rois', exist_ok=True)

font = {'family' : 'normal',
        'size'   : 22}

matplotlib.rc('font', **font)


with Context() as ctx:
    
    roisize = 20
    sigma = 1.8
    pixelsize = .05
    area = (roisize*pixelsize)**2
    print(f'Area: {area:.1f}')
    
    def makeroi(N, psf,i):
        pos = np.zeros((N, 4))
        
        pos[:, [0,1]] = np.random.uniform(1,roisize-2,size=(N,2))
        pos[:, 2] = 1500
        pos[:, 3] = 10

        rois = psf.ExpectedValue(pos)        
        smp = np.random.poisson(np.sum(rois,0))
        plt.figure()
        plt.imshow(smp, extent=[0,pixelsize*roisize,0,pixelsize*roisize])
        plt.ylabel('Y [um]')
        plt.xlabel('X [um]')
        
        plt.savefig(f"rois/{N}-emitters-{i}.svg")
        return smp
        
    g = GaussianPSFMethods(ctx)
    psf = g.CreatePSF_XYIBg(roisize, sigma, True)
    
    for i in range(10):
        makeroi(4, psf, i)
        makeroi(6, psf, i)
        makeroi(8, psf, i)
        makeroi(10, psf, i)
    
    