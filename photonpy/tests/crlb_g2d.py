import matplotlib.pyplot as plt
import numpy as np
from photonpy.cpp.context import Context
import photonpy.cpp.gaussian as gaussian


with Context(debugMode=False) as ctx:
    g = gaussian.Gaussian(ctx)
      
    sigma=2
    roisize=16
    numspots = 10000
    theta=[[roisize/2, roisize/2, 1000, 5]]
    theta=np.repeat(theta,numspots,0)
    theta[:,3]  = np.linspace(0.001, 30,numspots)
      
    useCuda=True
      
    #    with g.CreatePSF_XYZIBg(roisize, calib, True) as psf:
    g_psf = g.CreatePSF_XYIBg(roisize, sigma, useCuda)
          
    crlb = g_psf.CRLB(theta)
    
    pixelsize=65
    
    plt.figure()
    plt.plot(theta[:,3],crlb[:,0]*pixelsize, label="CRLB X [nm]")
    plt.xlabel("Background [photons/pixel]")
    plt.legend()
    
    theta[:,3] = 20
    theta[:,2]  = np.logspace(np.log(50)/np.log(10), np.log(1000)/np.log(10),numspots)
    crlb = g_psf.CRLB(theta)

    plt.figure()
    plt.plot(theta[:,2],crlb[:,0]*pixelsize, label="CRLB X [nm]")
    plt.xlabel("Photon count [photons]")
    plt.legend()
    