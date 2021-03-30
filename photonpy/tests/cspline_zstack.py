
# Steps to run this:
# 
# 1. Download the software from "Real-time 3D single-molecule localization using experimental point spread functions"
#    https://www.nature.com/articles/nmeth.4661#Sec21
# 
# 2. Download the bead calibration stack from 
#    http://bigwww.epfl.ch/smlm/challenge2016/datasets/Tubulin-A647-3D/Data/data.html
# 
# 3. Use above software to generate the cspline calibration .mat file
#    Adjust the path and run this code
#
import numpy as np
import matplotlib.pyplot as plt

from photonpy.cpp.context import Context
from photonpy.cpp.cspline import CSplineCalibration, CSplineMethods
import napari

import time

# Change this to your cubic spline PSF calibration file..
def cspline_calib_fn():
    cspline_fn = 'cspline-nm-astig.mat'
    #cspline_fn = 'Tubulin-A647-cspline.mat'
    import os
    if not os.path.exists(cspline_fn):
        try:
            import urllib.request
            url=f'http://homepage.tudelft.nl/f04a3/{cspline_fn}'
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, cspline_fn)
            
            if not os.path.exists(cspline_fn):
                print('Skipping CSpline 3D PSF (no coefficient file found)')
                cspline_fn = None
        finally:
            ...
    
    return cspline_fn




with Context() as ctx:
    roisize= 20
    fn = 'C:/data/sfsf/sf3D/period_cal_12-16/pos1/x0y300(523)_summed_3Dcorr.mat'
    
    if False:
        filename = 'psfsim.zstack' 
        t0 = time.time()
        psf = CSplineMethods(ctx).CreatePSFFromFile(roisize, filename)
        calib = psf.calib
        t1 = time.time()
        print(f"CSpline time to compute coefficients: {t1-t0:.2f}")
    else:
        calib = CSplineCalibration.from_file_nmeth(fn)
        psf = CSplineMethods(ctx).CreatePSF_XYZIBg(roisize, calib)
    
    N = 200
    theta = np.repeat([[roisize/2,roisize/2,0,3000,0.1]], N, axis=0)
    theta[:,[0,1]] =roisize/2#  np.random.uniform(-2,2,size=(N,2))+roisize/2
    theta[:,2]= np.linspace(calib.zrange[0]*0.9,calib.zrange[1]*0.9,N)
    #theta[:,2] = np.linspace(-0.5,0.5, N)

    smp = psf.ExpectedValue(theta)
    fi = psf.FisherMatrix(theta)

    crlb = psf.CRLB(theta)
    pixelsize=52
    plt.figure()    
    plt.plot(theta[:,2], crlb[:,2] * 1000, label="Z")
    plt.plot(theta[:,2], crlb[:,0]  * pixelsize, label="X")
    plt.plot(theta[:,2], crlb[:,1] * pixelsize, label="Y")
    plt.legend()
    plt.ylabel('CRLB [nm]')
    plt.xlabel('Z position [um]')
    plt.show()
    
    plt.figure()
    #zpos = np.repeat(np.linspace(-1,1,50))
    estim_z = psf.Estimate(smp)[0][:,2]
    plt.plot(theta[:,2], estim_z, label="Estimated")
    plt.plot(theta[:,2], theta[:,2], label='Ground truth')
    plt.xlabel("Z position [um]")
    plt.ylabel("Z position [um]")
    plt.title('Estimated Z vs ground truth Z')
    
    with napari.gui_qt():
        napari.view_image(smp)
