import matplotlib.pyplot as plt
import numpy as np
import os
import time

import photonpy.smlm.util as su
from photonpy.cpp.context import Context
import photonpy.cpp.gaussian as gaussian

from photonpy.cpp.cspline import CSplineMethods,CSplineCalibration
from photonpy.cpp.simflux import SIMFLUX
from photonpy.cpp.estimator import Estimator
import photonpy.cpp.com  as com
import photonpy.cpp.phasor as phasor
import tqdm


def plot_traces(traces, theta, psf, axis):
    plt.figure()
    for k in range(len(traces)):
        plt.plot(np.arange(len(traces[k])),traces[k][:,axis])
        plt.plot([len(traces[k])],[theta[k,axis]],'o')
    plt.title(f"Axis {axis}[{psf.param_names[axis]}]")

def estimate_precision(psf:Estimator, estimate_fn, thetas, photons):
    prec = np.zeros((len(photons),thetas.shape[1]))
    bias = np.zeros((len(photons),thetas.shape[1]))
    crlb = np.zeros((len(photons),thetas.shape[1]))
    
    Iidx = psf.ParamIndex('I')
                
    com_estim = com.CreateEstimator(psf.sampleshape[-1], psf.ctx)
    
    for i in tqdm.trange(len(photons)):
        thetas_ = thetas*1
        thetas_[:, Iidx] = photons[i]
        
        roipos = np.random.randint(0,20,size=(len(thetas_), psf.indexdims))
        roipos[:,0] = 0
        smp = psf.GenerateSample(thetas_,roipos=roipos)
        roisize = smp.shape[-1]

        estim,diag,traces = estimate_fn(smp,roipos=roipos)
            
        crlb_ = psf.CRLB(thetas_,roipos=roipos)
        err = estim-thetas_
        prec[i] = np.std(err,0)
        bias[i] = np.mean(err,0)
        crlb[i] = np.mean(crlb_,0)

#    print(f'sigma bias: {bias[:,4]}')        
    return prec,bias,crlb

pitch = 221/65
k = 2*np.pi/pitch

mod = np.array([
           [0, k, 0,  0.95, 0, 1/6],
           [k, 0, 0, 0.95, 0, 1/6],
           [0, k, 0,   0.95, 2*np.pi/3, 1/6],
           [k, 0, 0, 0.95, 2*np.pi/3, 1/6],
           [0, k, 0,  0.95, 4*np.pi/3, 1/6],
           [k, 0, 0, 0.95, 4*np.pi/3, 1/6]
          ])

def cspline_calib_fn():
    cspline_fn = 'cspline-nm-astig.mat'
    #cspline_fn = "C:/data/beads/Tubulin-A647-cspline.mat"
    if not os.path.exists(cspline_fn):
        try:
            import urllib.request
            url='http://homepage.tudelft.nl/f04a3/Tubulin-A647-cspline.mat'
            print(f"Downloading {url}")
            urllib.request.urlretrieve(url, cspline_fn)
            
            if not os.path.exists(cspline_fn):
                print('Skipping CSpline 3D PSF (no coefficient file found)')
                cspline_fn = None
        finally:
            ...
    
    return cspline_fn

    

calib = gaussian.Gauss3D_Calibration()#.from_file('../../data/simulated_as_gaussian.npy')

#calib = gaussian.Gauss3D_Calibration()#.from_file('../../data/simulated_as_gaussian.npy')

with Context(debugMode=False) as ctx:
    g = gaussian.Gaussian(ctx)
      
    sigma=2.5
    roisize=16
    bg=10
    numspots = 500
    theta=[[roisize/2, roisize/2, 10009, bg]]
    theta=np.repeat(theta,numspots,0)
    theta[:,[0,1]] += np.random.uniform(-pitch/2,pitch/2,size=(numspots,2))
      
    useCuda=True
      
    #    with g.CreatePSF_XYZIBg(roisize, calib, True) as psf:
    g_psf = g.CreatePSF_XYIBg(roisize, sigma, useCuda)
    g_psf.SetLevMarParams(-10, 40)
    g_s_psf = g.CreatePSF_XYIBgSigma(roisize, sigma+1, useCuda)
    g_s_psf.SetLevMarParams(1e-5,50)
    g_sxy_psf = g.CreatePSF_XYIBgSigmaXY(roisize, [sigma+1,sigma+1], useCuda)
    g_z_psf = g.CreatePSF_XYZIBg(roisize, calib, useCuda)
    g_z_psf.SetLevMarParams(1e-15, 30)
    
    com_psf = com.CreateEstimator(roisize, ctx)
    
    theta_as=np.zeros((numspots,5)) # x,y,z,I,bg
    theta_as[:,[0,1]]=theta[:,[0,1]]
    theta_as[:,2] = np.random.uniform(-0.3,0.3,size=numspots)
    theta_as[:,4]=bg
    
    
    theta_sig=np.zeros((numspots,6))
    theta_sig[:,0:4]=theta
    theta_sig[:,[4,5]]=sigma
    theta_sig[:,[4,5]] += np.random.uniform(-0.5,0.5,size=(numspots,2))
        
    photons = np.logspace(2, 5, 20)
    
    data = [
        (g_psf, "2D Gaussian", estimate_precision(g_psf, g_psf.Estimate, theta, photons)),
        (g_z_psf, 'Astig Gaussian',estimate_precision(g_z_psf, g_z_psf.Estimate, theta_as, photons))
            ]
    
    cspline_fn = cspline_calib_fn()
    if cspline_fn is not None:
        print('CSpline 3D PSF:')
        cs_calib = CSplineCalibration.from_file_nmeth(cspline_fn)
        cs_psf = CSplineMethods(ctx).CreatePSF_XYZIBg(roisize, cs_calib, CSplineMethods.FlatBg)
        cs_psf.SetLevMarParams(1e-18, 50)
        
        theta_cs = theta_as*1
        theta_cs[:,2] = np.linspace(-0.05,0.05, numspots)
       
        data.append((cs_psf, "Cubic spline PSF", estimate_precision(cs_psf, cs_psf.Estimate, theta_cs, photons)))
        

    axes=['x']
    axes_unit=['pixels', 'photons','photons/pixel']
    axes_scale=[1, 1, 1, 1]
    for i,ax in enumerate(axes):
        plt.figure(dpi=150,figsize=(12,8))
        for psf,name,(prec,bias,crlb) in data:
            ai = psf.ParamIndex(ax)
            line, = plt.gca().plot(photons,axes_scale[i]*prec[:,ai],label=f'Precision {name}')
            plt.plot(photons,axes_scale[i]*crlb[:,ai],label=f'CRLB {name}', color=line.get_color(), linestyle='--')

        plt.title(f'{ax} axis')
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel('Signal intensity [photons]')
        plt.ylabel(f"{ax} [{axes_unit[i]}]")
        plt.grid(True)
        plt.legend()

        plt.figure()
        for psf,name,(prec,bias,crlb) in data:
            ai = psf.ParamIndex(ax)
            plt.plot(photons,bias[:,ai],label=f'Bias {name}')

        plt.title(f'{ax} axis')
        plt.grid(True)
        plt.xscale("log")
        plt.legend()
        plt.show()

