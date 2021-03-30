import matplotlib.pyplot as plt
import numpy as np
import time

from photonpy.cpp.context import Context
import photonpy.cpp.gaussian as gaussian
import photonpy.cpp.cspline as cspline

from photonpy.cpp.estimator import Estimator
from photonpy.cpp.estim_queue import EstimQueue

import photonpy.cpp.glrt as glrt
import photonpy.cpp.com as com
import photonpy.cpp.phasor as phasor

from photonpy.cpp.simflux import SIMFLUX

from photonpy.cpp.calib import sCMOS_Calib

import os

as_calib = gaussian.Gauss3D_Calibration()

sf_mod = np.array([
     [0, 1.8, 0, 0.95, 0, 1/6],
     [1.9, 0, 0, 0.95, 0, 1/6],
     [0, 1.8, 0, 0.95, 2*np.pi/3, 1/6],
     [1.9, 0, 0, 0.95, 2*np.pi/3, 1/6],
     [0, 1.8, 0, 0.95, 4*np.pi/3, 1/6],
     [1.9, 0, 0, 0.95, 4*np.pi/3, 1/6]
])

def sfsf_offsets():
    offsets = np.zeros((len(sf_mod),2))
    ang = np.linspace(0,2*np.pi, len(sf_mod), endpoint=False)
    R = 4
    offsets[:,0] = R*np.cos(ang)
    offsets[:,1] = R*np.sin(ang)
    return offsets


def test_queue_output(ctx: Context, psf:Estimator, theta):
    numspots = 5
    theta_ = np.repeat(theta,numspots,axis=0)
    smp = np.random.poisson(psf.ExpectedValue(theta_))
    estim1 = psf.Estimate(smp)[0]
    
    q = EstimQueue(psf, batchSize=4, numStreams=3)
    q.Schedule(smp, ids=np.arange(numspots))
    q.WaitUntilDone()
    results = q.GetResults()
    results.SortByID()
    print(estim1)
    print(results.estim)
    print(results.ids)
    
    assert( np.sum( np.abs(results.estim-estim1) ) < 0.01)
    
    
def test_psf_speed(ctx: Context, smp_psf:Estimator, est_psf:Estimator,
                   theta, batchSize=1024*4,repeats=1,nstreams=4, binsize=10000):
    img = smp_psf.ExpectedValue(theta)
    smp = np.random.poisson(img)
    plt.figure()
    if len(smp[0].shape)>2: 
        plt.imshow(np.concatenate(smp[0],-1))
    else:
        plt.imshow(smp[0])

    queue = EstimQueue(est_psf, batchSize=batchSize, numStreams=nstreams)
    n = binsize
    if ctx.smlm.debugMode:
        n = 200
    repd = np.ascontiguousarray(np.repeat(smp,n,axis=0),dtype=np.float32)
    initial = np.ascontiguousarray(np.repeat(np.array(theta)*1.05,n,axis=0),dtype=np.float32)

    t0 = time.time()
    total = 0
    for i in range(repeats):
        queue.Schedule(repd, initial=initial)
        results = queue.GetResults()
        total += n
        
    queue.Flush()
    while not queue.IsIdle():
        time.sleep(0.05)

    results = queue.GetResults(getSampleData=True)
#    print(results.CRLB())
    t1 = time.time()
    
    queue.Destroy()
            
    print(f"Finished. Processed {total} in {t1-t0:.2f} s. {total/(t1-t0):.1f} spots/s")


with Context(debugMode=False) as ctx:
    sigma=1.5
    w = 512
    roisize=16
    theta=[[roisize//2, roisize//2, 1000, 5]]
    g_api = gaussian.Gaussian(ctx)
    sf_api = SIMFLUX(ctx)
    psf = g_api.CreatePSF_XYIBg(roisize, sigma, True)
    #scmos = sCMOS_Calib(ctx, np.zeros((w,w)), np.ones((w,w)), np.ones((w,w))*5)
    #psf_sc = g_api.CreatePSF_XYIBg(roisize, sigma, True, scmos)
            
    print('Phasor:')
    phasor_est= phasor.CreateEstimator(roisize, ctx)
    test_psf_speed(ctx,psf,phasor_est,theta, repeats=100,batchSize=10*1024)

    print('COM:')
    com_est = com.CreateEstimator(roisize, ctx)
    test_psf_speed(ctx,psf,com_est,theta, batchSize=10*1024, repeats=100)
    
#    test_queue_output(ctx, psf, theta)
    print('2D Gaussian fit:')
    test_psf_speed(ctx,psf,psf,theta,repeats=100)

    print('Astigmatic 2D Gaussian PSF:')
    as_psf = g_api.CreatePSF_XYZIBg(roisize, as_calib, True)
    as_theta=[[roisize//2, roisize//2, 0, 1000, 5]]
    test_psf_speed(ctx, as_psf, as_psf, as_theta,repeats=100)

    if False:    
        print('SIMFLUX 2D Gaussian PSF:')
        sf_psf = sf_api.CreateEstimator_Gauss2D(sigma, roisize, len(sf_mod),  simfluxEstim=True)
        test_psf_speed(ctx, sf_psf, sf_psf, theta, repeats=10)
    
        print('SIMFLUX Astigmatic gaussian PSF (3D):')
        sf_as_psf = sf_api.CreateEstimator(as_psf, sf_mod)
        test_psf_speed(ctx, sf_as_psf, sf_as_psf, as_theta, repeats=50, nstreams=10, binsize=512, batchSize=512)
        
        print('Single-Frame SIMFLUX 2D PSF:')
        sfsf_psf = sf_api.CreateSFSFEstimator(psf, sf_mod, sfsf_offsets())
        test_psf_speed(ctx, sfsf_psf, sfsf_psf, theta, repeats=50, nstreams=10, binsize=512, batchSize=512)

    cspline_fn = 'Tubulin-A647-cspline.mat'
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
    

    if cspline_fn is not None:
        print('CSpline 3D PSF:')
        calib = cspline.CSpline_Calibration.from_file_nmeth(cspline_fn)
        cs_psf = cspline.CSpline(ctx).CreatePSF_XYZIBg(roisize, calib, True)
        cs_theta=[[roisize//2, roisize//2, 0, 1000, 5]]
        test_psf_speed(ctx,cs_psf,cs_psf, cs_theta,repeats=10)

    #print('2D Gaussian fit + sCMOS:')
    #test_psf_speed(ctx,psf_sc,psf_sc,theta)

    print('2D Gaussian fit + GLRT:')
    psf_glrt = glrt.CreateEstimator(psf, ctx)
    test_psf_speed(ctx,psf_glrt,psf_glrt,theta,repeats=50)
    
        
        