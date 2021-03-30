# -*- coding: utf-8 -*-

from photonpy.cpp.lib import SMLM
from photonpy.cpp.simflux import SIMFLUX
import numpy as np
from .findpeak1D import quadraticpeak
from photonpy.cpp.context import Context

def dft_points(xyI, kx, ky, ctx:Context, useCuda=True):
    KX,KY = np.meshgrid(kx,ky)
    klist = np.zeros((len(kx)*len(ky),2),dtype=np.float32)
    klist[:,0] = KX.flatten()
    klist[:,1] = KY.flatten()
    return SIMFLUX(ctx).SIMFLUX_DFT2D_Points(xyI, klist, useCuda=useCuda).reshape((len(kx),len(ky)))

def find_freq_peak(xyI, kx, ky, freq_range, ctx:Context, N=100, plot=False):
    kxrange = np.linspace(kx-freq_range, kx+freq_range, N)
    kyrange = np.linspace(ky-freq_range, ky+freq_range, N)
    
    sig = np.abs(dft_points(xyI, kxrange, kyrange, ctx)**2)
    
    peak = np.argmax(sig)
    peak = np.unravel_index(peak, sig.shape)
        
    kx_peak = quadraticpeak(sig[peak[0], :], kxrange, plotTitle='X peak' if plot else None)
    ky_peak = quadraticpeak(sig[:, peak[1]], kyrange, plotTitle='Y peak' if plot else None)
    
    return kx_peak, ky_peak, sig

