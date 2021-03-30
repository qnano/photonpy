

from .context import Context
from .estimator import Estimator
import numpy as np

import ctypes as ct

def Localize(roi):
    fx = np.sum(np.sum(roi,0)*np.exp(-2j*np.pi*np.arange(roi.shape[1])/roi.shape[1]))
    fy = np.sum(np.sum(roi,1)*np.exp(-2j*np.pi*np.arange(roi.shape[0])/roi.shape[0]))
            
    #Calculate the angle of the X-phasor from the first Fourier coefficient in X
    angX = np.angle(fx)
    if angX>0: angX=angX-2*np.pi
    #Normalize the angle by 2pi and the amount of pixels of the ROI
    posx = np.abs(angX)/(2*np.pi/roi.shape[1])
    #Calculate the angle of the Y-phasor from the first Fourier coefficient in Y
    angY = np.angle(fy)
    #Correct the angle
    if angY>0: angY=angY-2*np.pi
    #Normalize the angle by 2pi and the amount of pixels of the ROI
    posy = np.abs(angY)/(2*np.pi/roi.shape[1])
    
    return posx,posy


def CreateEstimator(roisize, ctx: Context=None):
    smlmlib = ctx.smlm.lib

    fn = smlmlib.CreatePhasorEstimator
    fn.argtypes = [
            ct.c_int32,
            ct.c_void_p]
    fn.restype = ct.c_void_p

    inst = fn(roisize, ctx.inst)
    return Estimator(ctx, inst)

