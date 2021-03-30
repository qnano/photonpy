# -*- coding: utf-8 -*-


from photonpy.cpp.context import Context
from photonpy import GaussianPSFMethods
import numpy as np


def render_gaussians(xyI, imageOrShape, zoom, sigma, ctx=None):
    
    if type(imageOrShape) == tuple:
        shape = imageOrShape
    else:
        shape = imageOrShape.shape
        
    imgsize = np.array(shape)
        
    with Context(ctx.smlm if ctx else None) as ctx:
        
        roisize = int(3+np.mean(sigma)*4)
        
        psf = GaussianPSFMethods(ctx).CreatePSF_XYIBg(roisize, sigma, True)
        
        params = np.zeros((len(xyI), 4))
        params[:,:3] = xyI
        
        roipos = np.clip((params[:,[1,0]] - roisize/2).astype(int), 0, imgsize-roisize)
        params[:,[1,0]] -= roipos
        
        rois = psf.ExpectedValue(params)
        return ctx.smlm.DrawROIs(imageOrShape, rois, roipos)
