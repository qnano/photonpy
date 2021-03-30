# -*- coding: utf-8 -*-

from photonpy.cpp.gaussian import Gaussian, Gauss3D_Calibration
from photonpy.cpp.context import Context

import numpy as np
import matplotlib.pyplot as plt

import napari

# Calibration coefficients
s0_x = 1.3
gamma_x = -20
d_x = 20
A_x = 0#2e-04

s0_y = 1.3
gamma_y = 20
d_y = 20
A_y = 0#1e-05

x=[1.3,  2, 3, 0]
y=[1.3, -2, 3, 0]
zrange=[-3, 3]

calib = Gauss3D_Calibration(x,y,zrange)#[s0_x, gamma_x, d_x, A_x], [s0_y, gamma_y, d_y, A_y])

with Context() as ctx:
    g = Gaussian(ctx)
    
    roisize=20
    psf = g.CreatePSF_XYZIBg(roisize, calib, True)
    
    N = 200
    theta = np.repeat([[roisize/2,roisize/2,50,1000,3]], N, axis=0)
    theta[:,2] = np.linspace(zrange[0],zrange[1],N)
    
    smp = np.random.poisson(psf.ExpectedValue(theta))
    
    estim, diag, traces = psf.Estimate(smp)
    
    with napari.gui_qt():
        napari.view_image(smp)

    plt.figure()
    plt.plot(theta[:,2], estim[:,2])
    