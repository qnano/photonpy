import matplotlib.pyplot as plt
import numpy as np

from photonpy.cpp.context import Context
import photonpy.cpp.gaussian as gaussian

# Calibration coefficients
s0_x = 1.3
gamma_x = -20
d_x = 20
A_x = 0#2e-04

s0_y = 1.3
gamma_y = 20
d_y = 20
A_y = 0#1e-05

calib = gaussian.Gauss3D_Calibration([s0_x, gamma_x, d_x, A_x], [s0_y, gamma_y, d_y, A_y])


with Context() as ctx:
    sigma=1.5
    roisize=12
    theta=[[roisize//2, roisize//2, 3, 1000, 5]]
    g_api = gaussian.Gaussian(ctx)
    psf = g_api.CreatePSF_XYZIBg(roisize, calib, True)

    imgs = psf.ExpectedValue(theta)
    plt.figure()
    plt.imshow(imgs[0])
    
    sample = np.random.poisson(imgs)

    # Run localization on the sample
    estimated,diag,traces = psf.Estimate(sample)        
        
    print(f"Estimated position: {estimated[0]}")
    
    