import matplotlib.pyplot as plt
import numpy as np

from photonpy.cpp.context import Context
import photonpy.cpp.gaussian as gaussian
import photonpy.cpp.glrt as glrt

from scipy.stats import norm

with Context() as ctx:
    g = gaussian.Gaussian(ctx)

    sigma=1.5
    roisize=9

    psf = g.CreatePSF_XYIBg(roisize, sigma, True)
    psf = glrt.CreateEstimator(psf, ctx)

    N = 10000
    theta = [[roisize/2,roisize/2,50,5]]
    theta = np.repeat(theta,N,0)

    img = psf.ExpectedValue(theta)

    smp = np.random.poisson(img)
    plt.figure()
    plt.set_cmap('inferno')
    plt.imshow(smp[0])

    estim, diag, trace = psf.Estimate(smp)

    ll_on = diag[:,0]
    ll_off = diag[:,1]
    
    #float Tg = 2.0f*(h1_ll - h0_ll);	// Test statistic (T_g)
	#float pfa = 2.0f * cdf(-sqrt(Tg)); // False positive probability (P_fa)

    Tg = 2*(ll_on-ll_off)
    Tg = np.maximum(0,Tg)
    pfa = 2 * norm.cdf(-np.sqrt(Tg))
        
    print(f"p(false positive): {np.mean(pfa)}")
    
    
    