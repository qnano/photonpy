# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize


# Curve-fit y around the peak to find subpixel peak x
def quadraticpeak(y, x=None, npts=7, plotTitle=None):
    if x is None:
        x = np.arange(len(y))
    xmax = np.argmax(y)
    W = int((npts + 1) / 2)
    window = np.arange(xmax - W + 1, xmax + W)
    window = np.clip(window, 0, len(x) - 1)
    coeff = np.polyfit(x[window], y[window], 2)

    if plotTitle:
        plt.figure()
        plt.plot(x[window], y[window], label="data")
        sx = np.linspace(x[xmax - W], x[xmax + W], 100)
        plt.plot(sx, np.polyval(coeff, sx), label="fit")
        plt.legend()
        plt.title(plotTitle)

    return -coeff[1] / (2 * coeff[0])



def gaussianpeak(y, plotTitle=False, bounds=None):
    x = np.arange(len(y))

    npts = 20
    xmax = np.argmax(y)
    W = int((npts + 1) / 2)
    window = np.arange(xmax - W + 1, xmax + W)
    window = np.clip(window, 0, len(y) - 1)
        
    qc = np.polyfit(x[window], y[window], 2)
    peakx = -qc[1] / (2 * qc[0])
    peaky = np.polyval(qc, peakx)

    f = lambda x,a,b,c,d: a * np.exp(-d*(x-b)**2) + c
    c = np.min(y[window])
    p0 = [peaky,peakx,c,0.01]
    
    if bounds is None:
        bounds=([0.1,-np.inf,-np.inf,0], [np.inf,np.inf,np.inf,np.inf])
    
    obj = lambda p: np.sum( (f(x,*p)-y)** 2)
   
    try:
        #r = minimize(obj, p0).x
        r = curve_fit(f, x, y, p0, method='trf', maxfev=10000, bounds=bounds)[0]
    except RuntimeError as err:
        
        plt.figure()
        plt.plot(y,label='data')
        sx = np.linspace(x[window[0]], x[window[-1]], 100)
        plt.plot(sx, np.polyval(qc, sx), label="quadratic fit")
        plt.legend()
        plt.title(plotTitle)
        plt.show()

        raise err
        
    #r = curve_fit(f, window, y[window], p0)[0]
    
    if plotTitle is not None:
        plt.figure()
        plt.plot(y,label='data')
        plt.plot(f(x,*r),label=f'gaussian fit {r}')
        sx = np.linspace(x[window[0]], x[window[-1]], 100)
        plt.plot(sx, np.polyval(qc, sx), label="quadratic fit")
        plt.legend()
        plt.title(plotTitle)
    
    return r[1]

if __name__ == "__main__":

    x = np.arange(100)
    max = 50
    y = -(x - max) ** 2 + np.random.uniform(-5, 5, size=len(x))

    plt.figure()
    plt.plot(x, y)
    print(quadraticpeak(y, x, npts=12, plotTitle="Fit"))
