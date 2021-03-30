# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import matplotlib.mlab as mlab
from scipy.stats import norm

def load_hdf5(fn):
    with h5py.File(fn, 'r') as f:
        locs = f['locs']
#        print(locs.dtype)
        x = locs['x']
        y = locs['y']
        I = locs['photons']
        print(locs.dtype)#locs['group']

        xyI = np.zeros((len(x),3))
        xyI[:,0] = x
        xyI[:,1] = y
        xyI[:,2] = I
        
        if 'group' in locs.dtype.fields.keys():
            return xyI, locs['group']
        
        return xyI,None
    
def fit(x,title, nbins=50):

    # best fit of data
    (mu, sigma) = norm.fit(x)
    
    x -= mu
    
    # the histogram of the data
    n, bins, patches = plt.hist(x, nbins, normed=1, alpha=0.75)
    
    # add a 'best fit' line
    px = np.linspace(-0.4,0.4)
    y = mlab.normpdf(px, 0, sigma)
    plt.plot(px, y, 'r--', linewidth=2)

    plt.grid(True)
    plt.title(f"{title} - Sigma = {sigma:.3f} pixels")

    return mu,sigma

if __name__ == "__main__":

    files = [
        '../../../SMLM/data/sim4_1/results/sim4_1/mw100-g2d_picked.hdf5',
        '../../../SMLM/data/sim4_1/results/sim4_1/mw100-silm_picked.hdf5']
            
#    '../../../SMLM/data/80nm/results/80nm-3rdshift-2_0/mw20-drift-corrected-g2d_picked.hdf5',
 #   '../../../SMLM/data/80nm/results/80nm-3rdshift-2_0/mw20-drift-corrected-silm_picked.hdf5']
#    files = [
 #           '../../data/80nm/results/80nm-3rdshift-2/mw50-g2d_picked.hdf5'
  #          , '../../data/80nm/results/80nm-3rdshift-2/mw50-silm_picked.hdf5']

#    files = [
 #           '../../data/sim-silm/results/sim0 - Copy/mw100-g2d_picked.hdf5',
  #          '../../data/sim-silm/results/sim0 - Copy/mw100-silm_picked.hdf5'
   #         
    #        ]
            
    axis = 0

    cr = []    
    for f in range(len(files)):
        d,groups = load_hdf5(files[f])
        
        centered = []
        
        ngroups = np.max(groups)+1
        for g in range(ngroups):
            xy = d[groups==g][:,[0,1]]*1
        
            xy -= np.mean(xy,0)
            centered.append(xy)
        
        cr.append(np.concatenate(centered))
            
#    fig,axes = plt.subplots(2,2)

    if True:
#        plt.figure()
#        for k in range(2):
 #           plt.scatter(cr[k][:,0],cr[k][:,1])
    
        plt.figure()
        plt.subplot(211)    
        _,s=fit(cr[0][:,1], 'Y (G2D)')
    
        plt.subplot(212) 
        _,s=fit(cr[1][:,1], 'Y (SIMFLUX)')
        
        plt.figure()
        plt.subplot(211) 
        _,s=fit(cr[0][:,0], 'X (G2D)')
        
        plt.subplot(212) 
        _,s=fit(cr[1][:,0], 'X (SIMFLUX)')
        
        
        