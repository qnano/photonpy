# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from shutil import copyfile


import scipy.spatial.ckdtree

def count_neighbors(xy, searchdist):
    """
    Return the number of neighbors for each localization
    """
    kdtree = scipy.spatial.ckdtree.cKDTree(xy)
    pairs = kdtree.query_pairs(searchdist) # output_type='ndarray')
    
    counts = np.zeros((len(xy)),dtype=np.int)
    for p in pairs:
        counts[p[0]] += 1
    
    return counts

def filter_hdf5(inputfile, outputfile, searchdist, mincount):
    
    with h5py.File(inputfile, 'r') as f:
        with h5py.File(outputfile, 'w') as fo:
#            fo['locs'] = f['locs']
 
            d = f['locs'][:]
            
            xy = np.zeros((len(d),2))
            xy[:,0] = d['x']
            xy[:,1] = d['y']
            nbcounts = count_neighbors(xy, searchdist)
            
            plt.figure()
            plt.hist(nbcounts)
            
            filtered = d[nbcounts>=mincount]
            print(f"removed {len(d)-len(filtered)} out of {len(d)} localizations")
            fo.create_dataset('locs',filtered.shape,filtered.dtype,filtered)
            
            copyfile(os.path.splitext(inputfile)[0]+".yaml",
                     os.path.splitext(outputfile)[0]+".yaml")
            
            return d
        
        
if __name__ == "__main__":
    
    fn = 'O:/sim1_1/results/sim1_1/g2d.hdf5'
    d=filter_hdf5(fn, 'O:/sim1_1/results/sim1_1/g2d_filtered.hdf5', 0.2, 2)
    
    
    