import os
import numpy as np
from photonpy.smlm.picasso_hdf5 import load as load_hdf5

import matplotlib.pyplot as plt

from photonpy import Context
from photonpy.cpp.postprocess import PostProcessMethods


def estimate_on_time(locs_hdf5_fn, maxdist=3, frameskip=1, maxintensitydist=5, maxspots=None,dataset_title=None):
    estim, framenum, crlb, imgshape,sx,sy = load_hdf5(locs_hdf5_fn)
    
    if maxspots is not None:
        if len(estim) > maxspots:
            estim = estim[:maxspots]
            framenum = framenum[:maxspots]
            crlb = crlb[:maxspots]
            
    with Context() as ctx:
        pp = PostProcessMethods(ctx)
        xyI = estim[:,:3]
        linked, framecounts, startframes, linkedXYI, crlb = pp.LinkLocalizations(
                xyI,crlb[:,0:3],framenum,maxdist,maxintensitydist,frameskip)
        
        dataname = os.path.splitext(locs_hdf5_fn)[0]
        
        bins = np.bincount(framecounts)
        print(f"Total spots: {len(xyI)}. #bins={len(bins)} Linked spots: {len(startframes)}. Median on-time: {np.median(framecounts)}. Mean on-time: {np.mean(framecounts)}")

        bins = bins[:np.minimum(50,len(bins))]
        
        startbin = 2
        endbin = None
        
        fit_data = bins[startbin:endbin]
        estsum = np.sum(fit_data) 
        
        mean = np.sum(fit_data * (startbin+np.arange(len(fit_data))) ) / estsum
        print(f'Exponential fit on time: {mean:.3f}')
        lam =  1 / mean
        
        x = np.arange(0, len(bins))
        prob = bins / np.sum(bins)
        fig,axes = plt.subplots(2,1,sharex=True)
        axes[0].bar(np.arange(0, len(bins)), np.sum(bins) * prob)
        axes[0].plot(x, estsum * lam * np.exp(-lam * x), 'k')
        #plt.hist(framecounts,bins=100,range=(2,60))
#        plt.hist(bins[(bins<=60)],bins=100)
        if dataset_title is None:
            dataset_title = os.path.split(dataname)[1]

        axes[0].set_title(f"Estimated on times for spots in {dataset_title}")

        axes[1].bar(np.arange(2, len(bins)), np.sum(bins) * prob[2:])
        axes[1].plot(x[2:], estsum * lam * np.exp(-lam * x[2:]), 'k')
        axes[1].set_xlabel(f"On-time [frames]. Mean on-time={mean:.2f}")

        return fig, bins, framecounts


if __name__ == "__main__":
    
    import matplotlib as mpl
    #mpl.use('svg')
    new_rc_params = {
    #    "font.family": 'Times',
        "font.size": 15,
    #    "font.serif": [],
        "svg.fonttype": 'none'} #to store text as text, not as path
    mpl.rcParams.update(new_rc_params)

    
    files = [
            ('C:/data/storm-wf/60 mw Pow COS7_aTub_A647_sulfite_10mM_MEA_50Glyc/1', '60 mW'),
            ('C:\data\storm-wf/COS7 ABBELIGHT A657N - 7-17-2019/Pos1_WF_merge', 'Full power'),
            ('C:\data\storm-wf\Half Pow COS7_aTub_A647_sulfite_10mM_MEA_50Glyc/2', 'Half power [pos2]'),
            ('C:\data\storm-wf\Half Pow COS7_aTub_A647_sulfite_10mM_MEA_50Glyc/1', 'Half power [pos1]'),
            ('C:\data\storm-wf\Half Pow COS7_aTub_A647_sulfite_10mM_MEA_50Glyc/3', 'Half power [pos3]'),
            ('C:/data/gattabeads/gattabeads-2_ld_1', 'WF DNA PAINT')
            ]
    

    for k in range(len(files)):
        locs_fn = files[k][0]+".hdf5"
        fig,bins,framecounts=estimate_on_time(locs_fn, frameskip=1, maxdist=1, dataset_title=files[k][1])
    #fig,bins,framecounts=estimate_on_time(
     #       '/dev/simflux/data/7-17-2019 STORM COS/Pos1_WF_merge.hdf5')
