# -*- coding: utf-8 -*-


if __name__ == "__main__":
    import sys
    sys.path.append('../..')

import os
import numpy as np
import matplotlib.pyplot as plt

from smlmlib.context import Context
from smlmlib.base import SMLM
import smlmlib.gaussian as gaussian

from smlmlib.simflux import SIMFLUX


def radon(xyI, framenum, pattern_frames, scale, num_angle_bins, freq_range):
    with SMLM() as smlm, Context(smlm) as ctx:
        sampleWidth = int(np.ceil(np.max(xyI[:,0])-np.min(xyI[:,0])))
        
        sf = SIMFLUX(ctx)

        pattern_frames = np.array(pattern_frames)
        numpat = pattern_frames.size
        angles = np.linspace(0,np.pi,num_angle_bins)
        projWidth = scale * sampleWidth

        for i, ang_steps in enumerate(pattern_frames):
            result = np.zeros((num_angle_bins, projWidth),dtype=np.float32)
            
            for j, frame_index in enumerate(ang_steps):
                indices = (framenum % numpat) == frame_index
                output,shifts = sf.ProjectPoints(xyI[indices], projWidth, scale, angles)
                f_out = np.abs(np.fft.fftshift(np.fft.fft(output),axes=1))
#                f_out = f_out[len(f_out)//2-W//2:len(f_out)//2+W//2]
                result += f_out

            deg = np.rad2deg(angles).astype(int)
            proj_freq = np.sum(result-np.mean(result),0)
            proj_ang = np.sum(result-np.mean(result),1)
            
            plt.figure()
            W=200
            proj_freq[np.abs(np.fft.fftfreq(len(proj_freq)))*2*np.pi<1]=0
            #proj_freq[len(proj_freq)//2]=0
            plt.plot(proj_freq[len(proj_freq)//2-W//2:len(proj_freq)//2+W//2])
            plt.xlabel('Freq')
            plt.figure()
            plt.plot(deg,proj_ang)
            plt.xlabel('Ang')
#            plt.imshow(result, extent=(0, sampleWidth, deg[0],deg[-1]))
 #           plt.title(f"Radon FFT plot for angle {i}")
            #plt.xlabel('Frequency')
        
    
def radon_hdf5(locs_fn,pattern_frames):
    from utils.picasso_hdf5 import load as load_hdf5
    
    estim,framenum,crlb,imgshape = load_hdf5(locs_fn)
    print(f"{locs_fn} contains {len(estim)} spots")
    
    radon(estim[:,0:3], framenum, pattern_frames, 10, 200, np.linspace(1.7,1.8))
    
if __name__ == "__main__":    
    
    #locs_fn = 'C:/dev/simflux/data/7-23-2019/Pos5/1_merge.hdf5'
    locs_fn = 'C:/dev/simflux/oxford/20190729-132205_SIMFlux_561ex_Rbead.hdf5'
    #pattern_frames = [[0,2,4],[1,3,5]]
    pattern_frames = np.arange(6)[:,None]
    
    radon_hdf5(locs_fn,pattern_frames)
    