# -*- coding: utf-8 -*-

import numpy as np
from photonpy import Context,GaussianPSFMethods
import matplotlib.pyplot as plt
from scipy.signal.windows import tukey

def _getfft(xy,photons,imgshape,zoom,ctx:Context):
    spots = np.zeros((len(xy),5))
    spots[:,[0,1]] = xy * zoom
    spots[:,4] = photons 
    spots[:,[2,3]] = 0.5
    
    w = np.max(imgshape)*zoom
    img = np.zeros((w,w))
    img = GaussianPSFMethods(ctx).Draw(img, spots)
    img = np.array(img, dtype=np.float32)
    
    # Image width / Width of edge region
    wnd = tukey(w, 1/4).astype(np.float32)
    #plt.plot(wnd)
    img = (img * wnd[:,None]) * wnd[None,:]
    f_img = np.fft.fftshift(ctx.smlm.FFT2(img))
    #f_img = np.fft.fftshift(np.fft.fft2(img))

    return f_img

def radialsum(sqimg):
    W = len(sqimg)
    Y,X = np.indices(sqimg.shape)
    R = np.sqrt((X-W//2)**2+(Y-W//2)**2)
    R = R.astype(np.int32)
    return np.bincount(R.ravel(), sqimg.ravel()) # / np.bincount(R.ravel())


def FRC(xy, photons, zoom, imgshape, pixelsize, display=True):
    with Context() as ctx:
        set1 = np.random.binomial(1, 0.5, len(xy))==1
        set2 = np.logical_not(set1)
        f1 = _getfft(xy[set1],photons[set1],imgshape,zoom,ctx)
        f2 = _getfft(xy[set2],photons[set2],imgshape,zoom,ctx)
            
    frc_num = radialsum(np.real(f1*np.conj(f2)))
    frc_denom = np.sqrt(radialsum(np.abs(f1)**2) * radialsum(np.abs(f2)**2))
    
    frc = frc_num / frc_denom
    
    freq = np.fft.fftfreq(len(f1))
    frc = frc[:imgshape[0]*zoom//2]
    freq = freq[:imgshape[0]*zoom//2]
    

    b = np.where(frc<1/7)[0]
    frc_res =  freq[b[0]] if len(b)>0 else freq[0] 
    
    if display:
        plt.figure()
        plt.plot(freq/zoom, frc)
        plt.title(f'FRC resolution: {pixelsize / (zoom*frc_res):.2f} nm')
        plt.xlabel('Frequency [1/pixels]')
    
    return pixelsize / (zoom*frc_res),frc
    
if __name__ == "__main__":
    
    fn= 'C:/data/simflux/sim4_1/results/sim4_1/g2d-dc-fbp10.hdf5' 
    fn  = 'C:/data/simflux/sim4_1/results/sim4_1/sf-dc-fbp10.hdf5' 
    #fn='C:/data/drift/gatta RY/3.hdf5'
    
    from photonpy.smlm.dataset import Dataset
    ds = Dataset.load(fn)
    
    frc,frc_res,freq = FRC(ds.pos, ds.photons, 30, ds.imgshape, pixelsize=65)
    