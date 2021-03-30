"""
PSF Utilities
"""


from photonpy.cpp.estimator import Estimator
import numpy as np
import matplotlib.pyplot as plt

from photonpy.utils.findpeak1D import gaussianpeak
import tqdm

from photonpy.cpp.phasor import Localize as PhasorLocalize

def upscale_fft(A, axes, factor):
    fA = np.fft.fftshift(np.fft.fftn(A,axes=axes),axes=axes)
    
    shape = np.array(A.shape)
    newshape = shape*1
    newshape[list(axes)] *= factor
    
    padding = np.zeros((len(shape),2),dtype=int)
    for i in range(len(shape)):
        padding[i, 0] = (newshape[i]-shape[i])//2
        padding[i, 1] = newshape[i]-shape[i]-padding[i,0]
    
    padded = np.pad(fA,padding)

    return np.real( np.fft.ifftn(np.fft.ifftshift(padded,axes=axes), axes=axes))


    

def find_shift(A=None, B=None, fA=None, fB=None, autocor=True, plotTitle=None):
    """
    Find a shift between two images with arbitrary number of dimensions.
    """
    if A is not None:
        fA = np.fft.fftn(A)
        
    if B is not None:
        fB = np.fft.fftn(B)
    
    if autocor:
        fA = np.conj(fA)
    
    conv = np.real(np.fft.ifftn(fA*fB))
    conv = np.fft.fftshift(conv)
    
    peak = np.unravel_index(np.argmax(conv),conv.shape)
    fittedpeak = np.array(peak)*0.0

    W=10
    numax = len(fA.shape)
    for i in range(numax):
        idx = list(peak)
        start=np.maximum(0,peak[i]-W)
        idx[i] = np.s_[start:start+W*2]
        #
        x = gaussianpeak(conv[tuple(idx)], plotTitle=plotTitle)
        #x = quadraticpeak(conv[tuple(idx)],npts=5)#,plotTitle=f"ax{i}")
        
        fittedpeak[i] = peak[i]-W+x
        
    fittedpeak -= np.array(conv.shape)//2
    return fittedpeak



def apply_shift_3D(img, zyx):
    f_img = np.fft.fftn(img)
    Z,Y,X=np.mgrid[:img.shape[0],:img.shape[1],:img.shape[2]]
    tilt = np.fft.fftshift(np.exp(-2j*np.pi*(
        X*zyx[2]/img.shape[2]+
        Y*zyx[1]/img.shape[1]+
        Z*zyx[0]/img.shape[0])))
    shifted = np.real(np.abs(np.fft.ifftn(f_img*tilt)))
    return shifted





def apply_shift_z(img, value):
    f_img = np.fft.fft(img,axis=0)
    X = np.arange(img.shape[0])
    tilt = np.fft.fftshift(np.exp(-2j*np.pi/img.shape[0]*(X*value)),axes=0)
    shifted = np.real(np.abs(np.fft.ifft(f_img*tilt[:,None,None],axis=0)))
    return shifted



def apply_shift_2D(img, yx):
    f_img = np.fft.fft2(img)
    W=img.shape[-1]
    X,Y = np.meshgrid(np.arange(W),np.arange(W))
    tilt = np.fft.fftshift(np.exp(-2j*np.pi/W*(X*yx[1]+Y*yx[0])))
    shifted = np.real(np.abs(np.fft.ifft2(f_img*tilt)))
    return shifted


def fft_align_zstacks(zstack, xy_upscale=4):
    """
    Align a set of zstacks and compute the average. Expected shape of zstack is:
        [numbeads, numzsteps, roisize, roisize]

    Steps:
    - upscale
    - run phasor localization on zstack sums to find 2D pos
    - shift in 2D
    - align in Z by cross-correlation using FFT in z
    """
    zstack=zstack.astype(np.float32)
    zstack=np.maximum(0, zstack-np.median(zstack))

    numbeads = zstack.shape[0]
    zsteps = zstack.shape[1]
    zstack_upscaled = np.zeros((numbeads, zsteps,zstack.shape[2]*xy_upscale, 
                           zstack.shape[3]*xy_upscale),dtype=np.float32)
    
    print(f'\nupsampling {xy_upscale}x...\n',flush=True)
    for i in tqdm.trange(numbeads):
        zstack_upscaled[i] = upscale_fft(zstack[i], (1,2), xy_upscale)

    print(f'\naligning in 2D...\n',flush=True)    
    zsum = np.sum(zstack_upscaled, 1) # sum zstacks over z
    roisize = zstack_upscaled.shape[-1]
    
    sumall = np.sum(zsum, 0)
        
    # Sum in Z direction and align in 2D
    beadpos = np.zeros((len(zstack),2))
    for bead in tqdm.trange(len(zsum)):
         
        beadpos[bead] = find_shift(sumall, zsum[bead])
        #beadpos[bead] = PhasorLocalize(zsum[bead])
        #beadpos[bead] -= roisize/2
        
        for z in range(zstack.shape[1]):
            zstack_upscaled[bead,z] = apply_shift_2D(zstack_upscaled[bead, z], -beadpos[bead])
        
    # sum all beads to do z-alignment against
    avg = np.mean(zstack_upscaled, 0)
    
    #show_napari( np.concatenate([zstack_upscaled[0],avg ],-1))
    
    zstack_shifted = zstack_upscaled*0
    
    print('\naligning 3D',flush=True)

    beadz = np.zeros((len(zstack),2))
    for bead in tqdm.trange(len(zsum)):

        sh = find_shift(zstack_upscaled[bead],avg)
        #print(sh)
        beadz[bead,0] = sh[0]
        #beadz[bead], _= align_z(zstack_upscaled[bead], avg )
        # Pad zeros in Z direction
        padding = ((zsteps//2,zsteps//2), (0,0), (0,0))
        zstack_padded = np.pad(zstack_upscaled[bead], padding)

        img = apply_shift_z(zstack_padded,sh[0])[zsteps//2:zsteps//2+avg.shape[0]]
        zstack_shifted[bead] = apply_shift_2D(img, sh[:2])
        #beadz[bead,1] = find_shift(zstack_shifted[bead],avg)[0]

    if True:
        print('\naligning 3D - pass 2/2\n',flush=True)
    
        avg2 = np.mean(zstack_shifted, 0)
        for bead in tqdm.trange(len(zsum)):
            sh = find_shift(zstack_upscaled[bead],avg2)
            beadz[bead,1] = sh[0]
            padding = ((zsteps//2,zsteps//2), (0,0), (0,0))
            zstack_padded = np.pad(zstack_upscaled[bead], padding)
            img = apply_shift_z(zstack_padded,sh[0])[zsteps//2:zsteps//2+avg.shape[0]]
            zstack_shifted[bead] = apply_shift_2D(img, sh[:2])
            #beadz[bead,3] = find_shift(zstack_shifted[bead],avg)[0]
    
    result = np.sum(zstack_shifted,0)
    W = result.shape[-1]
    M = 4
    result = result.reshape((len(result), W//M,M,W//M,M)).sum((2,4))
    
    return result, zstack_shifted, beadpos, beadz



def align_zstacks_simple(zstack, xy_upscale):
    """
    Align a set of zstacks and compute the average. Expected shape of zstack is:
        [numbeads, numzsteps, roisize, roisize]
    """
    numbeads = zstack.shape[0]
    zsteps = zstack.shape[1]
    zstack_upscaled = np.zeros((numbeads, zsteps,zstack.shape[2]*xy_upscale, 
                           zstack.shape[3]*xy_upscale),dtype=np.float32)
    for i in tqdm.trange(numbeads):
        zstack_upscaled[i] = upscale_fft(zstack[i], (1,2), xy_upscale)

    zstack_avg = np.mean(zstack_upscaled,0)

    # Pad zeros in Z direction
    padding = ((zsteps//2,zsteps//2), (0,0), (0,0))
    zstack_avg = np.pad(zstack_avg, padding)

    summed_zstack = zstack_avg*0
    for i in tqdm.trange(numbeads):
        zstack_padded = np.pad(zstack_upscaled[i], padding)
        sh = find_shift(zstack_padded, zstack_avg, plotTitle="" if i==0 else None)
        summed_zstack += apply_shift_3D(zstack_padded, sh)

    summed_zstack /= numbeads
    return summed_zstack[zsteps//2:-zsteps//2]



def psf_to_zstack(psf:Estimator, zrange, intensity=1, bg=0, plot=False):
    
    assert (psf.numparams == 5) or (psf.numparams == 7)  # 3D psf with or without tilted bg
    
    params = np.zeros((len(zrange),psf.numparams))
    params[:,0] = psf.sampleshape[0]/2
    params[:,1] = psf.sampleshape[1]/2
    params[:,2] = zrange
    params[:,3] = intensity
    params[:,4] = bg
    
    ev = psf.ExpectedValue(params)
    
    if plot:
        plt.figure()
        normalized = ev/np.max(ev,(1,2))[:,None,None]
        plt.imshow(np.concatenate(normalized,-1))
        plt.colorbar()
    
    return ev


def render_to_image(psf:Estimator, imgshape, emitters, constants=None):
    
    assert(len(psf.sampleshape)==2) and psf.sampleshape[0]==psf.sampleshape[1]
    
    yx = psf.ParamIndex(['y','x'])
    
    roi_em = emitters*1
    roisize = psf.sampleshape[0]
    roipos = np.clip((roi_em[:,yx] - roisize/2).astype(int), [0,0], imgshape-roisize)
    roi_em[:,yx] -= roipos
        
    rois = psf.ExpectedValue(roi_em, roipos, constants)
    
    return psf.ctx.smlm.DrawROIs(imgshape, rois, roipos)
    
    
header_dtype = [('version', '<i4'), ('dims','<i4', 3), ('zrange', '<f4', 2)]

def save_zstack(zstack, zrange, fn):
    """
    Save a ZStack to a binary file.
    """
    shape = zstack.shape
    with open(fn, "wb") as f:
        version = 1
        np.array([(version, shape, (zrange[0],zrange[-1]))],dtype=header_dtype).tofile(f,"")
        np.ascontiguousarray(zstack,dtype=np.float32).tofile(f,"")


def load_zstack(fn):
    """
    Returns zstack, [zmin, zmax]
    """
    with open(fn, "rb") as f:
        d = np.fromfile(f,dtype=header_dtype,count=1,sep="")
        version, shape, zrange = d[0]
        zstack = np.fromfile(f,dtype='<f4',sep="").reshape(shape)
        return  zstack, zrange
    
