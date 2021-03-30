import numpy as np
import matplotlib.pyplot as plt

from photonpy.cpp.context import Context
from photonpy.cpp.cspline import CSplineCalibration, CSplineMethods
from photonpy.cpp.gaussian import GaussianPSFMethods,Gauss3D_Calibration
from photonpy.cpp.estimator import Estimator
from photonpy.cpp.estim_queue import EstimQueue
from photonpy.cpp.spotdetect import PSFCorrelationSpotDetector, SpotDetectionMethods
import math
from photonpy.smlm.psf import psf_to_zstack
from photonpy.smlm.util import imshow_hstack
import time
import tqdm
import tifffile



def generate_storm_movie(psf:Estimator, xyzI, numframes=100, 
                         imgsize=512, bg=5, p_on=0.1):
    
    frames = np.zeros((numframes, imgsize, imgsize), dtype=np.float32)
    on_counts = np.zeros(numframes, dtype=np.int32)

    for f in range(numframes):
        on = np.random.binomial(1, p_on, len(xyzI))

        roisize = psf.sampleshape[0]
        roipos = np.clip((xyzI[:,[1,0]] - roisize/2).astype(int), 0, imgsize-roisize)
        theta = np.zeros((len(xyzI),5)) # assuming xyzIb
        theta[:,0:4] = xyzI
        theta[:,[1,0]] -= roipos
        on_spots = np.nonzero(on)[0]

        rois = psf.ExpectedValue(theta[on_spots])
        
        frames[f] = ctx.smlm.DrawROIs((imgsize,imgsize), rois, roipos[on_spots])
        frames[f] += bg
        on_counts[f] = np.sum(on)   

    return frames, on_counts

def view_movie(mov):
    import napari    
    
    with napari.gui_qt():
        napari.view_image(mov)



def process_movie(mov, spotDetector, roisize, ctx:Context):
    imgshape = mov[0].shape
    roishape = [roisize,roisize]

    img_queue, roi_queue = SpotDetectionMethods(ctx).CreateQueue(imgshape, roishape, spotDetector)
    t0 = time.time()

    for img in mov:
        img_queue.PushFrameU16(img)
   
    while img_queue.NumFinishedFrames() < len(mov):
        time.sleep(0.1)
    
    dt = time.time() - t0
    print(f"Processed {len(mov)} frames in {dt:.2f} seconds. {len(mov)/dt:.1f} fps")
    
    rois, data = roi_queue.Fetch()
    roipos = np.array([rois['x'],rois['y'],rois['z']]).T
    return roipos, data

    
    
def localization(psf, rois, initial_guess):
    # Direct mode, simpler but slower
    #return psf.Estimate(rois, initial=initial_guess)[0], np.arange(len(rois))
    
    print(f"Running localization on {len(rois)} ROIs...",end="")
    t0 = time.time()
    est_queue = EstimQueue(psf)
    est_queue.Schedule(rois, ids=np.arange(len(rois)), initial=initial_guess)
    est_queue.Flush()
    est_queue.WaitUntilDone()
    r = est_queue.GetResults()
    est_queue.Destroy()
    print(f"{len(rois) / (time.time()-t0):.0f} localizations/s")

    r.SortByID(isUnique=True)
    return r.estim, r.ids

def generate_ground_truth_cross(img_width, num_emitters, object_size_um, pixelsize, emitter_intensity):
    N = num_emitters
    pixelsize = 0.1 #um/pixel
            
    xyzI = np.zeros((N,4))
    pos = np.random.uniform(0,object_size_um,N) - object_size_um/2
    xyzI[:,0] = pos / pixelsize + img_width*0.5
    xyzI[:,1] = np.random.uniform(0.2,0.8,N) * img_width
    xyzI[:,2] = pos
    xyzI[:,3] = emitter_intensity
    xyzI[:N//2,2] = -xyzI[:N//2,2]
    return xyzI    


def generate_ground_truth_cylinder(img_width, num_emitters, object_size_um, pixelsize, emitter_intensity):
    N = num_emitters
    pixelsize = 0.1 #um/pixel
    angle = np.random.uniform(0, 2 * math.pi, N)
            
    xyzI = np.zeros((N,4))
    xyzI[:,0] = object_size_um/2 / pixelsize * np.cos(angle) + img_width / 2
    xyzI[:,1] = np.linspace(0.2,0.8,N) * img_width
    xyzI[:,2] = np.sin(angle) * object_size_um/2
    xyzI[:,3] = emitter_intensity
    
    return xyzI

def write_tif(mov,fn):
    with tifffile.TiffWriter(fn) as t:
        for m in mov:
            t.save(np.ascontiguousarray(m,dtype=np.uint16))


def show_zstack(psf):
    roisize = psf.sampleshape[-1]
    calib = psf.calib
    
    print(f'Z range according to calibration file: {calib.zrange}')
    
    z_range = [-0.5, 0.5]
    
    N = 200
    theta = np.repeat([[roisize/2,roisize/2,0,1,0.0]], N, axis=0)
    theta[:,[0,1]] =roisize/2#  np.random.uniform(-2,2,size=(N,2))+roisize/2
    theta[:,2]= np.linspace(z_range[0],z_range[1],N)
    
    ev = psf.ExpectedValue(theta)
    
    plt.figure()
    plt.plot(np.sum(ev,(1,2)))
    plt.title("Summed PSFs - should be ~1")

    theta[:,3]=1000
    theta[:,4]=1
    crlb = psf.CRLB(theta)
    pixelsize=100

    plt.figure()    
    plt.plot(theta[:,2], crlb[:,2] * 1000, label="Z")
    plt.plot(theta[:,2], crlb[:,0] * pixelsize, label="X")
    plt.plot(theta[:,2], crlb[:,1] * pixelsize, label="Y")
    plt.legend()
    plt.ylabel('CRLB [nm]')
    plt.xlabel('Z position [um]')
    plt.title('CRLB for 1000 photon emitter')
    plt.show()

    return ev
    

with Context(debugMode=False) as ctx:
    
    np.random.seed(0)
    
    # Use cubic spline PSFs, or a default astigmatic Gaussian?
    if False:
        roisize = 18
        psf = GaussianPSFMethods(ctx).CreatePSF_XYZIBg(roisize, Gauss3D_Calibration(), cuda=True)
        detection_threshold = 20
        emitter_intensity = 3000
    else:
        fn = 'tetrapod-psf.zstack'
        roisize = 18
        psf = CSplineMethods(ctx).CreatePSFFromFile(roisize, fn)
        detection_threshold = 10
        emitter_intensity = 2000

        show_zstack(psf)
        
    img_width = 64
    pixelsize = 0.100 # um/pixel
    background = 4 # photons/pixel
    object_size_um = 1 #um
    num_emitters = 1000
    xyzI = generate_ground_truth_cylinder(img_width, num_emitters, object_size_um = object_size_um,
                                 pixelsize = pixelsize, emitter_intensity=emitter_intensity)
    
    print("Generating SMLM example movie")
    mov_expval, on_counts = generate_storm_movie(psf, xyzI, numframes=2000, 
                                          imgsize=img_width,bg=background, p_on=1 / len(xyzI))

    print("Applying poisson noise")
    mov = np.random.poisson(mov_expval)
    
    write_tif(mov, 'cylinder.tif')
    
    bgimg = mov[0]*0
    psf_zrange = np.linspace(-object_size_um*0.6, object_size_um*0.6, 100)
    psfstack = psf_to_zstack(psf, psf_zrange)

    # this sets up the template-based spot detector. MinPhotons is not actually photons, still just AU.
    sd = PSFCorrelationSpotDetector(psfstack, bgimg, minPhotons=detection_threshold, maxFilterSizeXY=5, debugMode=False)
    roipos, rois = process_movie(mov, sd, roisize, ctx)
            
    plt.figure()
    hist = np.histogram(roipos[:,2],bins=len(psf_zrange),range=[0,len(psf_zrange)])
    plt.bar(psf_zrange, hist[0], width=(psf_zrange[-1]-psf_zrange[0])/len(hist[0]))
    plt.xlabel('Z position [um]')
    plt.title('Z position initial estimate from PSF convolutions')

    imshow_hstack(rois)

    initial_guess = np.ones((len(rois), 5)) * [roisize/2,roisize/2,0,0,1]
    initial_guess[:,2] = psf_zrange[roipos[:,2]]
    initial_guess[:,3] = np.sum(rois,(1,2))

    estim, ids = localization(psf, rois, initial_guess)
    rois = rois[ids]
    roipos = roipos[ids]
    
    # Filter out all ROIs with chi-square, this gets rid of ROIs with multiple spots.
    expval = psf.ExpectedValue(estim)
    chisq = np.sum( (rois-expval)**2 / (expval+1e-9), (1,2))
    
    std_chisq = np.sqrt(2*psf.samplecount + np.sum(1/np.mean(expval,0)))

    # Filter out all spots that have a chi-square > expected value + 2 * std.ev.
    chisq_threshold = psf.samplecount + 2*std_chisq
    accepted = chisq < chisq_threshold
    rejected = accepted == False
    print(f"Chi-Square threshold: {chisq_threshold:.1f}. Removing {np.sum(rejected)}/{len(rois)} spots")

    plt.figure()
    plt.hist(chisq,bins=100,range=[0,1000])
    plt.gca().axvline(chisq_threshold,color='r', label='threshold')
    plt.title('Chi-Square values for each localization')
    plt.legend()

    estim[:,[0,1]] += roipos[:,[0,1]]

    plt.figure()    
    plt.scatter(estim[accepted][:,0]*pixelsize, estim[accepted][:,2],c='b',marker='o',label='Estimated [chi-square accepted]')
    plt.scatter(estim[rejected][:,0]*pixelsize, estim[rejected][:,2],c='r',marker='x',label='Estimated [chi-square rejected]')
    plt.scatter(xyzI[:,0]*pixelsize, xyzI[:,2], s=1,label='Ground truth')
    plt.xlabel('X [microns]'); plt.ylabel('Z [microns]')
    plt.xlim([img_width*0.5*pixelsize-object_size_um*0.6, img_width*0.5*pixelsize+object_size_um*0.6])
    plt.ylim([-object_size_um*0.7,object_size_um*0.7])
    plt.legend()
    plt.title(f'X-Z section [{len(estim)} spots]')
    print(f"#spots: {len(estim)}")

    crlb = psf.CRLB([[roisize/2,roisize/2,np.min(estim[:,2]),xyzI[0,3],background]])
    print(f"CRLB Z at z={psf.calib.zrange[0]:.1f}: {crlb[0,2]:.2f} um")
    
    
    