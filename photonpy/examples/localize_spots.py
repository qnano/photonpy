import numpy as np
import matplotlib.pyplot as plt
from photonpy import GaussianPSFMethods
from photonpy import Context, Dataset
from photonpy.cpp.estim_queue import EstimQueue
import photonpy.cpp.spotdetect as spotdetect
import photonpy.smlm.process_movie as process_movie
import photonpy.utils.multipart_tiff as tiff
from photonpy.cpp.calib import GainOffset_Calib
import math
import os
import tifffile
import time

def generate_storm_movie(ctx:Context, emitterList, numframes=100, 
                         imgsize=512, intensity=500, bg=2, sigma=1.5, p_on=0.1):
    frames = np.zeros((numframes, imgsize, imgsize), dtype=np.float32)
    emitters = np.array([[e[0], e[1], sigma, sigma, intensity] for e in emitterList])

    on_counts = np.zeros(numframes, dtype=np.int32)

    gaussianApi = GaussianPSFMethods(ctx) 
    for f in range(numframes):
        frame = bg * np.ones((imgsize, imgsize), dtype=np.float32)
        frame_emitters = emitters * 1
        on = np.random.binomial(1, p_on, len(emitters))
        frame_emitters[:, 4] *= on

        frames[f] = gaussianApi.Draw(frame, frame_emitters)
        on_counts[f] = np.sum(on)

    return frames, on_counts



def localize(fn, sigma, roisize, threshold, gain, offset, context:Context, output_file=None, estimate_sigma=False):
    imgshape = tiff.tiff_get_image_size(fn)

    gaussian = GaussianPSFMethods(ctx)
    
    #cameraCalib = GainOffset_Calib(3, 100, context)
        
    spotDetectorType = spotdetect.SpotDetector(np.mean(sigma), roisize, threshold)

    if estimate_sigma:
        psf = gaussian.CreatePSF_XYIBgSigmaXY(roisize, sigma, True)
    else:
        psf = gaussian.CreatePSF_XYIBg(roisize, sigma, True)

    calib = process_movie.create_calib_obj(gain, offset, imgshape, ctx)
    
    sumframes=1

    with Context(context.smlm) as sd_ctx:
        roishape = [roisize,roisize]
    
        img_queue, roi_queue = spotdetect.SpotDetectionMethods(ctx).CreateQueue(
            imgshape, roishape, spotDetectorType, calib, sumframes=1, ctx=sd_ctx)

        numframes = 0
        for img in tiff.tiff_read_file(fn, 100, 200):
            img_queue.PushFrameU16(img)
            numframes += 1
            
        print(f"Numframes: {numframes}")
       
        while img_queue.NumFinishedFrames() < numframes//sumframes:
            time.sleep(0.1)
    
        info, rois = roi_queue.Fetch()

    # the ROIs and their position in the larger frame are now known, so we can run MLE estimation on them

    roipos = np.zeros((len(rois),2))
    roipos[:,0] = info['y']
    roipos[:,1] = info['x']
    
    count = 0    
    queue = EstimQueue(psf, batchSize=1024)
    queue.Schedule(rois, ids=np.arange(count, count+len(rois)), roipos=roipos)
    queue.WaitUntilDone()

    r = queue.GetResults()
    r.SortByID() # sort by frame numbers
    
    print(f"Filtering {len(r.estim)} spots...")
    minX = 2.1
    minY = 2.1
    r.FilterXY(minX,minY,roisize-minX-1, roisize-minY-1)
    r.Filter(r.iterations<50)
    
    plt.figure()
    plt.hist(r.chisq,bins=50)
    
    nframes = np.max(r.ids)+1 if len(r.ids)>0 else 1
    print(f"Num spots: {len(r.estim)}. {len(r.estim) / nframes:.1f} spots/frame")
    
    if output_file is not None:
        Dataset.fromQueueResults(r, imgshape).save(output_file)
    
    return r,imgshape



psfSigma = 1.8
roisize = int(4 + (psfSigma + 1) * 2)
w = 40
N = 2000
numframes = 2000
R = w * 0.3
angle = np.random.uniform(0, 2 * math.pi, N)
emitters = np.vstack((R * np.cos(angle) + w / 2, R * np.sin(angle) + w / 2)).T


with Context(debugMode=False) as ctx:
    
    fn = "generated-movie.tif"
    if True:
        print("Generating SMLM example movie")
        mov_expval, on_counts = generate_storm_movie(ctx, emitters, numframes, 
                                              imgsize=w, intensity=300,bg=10,sigma=psfSigma, p_on=20 / N)
        print("Applying poisson noise")
        mov = np.random.poisson(mov_expval) * 3 + 100
    
        print("Saving movie to {0}".format(fn))
        tiff.imsave(fn, np.array(mov, dtype=np.uint16))
    
        plt.figure()
        plt.imshow(mov[0])
        plt.title("Frame 0")
    
    print(f"Performing localization (roisize={roisize})",flush=True)#flush is needed so the tqdm progress bar is not messed up
    r,imgshape = localize(fn, [2,2], roisize, threshold=2, gain=3, offset=100,
             context=ctx, output_file="generated-movie.hdf5", 
             estimate_sigma=True)
    
    crlb = r.CRLB()
    print(f"Estimated sigmas: {np.median(r.estim[:,4:],0)}. Mean CRLB X: {np.mean(crlb[:,0]):.2f} pixels")

    plt.figure()
    plt.scatter(r.estim[:,0] + r.roipos[:,1],r.estim[:,1]+r.roipos[:,0])
    plt.title('Localized spots')
    plt.show()    

