"""
Spot detection test without fitting
"""
import numpy as np
import matplotlib.pyplot as plt
from photonpy.cpp.gaussian import Gaussian
from photonpy.cpp.context import Context
import photonpy.cpp.spotdetect as spotdetect
import math
import photonpy.smlm.util as smlmutil
import time
import tqdm

def generate_storm_movie(gaussian, emitterList, numframes=100, imgsize=512, intensity=500, bg=2, sigma=1.5, p_on=0.1):
    frames = np.zeros((numframes, imgsize, imgsize), dtype=np.uint16)
    emitters = np.array([[e[0], e[1], sigma, sigma, intensity] for e in emitterList])

    on_counts = np.zeros(numframes, dtype=np.int32)

    for f in range(numframes):
        frame = bg * np.ones((imgsize, imgsize), dtype=np.float32)
        frame_emitters = emitters * 1
        on = np.random.binomial(1, p_on, len(emitters))
        frame_emitters[:, 4] *= on

        frame = gaussian.Draw(frame, frame_emitters)
        frames[f] = frame
        on_counts[f] = np.sum(on)

    return frames, on_counts


psfSigma = 1.8
roisize = 10
w = 256
N = 2000
numframes = 50
R = np.random.normal(0, 0.2, size=N) + w * 0.3
angle = np.random.uniform(0, 2 * math.pi, N)
emitters = np.vstack((R * np.cos(angle) + w / 2, R * np.sin(angle) + w / 2)).T


with Context(debugMode=True) as ctx:
    gaussian = Gaussian(ctx)
    mov, on_counts = generate_storm_movie(gaussian, emitters, numframes, imgsize=w, sigma=psfSigma, p_on=20 / N)
    mov = np.random.poisson(mov)

    spotDetector = spotdetect.GLRTSpotDetector(psfSigma, roisize, fdr=0.5)
    
    processFrame = spotdetect.SpotDetectionMethods(ctx).ProcessFrame

    repeats=1
    numspots=0
    t0 = time.time()
    for r in tqdm.trange(repeats):
        for f in range(len(mov)):
            rois, cornerpos, scores = processFrame(mov[f], spotDetector, 100)
            numspots += len(rois)

    t1 = time.time()
    
    print(f"Time: {t1-t0} s. {repeats*len(mov)/(t1-t0)} fps. numspots={numspots}")
    smlmutil.imshow_hstack(rois)
    