# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 13:06:46 2018

@author: jcnossen1
"""
import numpy as np
import math
import matplotlib.pyplot as plt
import sys

# smp = [numspots, width, height]
def compute_com(smp):
    numdims=len(smp.shape)
    if numdims==2:
        smp=np.array([smp])
    w = smp.shape[1]
    h = smp.shape[2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    X = np.tile(X, (smp.shape[0], 1, 1))
    Y = np.tile(Y, (smp.shape[0], 1, 1))
    moment_x = np.sum(smp * X, (1, 2))
    moment_y = np.sum(smp * Y, (1, 2))
    sums = np.sum(smp, (1, 2))

    bgfraction = 0.2

    theta = np.ascontiguousarray(
        np.array([moment_x / sums, moment_y / sums, sums * (1 - bgfraction), bgfraction * sums / (w * h)], np.float32).T
    )
    
    if numdims==2:
        return theta[0]
    
    return theta


def imshow_many(imgs, title=None, cols=6, maxrows=5, **fig_kw):
    n = imgs.shape[0]
    nrows = math.ceil(n / cols)

    if nrows > maxrows:
        nrows = maxrows

    fig, axes = plt.subplots(nrows, cols, sharex=True, sharey=True,squeeze=False, **fig_kw)

    for y in range(nrows):
        for x in range(cols):
            if y * cols + x < len(imgs):
                axes[y][x].imshow(imgs[y * cols + x])

    if title:
        fig.suptitle(title)

    return fig


def imshow_hstack(imgs, title=None, max_num_img=10,colorbar=False):
    if len(imgs) == 0:
        return
    
    n = np.minimum(max_num_img, len(imgs))
    fig=plt.figure()
    plt.imshow(np.hstack(imgs[:n]))
    if title:
        plt.title(title)
    return fig

def imshow_rois(imgs, title=None, cols=10, maxrows=10,colorbar=False):
    n = imgs.shape[0]
    imgh = imgs.shape[1]
    imgw = imgs.shape[2]

    nrows = (n+cols-1)//cols
    
    if nrows > maxrows:
        nrows = maxrows
        n=maxrows*cols
        
    img = np.zeros((imgh*nrows,imgw*cols))
    for k in range(n):
        y = k//cols
        x = k%cols
        img[imgh*y:imgh*(y+1),imgw*x:imgw*(x+1)]=imgs[k]

    fig=plt.figure()
    plt.imshow(img)
    if title:
        plt.title(title)
    plt.show()
    return fig

def save_movie(imgs, fn, fps=15):
    import matplotlib.animation as manimation

    print(f"saving {len(imgs)} images to {fn}")
    FFMpegWriter = manimation.writers["ffmpeg"]
    metadata = dict(title="Movie Test", artist="Matplotlib", comment="Movie support!")
    writer = FFMpegWriter(fps=fps, metadata=metadata)

    fig = plt.figure()
    with writer.saving(fig, fn, dpi=100):
        for i in range(len(imgs)):
            sys.stdout.write(".")
            plt.imshow(imgs[i])
            writer.grab_frame()
            plt.clf()

    print(f"done")


def sum_epp(img):
    return np.sum(img, 1)


def chisq(mu, smp):
    return np.sum(mu - smp - smp * np.log(mu / np.maximum(smp, 1)))


def loglikelihood(mu, smp):
    return np.sum(smp * np.log(np.maximum(mu, 1e-9)) - mu)


def extract_roi(images, x, y, roisize):
    hs = roisize // 2
    r = np.zeros((len(images), roisize, roisize))
    for f in range(len(images)):
        r[f] = images[f, y - hs : y + hs, x - hs : x + hs]
    return r


# x and y are arrays
def extract_rois(images, x, y, roisize):
    images = np.array(images)
    r = np.zeros((len(x), len(images), roisize, roisize), dtype=images.dtype)
    halfroi = roisize // 2
    for s in range(len(x)):
        for f in range(len(images)):
            r[s, f] = images[f, y[s] - halfroi : y[s] - halfroi + roisize, x[s] - halfroi : x[s] - halfroi + roisize]

    return r



def plot_traces(traces, theta, psf, axis):
    plt.figure()
    for k in range(len(traces)):
        plt.plot(np.arange(len(traces[k])),traces[k][:,axis])
        plt.plot([len(traces[k])],[theta[k,axis]],'o')
    plt.title(f"Axis {axis}[{psf.param_names[axis]}]")
