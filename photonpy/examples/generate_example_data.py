"""
Generates a test dataset for estimate_traces.py
Use pip install photonpy=1.0.24
"""
import numpy as np
import matplotlib.pyplot as plt
from photonpy import Context, Gaussian
import tifffile
import tqdm # progress bar

def generate_movie(pos_xyI, psf, imgwidth, ctx:Context, background=2, numframes=2000, avg_on_time = 20, on_fraction=0.1):
    numspots = len(pos_xyI)

    p_off = 1-on_fraction
    k_off = 1/avg_on_time
    k_on =( k_off - p_off*k_off)/p_off 
    
    print(f"p_off={p_off}, k_on={k_on}, k_off={k_off}",flush=True)
    
    blinkstate = np.random.binomial(1, on_fraction, size=numspots)

    frames = np.zeros((numframes, imgwidth, imgwidth), dtype=np.uint16)

    for f in tqdm.trange(numframes):
        turning_on = (1 - blinkstate) * np.random.binomial(1, k_on, size=numspots)
        remain_on = blinkstate * np.random.binomial(1, 1 - k_off, size=numspots)
        blinkstate = remain_on + turning_on
        
        pos_on = pos_xyI[blinkstate==1]

        roisize = psf.sampleshape[0]
        roipos = np.clip((pos_on[:,[1,0]] - roisize/2).astype(int), 0, imgwidth-roisize)
        theta = np.zeros((len(pos_on),4))
        theta[:,:3] = pos_on
        theta[:,[1,0]] -= roipos

        rois = psf.ExpectedValue(theta)
        frame = ctx.smlm.DrawROIs((imgwidth,imgwidth), rois, roipos)
        frame += background
        frames[f] = np.random.poisson(frame)
    
    return frames


def generate(image_width=200, num_positions=500, avg_on_time=20):
    
    # generate some random binding site positions
    image_width = 200
    num_positions = 1000
    # generate positions [x, y, intensity]
    positions = np.random.uniform([0,0,100],[image_width,image_width,300], (num_positions,3))
    plt.figure(); plt.scatter(positions[:,0],positions[:,1]); plt.title('Binding site positions')
    
    # optical model defined by a 2D Gaussian:
    psf_sigma = 1.4     # [pixels]
    roisize = 10        # [pixels]
    
    with Context() as ctx:
        psf = Gaussian(ctx).CreatePSF_XYIBg(roisize, psf_sigma, cuda=True)
        mov = generate_movie(positions, psf, image_width, ctx, 
                       numframes = 2000,
                       avg_on_time = avg_on_time,
                       on_fraction = 0.05,
                       background = 5)
    
    tifffile.imsave('movie.tif', mov)
    return mov    
    
if __name__ == "__main__":
    mov = generate()

    # Napari is recommended as a quick way to view image data in python:
    import napari
    with napari.gui_qt():
        napari.view_image(mov)
    
