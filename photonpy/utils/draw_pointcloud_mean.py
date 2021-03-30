# -*- coding: utf-8 -*-
# Render a 2D pointcloud into an image, applying gaussian smoothing on the values

import sys
sys.path.append("..")
import numpy as np
from smlmlib.base import SMLM
from smlmlib.gaussian import Gaussian


def draw(pts, image_size, filter_sigma=10,smlm=None):
    if smlm is None:
        with SMLM() as smlm:
            return _draw(pts, image_size, filter_sigma, smlm)
    else:
        return _draw(pts, image_size, filter_sigma, smlm)



def _add_gauss2D(img, x,y,sx,sy, intensity):
    X,Y=np.meshgrid(range(img.shape[1]),range(img.shape[0]))
    img += np.exp((X-x)**2/(2*sx**2) + (Y-y)**2/(2*sy**2)) 
    

def _draw(pts, image_size, filter_sigma, smlm):
    g = Gaussian(smlm)
    
    # Spots is an array with rows: [ x,y, sigmaX, sigmaY, intensity ]
    img = np.zeros(image_size, dtype=np.float32)

    spots = np.zeros((len(pts), 5), dtype=np.float32)
    spots[:, 0] = pts[:,0]
    spots[:, 1] = pts[:,1]
    spots[:, 2] = filter_sigma
    spots[:, 3] = filter_sigma
    spots[:, 4] = pts[:,2]
    
    img = g.Draw(img, spots, 100)
    
    density = img*0+1e-6
    spots[:, 4] = 1
    g.Draw(density, spots)

    return img / density,img,density

   
if __name__ == "__main__":
    pts = np.random.uniform(np.array([-50,-50,800]),np.array([150,150,1000]),size=(300,3))
    
    mean,img,density = draw(pts, [100,100])
    import matplotlib.pyplot as plt
    
    plt.figure()
    plt.imshow(mean)
    plt.scatter(pts[:,0],pts[:,1], c=pts[:,2])
    plt.figure()
    plt.imshow(img)
    plt.scatter(pts[:,0],pts[:,1], c=pts[:,2])
    
            