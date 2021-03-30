# -*- coding: utf-8 -*-

from photonpy import Context,Dataset
import numpy as np
import matplotlib.pyplot as plt

W=200
X,Y = np.meshgrid(np.linspace(0,W), np.linspace(0,W))
N = X.size
ds = Dataset(N, 2, [W,W])

ds.pos[:,0] = X.flatten()
ds.pos[:,1] = Y.flatten()
ds.crlb.pos = 1
ds.photons[:] = 1

idx = ds.pick([[W/2,W/2]], W/4)

plt.figure()
plt.imshow(ds[idx[0]].renderGaussianSpots(4,3))
plt.show()



