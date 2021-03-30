# Drift estimation example using Minimum Entropy
# use pip install photonpy==1.0.35

from photonpy import Dataset
import os

fn = 'sim4_1.hdf5'
ds = Dataset.load(fn, pixelsize=100)

ds.estimateDriftMinEntropy(initializeWithRCC=5, framesPerBin=4, debugMode=False,
                           maxdrift=5, apply=True, useCuda=True)

#ds.save(os.path.splitext(fn)[0]+"_undrifted_dme.hdf5")
