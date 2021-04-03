import setuptools
import os,shutil

#https://stackoverflow.com/questions/24071491/how-can-i-make-a-python-wheel-from-an-existing-native-library
class BinaryDistribution(setuptools.Distribution):
    def has_ext_modules(foo):
        return True

long_description = """
A C++/CUDA toolbox with python bindings for doing single molecule microscopy image processing.

Features include:    
    - Super fast spot detection and extraction on tiff files (~1k frames/s on my dell xps laptop)
    - Max-Likelihood fitting of various 2D Gaussian models
        (XY with or without fixed sigma, 3D using astigmatism)
    - SIMFLUX (https://www.nature.com/articles/s41592-019-0657-7)
    - Easy rendering of results: Piccaso Render compatible HDF5 export (https://github.com/jungmannlab/picasso)
    - Phasor-based SMLM (https://aip.scitation.org/doi/full/10.1063/1.5005899)
    - A C++ templated approach to quickly implement new PSF models as long as the first derivative can be computed.
    - Drift correction in 2D and 3D
    - Localization using cubic spline PSFs from https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6009849/

Currently compiled for CUDA 11.2 for Windows 10

Credits to Willem Melching (@pd0wm) for general debugging and implementation of astigmatic Gaussian PSF models, as well as yet unpublished code.

"""
#    - Cubic spline PSF model fitting (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6009849/)


pk = setuptools.find_packages(exclude=['utils'])

print(pk)

dstdir = os.path.dirname(__file__) + "/photonpy/x64/Release"
print(dstdir)
#os.makedirs(dstdir,exist_ok=True)
#shutil.copy(os.path.dirname(__file__)+"/x64/Release/photonpy.dll",dstdir)

setuptools.setup(
    name="photonpy",
    version="1.0.38",
    author="Jelmer Cnossen",
    author_email="j.p.cnossen@tudelft.nl",
    description="CUDA-Based image processing for single molecule localization microscopy",
    long_description=long_description,
 #   long_description_content_type="text/markdown",
    url="https://github.com/qnano/photonpy",
    packages=pk,
    #package_data={'smlmlib':['x64/Release/photonpy.dll'] },
	data_files=[('lib/site-packages/photonpy/x64/Release', 
              ['photonpy/x64/Release/photonpy.dll', 
		'photonpy/x64/Release/cudart64_110.dll',
		'photonpy/x64/Release/vcruntime140.dll',
		'photonpy/x64/Release/vcruntime140_1.dll'
		])],
    classifiers=[
        "Programming Language :: Python :: 3",
		"Programming Language :: C++",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Microsoft :: Windows"
    ],
	install_requires=[
		'scipy', 
		'numpy',
		'matplotlib',
		'tqdm',
		'tifffile',
        'h5py',
		'pyyaml',
        'scikit-image',
        'pyqtgraph'
	]#,    distclass=BinaryDistribution
)
