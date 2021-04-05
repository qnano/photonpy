README
----------------------------------------------------------------------------

Photonpy is a Python-wrapped C++/CUDA library for processing SMLM (single molecule localization microscopy data). 
It currently only runs on Windows 64-bit with CUDA, but I'm working on making the CUDA optional. Help with porting to GCC/linux is appreciated.

Features include:
* Super fast spot detection and extraction on tiff files (~1k frames/s on my dell xps laptop)
* Max-Likelihood fitting of various 2D Gaussian models (XY with or without fixed sigma, 3D using astigmatism), Center-Of-Mass and [Phasor-based SMLM](https://aip.scitation.org/doi/full/10.1063/1.5005899)
  * Optimized CUDA code results in > 1M localizations/s on any reasonable CUDA card, on typical [2D Gaussian MLE](https://www.nature.com/articles/nmeth.1449)
* [SIMFLUX](https://www.nature.com/articles/s41592-019-0657-7)
* Easy rendering of results: [Piccaso Render](https://github.com/jungmannlab/picasso) compatible HDF5 import/export
* A C++ templated approach to quickly implement new PSF models as long as the first derivative can be computed.
* Drift correction in 2D and 3D
* 3D Localization using [cubic spline PSFs](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6009849/)

Currently compiled for CUDA 11.2 / Windows x64 using Visual Studio 2019

Credits to Willem Melching (@pd0wm) for general debugging and implementation of astigmatic Gaussian PSF models, as well as yet unpublished code.



Installing as a user
----------------------------------------------------------------------------

Step 1. 
Install python. Anaconda is recommended: https://www.anaconda.com/distribution/

Step 2.
Create a virtual environment, such as an anaconda environment:

conda create -n myenv anaconda python=3.8
conda activate myenv

Step 3.
Install photonpy:

pip install photonpy

Step 4. 
Install CUDA Toolkit 10.1 update 2. We had some problems running the build from 10.2 on some PCs.
https://developer.nvidia.com/cuda-10.1-download-archive-update2

Step 5.
You should now be able to run some examples:
python -m photonpy.examples.localize_spots.py


Installing as developer / Building from source
----------------------------------------------------------------------------
Perform step 1 + 2 listed above.

Step 3. Install Visual studio 2019 Community Edition: https://visualstudio.microsoft.com/downloads/

Step 4. Extract the external libraries (cpp/external.zip) so cpp contains a folder called "external"

Step 5. In visual studio, open the smlm.sln, set the build to release mode and x64 platform, and build SMLMLib. 

Step 6.
Open this directory and install the photonpy python package in developer mode (this way your python environment knows photonpy is located in the path you installed it in)
python setup.py develop

Step 7.
You should be able to run photonpy/examples/localize_spots.py now


Possible fixes should above steps not lead to the desired result:
- Change CUDA setting: search for NSight Monitor, run as admin. It will probably not open a new window but show up as background process hid under the arrow at the bottomright part of your screen. Rightclick icon and go to 'Options'. Change 'WDDM TDR Delay' to '180' and 'WDDM TDR Enabled' to 'False'. You will probably have to reboot. 
- Update GPU driver: search for 'device manager', go to 'Display adapters', right click your video card (usually NVIDIA XXXX), click properties -> Driver -> Update Driver..., you will probably have to reboot.
