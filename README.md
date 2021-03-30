README
----------------------------------------------------------------------------

Photonpy is a Python-wrapped C++/CUDA library for processing SMLM (single molecule localization microscopy data). 
It currently only runs on Windows 64-bit with CUDA, but I'm working on making the CUDA optional. Help with porting to GCC/linux is appreciated.

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
