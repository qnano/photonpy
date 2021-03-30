README
----------------------------------------------------------------------------

Step 1. Download and install Anaconda for Windows 64 bit
https://www.anaconda.com/distribution/
https://repo.anaconda.com/archive/Anaconda3-2018.12-Windows-x86_64.exe

Step 2: Install Visual studio 2019 Community Edition: https://visualstudio.microsoft.com/downloads/

Step 3. Install CUDA Toolkit 10.1 update 2. We had some problems running the build from 10.2 on some PCs.
https://developer.nvidia.com/cuda-10.1-download-archive-update2
The toolkit needs to be installed after installing Visual Studio

Step 4. Extract the external libraries (cpp/external.zip) so cpp contains a folder called "external"

Step 5. In visual studio, open the smlm.sln, set the build to release mode, and build SMLMLib. 

Step 6.
Create a virtual environment in anaconda:
You should have an "Anaconda Prompt" somewhere in the windows apps now.
Open this anaconda prompt and run the following to create an anaconda environment named myenv:
conda create -n myenv anaconda python=3.8
conda activate myenv

Step 7.
Open this directory and install the photonpy python package in developer mode:
python setup.py develop

Step 8.
You should be able to run photonpy/examples/localize_spots.py now





Possible debugs should above steps not lead to the desired result:
- Update GPU driver: search for 'device manager', go to 'Display adapters', right click your video card (usually NVIDIA XXXX), click properties -> Driver -> Update Driver..., you will probably have to reboot.
- Change CUDA setting: search for NSight Monitor, run as admin. It will probably not open a new window but show up as background process hid under the arrow at the bottomright part of your screen. Rightclick icon and go to 'Options'. Change 'WDDM TDR Delay' to '180' and 'WDDM TDR Enabled' to 'False'. You will probably have to reboot. 
