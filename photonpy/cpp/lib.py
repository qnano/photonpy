# -*- coding: utf-8 -*-

import ctypes
import os
import math
import numpy as np
import numpy.ctypeslib as ctl
import matplotlib.pyplot as plt
import sys

# https://stackoverflow.com/questions/32120178/how-can-i-pass-null-to-an-external-library-using-ctypes-with-an-argument-decla/32138619


def debugPrint(msg):
    sys.stdout.write(msg.decode("utf-8"))
    return 1 # also print using OutputDebugString on C++ side

#def debugImage(w,h,n,data,label):
#    sys.stdout.write(label.decode('utf-8'))

FloatArrayTypeBase = np.ctypeslib.ndpointer(dtype=np.float32, flags="C_CONTIGUOUS")

def _from_param_F32(cls, obj):
    if obj is None:
        return obj
    return FloatArrayTypeBase.from_param(obj)


NullableFloatArrayType = type("FloatArrayType", (FloatArrayTypeBase,), {"from_param": classmethod(_from_param_F32)})

IntArrayTypeBase = np.ctypeslib.ndpointer(dtype=np.int32, flags="C_CONTIGUOUS")

def _from_param_I32(cls, obj):
    if obj is None:
        return obj
    return IntArrayTypeBase.from_param(obj)

NullableIntArrayType = type("IntArrayType", (IntArrayTypeBase,), {"from_param": classmethod(_from_param_I32)})


class SMLM:
    def __init__(self, debugMode=False):
        thispath = os.path.dirname(os.path.abspath(__file__))

        if ctypes.sizeof(ctypes.c_voidp) == 4:
            raise RuntimeError(f"The SMLM library can only be used with 64-bit python.")

        if debugMode:
            dllpath = "/../x64/Debug/photonpy.dll"
        else:
            dllpath = "/../x64/Release/photonpy.dll"

        abs_dllpath = os.path.abspath(thispath + dllpath)
        
        #if not os.path.exists(abs_dllpath):
        #    abs_dllpath = pkg_resources.resource_filename('photonpy', 'data/')

        if debugMode:
            print("Using " + abs_dllpath)
        self.debugMode = debugMode
        
        currentDir = os.getcwd()
        os.chdir(os.path.dirname(abs_dllpath))

        smlmlib = ctypes.CDLL(abs_dllpath)
        os.chdir(currentDir)
        
        self.lib_path = abs_dllpath
        self.lib = smlmlib
        
        CudaGetNumDevices=smlmlib.CudaGetNumDevices
        CudaGetNumDevices.argtypes=[]
        CudaGetNumDevices.restype=ctypes.c_int32
        self.numDevices = CudaGetNumDevices()
        
        self._CudaSetDevice = smlmlib.CudaSetDevice
        self._CudaSetDevice.argtypes=[ctypes.c_int32]
        self._CudaSetDevice.restype = ctypes.c_bool
        
        self._CudaGetDeviceInfo = smlmlib.CudaGetDeviceInfo
        self._CudaGetDeviceInfo.argtypes=[
                ctypes.c_int32,
                ctypes.POINTER(ctypes.c_int32),
                ctypes.c_char_p,
                ctypes.c_int32]
        self._CudaGetDeviceInfo.restype =ctypes.c_bool
        
            
        #void CudaGetMemoryUse(int& pinnedBytes, int& devicePitchedBytes, int& deviceBytes, int& pitchedNumAllocs)
        self._CudaGetMemoryUse = smlmlib.CudaGetMemoryUse
        self._CudaGetMemoryUse.argtypes = [
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64),
            ctypes.POINTER(ctypes.c_uint64)
        ]
        
        self.DebugPrintCallback = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_char_p)
        self._SetDebugPrintCallback = smlmlib.SetDebugPrintCallback
        self._SetDebugPrintCallback.argtypes = [self.DebugPrintCallback]

#void(*cb)(int width,int height, int numImg, const float* data, const char* title));

        self.DebugImageCallbackType = ctypes.CFUNCTYPE(
            None, ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, 
                ctypes.POINTER(ctypes.c_float), ctypes.c_char_p
        )
        self._SetDebugImageCallback = smlmlib.SetDebugImageCallback
        self._SetDebugImageCallback.argtypes = [self.DebugImageCallbackType]
        
        #CDLL_EXPORT void ShowDebugImage(int w, int h, int numImages, const float* data, const char *title)

        self._ShowDebugImage = smlmlib.ShowDebugImage
        self._ShowDebugImage.argtypes = [
             ctypes.c_int32, ctypes.c_int32, ctypes.c_int32, 
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
            ctypes.c_char_p
            ]
        
#        self._GetDeviceMemoryAllocation = smlmlib.GetDeviceMemoryAllocation

        # CDLL_EXPORT void FFT(const cuFloatComplex* src, cuFloatComplex* dst, int batchsize, int siglen, int forward)
        self._FFT = smlmlib.FFT
        self._FFT.argtypes = [
            ctl.ndpointer(np.complex64, flags="aligned, c_contiguous"),
            ctl.ndpointer(np.complex64, flags="aligned, c_contiguous"),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        self._FFT2 = smlmlib.FFT2
        self._FFT2.argtypes = [
            ctl.ndpointer(np.complex64, flags="aligned, c_contiguous"),
            ctl.ndpointer(np.complex64, flags="aligned, c_contiguous"),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_int32,
        ]
        
        #
#CDLL_EXPORT void DrawROIs(float* image, int width, int height, const float* rois, 
#int numrois, int roisize, Int2* roiposYX)
        self._DrawROIs = smlmlib.DrawROIs
        self._DrawROIs.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
            ctypes.c_int32,
            ctypes.c_int32,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
            ctypes.c_int32,
            ctypes.c_int32,
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous")
        ]

        self.SetDebugPrintCallback(debugPrint)
        
        def imgcb(w,h,n,data,label):
            #ptr = ctypes.pointer(c_arr[0])
            images = np.ctypeslib.as_array(data, shape=(n,h,w))*1
            if self.debugImageCallback is not None:
                self.debugImageCallback(images, label.decode('utf-8'))

        self.debugImageCallback = None
        self.debugImageCFunction = self.DebugImageCallbackType(imgcb)
        self._SetDebugImageCallback(self.debugImageCFunction)

    def GetCudaMemUse(self):
        #void CudaGetMemoryUse(int& pinnedBytes, int& devicePitchedBytes, int& deviceBytes, int& pitchedNumAllocs)

        pinned = ctypes.c_uint64()
        pitched = ctypes.c_uint64()
        device = ctypes.c_uint64()
        numallocs = ctypes.c_uint64()
        self._CudaGetMemoryUse(pinned, pitched, device, numallocs)
        
        return pinned.value, pitched.value, device.value, numallocs.value

    def GetCudaDevices(self):
        names = []
        for d in range(self.numDevices):
            mpu = ctypes.c_int32()
            namebuf = ctypes.create_string_buffer(100)
            self._CudaGetDeviceInfo(d, ctypes.byref(mpu), namebuf, 100)
            name = namebuf.value.decode("utf-8")
            print(f"Device {d}: {name}. #MPU={mpu.value}")
            names.append(name)
        return names
    
    def SetCudaDevice(self, idx):
        self._CudaSetDevice(idx)

    def ListDebugImages(self):
        for i, v in enumerate(self.debugImages):
            print(f"{v[1]}: {v[0].shape}")

    def SetDebugPrintCallback(self, fn):
        self.dbgPrintCallback = self.DebugPrintCallback(fn)  # make sure the callback doesnt get garbage collected
        self._SetDebugPrintCallback(self.dbgPrintCallback)

    def SetDebugImageCallback(self, fn):
        self.debugImageCallback = fn
        
    #  To test the callbacks
    def ShowDebugImage(self, img, label):
        if len(img.shape)==2: img=np.array([img])
        img = np.ascontiguousarray(img, dtype=np.float32)
        self._ShowDebugImage(img.shape[2],img.shape[1],img.shape[0],img,label.encode('utf-8'))

    def FFT(self, src, forward=True):
        src_shape = src.shape
        if len(src.shape) == 1:
            src = [src]

        src = np.ascontiguousarray(src, dtype=np.complex64)
        dst = np.zeros(src.shape, dtype=np.complex64)

        batchsize = src.shape[0]
        siglen = src.shape[1]

        self._FFT(src, dst, batchsize, siglen, forward)
        if len(src_shape) == 1:
            return dst[0]
        return dst

    def IFFT(self, src):
        return self.FFT(src, forward=False)

    def FFT2(self, src, forward=True):
        src_shape = src.shape
        if len(src.shape) == 2:
            src = [src]

        src = np.ascontiguousarray(src, dtype=np.complex64)
        dst = np.zeros(src.shape, dtype=np.complex64)

        batchsize = src.shape[0]
        sigw = src.shape[1]
        sigh = src.shape[2]

        self._FFT2(src, dst, batchsize, sigw, sigh, forward)

        if len(src_shape) == 2:
            return dst[0]
        return dst

    def IFFT2(self, src):
        return self.FFT2(src, forward=False)
    
    def DrawROIs(self, imageOrShape, rois, roiposYX):
        if len(np.array(imageOrShape).shape) == 1 and len(imageOrShape)==2:
            image = np.zeros(imageOrShape,dtype=np.float32)
        else:
            image = imageOrShape

        if len(rois)>0:
            rois = np.ascontiguousarray(rois,dtype=np.float32)
            roiposYX = np.ascontiguousarray(roiposYX, dtype=np.int32)
            
            assert(len(rois.shape)==3)
            roisize = rois.shape[2]
            assert(rois.shape[1]==rois.shape[2])
            assert(np.array_equal(roiposYX.shape, (len(rois),2)))
            
            self._DrawROIs(image, image.shape[1], image.shape[0], rois, len(rois), roisize, roiposYX)
        return image

    def Close(self):
        if self.lib is not None:
            # Free DLL so we can overwrite the file when we recompile
            ctypes.windll.kernel32.FreeLibrary.argtypes = [ctypes.wintypes.HMODULE]
            ctypes.windll.kernel32.FreeLibrary(self.lib._handle)
            self.lib = None
        

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.Close()
        
