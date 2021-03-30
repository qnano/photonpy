import ctypes as ct
from .lib import SMLM
import numpy as np
import numpy.ctypeslib as ctl
from .context import Context

#CDLL_EXPORT sCMOS_CalibrationTransform* sCMOS_Calib_Create(int w, int h, 
#const float* offset, const float* gain, const float *variance, Context* ctx);
#CDLL_EXPORT GainOffsetTransform* GainOffsetCalib_Create(float gain, float offset, Context* ctx);


class sCMOS_Calib:
    def __init__(self, ctx, offset, gain, variance):
        self._sCMOS_Calib_Create = ctx.lib.sCMOS_Calib_Create
        self._sCMOS_Calib_Create.argtypes = [
                ct.c_int32, ct.c_int32,
                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), 
                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                ct.c_void_p
                ]
        self._sCMOS_Calib_Create.restype = ct.c_void_p

        offset = np.ascontiguousarray(offset,dtype=np.float32)
        gain = np.ascontiguousarray(gain,dtype=np.float32)
        variance = np.ascontiguousarray(variance,dtype=np.float32)
        
        assert(len(offset.shape)==2)
        assert(np.array_equal(offset.shape,gain.shape))
        assert(np.array_equal(offset.shape,variance.shape))
        
        self.inst = self._sCMOS_Calib_Create(offset.shape[1],offset.shape[0],offset,gain,variance,ctx.inst)
        
# Constant global Gain/offset 
class GainOffset_Calib:
    def __init__(self, gain, offset, ctx):
        self._GainOffsetCalib_Create = ctx.lib.GainOffsetCalib_Create
        self._GainOffsetCalib_Create.argtypes=[
                ct.c_float, 
                ct.c_float, 
                ct.c_void_p]
        self._GainOffsetCalib_Create.restype=ct.c_void_p
        self.inst = self._GainOffsetCalib_Create(gain,offset,ctx.inst)


# Gain/offset supplied by image
class GainOffsetImage_Calib:
    def __init__(self, gain, offset, ctx):
        self._GainOffsetImageCalib_Create = ctx.lib.GainOffsetImageCalib_Create
        self._GainOffsetImageCalib_Create.argtypes=[
                ct.c_int32, ct.c_int32,
                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
                ct.c_void_p]
        self._GainOffsetImageCalib_Create.restype=ct.c_void_p

        gain = np.ascontiguousarray(gain,dtype=np.float32)
        offset = np.ascontiguousarray(offset,dtype=np.float32)
        assert(np.array_equal(gain.shape,offset.shape))
        self.inst = self._GainOffsetImageCalib_Create(gain.shape[1], gain.shape[0], gain,offset,ctx.inst)
