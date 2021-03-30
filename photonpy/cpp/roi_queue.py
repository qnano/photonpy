import ctypes as ct
import numpy as np
import numpy.ctypeslib as ctl

from .context import Context

ROIInfoDType = np.dtype([
    ('id','<i4'),('score','<f4'),('x','<i4'),('y','<i4'), ('z','<i4')
])


class ROIQueue:
    def __init__(self, roishape, ctx:Context=None):
        self.ctx = ctx
        lib = self.ctx.smlm.lib

        InstancePtrType = ct.c_void_p
        
        #ROIQueue* RQ_Create(const Int3& shape);
        _RQ_Create = lib.RQ_Create
        _RQ_Create.argtypes = [
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"), # zyx
            ct.c_void_p
            ]
        _RQ_Create.restype = ct.c_void_p

        rs = roishape        
        if len(roishape)==2:
            rs=[1,*roishape]
            
        rs = np.array(rs,dtype=np.int32)
        self.inst = _RQ_Create(rs, ctx.inst)
        self.roishape = roishape
        self.smpcount = np.prod(roishape)
        
        self._RQ_Pop = lib.RQ_Pop
        self._RQ_Pop.argtypes =[
            InstancePtrType,
            ct.c_int32,
            ctl.ndpointer(ROIInfoDType, flags="aligned, c_contiguous"), # ids
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # data
        ]
        
        self._RQ_Length = lib.RQ_Length
        self._RQ_Length.argtypes= [ InstancePtrType]
        self._RQ_Length.restype = ct.c_int32
        
        self._RQ_Push = lib.RQ_Push
        self._RQ_Push.argtypes=[
            InstancePtrType,
            ct.c_int32,
            ctl.ndpointer(ROIInfoDType, flags="aligned, c_contiguous"), # info
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # data
            ]
        
        self._RQ_SmpShape = lib.RQ_SmpShape
        self._RQ_SmpShape.argtypes=[
            InstancePtrType,
            ctl.ndpointer(ct.c_int32, flags="aligned, c_contiguous") # zyx
        ]
        
        self._RQ_Delete = lib.RQ_Delete
        self._RQ_Delete.argtypes=[InstancePtrType]
        
    def Length(self):
        return self._RQ_Length(self.inst)
    
    def __len__(self):
        return self.Length()
        
    def Push(self, roi_data, **kwargs):
        n = len(roi_data)
        roi_data = np.ascontiguousarray(roi_data, dtype=np.float32)
        info = np.zeros(n,dtype=ROIInfoDType)
        
        for k in kwargs.keys():
            info[k] = kwargs[k]
            
        self._RQ_Push(self.inst, n, info, roi_data)
        

    def Fetch(self, count=-1):
        if count < 0:
            count = len(self)
            
        rs = self.roishape
        if rs[0] == 1:
            rs = rs[1:]
        
        data = np.zeros((count, *rs),dtype=np.float32)
        rois_info = np.zeros((count), dtype=ROIInfoDType)
        
        self._RQ_Pop(self.inst, count, rois_info, data)
        return rois_info, data
    
    def Destroy(self):
        if self.inst is not None:
            self._RQ_Delete(self.inst)
            self.inst = None
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.Destroy()
