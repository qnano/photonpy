# -*- coding: utf-8 -*-


import ctypes as ct
import numpy as np
import numpy.ctypeslib as ctl
import time
import copy

from .estimator import Estimator
from .lib import NullableFloatArrayType


class EstimQueue_Results:
    def __init__(self, param_names, sampleshape, estim, diag, crlb, 
                 iterations, chisq, roipos, samples, ids):
        self.estim = estim
        self.sampleshape = sampleshape
        self.diagnostics = diag
        self.crlb = crlb
        self.roipos = roipos
        self.iterations = iterations
        self.chisq = chisq
        self.ids = ids
        self.samples = samples
        self.param_names = param_names

    def CRLB(self):
        return self.crlb
    
    def SortByID(self, isUnique=False):
        if isUnique:
            order = np.arange(len(self.ids))
            order[self.ids] = order*1
        else:
            order = np.argsort(self.ids)
        self.Filter(order)
        return order       
        
    def Filter(self, indices):
        if indices.dtype == bool:
            indices = np.nonzero(indices)[0]
        
        if len(indices) != len(self.ids):
            print(f"Removing {len(self.ids)-len(indices)}/{len(self.ids)}")
        self.estim = self.estim[indices]
        self.diagnostics = self.diagnostics[indices]
        self.crlb = self.crlb[indices]
        self.roipos = self.roipos[indices]
        self.iterations = self.iterations[indices]
        self.chisq = self.chisq[indices]
        self.ids = self.ids[indices]
        if self.samples is not None:
            self.samples = self.samples[indices]
        
        return indices
        
    def FilterXY(self, minX, minY, maxX, maxY):
        return self.Filter(np.where(
            np.logical_and(
                np.logical_and(self.estim[:,0]>minX, self.estim[:,1]>minY),
                np.logical_and(self.estim[:,0]<maxX, self.estim[:,1]<maxY)))[0])

    def Clone(self):
        return copy.deepcopy(self)
    
    def ColIdx(self, *names):
        return np.squeeze(np.array([self.param_names.index(n) for n in names],dtype=np.int))
        

EstimResultDType = np.dtype([
    ('id','<i4'),('chisq','<f4'),('iterations','<i4')
])


class EstimQueue:
    def __init__(self, estim:Estimator, batchSize=256, maxQueueLenInBatches=5, numStreams=-1, keepSamples=False, ctx=None):
        if ctx is None:
            self.ctx = estim.ctx
        else:
            self.ctx = ctx
        lib = self.ctx.smlm.lib
        self.estim = estim
        self.batchSize = batchSize

        InstancePtrType = ct.c_void_p

#        DLL_EXPORT LocalizationQueue* EstimQueue_CreateQueue(PSF* psf, int batchSize, int maxQueueLen, int numStreams);
        self._EstimQueue_Create = lib.EstimQueue_Create
        self._EstimQueue_Create.argtypes = [
                InstancePtrType, 
                ct.c_int32,
                ct.c_int32,
                ct.c_bool,
                ct.c_int32,
                ct.c_void_p]
        self._EstimQueue_Create.restype = InstancePtrType
        
#        DLL_EXPORT void EstimQueue_Delete(LocalizationQueue* queue);
        self._EstimQueue_Delete= lib.EstimQueue_Delete
        self._EstimQueue_Delete.argtypes = [InstancePtrType]
        
#        DLL_EXPORT void EstimQueue_Schedule(LocalizationQueue* q, int numspots, const int *ids, const float* h_samples,
 #       	const float* h_constants, const int* h_roipos);

        self._EstimQueue_Schedule = lib.EstimQueue_Schedule
        self._EstimQueue_Schedule.argtypes = [
                InstancePtrType,
                ct.c_int32,
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"), # ids
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # samples
            NullableFloatArrayType, #initial
            NullableFloatArrayType, #const
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # roipos
                ]
            
#        DLL_EXPORT void EstimQueue_Flush(LocalizationQueue* q);
        self._EstimQueue_Flush = lib.EstimQueue_Flush
        self._EstimQueue_Flush.argtypes = [InstancePtrType]
        
#        DLL_EXPORT bool EstimQueue_IsIdle(LocalizationQueue* q);
        self._EstimQueue_IsIdle = lib.EstimQueue_IsIdle
        self._EstimQueue_IsIdle.argtypes = [InstancePtrType]
        self._EstimQueue_IsIdle.restype = ct.c_bool
        
#        DLL_EXPORT int EstimQueue_GetResultCount(LocalizationQueue* q);
        self._EstimQueue_GetResultCount = lib.EstimQueue_GetResultCount
        self._EstimQueue_GetResultCount.argtypes = [InstancePtrType]
        self._EstimQueue_GetResultCount.restype = ct.c_int32

        self._EstimQueue_GetQueueLength = lib.EstimQueue_GetQueueLength
        self._EstimQueue_GetQueueLength.argtypes = [InstancePtrType]
        self._EstimQueue_GetQueueLength.restype = ct.c_int32
        
#        // Returns the number of actual returned localizations. 
 #       // Results are removed from the queue after copying to the provided memory
#        DLL_EXPORT int EstimQueue_GetResults(LocalizationQueue* q, int maxresults, float* estim, float* diag, float *fi);
        self._EstimQueue_GetResults = lib.EstimQueue_GetResults
        self._EstimQueue_GetResults.argtypes = [
                InstancePtrType,
                ct.c_int32,
                ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # estim
                NullableFloatArrayType, # diag
                NullableFloatArrayType, # crlb
                ctl.ndpointer(np.int32, flags="aligned, c_contiguous"), # roipos
                NullableFloatArrayType, # samples
                ctl.ndpointer(EstimResultDType, flags="aligned, c_contiguous"), # results
                ]
        self._EstimQueue_GetResults.restype = ct.c_int32
        
        self.param_names = estim.param_names
        
        self.inst = self._EstimQueue_Create(estim.inst,batchSize, maxQueueLenInBatches, keepSamples, numStreams,
                                           self.ctx.inst if self.ctx else None)
        if not self.inst:
            raise RuntimeError("Unable to create PSF MLE Queue with given PSF")
            
    
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.Destroy()

    def Destroy(self):
        self._EstimQueue_Delete(self.inst)
        
    def Flush(self):
        self._EstimQueue_Flush(self.inst)
        
    def WaitUntilDone(self):
        self.Flush()
        while not self.IsIdle():
            time.sleep(0.05)

        
    def IsIdle(self):
        return self._EstimQueue_IsIdle(self.inst)
    
    def Schedule(self, samples, roipos=None, ids=None, initial=None, constants=None):
        samples = np.ascontiguousarray(samples,dtype=np.float32)
        numspots = len(samples)

        if roipos is None:
            roipos = np.zeros((numspots,self.estim.indexdims),dtype=np.int32)
            
        if constants is not None:
            constants = np.ascontiguousarray(constants,dtype=np.float32)
            if constants.size != numspots*self.estim.numconst:
                raise ValueError(f'Estimator is expecting constants array with shape {(numspots,self.estim.numconst)}. Given: {constants.shape}')
        else:
            assert(self.estim.numconst==0)
            
        if initial is not None:
            initial = np.ascontiguousarray(initial, dtype=np.float32)
            assert(np.array_equal(initial.shape, [numspots, self.estim.numparams]))

        roipos = np.ascontiguousarray(roipos, dtype=np.int32)
        
        if not np.array_equal(roipos.shape, [numspots, self.estim.indexdims]):
            raise ValueError(f'Incorrect shape for ROI positions: {roipos.shape} given, expecting {[numspots, self.estim.indexdims]}')

        assert self.estim.samplecount*numspots==samples.size
        
        if ids is None:
            ids = np.zeros(numspots,dtype=np.int32)
        else:
            assert len(ids) == len(samples)
            ids = np.ascontiguousarray(ids,dtype=np.int32)
        
        self._EstimQueue_Schedule(self.inst, numspots, ids, samples, initial, constants, roipos)
        
    def GetQueueLength(self):
        return self._EstimQueue_GetQueueLength(self.inst)
    
    def GetResultCount(self):
        return self._EstimQueue_GetResultCount(self.inst)
        
    def GetResults(self,maxResults=None, getSampleData=False) -> EstimQueue_Results:  # 
        count = self._EstimQueue_GetResultCount(self.inst)
        
        if maxResults is not None and count>maxResults:
            count=maxResults
        
        K = self.estim.NumParams()
        estim = np.zeros((count, K),dtype=np.float32)
        diag = np.zeros((count, self.estim.NumDiag()), dtype=np.float32)
        crlb = np.zeros((count, K), dtype=np.float32)
        roipos = np.zeros((count, self.estim.indexdims),dtype=np.int32)
        results = np.zeros(count, dtype=EstimResultDType)
        
        if getSampleData:
            samples = np.zeros((count, *self.estim.sampleshape), dtype=np.float32)
        else:
            samples = None

        copied = self._EstimQueue_GetResults(self.inst, count, estim, 
                                             diag, crlb, roipos, samples, results)
        assert(count == copied)
                
        r = EstimQueue_Results(self.param_names, self.estim.sampleshape, 
                                 estim, diag, crlb, results['iterations'], 
                                 results['chisq'], roipos, samples, results['id'])
                    
        return r
        


