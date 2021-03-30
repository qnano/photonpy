import ctypes as ct
import numpy as np
import numpy.ctypeslib as ctl
from .context import Context


class ImageProcessor:
    def __init__(self, imgshape, inst, ctx:Context):
        self.frameCounter = 0
        self.ctx = ctx
        InstancePtrType = ct.c_void_p
                
        self._ImgProc_NumFinishedFrames = ctx.lib.ImgProc_NumFinishedFrames
        self._ImgProc_NumFinishedFrames.argtypes = [
                InstancePtrType
                ]

        self._ImgProc_AddFrameU16 = ctx.lib.ImgProc_AddFrameU16
        self._ImgProc_AddFrameU16.argtypes = [
            InstancePtrType,
            ctl.ndpointer(np.uint16, flags="aligned, c_contiguous"), #image
            ]

        self._ImgProc_AddFrameF32 = ctx.lib.ImgProc_AddFrameF32
        self._ImgProc_AddFrameF32.argtypes = [
            InstancePtrType,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), #image
            ]

#CDLL_EXPORT bool ImgFilterQueue_ReadFrame(ImgFilterQueue* q, float* filtered, float* image, int& framenum)
        
        self._ImgProc_ReadFrame = ctx.lib.ImgProc_ReadFrame 
        self._ImgProc_ReadFrame.argtypes =[
            InstancePtrType,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), #timefilter
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), #timefilter
                ]
        
        self._ImgProc_GetQueueLength = ctx.lib.ImgProc_GetQueueLength 
        self._ImgProc_GetQueueLength.argtypes=[
                InstancePtrType]
        
        self._ImgProc_IsIdle = ctx.lib.ImgProc_IsIdle
        self._ImgProc_IsIdle.argtypes = [ InstancePtrType]
        self._ImgProc_IsIdle.restype = ct.c_bool
        
        self.imgshape = imgshape
        self.inst = inst
        
    def PushFrameU16(self,img):
        assert np.array_equal(img.shape, self.imgshape)
        self._ImgProc_AddFrameU16(self.inst, np.ascontiguousarray(img,dtype=np.uint16))

    def PushFrameF32(self,img):
        assert np.array_equal(img.shape, self.imgshape)
        self._ImgProc_AddFrameF32(self.inst, np.ascontiguousarray(img,dtype=np.float32))

    def GetQueueLength(self):
        return self._ImgProc_GetQueueLength(self.inst)
    
    def NumFinishedFrames(self):
        return self._ImgProc_NumFinishedFrames(self.inst)
    
    def ReadResultFrame(self):
        filtered = np.empty(self.imgshape,dtype=np.float32)
        img = np.empty(self.imgshape,dtype=np.float32)
        
        num = self._ImgProc_ReadFrame(self.inst, img, filtered)
        
        if num > 0:
            return img, filtered
        
        return None
    
    def IsIdle(self):
        return self._ImgProc_IsIdle(self.inst)
    


class ROIExtractor(ImageProcessor):
    
    ROIType = np.dtype([('cornerpos','<i4',2),('startframe','<i4'),('numframes','<i4')])

    def __init__(self, imgshape, rois, roiframes, roisize,calib, ctx:Context):
        
        ROIExtractor_Create = ctx.lib.ROIExtractor_Create
        ROIExtractor_Create.argtypes =[
                ct.c_int32, # w
                ct.c_int32, # h
                ctl.ndpointer(ROIExtractor.ROIType, flags="aligned, c_contiguous"),  # rois
                ct.c_int32, #numrois
                ct.c_int32, #roiframes
                ct.c_int32, # roisize
                ct.c_void_p, # calib
                ct.c_void_p # ctx
                ]
        ROIExtractor_Create.restype = ct.c_void_p
        
#        CDLL_EXPORT int ROIExtractor_GetResultCount(ROIExtractor *re) {

        self._GetResultCount = ctx.lib.ROIExtractor_GetResultCount
        self._GetResultCount.argtypes=[ct.c_void_p]
        self._GetResultCount.restype = ct.c_int32

        #CDLL_EXPORT int ROIExtractor_GetResults(ROIExtractor* re, int numrois, ExtractionROI* rois, float* framedata)
        self._GetResults = ctx.lib.ROIExtractor_GetResults
        self._GetResults.argtypes=[
                ct.c_void_p,
                ct.c_int32,
                ctl.ndpointer(ROIExtractor.ROIType, flags="aligned, c_contiguous"),  # rois
                ctl.ndpointer(np.float32, flags="aligned, c_contiguous")
                ]
        
        inst = ROIExtractor_Create(imgshape[1],imgshape[0],rois,len(rois),roiframes,roisize,calib.inst if calib else None,ctx.inst)

        self.roiframes = roiframes
        self.roisize = roisize

        super().__init__(imgshape,inst,ctx)
#CDLL_EXPORT ROIExtractor* ROIExtractor_Create(int imgWidth, int imgHeight, ExtractionROI* rois, int numrois, 
#int roiframes, int roisize, IDeviceImageProcessor* imgCalibration, Context* ctx)
    
    def GetResultCount(self):
        return self._GetResultCount(self.inst)
    
    def GetResults(self, numrois):
        rois = np.zeros(numrois, dtype=ROIExtractor.ROIType)
        framedata = np.zeros((numrois,self.roiframes,self.roisize,self.roisize),dtype=np.float32)
        #print(f'framedata: {framedata.size} elements. ({framedata.shape})')
        count = self._GetResults(self.inst,numrois,rois,framedata)
        return rois[:count],framedata[:count]
    

class RollingMedianImageFilter(ImageProcessor):
    def __init__(self, imgshape, windowsize, ctx:Context):
        #MedianFilterImgQueue<float>* MedianFilterQueue_CreateF32(int w, int h, int nframes)
        Create = ctx.lib.MedianFilterQueue_CreateF32
        Create.argtypes = [
            ct.c_int32, # w
            ct.c_int32, # h
            ct.c_int32, # h
            ct.c_void_p # ctx
        ]
        Create.restype = ct.c_void_p
        inst = Create(imgshape[1],imgshape[0],windowsize,ctx.inst)
        super().__init__(imgshape, inst, ctx)
        