import ctypes
from .lib import SMLM
import numpy as np
import numpy.ctypeslib as ctl


class PostProcessMethods:
    def __init__(self, ctx):
        self.lib = ctx.smlm.lib

        #CDLL_EXPORT void LinkLocalizations(int numspots, int* frames, Vector2f* xyI, float maxDist, int frameskip, int *linkedSpots)

        self._LinkLocalizations = self.lib.LinkLocalizations
        self._LinkLocalizations.argtypes = [
            ctypes.c_int32,  # numspots
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # framenum
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # xyI
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # crlbXYI
            ctypes.c_float,  # maxdist (in crlbs)
            ctypes.c_float, # max intensity distance (in crlb's)
            ctypes.c_int32,  # frameskip
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # linkedspots
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"), # startframes
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # framecounts
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # linkedXYI
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # linkedCRLBXYI
        ]
        self._LinkLocalizations.restype = ctypes.c_int32
        

        #(const Vector2f* xy, const int* spotFramenum, int numspots,
        #float sigma, int maxiterations, Vector2f* driftXY,  float gradientStep, float maxdrift, float* scores, int flags)

        self.ProgressCallback = ctypes.CFUNCTYPE(
            ctypes.c_int32,  # continue
            ctypes.c_int32,  # iteration
            ctypes.c_char_p
        )
        
        
        self._MinEntropyDriftEstimate = self.lib.MinEntropyDriftEstimate
        self._MinEntropyDriftEstimate.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # xy: float[numspots, dims]
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # crlb: float[numspots, dims] or float[dims]
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # framenum
            ctypes.c_int32,  # numspots
            ctypes.c_int32, #maxit
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # drift XY
            ctypes.c_int32, # framesperbin
            ctypes.c_float, # gradientstep
            ctypes.c_float, # maxdrift
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # scores
            ctypes.c_int32, # flags
            ctypes.c_int32, # maxneighbors
            self.ProgressCallback] # flags
        self._MinEntropyDriftEstimate.restype = ctypes.c_int32
                
    
        #void ComputeContinuousFRC(const float* data, int dims, int numspots, const float* rho, int nrho, float* frc, float maxDistance, bool useCuda)
        self._ComputeContinuousFRC = self.lib.ComputeContinuousFRC
        self._ComputeContinuousFRC.argtypes = [
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # data (xy or xyz)
            ctypes.c_int32,  # number of dimensions
            ctypes.c_int32,  # numspots
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # rho
            ctypes.c_int32,  # nrho
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # output frc
            ctypes.c_float, # maxdistance
            ctypes.c_bool, # usecuda
            ctypes.c_float, # cutoffDist
            ctypes.c_float # cutoffSigma
        ]
        
        
        #(int startA, int numspotsA, const int* counts, const int* indicesB, int numIndicesB);
        self.FindNeighborCallback = ctypes.CFUNCTYPE(
            ctypes.c_int32,  # continue
            ctypes.c_int32,  # startA
            ctypes.c_int32,  # numspotsA
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32  # numindices
        )
        
        
        #(int startA, int numspotsA, const int* counts, const int* indicesB, int numIndicesB);
        self._ClusterLocsCallback = ctypes.CFUNCTYPE(
            ctypes.c_int32,  # continue
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
        )
    
    
        # int FindNeighbors(int numspotsA, const float* coordsA, int numspotsB, const float* coordsB, int dims, float maxDistance, int minBatchSize,
	#FindNeighborCallback cb)

        self._FindNeighbors = self.lib.FindNeighbors
        self._FindNeighbors.argtypes = [
            ctypes.c_int32,  # numspotsA
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # coordsA
            ctypes.c_int32,  # numspotsB
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # coordsB
            ctypes.c_int32,  # dims
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # maxDistance[dims]
            ctypes.c_int32,  # minBatchSize
            self.FindNeighborCallback
        ]
        self._FindNeighbors.restype = ctypes.c_int32
        
        #int ClusterLocs(int dims, float* pos, int* mappingToNew, const float* distance, int numspots, ClusterLocsCallback callback)
        self._ClusterLocs = self.lib.ClusterLocs
        self._ClusterLocs.argtypes = [
            ctypes.c_int32,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # pos
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # mapping
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # distance
            ctypes.c_int32,
            self._ClusterLocsCallback
        ]
        self._ClusterLocs.restype= ctypes.c_int32
        
    def ClusterLocs(self, points, crlb, dist, cb=None):
        """
        points: [numspots, numdims]
        crlb: [numspots, numdims]
        """
        numdims = points.shape[1]
        newpts = np.ascontiguousarray(points*1,dtype=np.float32)
        newcrlb = np.ascontiguousarray(crlb*1,dtype=np.float32)
        mapping = np.zeros(len(points), dtype=np.int32)
        
        if np.isscalar(dist):
            dist = (np.ones(numdims)*dist).astype(np.float32)
        else:
            dist = np.ascontiguousarray(dist,dtype=np.float32)
        
        def callback_(mappingPtr, centerPtr):
            mapping = ctl.as_array(mappingPtr, (len(points),))
            nclust=np.max(mapping)+1
            centers = ctl.as_array(centerPtr, (nclust,numdims))
            if cb is not None:
                r = cb(mapping,centers)
                if r is None:
                    return 1
                return r
            return 1
        
        newcount = self._ClusterLocs(numdims, newpts, mapping, dist, len(points), self._ClusterLocsCallback(callback_))
        if newcount >= 0:
            return newpts[:newcount], newcrlb[:newcount], mapping
        raise ValueError('Something went wrong in ClusterLocs')
        
        
    def FindNeighborsIterative(self, pointsA, pointsB, maxDistance, minBatchSize, callback):
                
        pointsA = np.ascontiguousarray(pointsA,dtype=np.float32)
        pointsB = np.ascontiguousarray(pointsB,dtype=np.float32)

        assert len(pointsA.shape)==2
        assert len(pointsB.shape)==2
        
        dims = pointsA.shape[1]
        
        assert(pointsB.shape[1] == dims)
        
        def callback_(startA, numspotsA, countsPtr, indicesPtr, numIndices):
            #print(f"num indices: {numIndices}. numspotsA: {numspotsA}")
            counts = ctl.as_array(countsPtr, (numspotsA,))
            if numIndices == 0:
                indices = np.zeros(0,dtype=np.int32)
            else:
                indices = ctl.as_array(indicesPtr, (numIndices,))
            
            r = callback(startA, counts, indices)
            if r is None:
                return 1
            return r
        
        if np.isscalar(maxDistance):
            maxDistance = np.ones(dims,dtype=np.float32)*maxDistance
        else:
            maxDistance = np.ascontiguousarray(maxDistance, dtype=np.float32)
            assert(len(maxDistance) == dims)
            
        cb = self.FindNeighborCallback(callback_)
        self._FindNeighbors(len(pointsA), pointsA, len(pointsB), pointsB, pointsA.shape[1], maxDistance, minBatchSize, cb)

    def FindNeighbors(self, pointsA, pointsB, maxDistance):
        counts = []
        indices = []
        
        def callback(startA, counts_, indices_):
            counts.append(counts_.copy()) 
            indices.append(indices_.copy())
        
        self.FindNeighborsIterative(pointsA, pointsB, maxDistance, minBatchSize=10000, callback=callback)
        
        if len(counts) == 0:
            return [],[]
        
        return np.concatenate(counts), np.concatenate(indices)
    
    def LinkLocalizations(self, xyI, crlbXYI, framenum, maxdist, maxIntensityDist, frameskip):
        """
        linked: int [numspots], all spots that are linked will have the same index in linked array.
        """
        xyI = np.ascontiguousarray(xyI,dtype=np.float32)
        crlbXYI = np.ascontiguousarray(crlbXYI,dtype=np.float32)
        framenum = np.ascontiguousarray(framenum, dtype=np.int32)
        linked = np.zeros(len(xyI),dtype=np.int32)
        framecounts = np.zeros(len(xyI),dtype=np.int32)
        startframes = np.zeros(len(xyI),dtype=np.int32)
        resultXYI = np.zeros(xyI.shape,dtype=np.float32)
        resultCRLBXYI = np.zeros(crlbXYI.shape,dtype=np.float32)
        
        assert crlbXYI.shape[1] == 3
        assert xyI.shape[1] == 3
        assert len(xyI) == len(crlbXYI)
        
        nlinked = self._LinkLocalizations(len(xyI), framenum, xyI, crlbXYI, maxdist, maxIntensityDist, 
                                          frameskip, linked, startframes, framecounts, resultXYI, resultCRLBXYI)
        startframes = startframes[:nlinked]
        framecounts = framecounts[:nlinked]
        resultXYI = resultXYI[:nlinked]
        resultCRLBXYI = resultCRLBXYI[:nlinked]
        return linked, framecounts,startframes, resultXYI, resultCRLBXYI

    def MinEntropyDriftEstimate(self, positions, framenum, drift, crlb, iterations, 
                                stepsize, maxdrift, framesPerBin=1, cuda=False, progcb=None,flags=0, 
                                maxneighbors=10000):
        
        positions = np.ascontiguousarray(positions,dtype=np.float32)
        framenum = np.ascontiguousarray(framenum,dtype=np.int32)
        drift = np.ascontiguousarray(drift,dtype=np.float32)
        
        nframes = np.max(framenum)+1
        
        assert len(drift)>=nframes and drift.shape[1]==positions.shape[1]

        if len(drift)>nframes:
            drift = drift[:nframes]
            drift = np.ascontiguousarray(drift,dtype=np.float32)

        if cuda:
            flags |= 2
                    
        scores = np.zeros(iterations,dtype=np.float32)
        
        if positions.shape[1] == 3:
            flags |= 1 # 3D

        if np.isscalar(crlb):
            crlb=np.ones(positions.shape[1])*crlb

        crlb = np.array(crlb,dtype=np.float32)
        if len(crlb.shape) == 1: # constant CRLB values, all points have the same CRLB
            flags |= 4
            assert len(crlb) == positions.shape[1]
            #print(f"DME: Using constant crlb")
        else:
            assert np.array_equal(crlb.shape,positions.shape)
            #print(f"DME: Using variable crlb")
            
        crlb=np.ascontiguousarray(crlb,dtype=np.float32)
                
        if progcb is None:
            progcb = lambda i,txt: 1

        nIterations = self._MinEntropyDriftEstimate(
            positions, crlb, framenum, len(positions), iterations, drift, framesPerBin,
            stepsize, maxdrift, scores, flags, maxneighbors, self.ProgressCallback(progcb))

        return drift, scores[:nIterations]
    
    
    def ContinuousFRC(self, points, maxdistance, freq, cutoffDist, cutoffSigma, useCuda=True):
        points = np.ascontiguousarray(points,dtype=np.float32)
        
        npts = len(points)
        if not np.array_equal(points.shape, [npts,2]) and not np.array_equal(points.shape,[npts,3]):
            raise ValueError(f'Expected points to have shape [numpoints, dimensions] where dimensions is either 2 or 3. Given:{points.shape}')
            
        rho = np.ascontiguousarray(freq * 2*np.pi,dtype=np.float32)
        frc = np.zeros(len(rho),dtype=np.float32)
        
        self._ComputeContinuousFRC(points, points.shape[1], npts, rho, len(rho), frc, maxdistance, useCuda, cutoffDist, cutoffSigma)
        return frc
        

        