import ctypes as ct
import numpy as np
import numpy.ctypeslib as ctl
from .lib import NullableFloatArrayType
from .context import Context


class EstimatorProperties(ct.Structure):
    _fields_ = [
	#int numParams, sampleCount, numDiag, numConst, sampleIndexDims;
        ("numParams", ct.c_int),
        ("sampleCount", ct.c_int),
        ("numDiag", ct.c_int),
        ("numConst", ct.c_int),
        ("sampleIndexDims", ct.c_int)
    ]
    
class LevMarParams(ct.Structure):
    _fields_ = [
        ("iterations", ct.c_int32),
        ("lambda_", ct.c_float),
        ("factor", ct.c_float)
    ]
    
    

class Estimator:
    def __init__(self, ctx:Context, psfInst, calib=None):
        self.inst = psfInst
        self.ctx = ctx
        self.calib = calib
        lib = ctx.smlm.lib

        InstancePtrType = ct.c_void_p

        self._Estimator_Delete = lib.Estimator_Delete
        self._Estimator_Delete.argtypes = [InstancePtrType]

        self._Estimator_ParamFormat = lib.Estimator_ParamFormat
        self._Estimator_ParamFormat.restype = ct.c_char_p
        self._Estimator_ParamFormat.argtypes = [InstancePtrType]
        
        self._Estimator_GetProperties = lib.Estimator_GetProperties
        self._Estimator_GetProperties.argtypes = [
            InstancePtrType,
            ct.POINTER(EstimatorProperties)
        ]
        
        self._Estimator_SampleDims = lib.Estimator_SampleDims
        self._Estimator_SampleDims.argtypes = [
            InstancePtrType,
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous")
        ]        
        
        #CDLL_EXPORT void Estimator_GetParamLimits(Estimator* estim, ParamLimit* limits)

        self._Estimator_GetParamLimits = lib.Estimator_GetParamLimits
        self._Estimator_GetParamLimits.argtypes = [
            InstancePtrType,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous") #Param 
            ]
        
        
        #CDLL_EXPORT void Estimator_ChiSquareAndCRLB(Estimator* estim, 
        #const float* params, const float* sample, const float* h_const, const int* spot_pos, int numspots, float* crlb, float* chisq);

        self._Estimator_ChiSquareAndCRLB = lib.Estimator_ChiSquareAndCRLB
        self._Estimator_ChiSquareAndCRLB.argtypes = [
            InstancePtrType,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), #Param 
            NullableFloatArrayType,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), #Param 
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # spotpos
            ct.c_int32,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), #Param 
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), #Param 
            ]

        # CDLL_EXPORT void Estimator_ComputeExpectedValue(PSF* psf, int numspots, const float* params, const float* constants, const int* spotpos, float* ev);
        self._Estimator_ComputeExpectedValue = lib.Estimator_ComputeExpectedValue
        self._Estimator_ComputeExpectedValue.argtypes = [
            InstancePtrType,
            ct.c_int32,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), #Param 
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # constants
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # spotpos
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # ev
        ]

        # CDLL_EXPORT void Estimator_ComputeDerivatives(PSF* psf, int numspots, const float* Param, const float* constants, const int* spotpos, float* derivatives, float* ev);
        self._Estimator_ComputeDerivatives = lib.Estimator_ComputeDerivatives
        self._Estimator_ComputeDerivatives.argtypes = [
            InstancePtrType,
            ct.c_int32,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # spotpos
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),
        ]
        
        self.fitInfoDType = np.dtype([('likelihood', '<f4'), ('iterations', '<i4')])

        # CDLL_EXPORT void Estimator_ComputeMLE(PSF* psf, int numspots, const float* sample, const float* constants,const int* spotpos, 
        # const float* initial, float* params, int* iterations, int maxiterations, float levmarAlpha, float* trace, int traceBufLen);
        self._Estimator_Estimate = lib.Estimator_Estimate
        self._Estimator_Estimate.argtypes = [
            InstancePtrType,
            ct.c_int32,
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # sample
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # constants
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # spotpos
            NullableFloatArrayType,  # initial
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # Param
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"),  # diagnostics
            ctl.ndpointer(np.int32, flags="aligned, c_contiguous"),  # iterations
            ctl.ndpointer(np.float32, flags="aligned, c_contiguous"), # tracebuf [params*numspots*tracebuflen]
            ct.c_int32
        ]
        
        #CDLL_EXPORT void Estimator_SetLMParams(Estimator* estim, LMParams&)

        self._Estimator_SetLMParams = lib.Estimator_SetLMParams
        self._Estimator_SetLMParams.argtypes = [
            InstancePtrType,
            ct.POINTER(LevMarParams)
            ]
        
        self._Estimator_GetLMParams = lib.Estimator_GetLMParams 
        self._Estimator_GetLMParams.argtypes = [
            InstancePtrType,
            ct.POINTER(LevMarParams)
        ]
        
        props = EstimatorProperties()
        self._Estimator_GetProperties(self.inst, props)
        
        self.numparams = props.numParams
        self.numconst = props.numConst
        self.samplecount = props.sampleCount
        self.indexdims = props.sampleIndexDims
        self.sampleshape = np.zeros(self.indexdims,dtype=np.int32)
        self._Estimator_SampleDims(self.inst, self.sampleshape)
        self.numdiag = props.numDiag
        self.fmt = self.ParamFormat()
        self.param_names = [s.strip() for s in self.fmt.split(',')]
        
        self.limits = np.zeros( ( self.numparams, 2),dtype=np.float32)
        self._Estimator_GetParamLimits(self.inst, self.limits)

        self.SetLevMarParams(1e-18, 50)

        #print(f"Created PSF Parameters: {self.fmt}, #const={self.numconst}. #diag={self.numdiag} samplesize={self.sampleshape}." )

    def _checkparams(self, params, constants, roipos):
        params = np.array(params)
        
        if len(params.shape) == 1:
            params = [params]
        params = np.ascontiguousarray(params, dtype=np.float32)
        numspots = len(params)
        if constants is None:
            if self.numconst != 0:
                raise ValueError(f'Estimator is expecting constant array with shape {[len(params), self.numconst]} (None given) ')
            constants = np.zeros((numspots, self.numconst), dtype=np.float32)
        else:
            constants = np.ascontiguousarray(constants, dtype=np.float32)
            if constants.size != numspots*self.numconst:
                
                # if its not the same size, try broadcasting it into the right shape
                try:
                    constants_ = np.zeros([numspots,self.numconst],dtype=np.float32)
                    constants_[:,:] = constants
                    constants = constants_
                except:                
                    raise ValueError(f'Provided constants with shape {constants.shape}, expecting {[numspots,self.numconst]}')

        if roipos is None:
            roipos = np.zeros((numspots, self.indexdims), dtype=np.int32)
        else:
            roipos = np.ascontiguousarray(roipos, dtype=np.int32)
            if not np.array_equal( roipos.shape, [numspots, self.indexdims]):
                raise ValueError(f'Provided roi positions have incorrect shape: {roipos.shape} should be {[numspots,self.indexdims]}')

        if params.shape[1] != self.numparams:
            raise ValueError(f"{self.fmt} expected, {params.shape[1]} parameters given")
        return params,constants,roipos

    def ExpectedValue(self, params, roipos=None, constants=None):           
        """
        Compute the expected value of each pixel the ROIs.
        Returns a matrix with shape [len(params), sample shape...]
        """
        params,constants,roipos=self._checkparams(params,constants,roipos)
        ev = np.zeros((len(params), *self.sampleshape), dtype=np.float32)
        if len(params)>0:
            #print(f"Calling expected value (e={type(self)}), params: {params.shape}")
            self._Estimator_ComputeExpectedValue(self.inst, len(params), params, constants, roipos, ev)
        return ev                       # mu

    def GenerateSample(self, params, roipos=None, constants=None):          
        ev = self.ExpectedValue(params, roipos, constants)
        return np.random.poisson(ev)    # smp

    def FisherMatrix(self, params, roipos=None, constants=None):
        params,constants,roipos=self._checkparams(params,constants,roipos)
        fi = np.zeros((len(params), self.numparams, self.numparams), dtype=np.float32)
        if len(params)>0:
            deriv, ev = self.Derivatives(params,roipos, constants)
            
            K = self.numparams
            ev = np.reshape(ev, (len(params), self.samplecount))
            deriv = np.reshape(deriv, (len(params), K, self.samplecount))
            
            ev[ev<1e-9] = 1e-9
            for i in range(K):
                for j in range(K):
                    fi[:,i,j] = np.sum(1/ev * (deriv[:,i] * deriv[:,j]), axis=1)

        return fi
    
    def CRLBAndChiSquare(self, params, samples=None,roipos=None, constants=None):
        """
        return (crlb,chisq) if samples are given, otherwise crlb
        """
        params,constants,roipos=self._checkparams(params,constants,roipos)
        
        crlb = np.zeros((len(params), self.numparams), dtype=np.float32)
        chisq = np.zeros(len(params),dtype=np.float32)
        
        if samples is not None:
            samples = np.ascontiguousarray(samples,dtype=np.float32)

            if not np.array_equal(samples.shape, [len(params), *self.sampleshape]):
                raise ValueError(f"Sample data was expected to have shape {[len(params), *self.sampleshape]}. Given: {samples.shape}")
        
        if len(params)>0:
            #(Estimator* estim, const float* params, const float* sample, const float* h_const, const int* spot_pos, int numspots, float* crlb, float* chisq);

            self._Estimator_ChiSquareAndCRLB(self.inst, params, samples, constants, roipos, len(params), crlb, chisq)
        
        if samples is not None:
            return crlb,chisq
        return crlb
    
    def ChiSquare(self, params, samples, roipos=None, constants=None):
        return self.CRLBAndChiSquare(params, samples, roipos, constants)[1]

    def CRLB(self, params, roipos=None, constants=None):
        fisher = self.FisherMatrix(params, roipos, constants)
        var = np.linalg.pinv(fisher)
        crlb = np.sqrt(np.abs(var[:,np.arange(self.numparams),np.arange(self.numparams)]))
        return crlb
    
    def Derivatives(self, params, roipos=None, constants=None):
        """
        Compute expected value and derivatives of each pixel w.r.t parameters.
        Returns a tuple (deriv, exp.val), where deriv has shape [numspots,numparam,*roishape]
        """
        params,constants,roipos=self._checkparams(params,constants,roipos)
        deriv = np.zeros((len(params), self.numparams, *self.sampleshape), dtype=np.float32)
        ev = np.zeros((len(params), *self.sampleshape), dtype=np.float32)
        if len(params)>0:
            self._Estimator_ComputeDerivatives(self.inst, len(params), params, constants, roipos, deriv, ev)
        return deriv, ev
    
    def NumDeriv(self, params, roipos=None, constants=None, eps=1e-6):
        """
        Compute numerical derivate based on the output of ExpectedValue
        """
        params,constants,roipos=self._checkparams(params,constants,roipos)
        ev = self.ExpectedValue(params,roipos,constants)
        deriv = np.zeros((len(params), self.numparams, *self.sampleshape), dtype=np.float32)
        
        for k in range(self.numparams):
            params_min = params*1
            params_min[:,k] -= eps
            params_max = params*1
            params_max[:,k] += eps
            ev_min = self.ExpectedValue(params_min, roipos, constants)
            ev_max = self.ExpectedValue(params_max, roipos, constants)
            deriv[:,k] = (ev_max-ev_min)/(2*eps)
        
        return deriv, ev
    
    def NumCRLB(self, params, roipos=None, constants=None, eps=1e-4, useNumDeriv=True):
        if useNumDeriv:
            deriv,ev = self.NumDeriv(params,roipos,constants,eps)
        else:
            deriv,ev = self.Derivatives(params,roipos,constants)

        K = self.numparams
        deriv = np.reshape(deriv, (len(params), K, self.samplecount))
        ev[ev<1e-9] = 1e-9
        fi = np.zeros((len(params), K,K))
        for i in range(K):
            for j in range(K):
                fi[:,i,j] = np.sum(1/ev * (deriv[:,i] * deriv[:,j]), axis=1)
        var = np.linalg.inv(fi)
        return np.sqrt(np.abs(var[:,np.arange(self.numparams),np.arange(self.numparams)]))
    
    def GetLevMarParams(self):
        lmp = LevMarParams()
        self._Estimator_GetLMParams(self.inst, lmp)
        self.lmparams = lmp
        return lmp
    
    def SetLevMarParams(self, lambda_ = 1e-15, iterations=50, lmFactor=0.0):
        
        self.lmparams = LevMarParams()
        self.lmparams.lambda_ = lambda_
        self.lmparams.iterations = iterations
        self.lmparams.factor = lmFactor
        
        self._Estimator_SetLMParams(self.inst, self.lmparams)

    def Estimate(self, imgs, roipos=None, constants=None, initial=None):
        numspots = len(imgs)

        traceBufLen = self.lmparams.iterations
        imgs = np.ascontiguousarray(imgs, dtype=np.float32)
        
        if not np.array_equal(imgs.shape[1:], self.sampleshape):
            raise ValueError(f"Sample images are expected to have shape {self.sampleshape}. Given: {imgs.shape[1:]}")

        iterations = np.zeros(numspots, np.int32)
        params = np.zeros((numspots, self.numparams), dtype=np.float32)
        trace = np.zeros((numspots, traceBufLen, self.numparams), dtype=np.float32)
        diag = np.zeros((numspots, self.numdiag), dtype=np.float32)

        params,constants,roipos = self._checkparams(params,constants,roipos)

        if initial is not None:
            initial = np.ascontiguousarray(initial, dtype=np.float32)
            
            if not np.array_equal(initial.shape, [numspots, self.numparams]):
                raise ValueError(f'Initial value array is expected to have shape {[numspots,self.numparams]}. Given: {initial.shape}')

        if numspots>0:

            #print(f"Calling estimate (e={type(self)}), params: {params.shape}. roipos={roipos.shape}")

            self._Estimator_Estimate(
                self.inst,
                numspots,
                imgs,
                constants,
                roipos,
                initial,
                params,
                diag,
                iterations,
                trace,
                traceBufLen
            )

        traces = []
        for k in range(numspots):
            traces.append(trace[k, 0 : iterations[k], :])

        return params, diag, traces

    def Destroy(self):
        if self.inst is not None:
            self._Estimator_Delete(self.inst)
            self.inst = None

    def NumConstants(self):
        return self.numconst

    def ParamFormat(self):
        return self._Estimator_ParamFormat(self.inst).decode("utf-8") 
    
    def ParamIndex(self, paramName):
        if type(paramName)==list:
            return [self.param_names.index(n) for n in paramName]
        return self.param_names.index(paramName)

    def NumParams(self):
        return self.numparams

    def SampleCount(self):
        return self.samplecount
    
    def NumDiag(self):
        return self.numdiag
    
    def ParamLimits(self):
        """ Returns a tuple with min[NumParams], and max[NumParams] """
        return self.limits[:,0], self.limits[:,1]
        
    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.Destroy()

