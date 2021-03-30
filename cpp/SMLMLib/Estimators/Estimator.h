// Estimator base classes
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include <driver_types.h>
#include "DLLMacros.h"
#include "Vector.h"
#include "ContainerUtils.h"
#include "CudaUtils.h"
#include <unordered_map>
#include "Context.h"

template<typename T>
class DeviceArray;

struct LMParams
{
	int iterations;
	float lambda; // levenberg-marquardt damping factor

	// Adaptive adjustment of lambda: if 0 no adjustment. 
	// If nonzero, improvement in likelihood will divide lambda by lmFactor, no improvement will multiply it
	float factor;
};

struct ParamLimit
{
	float min, max;
};

class EstimatorBase : public ContextObject
{
	std::string paramFormat; 
	std::vector<std::string> paramNames; // paramFormat but split up
	std::vector<int> sampleSize; // for example: [frames, height, width] or [height, width]
	int sampleCount; // = product(sampleSize)
	int numConstants;
	int numParams;
	int diagsize; // number of float in diagnostics / debug info per spot
	std::vector<ParamLimit> limits;

protected:

	LMParams lmParams;
public:
	DLL_EXPORT EstimatorBase(const std::vector<int>& sampleSize, int numConst, int diagsize, const char* paramFormat, std::vector<ParamLimit> limits, Context* ctx=0);
	virtual ~EstimatorBase();

	DLL_EXPORT int ParamIndex(const char *name);
	const char* ParamFormat() { return paramFormat.c_str(); }// Comma separated like "x,y,I,bg
	const std::vector<std::string>& ParamNames() { return paramNames; 	}
	const std::vector<ParamLimit>& ParamLimits() { return limits; }
	int NumParams() { return numParams;  }
	int SampleCount() { return sampleCount; }
	int NumConstants() { return numConstants; } // Parameters that vary per spot, but are not estimated (like roix,roiy)
	int SampleIndexDims() {	return (int)sampleSize.size(); } // Number of dimensions for a sample index (x,y or frame,x,y)
	int SampleSize(int dim) { return sampleSize[dim]; } // 0 <= dim < SampleIndexDims()
	int DiagSize() { return diagsize; }
	const std::vector<int>& SampleSize() { return sampleSize; }
	virtual void SetLMParams(LMParams p) { lmParams = p; }
	virtual LMParams GetLMParams() { return lmParams; }
};

class cuEstimator;

// Abstract Estimator Model. All pointers point to host memory
class Estimator : public EstimatorBase
{
public:
	Estimator(const std::vector<int>& sampleSize, int numConst, int diagsize, const char* paramFormat, std::vector<ParamLimit> limits) :
		EstimatorBase(sampleSize, numConst, diagsize, paramFormat, limits) {}

	// Return CUDA Estimator if this is a cuEstimatorWrapper
	virtual cuEstimator* Unwrap() { return 0; }

	// COmpute chi-square and crlb. If sample=0 then chi-square is not computed.
	virtual void ChiSquareAndCRLB(const float* params, const float* sample, const float* h_const, const int* spot_pos, float* crlb, float* chisq, int numspots) = 0;

	// d_image[numspots, SampleCount()], d_params[numspots, NumParams()]
	virtual void ExpectedValue(float* expectedvalue, const float* params, const float* _const, const int* spot_pos, int numspots) = 0;
	// d_deriv[numspots, NumParams(), SampleCount()], d_expectedvalue[numspots, SampleCount()], d_params[numspots, NumParams()]
	virtual void Derivatives(float* deriv, float *expectedvalue, const float* params, const float* _const, const int* spot_pos, int numspots) = 0;

	// d_sample[numspots, SampleCount()], d_initial[numspots, NumParams()], d_params[numspots, NumParams()], d_iterations[numspots]
	virtual void Estimate(const float* sample, const float* d_const, const int* spot_pos, 
		const float* initial, float* params, float* diagnostics, int* iterations, int numspots, float * trace, int traceBufLen) = 0;

	template<typename T>
	struct VectorType {
		typedef std::vector<T> type;
	};
};



// Abstract Estimator Model. All pointers with d_ prefix point to cuda memory. 
// Default implementation uses Derivatives() to implement Estimate() and FisherMatrix()
class cuEstimator : public EstimatorBase
{
public:
	cuEstimator(const std::vector<int>& sampleSize, int numConst, int diagsize, const char* paramFormat, std::vector<ParamLimit> limits);
	~cuEstimator();

	// Compute chi-square and crlb. If sample=0 then chi-square is not computed.
	virtual void ChiSquareAndCRLB(const float* params, const float* sample, const float* h_const, 
		const int* spot_pos, float* crlb, float* chisq, int numspots, cudaStream_t stream);

	// d_image[numspots, SampleCount()], d_params[numspots, NumParams()]
	virtual void ExpectedValue(float* expectedvalue, const float* d_params, const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream) = 0;
	// d_deriv[numspots, NumParams(), SampleCount()], d_expectedvalue[numspots, SampleCount()], d_params[numspots, NumParams()]
	// psf_deriv output format: [numspots, NumParams(), SampleCount()]
	virtual void Derivatives(float* deriv, float *expectedvalue, const float* params, const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream) = 0;

	// d_sample[numspots, SampleCount()], d_params[numspots, numparams], d_initial[numspots, NumParams()], d_params[numspots, NumParams()], d_iterations[numspots]
	virtual void Estimate(const float* d_sample, const float* d_const, const int* d_roipos, const float* d_initial, float* d_params, float* d_diagnostics, int* iterations,
		int numspots, float * trace, int traceBufLen, cudaStream_t stream);

	template<typename T>
	struct VectorType {
		typedef DeviceArray<T> type;
	};

protected:

	class DeviceBuffers
	{
	public:
		DeviceBuffers(int smpcount, int numspots, int thetasize);
		~DeviceBuffers();
		DeviceArray<float> psf_deriv, psf_ev, lm_alphabeta, lm_lu;
		DeviceArray<int> invert_temp;
		int numspots;
	};

	DeviceArray<ParamLimit> d_limits;

	std::unordered_map<cudaStream_t, DeviceBuffers> streamData;
	std::mutex streamDataMutex;

	DeviceBuffers* GetDeviceBuffers(cudaStream_t stream, int numspots);

	void LevMarIterate(const float* d_sample, const float* d_deriv, const float* d_expval, float* d_params,
		int numspots, float* trace, cudaStream_t stream);

};

CDLL_EXPORT Estimator* Estimator_WrapCUDA(cuEstimator* cuda_psf);

struct EstimatorProperties
{
	int numParams, sampleCount, numDiag, numConst, sampleIndexDims;
};

// C/Python API - All pointers are host memory
CDLL_EXPORT void Estimator_Delete(Estimator* estim);
CDLL_EXPORT const char* Estimator_ParamFormat(Estimator* estim);
CDLL_EXPORT void Estimator_GetProperties(Estimator* estim, EstimatorProperties& props);
CDLL_EXPORT void Estimator_SampleDims(Estimator* estim, int* dims);
CDLL_EXPORT void Estimator_GetParamLimits(Estimator* estim, ParamLimit* limits);
CDLL_EXPORT void Estimator_SetLMParams(Estimator* estim, const LMParams&);
CDLL_EXPORT void Estimator_GetLMParams(Estimator* estim, LMParams&);

CDLL_EXPORT void Estimator_ComputeExpectedValue(Estimator* estim, int numspots, const float* params, const float* constants, const int* spotpos, float* ev);
//CDLL_EXPORT void PSF_ComputeInitialEstimate(Estimator* estim, int numspots, const float* sample, const float* constants, float* params);
CDLL_EXPORT void Estimator_Estimate(Estimator* estim, int numspots, const float* sample, const float* constants, const int* spotpos,
	const float* initial, float* params, float* diagnostics, int* iterations, float* trace, int traceBufLen);
CDLL_EXPORT void Estimator_ComputeDerivatives(Estimator* estim, int numspots, const float* params, const float* constants, const int* spotpos, float* psf_deriv, float* ev);
// Compute chi-square and crlb. If sample=0 then chi-square is not computed.
CDLL_EXPORT void Estimator_ChiSquareAndCRLB(Estimator* estim, const float* params, const float* sample, const float* h_const, const int* spot_pos, int numspots, float* crlb, float* chisq);

