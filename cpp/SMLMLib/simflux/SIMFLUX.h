// SIMFLUX definitions
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "DLLMacros.h"

#include <vector>
#include "Vector.h"
#include "MathUtils.h"
#include "Estimators/Estimation.h"
#include "Estimators/Gaussian/GaussianPSF.h"


struct SIMFLUX_ASW_Params {
	int roisize;
	int numepp;
	float psfSigma;
	int maxLevMarIterations;
	float levMarLambdaStep;
};

// Q(x,y) = intensity*(1+depth*sin(kx*x + ky*y + kz*z - phase))
struct SIMFLUX_Modulation {
	float k[3];
	float depth;
	float phase;
	float intensity; // relative intensity
};


typedef Vector4f SIMFLUX_Theta;


template<typename TResult>
class ImageProcessingQueue;


struct SIMFLUX_EstimationResult
{
	typedef SIMFLUX_Theta Params;
	enum { MaxSwitchFrames=2 };

	template<typename TLevMarResult>
	PLL_DEVHOST SIMFLUX_EstimationResult(Params initialValue, TLevMarResult levMarResult, const Vector<float,16>& cov, const Vector<float,8>& crlb, int nsf) :
		estimate(levMarResult.estimate), initialValue(initialValue), crlb(crlb),
		iterations(levMarResult.iterations) {	}
	PLL_DEVHOST SIMFLUX_EstimationResult() { }

	Params estimate;
	Params initialValue;
	Vector<float, 4 + MaxSwitchFrames> crlb; // [X,Y,I,bg, SwitchFrameIntensities...]
	int iterations;
	uint64_t switchFrameMask, silmFrameMask;
	Int2 roiPosition;
};



#define SIMFLUX_MLE_CUDA 1				
#define SIMFLUX_MLE_FIXEDW 4

/*

imageData: float[imgw*imgw*epps*numspots]
backgroundMatrix: float[imgw*imgw*epps*numspots]
phi: float[epps*numspots]

*/
struct ImageQueueConfig;
class ISpotDetectorFactory;


/* SILM pipeline in pieces...
- Gaussian fit
- Estimate intensities/bg per frame
- Estimate initial SILM theta
- HMM
- SILM fit (with specific frames)
*/

//CDLL_EXPORT void Gauss2D_EstimateIntensityBg(const float* imageData, Vector2f *IBg, Vector2f* IBg_crlb,
//	int numspots, const Vector2f* xy, float sigma, int imgw, int maxIterations, bool cuda)



CDLL_EXPORT void SIMFLUX_ProjectPointData(const Vector3f *xyI, int numpts, int projectionWidth,
	float scale, int numProjAngles, const float *projectionAngles, float* output ,float* shifts);
