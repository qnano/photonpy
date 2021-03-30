// Cubic Spline PSF
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen, Willem Melching 2018-2021
#pragma once
#include "dllmacros.h"

class sCMOS_Calibration;
class Context;
class Estimator;

struct CSpline_Calibration {

	int NumCoefs() const
	{
		return shape.prod() * 64;
	}

	Int3 shape;
	float z_min;
	float z_max;
	const float* coefs;
};

#define CSPLINE_FLATBG 0  // fit a single flat bg
#define CSPLINE_TILTEDBG 1

// specify background as a set of roisize^2 constants for each ROI to be fitted
// expval = I * psf(x,y,z) + bg * constant[y*w+x]
#define CSPLINE_BGIMAGE 2 

// If coefs parameter is nonzero, it is used instead of the CSpline_Calibration
CDLL_EXPORT Estimator* CSpline_CreatePSF(int roisize, const CSpline_Calibration& calib, const float* coefs, int mode, bool cuda, Context* ctx);
CDLL_EXPORT Estimator* CSpline_CreatePSF_FromFile(int roisize, const char *filename, int mode, bool cuda, Context* ctx);

// Compute coefficients for a 3D spline
// splineCoefs[shape[0]-3, shape[1]-3, shape[2]-3, 64]
CDLL_EXPORT void CSpline_Compute(const Int3& shape, const float* data, float* splineCoefs, int border);

CDLL_EXPORT void CSpline_Evaluate(const Int3& shape, const float* splineCoefs, const Vector3f& shift, int border, float* values);
