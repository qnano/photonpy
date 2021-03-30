// Estimator C API
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "Estimator.h"
#include "StringUtils.h"
#include "CudaUtils.h"
#include "SolveMatrix.h"

EstimatorBase::EstimatorBase(
	const std::vector<int>& sampleSize, 
	int numConst, 
	int diagsize, 
	const char * paramFormat, 
	std::vector<ParamLimit> limits, Context* ctx
): 
	paramFormat(paramFormat), 
	sampleSize(sampleSize),
	sampleCount(1), 
	numConstants(numConst), 
	numParams(limits.size()), 
	limits(limits), 
	diagsize(diagsize), 
	lmParams{30, 1e-15f, 0.0f },
	ContextObject(ctx)
{
	for (int s : sampleSize) sampleCount *= s;
	paramNames = StringSplit(this->paramFormat, ',');
}

EstimatorBase::~EstimatorBase()
{
}

int EstimatorBase::ParamIndex(const char * name)
{
	for (int i = 0; i < paramNames.size(); i++)
		if (paramNames[i] == name)
			return i;
	return -1;
}



CDLL_EXPORT void Estimator_Delete(Estimator * e)
{
	delete e;
}

CDLL_EXPORT const char * Estimator_ParamFormat(Estimator * e)
{
	return e->ParamFormat();
}

CDLL_EXPORT void Estimator_GetProperties(Estimator* psf, EstimatorProperties& props)
{
	props.numConst = psf->NumConstants();
	props.numDiag = psf->DiagSize();
	props.numParams = psf->NumParams();
	props.sampleCount = psf->SampleCount();
	props.sampleIndexDims = psf->SampleIndexDims();
}

CDLL_EXPORT void Estimator_SampleDims(Estimator* e, int* dims)
{
	for (int i = 0; i < e->SampleIndexDims(); i++)
		dims[i] = e->SampleSize()[i];
}

CDLL_EXPORT void Estimator_GetParamLimits(Estimator* estim, ParamLimit* limits)
{
	try {
		for (int i = 0; i < estim->NumParams(); i++)
			limits[i] = estim->ParamLimits()[i];
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}

CDLL_EXPORT void Estimator_GetLMParams(Estimator* estim, LMParams& lmParams)
{
	try {
		lmParams = estim->GetLMParams();
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}

CDLL_EXPORT void Estimator_SetLMParams(Estimator* estim, const LMParams& lmParams)
{
	try {
		estim->SetLMParams(lmParams);
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}

}

CDLL_EXPORT void Estimator_ComputeExpectedValue(Estimator * e, int numspots, const float* params, const float* constants, const int* spot_pos, float * ev)
{
	//DebugPrintf("Estimator_ExpVal: %d spots. fmt: %s\n", numspots, e->ParamFormat());
	if (numspots == 0)
		return;
	try {
		e->ExpectedValue(ev, params, constants, spot_pos, numspots);
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}

CDLL_EXPORT void Estimator_Estimate(Estimator * e, int numspots, const float * sample, const float* constants, 
	const int* spot_pos, const float * initial, float * params,  float *diagnostics, int* iterations, 
	float* trace, int traceBufLen)
{
	//DebugPrintf("Estimator_Estimate: %d spots. fmt: %s\n", numspots, e->ParamFormat());
	if (numspots == 0)
		return;
	try {
		e->Estimate(sample, constants, spot_pos, initial, params, diagnostics, iterations, numspots, trace, traceBufLen);
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}


CDLL_EXPORT void Estimator_ComputeDerivatives(Estimator * e, int numspots, const float * params, const float* constants,
	const int* spot_pos, float * psf_deriv, float * ev)
{
	if (numspots == 0)
		return;
	//DebugPrintf("Estimator_ComputeDerivatives: numspots=%d. fmt=%s\n", numspots, e->ParamFormat());
	try {
		e->Derivatives(psf_deriv, ev, params, constants, spot_pos, numspots);
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}

}

CDLL_EXPORT void Estimator_ChiSquareAndCRLB(Estimator* estim, const float* params, const float* sample, const float* h_const, const int* spot_pos, int numspots, float* crlb, float* chisq)
{
	if (numspots == 0)
		return;

	try {
		estim->ChiSquareAndCRLB(params, sample, h_const, spot_pos, crlb, chisq, numspots);
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}
