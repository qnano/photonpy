// SIMFLUX Estimator model
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "DLLMacros.h"
#include "CameraCalibration.h"
#include "simflux/SIMFLUX.h"

class Estimator;
struct SIMFLUX_Modulation;


CDLL_EXPORT Estimator* SIMFLUX2D_PSF_Create(Estimator* original, int num_patterns,
	const int * xyIBg_indices, bool simfluxFit, Context* ctx=0);

CDLL_EXPORT Estimator* SIMFLUX2D_Gauss2D_PSF_Create(int num_patterns, float sigmaX, float sigmaY, 
	int roisize, int numframes, bool simfluxFit, bool defineStartEnd, Context* ctx=0);


