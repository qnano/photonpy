// SIMFLUX sine excitation model 
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "palala.h"
#include <cuda_runtime.h>

struct SIMFLUX_Modulation;

class SineWaveExcitation
{
public:
	const SIMFLUX_Modulation* modulation; // Each vector is [ kx, ky, phase ]
	int numep;

	PLL_DEVHOST int NumPatterns() const { return numep; }

	PLL_DEVHOST SineWaveExcitation(int numep, const SIMFLUX_Modulation* modulation)
		: numep(numep), modulation(modulation) {}

	PLL_DEVHOST float BackgroundPattern(int e, int x, int y) const
	{
//		SIMFLUX_Modulation mod = modulation[e];
		// this seems to match reality the most (tiny variations in bg per pattern, not as much as mod.intensity)
		return 1.0f;// / numep;
		//		return mod.intensity;
	}
	PLL_DEVHOST void ExcitationPattern(float& Q, float& dQdx, float& dQdy, int e, float2 xy) const {
		// compute Q, dQ/dx, dQ/dy
		SIMFLUX_Modulation mod = modulation[e];
		float A = mod.intensity;
		float w = mod.k[0] * xy.x + mod.k[1] * xy.y;
		float ang = w - mod.phase;
		Q = A * (1.0f + mod.depth * sin(ang));
		dQdx = A * mod.k[0] * mod.depth * cos(ang);
		dQdy = A * mod.k[1] * mod.depth * cos(ang);
	}
	PLL_DEVHOST void ExcitationPattern(float& Q, float deriv[3], int e, float const pos[3]) const {
		// compute Q, dQ/dx, dQ/dy
		SIMFLUX_Modulation mod = modulation[e];
		float A = mod.intensity;
		float w = mod.k[0] * pos[0] + mod.k[1] * pos[1] + mod.k[2] * pos[2];
		float ang = w - mod.phase;
		Q = A * (1.0f + mod.depth * sin(ang));
		for (int i=0;i<3;i++)
			deriv[i] = A * mod.k[i] * mod.depth * cos(ang);
	}
	PLL_DEVHOST float MeanBackgroundCoefficient() const
	{
		return 1.0f / numep;
	}
};
