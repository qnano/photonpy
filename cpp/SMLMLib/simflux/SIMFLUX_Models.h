// SIMFLUX 2D Gaussian PSF Model
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once
#include "CudaUtils.h"
#include "ExcitationModel.h"

#pragma warning(disable : 4503) // decorated name length exceeded, name was truncated

PLL_DEVHOST inline void ComputeImageSums(const float* frames, float* dst, int w, int h, int numframes)
{
	for (int i = 0; i < w*h; i++)
		dst[i] = 0;
	for (int f = 0; f < numframes; f++)
	{
		for (int i = 0; i < w*h; i++)
			dst[i] += frames[f*w*h + i];
	}
}

struct SIMFLUX_Calibration
{
	SineWaveExcitation epModel;
	float2 sigma;
};

struct SIMFLUX_Gauss2D_Model
{
	typedef float T;
	typedef SIMFLUX_Theta Params;
	enum { K = Params::K };

	typedef Int3 TSampleIndex;

	PLL_DEVHOST SIMFLUX_Gauss2D_Model(int roisize, const SIMFLUX_Calibration& calib, int startframe, int endframe, int numframes, Int3 roipos)
		: sigma(calib.sigma), roisize(roisize), startframe(startframe), endframe(endframe), epModel(calib.epModel), numframes(numframes), roipos(roipos)
	{}

	int roisize;
	int numframes;
	//int numframes; // ep = (frameNum + startPattern) % params.numepp
	int startframe, endframe;
	float2 sigma;
	Int3 roipos;
	SineWaveExcitation epModel;

	PLL_DEVHOST int SampleCount() const { return roisize * roisize * numframes; }

	static PLL_DEVHOST T StopLimit(int k)
	{
		const float deltaStopLimit[] = { 1e-5f, 1e-5f, 1e-2f,1e-4f };
		return deltaStopLimit[k];
	}

	PLL_DEVHOST void CheckLimits(Params& t) const {
		t.elem[0] = clamp(t.elem[0], 2.0f, roisize - 3.0f);
		t.elem[1] = clamp(t.elem[1], 2.0f, roisize - 3.0f);
		t.elem[2] = fmaxf(t.elem[2], 25.0f);
		t.elem[3] = fmaxf(t.elem[3], 0.0f);
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, SIMFLUX_Theta theta) const
	{
		int w = roisize;
		T _1oSq2PiSigmaX = 1.0f / (sqrtf(2 * MATH_PI) * sigma.x);
		T _1oSq2SigmaX = 1.0f / (sqrtf(2) * sigma.x);
		T _1oSq2PiSigmaY = 1.0f / (sqrtf(2 * MATH_PI) * sigma.y);
		T _1oSq2SigmaY = 1.0f / (sqrtf(2) * sigma.y);

		T thetaX = theta[0], thetaY = theta[1], thetaI = theta[2], thetaBg = theta[3];

		int e = roipos[0];
		for (int f = startframe; f < endframe; f++) {
			// compute Q, dQ/dx, dQ/dy
			float Q, dQdx, dQdy;
			epModel.ExcitationPattern(Q, dQdx, dQdy, e, { thetaX+roipos[2],thetaY+roipos[1] });

			for (int y = 0; y < w; y++) {
				// compute Ey,dEy/dy
				T Yexp0 = (y - thetaY + .5f) * _1oSq2SigmaY;
				T Yexp1 = (y - thetaY - .5f) * _1oSq2SigmaY;
				T Ey = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
				T dEy = _1oSq2PiSigmaY * (exp(-Yexp1 * Yexp1) - exp(-Yexp0 * Yexp0));
				for (int x = 0; x < w; x++) {
					// compute Ex,dEx/dx
					T Xexp0 = (x - thetaX + .5f) * _1oSq2SigmaX;
					T Xexp1 = (x - thetaX - .5f) * _1oSq2SigmaX;
					T Ex = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
					T dEx = _1oSq2PiSigmaX * (exp(-Xexp1 * Xexp1) - exp(-Xexp0 * Xexp0));

					const float exc_bg = epModel.BackgroundPattern(e, x, y);

					// combine
					const float Exy = Ex * Ey;
					const float mu = thetaI * (Q*Exy) + thetaBg * exc_bg;

					const float dmu_x = thetaI * (dQdx * Exy + Q * dEx * Ey);
					const float dmu_y = thetaI * (dQdy * Exy + Q * Ex * dEy);
					const float dmu_I = Q * Exy;
					const float dmu_bg = exc_bg;

					const float jacobian[] = { dmu_x, dmu_y,dmu_I,dmu_bg };

					cb(f*roisize*roisize+y*roisize+x, mu, jacobian);
				}
			}

			e++;
			if (e == epModel.NumPatterns()) e = 0;
		}
	}
};


