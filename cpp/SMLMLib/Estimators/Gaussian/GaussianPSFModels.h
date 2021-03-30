// 2D Gaussian PSF models
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "Vector.h"
#include "MathUtils.h"
#include "palala.h"
#include "GaussianPSF.h"
#include "CudaMath.h"

typedef Vector5f Theta_XYZIBg;
typedef Vector4f Theta_XYIBg;
typedef Vector5f Theta_XYIBgSigma;

#define Gauss2D_MinSigma (1.0f)
#define Gauss2D_Border (1.5f)

template<typename TNumber, int NumParams>
struct Gauss2D_PSFModel {
	enum { K = NumParams };
	typedef typename TNumber T;
	typedef Vector<T,K> Params;
	typedef Int2 TSampleIndex;
	Int2 roisize;

	PLL_DEVHOST Gauss2D_PSFModel(Int2 roisize) : roisize(roisize) {}

	PLL_DEVHOST int SampleCount() const { return roisize[0] * roisize[1]; }
	PLL_DEVHOST int Width() const { return roisize[1]; }
	PLL_DEVHOST int Height() const { return roisize[0]; }
};

struct Gauss2D_Model_XYIBg : public Gauss2D_PSFModel<float, 4>
{
	enum { NumStartPos = 1 };
	enum { NumConstants = 0 };
	typedef float2 Calibration;

	const float2 sigma;

	static const char* ParamFormat() { return "x,y,I,bg"; }

	static PLL_DEVHOST T StopLimit(int k)
	{
		const float deltaStopLimit[]={ 1e-5f, 1e-5f, 1e-1f,1e-5f };
		return deltaStopLimit[k];
	}

	PLL_DEVHOST Gauss2D_Model_XYIBg(Int2 roisize, float2 sigma, const float* sigmas=0) : 
		Gauss2D_PSFModel(roisize) , sigma(sigma) {}

	PLL_DEVHOST void CheckLimits(Params& t) const {
		t.elem[0] = clamp(t.elem[0], Gauss2D_Border, roisize[1] - Gauss2D_Border-1);
		t.elem[1] = clamp(t.elem[1], Gauss2D_Border, roisize[0] - Gauss2D_Border-1);

		if (t.elem[2] < 25.0f) t.elem[2] = 25.0f;
		if (t.elem[3] < 0.0f) t.elem[3] = 0.0f;
		//	if (t.elem[3] > t.elem[2]) t.elem[3] = t.elem[2];
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, const Params& theta) const
	{
		const T OneOverSqrt2PiSigmaX = 1.0f / (sqrtf(2 * MATH_PI) * sigma.x);
		const T OneOverSqrt2SigmaX = 1.0f / (sqrtf(2) * sigma.x);
		const T OneOverSqrt2PiSigmaY = 1.0f / (sqrtf(2 * MATH_PI) * sigma.y);
		const T OneOverSqrt2SigmaY = 1.0f / (sqrtf(2) * sigma.y);

		T tx = theta[0];
		T ty = theta[1];
		T tI = theta[2];
		T tbg = theta[3];

		for (int y = 0; y < Height(); y++)
		{
			T Yexp0 = (y - ty + .5f) * OneOverSqrt2SigmaY;
			T Yexp1 = (y - ty - .5f) * OneOverSqrt2SigmaY;
			T DeltaY = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
			T dEy = OneOverSqrt2PiSigmaY * (exp(-Yexp1 * Yexp1) - exp(-Yexp0 * Yexp0));
			for (int x = 0; x < Width(); x++)
			{
				T Xexp0 = (x - tx + .5f) * OneOverSqrt2SigmaX;
				T Xexp1 = (x - tx - .5f) * OneOverSqrt2SigmaX;
				T DeltaX = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
				T dEx = OneOverSqrt2PiSigmaX * (exp(-Xexp1 * Xexp1) - exp(-Xexp0 * Xexp0));

				T mu = tbg + tI * DeltaX * DeltaY;
				T dmu_dx = tI * dEx * DeltaY;
				T dmu_dy = tI * dEy * DeltaX;
				T dmu_dI0 = DeltaX * DeltaY;
				T dmu_dIbg = 1;
				const T jacobian[] = { dmu_dx, dmu_dy, dmu_dI0, dmu_dIbg };
				cb(y * Width() + x, mu, jacobian);
			}
		}
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeSecondDerivatives(TCallback cb, Params theta) const
	{
		const T OneOverSqrt2PiSigmaX = 1.0f / (sqrtf(2 * MATH_PI) * sigma.x);
		const T OneOverSqrt2SigmaX = 1.0f / (sqrtf(2) * sigma.x);
		const T OneOverSqrt2PiSigma3X = 1.0f / (sqrtf(2 * MATH_PI) * sigma.x*sigma.x*sigma.x);

		const T OneOverSqrt2PiSigmaY = 1.0f / (sqrtf(2 * MATH_PI) * sigma.y);
		const T OneOverSqrt2SigmaY = 1.0f / (sqrtf(2) * sigma.y);
		const T OneOverSqrt2PiSigma3Y = 1.0f / (sqrtf(2 * MATH_PI) * sigma.y*sigma.y*sigma.y);

		T tx = theta[0];
		T ty = theta[1];
		T tI = theta[2];
		T tbg = theta[3];

		for (int y = 0; y < imgw; y++)
		{
			T Yexp0 = (y - ty + .5f) * OneOverSqrt2SigmaY;
			T Yexp1 = (y - ty - .5f) * OneOverSqrt2SigmaY;
			T DeltaY = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
			T dEy = tI * OneOverSqrt2PiSigmaY * (exp(-Yexp1 * Yexp1) - exp(-Yexp0 * Yexp0));
			T dEy2 = tI * OneOverSqrt2PiSigma3Y * ((y - ty - 0.5f) * exp(-Yexp1 * Yexp1) - (y - ty + 0.5f) * exp(-Yexp0 * Yexp0));
			for (int x = 0; x < imgw; x++)
			{
				T Xexp0 = (x - tx + .5f) * OneOverSqrt2SigmaX;
				T Xexp1 = (x - tx - .5f) * OneOverSqrt2SigmaX;
				T DeltaX = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
				T dEx = tI * OneOverSqrt2PiSigmaX * (exp(-Xexp1 * Xexp1) - exp(-Xexp0 * Xexp0));
				T dEx2 = tI * OneOverSqrt2PiSigma3X * ((x - tx - 0.5f) * exp(-Xexp1 * Xexp1) - (x - tx + 0.5f) * exp(-Xexp0 * Xexp0));

				T mu = tbg + tI * DeltaX * DeltaY;
				T dmu_dx = dEx * DeltaY;
				T dmu_dy = dEy * DeltaX;
				T dmu_dI0 = DeltaX * DeltaY;

				T dmu2_dx2 = dEx2 * DeltaY;
				T dmu2_dy2 = dEy2 * DeltaX;

				const T firstOrder[] = { dmu_dx, dmu_dy, dmu_dI0, 1 };
				const T secondOrder[] = { dmu2_dx2, dmu2_dy2, 0, 0 };
				cb(y * imgw + x, mu, firstOrder, secondOrder);
			}
		}
	}


	PLL_DEVHOST Params ComputeInitialEstimate(const T* sample) const
	{
		Vector3f com = ComputeCOM(sample, roisize);
		return { com[0],com[1],com[2]*0.7f, 0.0f };
	}
};


// Same as X,Y,I,Bg but with sigma's defined by a per-spot constant value. X,Y,I,bg model above is maintained for backwards compatibility
struct Gauss2D_Model_XYIBgConstSigma : public Gauss2D_PSFModel<float, 4>
{
	enum { NumStartPos = 1 };
	enum { NumConstants = 2 };
	struct Calibration {};

	const float2 sigma;

	static const char* ParamFormat() { return "x,y,I,bg"; }

	static PLL_DEVHOST T StopLimit(int k)
	{
		const float deltaStopLimit[] = { 1e-5f, 1e-5f, 1e-1f,1e-5f };
		return deltaStopLimit[k];
	}

	PLL_DEVHOST Gauss2D_Model_XYIBgConstSigma(Int2 roisize, Calibration calib, const float* sigmas = 0) :
		Gauss2D_PSFModel(roisize), sigma(*(const float2*)sigmas) {}

	PLL_DEVHOST void CheckLimits(Params& t) const {
		t.elem[0] = clamp(t.elem[0], Gauss2D_Border, roisize[1] - Gauss2D_Border - 1);
		t.elem[1] = clamp(t.elem[1], Gauss2D_Border, roisize[0] - Gauss2D_Border - 1);

		if (t.elem[2] < 25.0f) t.elem[2] = 25.0f;
		if (t.elem[3] < 0.0f) t.elem[3] = 0.0f;
		//	if (t.elem[3] > t.elem[2]) t.elem[3] = t.elem[2];
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, const Params& theta) const
	{
		const T OneOverSqrt2PiSigmaX = 1.0f / (sqrtf(2 * MATH_PI) * sigma.x);
		const T OneOverSqrt2SigmaX = 1.0f / (sqrtf(2) * sigma.x);
		const T OneOverSqrt2PiSigmaY = 1.0f / (sqrtf(2 * MATH_PI) * sigma.y);
		const T OneOverSqrt2SigmaY = 1.0f / (sqrtf(2) * sigma.y);

		T tx = theta[0];
		T ty = theta[1];
		T tI = theta[2];
		T tbg = theta[3];

		for (int y = 0; y < Height(); y++)
		{
			T Yexp0 = (y - ty + .5f) * OneOverSqrt2SigmaY;
			T Yexp1 = (y - ty - .5f) * OneOverSqrt2SigmaY;
			T DeltaY = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
			T dEy = OneOverSqrt2PiSigmaY * (exp(-Yexp1 * Yexp1) - exp(-Yexp0 * Yexp0));
			for (int x = 0; x < Width(); x++)
			{
				T Xexp0 = (x - tx + .5f) * OneOverSqrt2SigmaX;
				T Xexp1 = (x - tx - .5f) * OneOverSqrt2SigmaX;
				T DeltaX = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
				T dEx = OneOverSqrt2PiSigmaX * (exp(-Xexp1 * Xexp1) - exp(-Xexp0 * Xexp0));

				T mu = tbg + tI * DeltaX * DeltaY;
				T dmu_dx = tI * dEx * DeltaY;
				T dmu_dy = tI * dEy * DeltaX;
				T dmu_dI0 = DeltaX * DeltaY;
				T dmu_dIbg = 1;
				const T jacobian[] = { dmu_dx, dmu_dy, dmu_dI0, dmu_dIbg };
				cb(y * Width() + x, mu, jacobian);
			}
		}
	}


	PLL_DEVHOST Params ComputeInitialEstimate(const T* sample) const
	{
		Vector3f com = ComputeCOM(sample, roisize);
		return { com[0],com[1],com[2] * 0.7f, 0.0f };
	}
};


// X,Y,I,Bg0,BgX,BgY
// mu = bg0 + bgx*(x-roisize*0.5f) + bgy*(y-roisize*0.5f) + I * PSF(x,y)
struct Gauss2D_Model_XYITiltedBg: public Gauss2D_PSFModel<float, 6>
{
	enum { NumStartPos = 1 };
	enum { NumConstants = 2 };
	struct Calibration {};

	float2 sigma;

	static const char* ParamFormat() { return "x,y,I,bg,bgx,bgy"; }

	static PLL_DEVHOST T StopLimit(int k)
	{
		const float deltaStopLimit[] = { 1e-5f, 1e-5f, 1e-1f,1e-5f,1e-5f,1e-5f };
		return deltaStopLimit[k];
	}

	PLL_DEVHOST Gauss2D_Model_XYITiltedBg(Int2 roisize, Calibration calib, const float* sigmas = 0) :
		Gauss2D_PSFModel(roisize), sigma(*(const float2*)sigmas) {}

	PLL_DEVHOST void CheckLimits(Params& t) const {
		t.elem[0] = clamp(t.elem[0], Gauss2D_Border, roisize[1] - Gauss2D_Border - 1);
		t.elem[1] = clamp(t.elem[1], Gauss2D_Border, roisize[0] - Gauss2D_Border - 1);

		if (t.elem[2] < 25.0f) t.elem[2] = 25.0f;
		if (t.elem[3] < 0.0f) t.elem[3] = 0.0f;
		//	if (t.elem[3] > t.elem[2]) t.elem[3] = t.elem[2];
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, const Params& theta) const
	{
		const T OneOverSqrt2PiSigmaX = 1.0f / (sqrtf(2 * MATH_PI) * sigma.x);
		const T OneOverSqrt2SigmaX = 1.0f / (sqrtf(2) * sigma.x);
		const T OneOverSqrt2PiSigmaY = 1.0f / (sqrtf(2 * MATH_PI) * sigma.y);
		const T OneOverSqrt2SigmaY = 1.0f / (sqrtf(2) * sigma.y);

		T tx = theta[0];
		T ty = theta[1];
		T tI = theta[2];
		T tbg = theta[3], bgx = theta[4], bgy = theta[5];
		float halfw = Width() * 0.5f;
		float halfh = Height() * 0.5f;

		for (int y = 0; y < Height(); y++)
		{
			T Yexp0 = (y - ty + .5f) * OneOverSqrt2SigmaY;
			T Yexp1 = (y - ty - .5f) * OneOverSqrt2SigmaY;
			T DeltaY = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
			T dEy = OneOverSqrt2PiSigmaY * (exp(-Yexp1 * Yexp1) - exp(-Yexp0 * Yexp0));
			for (int x = 0; x < Width(); x++)
			{
				T Xexp0 = (x - tx + .5f) * OneOverSqrt2SigmaX;
				T Xexp1 = (x - tx - .5f) * OneOverSqrt2SigmaX;
				T DeltaX = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
				T dEx = OneOverSqrt2PiSigmaX * (exp(-Xexp1 * Xexp1) - exp(-Xexp0 * Xexp0));

				T mu = tbg + bgx*(x-halfw) + bgy*(y-halfh) + tI * DeltaX * DeltaY;
				T dmu_dx = tI * dEx * DeltaY;
				T dmu_dy = tI * dEy * DeltaX;
				T dmu_dI0 = DeltaX * DeltaY;
				const T jacobian[] = { dmu_dx, dmu_dy, dmu_dI0, 1.0f, x - halfw, y - halfh};
				cb(y * Width() + x, mu, jacobian);
			}
		}
	}


	PLL_DEVHOST Params ComputeInitialEstimate(const T* sample) const
	{
		Vector3f com = ComputeCOM(sample, roisize);
		return { com[0],com[1],com[2] * 0.7f, 0.0f,0.0f, 0.0f };
	}
};



struct Gauss2D_Model_XYIBgSigma : public Gauss2D_PSFModel<float, 5>
{
	enum { NumStartPos = 1 };
	enum { NumConstants = 0 };

	typedef float Calibration;
	Calibration initialSigma;

	static const char* ParamFormat() { return "x,y,	I,bg,sigma"; }

	static PLL_DEVHOST T StopLimit(int k)
	{
		const float deltaStopLimit[] = { 1e-5f, 1e-5f, 1e-1f,1e-4f, 1e-5f };
		return deltaStopLimit[k];
	}

	PLL_DEVHOST Gauss2D_Model_XYIBgSigma(Int2 roisize, float initialSigma, const float* unused_constants = 0) : 
		Gauss2D_PSFModel(roisize), initialSigma(initialSigma) {}

	PLL_DEVHOST void CheckLimits(Params& t) const
	{
		t.elem[0] = clamp(t.elem[0], Gauss2D_Border, roisize[1] - Gauss2D_Border-1);
		t.elem[1] = clamp(t.elem[1], Gauss2D_Border, roisize[0] - Gauss2D_Border-1);

		if (t.elem[2] < 10.0f) t.elem[2] = 10.0f;
		if (t.elem[3] < 0.0f) t.elem[3] = 0.0f;
		if (t.elem[4] < Gauss2D_MinSigma) t.elem[4] = Gauss2D_MinSigma;
		if (t.elem[4] > roisize[0] * 0.5f) t.elem[4] = roisize[0] * 0.5f;
		//	if (t.elem[3] > t.elem[2]) t.elem[3] = t.elem[2];
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, const Params& theta) const
	{
		T tx = theta[0];
		T ty = theta[1];
		T tI = theta[2];
		T tbg = theta[3];
		T tsigma = theta[4];

		const T invSqrtPi = 1.0f/sqrtf(MATH_PI);
		const T OneOverSqrt2PiSigma = 1.0f / (sqrtf(2 * MATH_PI) * tsigma);
		const T OneOverSqrt2Sigma = 1.0f / (sqrtf(2) * tsigma);

		for (int y = 0; y < roisize[0]; y++)
		{
			T Yexp0 = (y - ty + .5f) * OneOverSqrt2Sigma;
			T Yexp1 = (y - ty - .5f) * OneOverSqrt2Sigma;
			T DeltaY = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
			T expY1 = exp(-Yexp1 * Yexp1), expY0 = exp(-Yexp0 * Yexp0);
			T dEy = OneOverSqrt2PiSigma * (expY1 - expY0);
			T dDeltaY_dsigma = (expY1*Yexp1 - expY0 * Yexp0)*invSqrtPi;

			for (int x = 0; x < roisize[1]; x++)
			{
				T Xexp0 = (x - tx + .5f) * OneOverSqrt2Sigma;
				T Xexp1 = (x - tx - .5f) * OneOverSqrt2Sigma;
				T DeltaX = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
				T expX1 = exp(-Xexp1 * Xexp1), expX0 = exp(-Xexp0 * Xexp0);
				T dEx = OneOverSqrt2PiSigma * (expX1 - expX0);
				T dDeltaX_dsigma = (expX1*Xexp1 - expX0 * Xexp0)*invSqrtPi;

				T mu = tbg + tI * DeltaX * DeltaY;
				T dmu_dx = tI * dEx * DeltaY;
				T dmu_dy = tI * dEy * DeltaX;
				T dmu_dI0 = DeltaX * DeltaY;
				T dmu_dIbg = 1;
				T dmu_dSigma = tI * (dDeltaX_dsigma*DeltaY + dDeltaY_dsigma * DeltaX);
				const T jacobian[] = { dmu_dx, dmu_dy, dmu_dI0, dmu_dIbg, dmu_dSigma };
				cb(y * roisize[1] + x, mu, jacobian);
			}
		}
	}

	PLL_DEVHOST Params ComputeInitialEstimate(const T* sample) const
	{
		Vector3f com = ComputeCOM(sample, roisize);
		return { com[0],com[1],com[2],0.0f, initialSigma };
	}
};



struct Gauss2D_Model_XYIBgSigmaXY : public Gauss2D_PSFModel<float, 6>
{
	enum { NumStartPos = 1 };
	enum { NumConstants = 0 };

	typedef Vector2f Calibration;
	Calibration initialSigma;

	static const char* ParamFormat() { return "x,y,I,bg,sx,sy"; }

	static PLL_DEVHOST T StopLimit(int k)
	{
		const float deltaStopLimit[] = { 1e-5f, 1e-5f, 1e-1f,1e-4f,1e-5f,1e-5f };
		return deltaStopLimit[k];
	}

	PLL_DEVHOST Gauss2D_Model_XYIBgSigmaXY(Int2 roisize, Vector2f initialSigma, const float* unused_constants = 0) :
		Gauss2D_PSFModel(roisize), initialSigma(initialSigma) {}

	PLL_DEVHOST void CheckLimits(Params& t) const {
		t.elem[0] = clamp(t.elem[0], Gauss2D_Border, roisize[1] - Gauss2D_Border -1);
		t.elem[1] = clamp(t.elem[1], Gauss2D_Border, roisize[0] - Gauss2D_Border -1);

		if (t.elem[2] < 10.0f) t.elem[2] = 10.0f;
		if (t.elem[3] < 0.0f) t.elem[3] = 0.0f;

		t.elem[4] = clamp(t.elem[4], Gauss2D_MinSigma, roisize[1] * 0.5f);
		t.elem[5] = clamp(t.elem[5], Gauss2D_MinSigma, roisize[0] * 0.5f);
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, const Params& theta) const
	{
		T tx = theta[0];
		T ty = theta[1];
		T tI = theta[2];
		T tbg = theta[3];
		T tsigmax = theta[4];
		T tsigmay = theta[5];

		const T invSqrtPi = 1.0f / sqrtf(MATH_PI);
		const T OneOverSqrt2PiSigmaX = 1.0f / (sqrtf(2 * MATH_PI) * tsigmax);
		const T OneOverSqrt2PiSigmaY = 1.0f / (sqrtf(2 * MATH_PI) * tsigmay);
		const T OneOverSqrt2SigmaX = 1.0f / (sqrtf(2) * tsigmax);
		const T OneOverSqrt2SigmaY = 1.0f / (sqrtf(2) * tsigmay);

		for (int y = 0; y < roisize[0]; y++)
		{
			T Yexp0 = (y - ty + .5f) * OneOverSqrt2SigmaY;
			T Yexp1 = (y - ty - .5f) * OneOverSqrt2SigmaY;
			T DeltaY = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
			T expY1 = exp(-Yexp1 * Yexp1), expY0 = exp(-Yexp0 * Yexp0);
			T dEy = OneOverSqrt2PiSigmaY * (expY1 - expY0);
			T dDeltaY_dsigma = (expY1*Yexp1 - expY0 * Yexp0)*invSqrtPi;

			for (int x = 0; x < roisize[1]; x++)
			{
				T Xexp0 = (x - tx + .5f) * OneOverSqrt2SigmaX;
				T Xexp1 = (x - tx - .5f) * OneOverSqrt2SigmaX;
				T DeltaX = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
				T expX1 = exp(-Xexp1 * Xexp1), expX0 = exp(-Xexp0 * Xexp0);
				T dEx = OneOverSqrt2PiSigmaX * (expX1 - expX0);
				T dDeltaX_dsigma = (expX1*Xexp1 - expX0 * Xexp0)*invSqrtPi;

				T mu = tbg + tI * DeltaX * DeltaY;
				T dmu_dx = tI * dEx * DeltaY;
				T dmu_dy = tI * dEy * DeltaX;
				T dmu_dI0 = DeltaX * DeltaY;
				T dmu_dSigmaX = tI * dDeltaX_dsigma * DeltaY;
				T dmu_dSigmaY = tI * dDeltaY_dsigma * DeltaX;
				const T jacobian[] = { dmu_dx, dmu_dy, dmu_dI0, 1.0f,dmu_dSigmaX, dmu_dSigmaY };
				cb(y*roisize[1]+x, mu, jacobian);
			}
		}
	}
	PLL_DEVHOST Params ComputeInitialEstimate(const T* sample) const
	{
		Vector3f com = ComputeCOM(sample, roisize);
		return { com[0],com[1],com[2],0.0f,  initialSigma[0], initialSigma[1] };
	}
};



// mu = bg0 + bgx*(x-roisize*0.5f) + bgy*(y-roisize*0.5f) + I * PSF(x,y)
struct Gauss2D_Model_XYITiltedBgSigmaXY : public Gauss2D_PSFModel<float, 8>
{
	enum { NumStartPos = 1 };
	enum { NumConstants = 0 };

	typedef Vector2f Calibration;
	Calibration initialSigma;

	static const char* ParamFormat() { return "x,y,I,bg,bgx,bgy,sx,sy"; }

	static PLL_DEVHOST T StopLimit(int k)
	{
		const float deltaStopLimit[] = { 1e-5f, 1e-5f, 1e-1f,1e-4f, 1e-4f,1e-4f,1e-5f,1e-5f };
		return deltaStopLimit[k];
	}

	PLL_DEVHOST Gauss2D_Model_XYITiltedBgSigmaXY(Int2 roisize, Vector2f initialSigma, const float* unused_constants = 0) :
		Gauss2D_PSFModel(roisize), initialSigma(initialSigma) {}

	PLL_DEVHOST void CheckLimits(Params& t) const {
		t.elem[0] = clamp(t.elem[0], Gauss2D_Border, roisize[1] - Gauss2D_Border - 1);
		t.elem[1] = clamp(t.elem[1], Gauss2D_Border, roisize[0] - Gauss2D_Border - 1);

		if (t.elem[2] < 10.0f) t.elem[2] = 10.0f;
		if (t.elem[3] < 0.0f) t.elem[3] = 0.0f;

		t.elem[6] = clamp(t.elem[6], Gauss2D_MinSigma, roisize[1] * 0.5f);
		t.elem[7] = clamp(t.elem[7], Gauss2D_MinSigma, roisize[0] * 0.5f);
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, const Params& theta) const
	{
		T tx = theta[0];
		T ty = theta[1];
		T tI = theta[2];
		T tbg = theta[3], bgx = theta[4], bgy = theta[5];
		T tsigmax = theta[6];
		T tsigmay = theta[7];
		float halfw = Width() * 0.5f;
		float halfh = Height() * 0.5f;

		const T invSqrtPi = 1.0f / sqrtf(MATH_PI);
		const T OneOverSqrt2PiSigmaX = 1.0f / (sqrtf(2 * MATH_PI) * tsigmax);
		const T OneOverSqrt2PiSigmaY = 1.0f / (sqrtf(2 * MATH_PI) * tsigmay);
		const T OneOverSqrt2SigmaX = 1.0f / (sqrtf(2) * tsigmax);
		const T OneOverSqrt2SigmaY = 1.0f / (sqrtf(2) * tsigmay);

		for (int y = 0; y < roisize[0]; y++)
		{
			T Yexp0 = (y - ty + .5f) * OneOverSqrt2SigmaY;
			T Yexp1 = (y - ty - .5f) * OneOverSqrt2SigmaY;
			T DeltaY = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
			T expY1 = exp(-Yexp1 * Yexp1), expY0 = exp(-Yexp0 * Yexp0);
			T dEy = OneOverSqrt2PiSigmaY * (expY1 - expY0);
			T dDeltaY_dsigma = (expY1 * Yexp1 - expY0 * Yexp0) * invSqrtPi;

			for (int x = 0; x < roisize[1]; x++)
			{
				T Xexp0 = (x - tx + .5f) * OneOverSqrt2SigmaX;
				T Xexp1 = (x - tx - .5f) * OneOverSqrt2SigmaX;
				T DeltaX = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
				T expX1 = exp(-Xexp1 * Xexp1), expX0 = exp(-Xexp0 * Xexp0);
				T dEx = OneOverSqrt2PiSigmaX * (expX1 - expX0);
				T dDeltaX_dsigma = (expX1 * Xexp1 - expX0 * Xexp0) * invSqrtPi;

				T mu = tbg + tI * DeltaX * DeltaY;
				T dmu_dx = tI * dEx * DeltaY;
				T dmu_dy = tI * dEy * DeltaX;
				T dmu_dI0 = DeltaX * DeltaY;
				T dmu_dSigmaX = tI * dDeltaX_dsigma * DeltaY;
				T dmu_dSigmaY = tI * dDeltaY_dsigma * DeltaX;
				const T jacobian[] = { dmu_dx, dmu_dy, dmu_dI0, 1.0f,x - halfw, y - halfh, dmu_dSigmaX, dmu_dSigmaY };
				cb(y * roisize[1] + x, mu, jacobian);
			}
		}
	}
	PLL_DEVHOST Params ComputeInitialEstimate(const T* sample) const
	{
		Vector3f com = ComputeCOM(sample, roisize);
		return { com[0],com[1],com[2],0.0f, 0.0f, 0.0f, initialSigma[0], initialSigma[1] };
	}
};


// A model that only fits signal vs background for a predefined Estimator
struct IntensityBgModel : Gauss2D_PSFModel<float, 2>
{
	const float* psf; // precomputed [imgw*imgw]

	// Precompute the integral of the gaussian over the pixel area, as this will stay constant during optimization
	PLL_DEVHOST static void ComputeGaussianPSF(float2 sigma, float tx, float ty, int imgw, float* psf)
	{
		const T OneOverSqrt2SigmaX = 1.0f / (sqrtf(2) * sigma.x);
		const T OneOverSqrt2SigmaY = 1.0f / (sqrtf(2) * sigma.y);

		for (int y = 0; y < imgw; y++)
		{
			T Yexp0 = (y - ty + .5f) * OneOverSqrt2SigmaY;
			T Yexp1 = (y - ty - .5f) * OneOverSqrt2SigmaY;
			T DeltaY = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
			for (int x = 0; x < imgw; x++)
			{
				T Xexp0 = (x - tx + .5f) * OneOverSqrt2SigmaX;
				T Xexp1 = (x - tx - .5f) * OneOverSqrt2SigmaX;
				T DeltaX = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
				psf[y*imgw + x] = DeltaX * DeltaY;
			}
		}
	}

	PLL_DEVHOST T StopLimit(int k) const
	{
		const float deltaStopLimit[]={ 1e-2f,1e-4f };
		return deltaStopLimit[k];
	}

	PLL_DEVHOST IntensityBgModel(Int2 roisize, const float* psf_space) : Gauss2D_PSFModel(roisize), psf(psf_space) {}

	PLL_DEVHOST void CheckLimits(Params& t) const 
	{
		if (t.elem[0] < 1e-5f) t.elem[0] = 1e-5f;
		if (t.elem[1] < 1e-8f) t.elem[1] = 1e-8f;
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, Params theta) const
	{
		for (int y = 0; y < Height(); y++)
		{
			for (int x = 0; x < Width(); x++)
			{
				int idx = y * Width() + x;
				T mu = theta[0] * psf[idx] + theta[1];

				const T firstOrder[] = { psf[idx], 1 };
				cb(idx, mu, firstOrder);
			}
		}
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeSecondDerivatives(TCallback cb, Params theta, const Gauss2D_PSFModel::TSampleIndex& roipos) const
	{
		for (int y = 0; y < height(); y++)
		{
			for (int x = 0; x < width(); x++)
			{
				int idx = y * width() + x;
				T mu = theta[0] * psf[idx] + theta[1];

				const T firstOrder[] = { psf[idx], 1 };
				const T secondOrder[] = { 0, 0 };
				cb(idx, mu, firstOrder, secondOrder);
			}
		}
	}
};


struct Gauss2D_Model_XYZIBg : Gauss2D_PSFModel<float, 5>
{
	enum { NumStartPos = 2 };
	static const char* ParamFormat() { return "x,y,z,I,bg"; }
	enum { NumConstants = 0 };
	typedef Gauss3D_Calibration Calibration;

	Calibration calibration;

	PLL_DEVHOST Gauss2D_Model_XYZIBg(Int2 roisize, const Calibration& calibration, const float* unused_constants=0) :
		Gauss2D_PSFModel(roisize), calibration(calibration) {}

	static PLL_DEVHOST T StopLimit(int k)
	{
		const float deltaStopLimit[] = { 1e-4f, 1e-4f, 1e-1f, 1e-4f, 1e-4f };
		return deltaStopLimit[k];
	}

	PLL_DEVHOST void CheckLimits(Params& t) const
	{
		t.elem[0] = clamp(t.elem[0], Gauss2D_Border, roisize[1] - Gauss2D_Border-1); // x
		t.elem[1] = clamp(t.elem[1], Gauss2D_Border, roisize[0] - Gauss2D_Border-1); // y
		t.elem[2] = clamp(t.elem[2], calibration.minz, calibration.maxz); // z
		t.elem[3] = fmaxf(t.elem[3], 10.f); //I
		t.elem[4] = fmaxf(t.elem[4], 0.0f); // bg
	}


	PLL_DEVHOST Params ComputeInitialEstimate(const T* sample) const
	{
		Vector3f com = ComputeCOM(sample, roisize); 
		Params estim;
		estim[0] = com[0];
		estim[1] = com[1];
		estim[2] = 0.0f;// index == 0 ? -0.3f : 0.3f; // Two different starting z
		estim[3] = com[2]; // I
		estim[4] = 0.01f; // bg
		return estim;
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, const Params& theta) const
	{
		T tx = theta[0];
		T ty = theta[1];
		T tz = theta[2];
		T tI = theta[3];
		T tbg = theta[4];

		T s0_x = calibration.x[0];
		T gamma_x = calibration.x[1];
		T d_x = calibration.x[2];
		T A_x = calibration.x[3];

		T s0_y = calibration.y[0];
		T gamma_y = calibration.y[1];
		T d_y = calibration.y[2];
		T A_y = calibration.y[3];

		T d_y2 = d_y * d_y;
		T d_x2 = d_x * d_x;

		const T tzx = tz - gamma_x; const T tzx2 = tzx * tzx; const T tzx3 = tzx2 * tzx;
		const T tzy = tz - gamma_y; const T tzy2 = tzy * tzy; const T tzy3 = tzy2 * tzy;

		const T sigma_x = s0_x * sqrt(1.0f + tzx2 / d_x2 + A_x * tzx3 / d_x2);
		const T sigma_y = s0_y * sqrt(1.0f + tzy2 / d_y2 + A_y * tzy3 / d_y2);

		const T OneOverSqrt2PiSigma_x = 1.0f / (sqrtf(2 * MATH_PI) * sigma_x);
		const T OneOverSqrt2Sigma_x = 1.0f / (sqrtf(2) * sigma_x);
		const T OneOverSqrt2PiSigma_y = 1.0f / (sqrtf(2 * MATH_PI) * sigma_y);
		const T OneOverSqrt2Sigma_y = 1.0f / (sqrtf(2) * sigma_y);

		for (int y = 0; y < Height(); y++)
		{
			T Yexp0 = (y - ty + .5f) * OneOverSqrt2Sigma_y;
			T Yexp1 = (y - ty - .5f) * OneOverSqrt2Sigma_y;
			T DeltaY = 0.5f * erf(Yexp0) - 0.5f * erf(Yexp1);
			T dEy = OneOverSqrt2PiSigma_y * (exp(-Yexp1 * Yexp1) - exp(-Yexp0 * Yexp0));
			T G21y = 1 / (sqrt(2.0f * MATH_PI) * sigma_y * sigma_y) * (
				(y - ty - 0.5f) * exp(-(y - ty - 0.5f)*(y - ty - 0.5f) / (2.0f * sigma_y * sigma_y)) -
				(y - ty + 0.5f) * exp(-(y - ty + 0.5f)*(y - ty + 0.5f) / (2.0f * sigma_y * sigma_y)));

			for (int x = 0; x < Width(); x++)
			{
				T Xexp0 = (x - tx + .5f) * OneOverSqrt2Sigma_x;
				T Xexp1 = (x - tx - .5f) * OneOverSqrt2Sigma_x;
				T DeltaX = 0.5f * erf(Xexp0) - 0.5f * erf(Xexp1);
				T dEx = OneOverSqrt2PiSigma_x * (exp(-Xexp1 * Xexp1) - exp(-Xexp0 * Xexp0));

				T mu = tbg + tI * DeltaX * DeltaY;
				T dmu_dx = tI * dEx * DeltaY;
				T dmu_dy = tI * dEy * DeltaX;

				T G21x = 1 / (sqrt(2.0f * MATH_PI) * sigma_x * sigma_x) * (
					(x - tx - 0.5f) * exp(-(x - tx - 0.5f)*(x - tx - 0.5f) / (2.0f * sigma_x * sigma_x)) -
					(x - tx + 0.5f) * exp(-(x - tx + 0.5f)*(x - tx + 0.5f) / (2.0f * sigma_x * sigma_x)));

				T dMuSigmaX = tI * DeltaY * G21x;
				T dMuSigmaY = tI * DeltaX * G21y;

				T dSigmaXThetaZ = s0_x * (2 * tzx / d_x2 + A_x * 3 * tzx2 / d_x2) /
					(2 * sqrt(1 + tzx2 / d_x2 + A_x * tzx3 / d_x2));
				T dSigmaYThetaZ = s0_y * (2 * tzy / d_y2 + A_y * 3 * tzy2 / d_y2) /
					(2 * sqrt(1 + tzy2 / d_y2 + A_y * tzy3 / d_y2));

				T dmu_dz = dMuSigmaX * dSigmaXThetaZ + dMuSigmaY * dSigmaYThetaZ;

				T dmu_dI0 = DeltaX * DeltaY;
				T dmu_dIbg = 1;
				const T jacobian[] = { dmu_dx, dmu_dy, dmu_dz, dmu_dI0, dmu_dIbg };
				cb(y*Width()+x, mu, jacobian);
			}
		}
	}
};



