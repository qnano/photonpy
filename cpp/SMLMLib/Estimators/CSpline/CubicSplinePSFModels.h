// Cubic Spline PSF
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen, Willem Melching 2018-2021
#pragma once

#include "Vector.h"
#include "Estimators/Estimation.h"
#include "CubicSplinePSF.h"

#define CSpline_Border (1.5f)


template<typename TNumber>
class CSpline_Model_XYZIBg
{
public:
	static const char* ParamFormat() { return "x,y,z,I,bg"; }
	typedef typename TNumber T;
	typedef Vector5f Params;
	typedef CSpline_Calibration Calibration;
	typedef Int2 TSampleIndex;
	enum { K = Params::K };
	enum { NumStartPos = 1 };
	enum { XDeg = 3, YDeg = 3, ZDeg = 3 };


	int imgw;
	Calibration calibration;

	PLL_DEVHOST CSpline_Model_XYZIBg(Int2 roisize, const Calibration& calib, const float* unused_constants=0)
		: calibration(calib), imgw(roisize[0]) {}

	PLL_DEVHOST float StopLimit(int k) const
	{
		const float deltaStopLimit[]={ 1e-4f, 1e-4f, 1e-7f, 1e-4f, 1e-4f };
		return deltaStopLimit[k];
	}
	
	PLL_DEVHOST int SampleCount() const {
		return imgw * imgw;
	}

	PLL_DEVHOST void CheckLimits(Params& t) const {
		t.elem[0] = clamp(t.elem[0], CSpline_Border, imgw - 1.0f - CSpline_Border); // x
		t.elem[1] = clamp(t.elem[1], CSpline_Border, imgw - 1.0f - CSpline_Border); // y
		t.elem[2] = clamp(t.elem[2], calibration.z_min + 1e-3f, calibration.z_max - 1e-3f); // z
		t.elem[3] = fmaxf(t.elem[3], 10.f); //I
		t.elem[4] = fmaxf(t.elem[4], 0.0f); // bg
	}

	PLL_DEVHOST Params ComputeInitialEstimate(const T* sample) const
	{
		Vector3f com = ComputeCOM(sample, { imgw,imgw });
		Params estim;
		estim[0] = com[0];
		estim[1] = com[1];
		estim[2] = 0.0f;
		estim[3] = com[2];
		estim[4] = 0.0f;
		return estim;
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, Params theta) const
	{
		//const int coeffPerVoxel = (XDeg + 1) * (YDeg + 1) * (ZDeg + 1);
		const float du = (calibration.z_max - calibration.z_min) / (calibration.shape[0]);
		const int Nx = calibration.shape[2];
		const int Ny = calibration.shape[1];
		const int Nz = calibration.shape[0];

		T tx = theta[0];
		T ty = theta[1];
		T tz = theta[2];
		T tI = theta[3];
		T tbg = theta[4];

		tx -= calibration.shape[2] * 0.5f;// = tx - (float)imgw / 2.0f + ((float)imgw - (float)calibration.shape[2] + 1) / 2.0f;
		ty -= calibration.shape[1] * 0.5f;// = ty - (float)imgw / 2.0f + ((float)imgw - (float)calibration.shape[1] + 1) / 2.0f;
		//tx += 0.5f;
		//ty += 0.5f;

		float delta_f[64] = { 0 };
		float delta_dxf[64] = { 0 };
		float delta_dyf[64] = { 0 };
		float delta_dzf[64] = { 0 };

		float spx = -tx; // subpixel x
		spx -= int(spx);
		float spy = -ty;
		spy -= int(spy);

		float spz = (tz - calibration.z_min) / du;
		int k = clamp((int)spz, 0, Nz - 1);
		spz = clamp(spz - k, 0.0f, 1.0f);


		// Note that coefficients are precomputed and not clamped as the indexes i,j,k are. 
		// This means in order to do proper clamping a one pixel border has to be added to the actual image stack.
		float zzz = 1.0f;
		for (int m = 0; m <= ZDeg; m++) {
			float yyy = 1.0f;
			for (int n = 0; n <= YDeg; n++) {
				float xxx = 1.0f;
				for (int o = 0; o <= ZDeg; o++) {
					const float xyz =  xxx * yyy * zzz;
					delta_f[m * 16 + n * 4 + o] = xyz;
					if (o < XDeg) delta_dxf[m * 16 + n * 4 + o + 1] =  ((float)o + 1) * xyz;
					if (n < YDeg) delta_dyf[m * 16 + (n + 1) * 4 + o] = ((float)n + 1) * xyz;
					if (m < ZDeg) delta_dzf[(m + 1) * 16 + n * 4 + o] = ((float)m + 1) * xyz;

					xxx *= spx;
				}
				yyy *= spy;
			}
			zzz *= spz;
		}

		for (int y = 0; y < imgw; y++)
		{
			float yy = y - ty;
			int j = clamp((int)yy, 0, Ny-1);

			for (int x = 0; x < imgw; x++)
			{
				float xx = x - tx;
				int i = clamp((int)xx, 0, Nx-1);

				float psf = 0.0f;
				float dx = 0.0f;
				float dy = 0.0f;
				float dz = 0.0f;

				for (int idx = 0; idx < 64; idx++) {
					float coef = calibration.coefs[64 * ( k * Nx * Ny + j * Nx + i) + idx];
					psf += delta_f[idx] * coef;
					dx += delta_dxf[idx] * coef;
					dy += delta_dyf[idx] * coef;
					dz += delta_dzf[idx] * coef; 
				}

				dx *= -tI;
				dy *= -tI;
				dz *= tI / du;

				if (psf < 0.0f) psf = 0.0f;
				float mu = tI * psf + tbg;
				float jacobian[5] = {dx, dy, dz, psf, 1 };
				cb(y * imgw + x, mu, jacobian);
			}
		}
	}
};


template<typename TNumber>
class CSpline_Model_XYZITiltedBg
{
public:
	static const char* ParamFormat() { return "x,y,z,I,bg,bgx,bgy"; }
	typedef typename TNumber T;
	typedef Vector<float,7> Params;
	typedef CSpline_Calibration Calibration;
	typedef Int2 TSampleIndex;
	enum { K = Params::K };
	enum { NumStartPos = 1 };
	enum { XDeg = 3, YDeg = 3, ZDeg = 3 };

	int imgw;
	Calibration calibration;

	PLL_DEVHOST CSpline_Model_XYZITiltedBg(Int2 roisize, const Calibration& calib, const float* unused_constants = 0)
		: calibration(calib), imgw(roisize[0]) {}

	PLL_DEVHOST float StopLimit(int k) const
	{
		const float deltaStopLimit[] = { 1e-4f, 1e-4f, 1e-7f, 1e-4f, 1e-4f,1e-4f,1e-4f };
		return deltaStopLimit[k];
	}

	PLL_DEVHOST int SampleCount() const {
		return imgw * imgw;
	}

	PLL_DEVHOST void CheckLimits(Params& t) const {
		t.elem[0] = clamp(t.elem[0], CSpline_Border, imgw - 1.0f - CSpline_Border); // x
		t.elem[1] = clamp(t.elem[1], CSpline_Border, imgw - 1.0f - CSpline_Border); // y
		t.elem[2] = clamp(t.elem[2], calibration.z_min + 1e-3f, calibration.z_max - 1e-3f); // z
		t.elem[3] = fmaxf(t.elem[3], 10.f); //I
		t.elem[4] = fmaxf(t.elem[4], 0.0f); // bg
	}

	PLL_DEVHOST Params ComputeInitialEstimate(const T* sample) const
	{
		Vector3f com = ComputeCOM(sample, { imgw,imgw });
		Params estim;
		estim[0] = com[0];
		estim[1] = com[1];
		estim[2] = 0.0f;
		estim[3] = com[2];
		estim[4] = 0.0f;
		estim[5] = 0.0f;
		estim[6] = 0.0f;
		return estim;
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, Params theta) const
	{
		//const int coeffPerVoxel = (XDeg + 1) * (YDeg + 1) * (ZDeg + 1);
		const float du = (calibration.z_max - calibration.z_min) / (calibration.shape[0]);
		const int Nx = calibration.shape[2];
		const int Ny = calibration.shape[1];
		const int Nz = calibration.shape[0];

		T tx = theta[0];
		T ty = theta[1];
		T tz = theta[2];
		T tI = theta[3];
		T tbg = theta[4], bgx = theta[5], bgy=theta[6];

		tx -= calibration.shape[2] * 0.5f;// = tx - (float)imgw / 2.0f + ((float)imgw - (float)calibration.shape[2] + 1) / 2.0f;
		ty -= calibration.shape[1] * 0.5f;// = ty - (float)imgw / 2.0f + ((float)imgw - (float)calibration.shape[1] + 1) / 2.0f;
		//tx += 0.5f;
		//ty += 0.5f;

		float delta_f[64] = { 0 };
		float delta_dxf[64] = { 0 };
		float delta_dyf[64] = { 0 };
		float delta_dzf[64] = { 0 };

		float spx = -tx; // subpixel x
		spx -= int(spx);
		float spy = -ty;
		spy -= int(spy);

		float spz = (tz - calibration.z_min) / du;
		int k = clamp((int)spz, 0, Nz - 1);
		spz = clamp(spz - k, 0.0f, 1.0f);

		float halfw = imgw * 0.5f;

		// Note that coefficients are precomputed and not clamped as the indexes i,j,k are. 
		// This means in order to do proper clamping a one pixel border has to be added to the actual image stack.
		float zzz = 1.0f;
		for (int m = 0; m <= ZDeg; m++) {
			float yyy = 1.0f;
			for (int n = 0; n <= YDeg; n++) {
				float xxx = 1.0f;
				for (int o = 0; o <= ZDeg; o++) {
					const float xyz = xxx * yyy * zzz;
					delta_f[m * 16 + n * 4 + o] = xyz;
					if (o < XDeg) delta_dxf[m * 16 + n * 4 + o + 1] = ((float)o + 1) * xyz;
					if (n < YDeg) delta_dyf[m * 16 + (n + 1) * 4 + o] = ((float)n + 1) * xyz;
					if (m < ZDeg) delta_dzf[(m + 1) * 16 + n * 4 + o] = ((float)m + 1) * xyz;

					xxx *= spx;
				}
				yyy *= spy;
			}
			zzz *= spz;
		}

		for (int y = 0; y < imgw; y++)
		{
			float yy = y - ty;
			int j = clamp((int)yy, 0, Ny - 1);

			for (int x = 0; x < imgw; x++)
			{
				float xx = x - tx;
				int i = clamp((int)xx, 0, Nx - 1);

				float psf = 0.0f;
				float dx = 0.0f;
				float dy = 0.0f;
				float dz = 0.0f;

				for (int idx = 0; idx < 64; idx++) {
					float coef = calibration.coefs[64 * (k * Nx * Ny + j * Nx + i) + idx];
					psf += delta_f[idx] * coef;
					dx += delta_dxf[idx] * coef;
					dy += delta_dyf[idx] * coef;
					dz += delta_dzf[idx] * coef;
				}

				dx *= -tI;
				dy *= -tI;
				dz *= tI / du;

				if (psf < 0.0f) psf = 0.0f;
				float mu = tI * psf + tbg + bgx * (x - halfw) + bgy * (y - halfw);
				float jacobian[7] = { dx, dy, dz, psf, 1, x-halfw, y-halfw };
				cb(y * imgw + x, mu, jacobian);
			}
		}
	}
};


template<typename TNumber>
class CSpline_Model_XYZIBgImage
{
public:
	static const char* ParamFormat() { return "x,y,z,I,bg"; }
	typedef typename TNumber T;
	typedef Vector5f Params;
	typedef CSpline_Calibration Calibration;
	typedef Int2 TSampleIndex;
	enum { K = Params::K };
	enum { NumStartPos = 1 };
	enum { XDeg = 3, YDeg = 3, ZDeg = 3 };

	int imgw;
	Calibration calibration;
	const float* bgimage;

	PLL_DEVHOST CSpline_Model_XYZIBgImage(Int2 roisize, const Calibration& calib, const float* bgimage = 0)
		: calibration(calib), imgw(roisize[0]), bgimage(bgimage) {}

	PLL_DEVHOST float StopLimit(int k) const
	{
		const float deltaStopLimit[] = { 1e-4f, 1e-4f, 1e-7f, 1e-4f, 1e-4f };
		return deltaStopLimit[k];
	}

	PLL_DEVHOST int SampleCount() const {
		return imgw * imgw;
	}

	PLL_DEVHOST void CheckLimits(Params& t) const {
		t.elem[0] = clamp(t.elem[0], CSpline_Border, imgw - 1.0f - CSpline_Border); // x
		t.elem[1] = clamp(t.elem[1], CSpline_Border, imgw - 1.0f - CSpline_Border); // y
		t.elem[2] = clamp(t.elem[2], calibration.z_min + 1e-3f, calibration.z_max - 1e-3f); // z
		t.elem[3] = fmaxf(t.elem[3], 10.f); //I
		t.elem[4] = fmaxf(t.elem[4], 0.0f); // bg
	}

	PLL_DEVHOST Params ComputeInitialEstimate(const T* sample) const
	{
		Vector3f com = ComputeCOM(sample, { imgw,imgw });
		Params estim;
		estim[0] = com[0];
		estim[1] = com[1];
		estim[2] = 0.0f;
		estim[3] = com[2];
		estim[4] = 0.0f;
		return estim;
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, Params theta) const
	{
		//const int coeffPerVoxel = (XDeg + 1) * (YDeg + 1) * (ZDeg + 1);
		const float du = (calibration.z_max - calibration.z_min) / (calibration.shape[0]);
		const int Nx = calibration.shape[2];
		const int Ny = calibration.shape[1];
		const int Nz = calibration.shape[0];

		T tx = theta[0];
		T ty = theta[1];
		T tz = theta[2];
		T tI = theta[3];
		T tbg = theta[4];

		tx = tx - (float)imgw / 2.0f + ((float)imgw - (float)calibration.shape[2] + 1) / 2.0f;
		ty = ty - (float)imgw / 2.0f + ((float)imgw - (float)calibration.shape[1] + 1) / 2.0f;
		tx += 0.5f;
		ty += 0.5f;

		float delta_f[64] = { 0 };
		float delta_dxf[64] = { 0 };
		float delta_dyf[64] = { 0 };
		float delta_dzf[64] = { 0 };

		float spx = -tx; // subpixel x
		spx -= int(spx);
		float spy = -ty;
		spy -= int(spy);

		float spz = (tz - calibration.z_min) / du;
		int k = clamp((int)spz, 0, Nz - 1);
		spz = clamp(spz - k, 0.0f, 1.0f);

		// Note that coefficients are precomputed and not clamped as the indexes i,j,k are. 
		// This means in order to do proper clamping a one pixel border has to be added to the actual image stack.
		float zzz = 1.0f;
		for (int m = 0; m <= ZDeg; m++) {
			float yyy = 1.0f;
			for (int n = 0; n <= YDeg; n++) {
				float xxx = 1.0f;
				for (int o = 0; o <= ZDeg; o++) {
					const float xyz = xxx * yyy * zzz;
					delta_f[m * 16 + n * 4 + o] = xyz;
					if (o < XDeg) delta_dxf[m * 16 + n * 4 + o + 1] = ((float)o + 1) * xyz;
					if (n < YDeg) delta_dyf[m * 16 + (n + 1) * 4 + o] = ((float)n + 1) * xyz;
					if (m < ZDeg) delta_dzf[(m + 1) * 16 + n * 4 + o] = ((float)m + 1) * xyz;

					xxx *= spx;
				}
				yyy *= spy;
			}
			zzz *= spz;
		}

		for (int y = 0; y < imgw; y++)
		{
			float yy = y - ty;
			int j = clamp((int)yy, 0, Ny - 1);

			for (int x = 0; x < imgw; x++)
			{
				float xx = x - tx;
				int i = clamp((int)xx, 0, Nx - 1);

				float psf = 0.0f;
				float dx = 0.0f;
				float dy = 0.0f;
				float dz = 0.0f;

				for (int idx = 0; idx < 64; idx++) {
					float coef = calibration.coefs[64 * (k * Nx * Ny + j * Nx + i) + idx];
					psf += delta_f[idx] * coef;
					dx += delta_dxf[idx] * coef;
					dy += delta_dyf[idx] * coef;
					dz += delta_dzf[idx] * coef;
				}

				dx *= -tI;
				dy *= -tI;
				dz *= tI / du;

				float bg = bgimage[y * imgw + x];
				if (psf < 0.0f) psf = 0.0f;
				float mu = tI * psf + tbg * bg;
				float jacobian[5] = { dx, dy, dz, psf, bg };
				cb(y * imgw + x, mu, jacobian);
			}
		}
	
	}
};
