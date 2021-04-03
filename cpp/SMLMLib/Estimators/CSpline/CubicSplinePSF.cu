// Cubic Spline PSF
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen, Willem Melching 2018-2021
#include "Estimators/EstimatorImpl.h"
#include "CubicSplinePSF.h"
#include "CubicSplinePSFModels.h"
#include "CameraCalibration.h"

#include <iostream>

template<typename TModel>
class cuCSplinePSF : public cuPSFImpl<TModel, CSpline_Calibration>
{
public:
	typedef cuPSFImpl<TModel, CSpline_Calibration> base;
	DeviceArray<float> splineCoefs;
	CSpline_Calibration calib;

	const CSpline_Calibration& GetCalibration()
	{
		return calib;
	}

	cuCSplinePSF(Int2 roisize, const CSpline_Calibration& c, std::vector<ParamLimit> limits, int numconst) :
		base(roisize, limits, numconst), 
		calib(c), 
		splineCoefs(c.NumCoefs(),c.coefs)
	{
		calib.coefs = splineCoefs.ptr();
	}
};


template<typename TModel>
class CSplinePSF : public PSFImpl<TModel, CSpline_Calibration>
{
public:
	typedef PSFImpl<TModel, CSpline_Calibration> base;
	std::vector<float> splineCoefs;
	CSpline_Calibration calib;

	const CSpline_Calibration& GetCalibration()
	{
		return calib; 
	}

	CSplinePSF(Int2 roisize, const CSpline_Calibration& c, std::vector<ParamLimit> limits, int numconst) :
		base(roisize, limits, numconst), calib(c), splineCoefs(c.coefs,c.coefs+c.NumCoefs())
	{
		calib.coefs = splineCoefs.data();
	}
};

CDLL_EXPORT Estimator* CSpline_CreatePSF(int roisize, const CSpline_Calibration& calib, 
	const float* coefs, int mode, bool cuda, Context* ctx)
{
	try {
		std::vector<ParamLimit> limits = {
			{CSpline_Border, roisize - 1 - CSpline_Border},
			{CSpline_Border, roisize - 1 - CSpline_Border},
			{calib.z_min, calib.z_max},
			{10.0f, 1e9f },
			{1e-6f, 1e9f },
		};

		if (mode == CSPLINE_TILTEDBG) {
			limits.push_back({ -1e6,1e6 });
			limits.push_back({ -1e6,1e6 });
		}

		CSpline_Calibration calib_ = calib;
		if (coefs) calib_.coefs = coefs;

		Estimator* psf;
		if (mode == CSPLINE_FLATBG) {
			typedef CSpline_Model_XYZIBg<float> Model;
			if (cuda)
				psf = new cuEstimatorWrapper(new cuCSplinePSF<Model>({ roisize,roisize }, calib_, limits,0));
			else
				psf = new CSplinePSF<Model>({ roisize,roisize }, calib_, limits, 0);
		}
		else if (mode == CSPLINE_TILTEDBG) {
			typedef CSpline_Model_XYZITiltedBg<float> Model;
			if (cuda)
				psf = new cuEstimatorWrapper(new cuCSplinePSF<Model>({ roisize,roisize }, calib_, limits,0));
			else
				psf = new CSplinePSF<Model>({ roisize,roisize }, calib_, limits, 0);
		}
		else if (mode == CSPLINE_BGIMAGE) {
			typedef CSpline_Model_XYZIBgImage<float> Model;
			if (cuda)
				psf = new cuEstimatorWrapper(new cuCSplinePSF<Model>({ roisize,roisize }, calib_, limits, roisize*roisize));
			else
				psf = new CSplinePSF<Model>({ roisize,roisize }, calib_, limits, roisize * roisize);
		}
		else
			throw std::runtime_error("Invalid model background mode");

		if (ctx) psf->SetContext(ctx);
		return psf;
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}

#pragma pack(push,4)
struct ZStackHeader {
	int version;
	int shape[3];
	float zmin, zmax;
	int numSplineCoefs(int border) { return (Int3(shape) - 3 + border).prod() * 64; }
};
#pragma pack(pop)

static std::pair< ZStackHeader, std::vector<float> > LoadZStack(const char* filename)
{
	ZStackHeader h;
	FILE* f = fopen(filename, "rb");
	if (!f) {
		throw std::runtime_error(SPrintf("Can't open file %s", filename).c_str());
	}
	fread(&h, sizeof(h), 1, f);
	if (h.version != 1) {
		throw std::runtime_error("Wrong ZStack file version (should be 1)");
	}
	std::vector<float> data(h.shape[0] * h.shape[1] * h.shape[2]);
	fread(data.data(), sizeof(float), data.size(), f);
	fclose(f);
	return { h, data };
}

CDLL_EXPORT Estimator* CSpline_CreatePSF_FromFile(int roisize, const char* filename, int mode, bool cuda, Context* ctx)
{
	auto zstack = LoadZStack(filename);
	int border = 0;
	std::vector<float> spline(zstack.first.numSplineCoefs(border));
	CSpline_Compute(Int3(zstack.first.shape), zstack.second.data(), spline.data(),0);
	CSpline_Calibration calib;
	calib.shape = zstack.first.shape;
	calib.z_min = zstack.first.zmin;
	calib.z_max = zstack.first.zmax;
	return CSpline_CreatePSF(roisize, calib, spline.data(), mode, cuda, ctx);
}




CDLL_EXPORT void CSpline_Compute(const Int3& shape, const float* data, float* splineCoefs, int border)
{
	double A[64 * 64];

	for (int i = 0; i < 4; i++) {
		double dx = i - 1.0;
		for (int j = 0; j < 4; j++) {
			double dy = j - 1.0;
			for (int k = 0; k < 4; k++) {
				double dz = k - 1.0;
				for (int l = 0; l < 4; l++)
					for (int m = 0; m < 4; m++)
						for (int n = 0; n < 4; n++)
							A[ 64*(i * 16 + j * 4 + k) + (l * 16 + m * 4 + n)] = powf(dx, l) * powf(dy, m) * powf(dz, n);
			}
		}
	}

	int P[65];
	LUPDecompose<double>(64, A, 1e-10f, P);

	// -3 because we go from grid to spline segments (4 points are needed to create 1 segment):
	//  0---1---2---3   ->   Spline from 1 to 2
	//  
	//  We add padding of one pixel on each side to make clamp the outer values, resulting in +2
	Int3 out(shape - 3 + border*2);

	auto getpixel = [&](int z, int y, int x) {
		z = clamp(z - border, 0, shape[0] - 1);
		y = clamp(y - border, 0, shape[1] - 1);
		x = clamp(x - border, 0, shape[2] - 1);
		return data[shape[1] * shape[2] * z + shape[2] * y + x];
	};

	ParallelFor(out[0], [&](int i) {
		double b[64];
		for (int j = 0; j < out[1]; j++) {
			for (int k = 0; k < out[2]; k++)
			{
				for (int m = 0; m < 4; m++)
					for (int n = 0; n < 4; n++)
						for (int o = 0; o < 4; o++)
							b[m * 16 + n * 4 + o] = getpixel(i+m, j+n, k+o);

				double coeff[64];
				LUPSolve<double, 64>(A, P, b, coeff);

				float* dst = &splineCoefs[64 * (out[1] * out[2] * i + out[2] * j + k)];
				for (int a = 0; a < 64; a++)
					dst[a] = (float) coeff[a];
			}
		}
	});
}

// Evaluate the 3D spline at a shifted position
// 
CDLL_EXPORT void CSpline_Evaluate(const Int3& shape, const float* splineCoefs, const Vector3f& shift, int border, float* values)
{
	float delta_f[64] = { 0 };
	float delta_dxf[64] = { 0 };
	float delta_dyf[64] = { 0 };
	float delta_dzf[64] = { 0 };

	Vector3f d = -shift - Int3(-shift); // subpixel pos

	float zzz = 1.0f;
	for (int m = 0; m <= 3; m++) {
		float yyy = 1.0f;
		for (int n = 0; n <= 3; n++) {
			float xxx = 1.0f;
			for (int o = 0; o <= 3; o++) {
				const float xyz = xxx * yyy * zzz;
				delta_f[m * 16 + n * 4 + o] = xyz;
				if (o < 3) delta_dxf[m * 16 + n * 4 + o + 1] = ((float)o + 1) * xyz;
				if (n < 3) delta_dyf[m * 16 + (n + 1) * 4 + o] = ((float)n + 1) * xyz;
				if (m < 3) delta_dzf[(m + 1) * 16 + n * 4 + o] = ((float)m + 1) * xyz;

				xxx *= d[2];
			}
			yyy *= d[1];
		}
		zzz *= d[0];
	}

	Int3 outshape = shape - 2 * border;
	for (int z = border; z < shape[0] - border; z++) {
		float zz = z - shift[0];
		int k = clamp((int)zz, 0, shape[0] - 1);

		for (int y = border; y < shape[1] - border; y++)
		{
			float yy = y - shift[1];
			int j = clamp((int)yy, 0, shape[1] - 1);
			for (int x = border; x < shape[2] - border; x++)
			{
				float xx = x - shift[2];
				int i = clamp((int)xx, 0, shape[2] - 1);

				float val = 0.0f;
				float dx = 0.0f;
				float dy = 0.0f;
				float dz = 0.0f;

				for (int idx = 0; idx < 64; idx++) {
					float coef = splineCoefs[64 * (k * shape[2] * shape[1] + j * shape[2] + i) + idx];
					val += delta_f[idx] * coef;
					dx += delta_dxf[idx] * coef;
					dy += delta_dyf[idx] * coef;
					dz += delta_dzf[idx] * coef;
				}

				int dstidx = (z - border) * outshape[1] * outshape[2] + (y - border) * outshape[2] + x - border;
				values[dstidx * 4 + 0] = val;
				values[dstidx * 4 + 1] = dx;
				values[dstidx * 4 + 2] = dy;
				values[dstidx * 4 + 3] = dz;
			}
		}
	}
}
