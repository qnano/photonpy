// Estimator wrapper class that also performs Generalized Likelihood Ratio Test
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "Estimators/Estimator.h"
#include "ThreadUtils.h"
#include "Estimation.h"
#include "Estimators/Gaussian/GaussianPSFModels.h"
#include "CameraCalibration.h"

// A model that only fits background. 
struct BgModel : public Gauss2D_PSFModel<float, 1>
{
	PLL_DEVHOST T StopLimit(int k) const
	{
		return 1e-4f;
	}

	PLL_DEVHOST BgModel(Int2 roisize) : Gauss2D_PSFModel(roisize) {}

	PLL_DEVHOST void CheckLimits(Params& t) const
	{
		if (t.elem[0] < 1e-8f) t.elem[0] = 1e-8f;
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeDerivatives(TCallback cb, Params theta) const
	{
		for (int y = 0; y < Height(); y++)
		{
			for (int x = 0; x < Width(); x++)
			{
				const T firstOrder[] = { 1 };
				cb(y * Width() + x, theta[0], firstOrder);
			}
		}
	}

	template<typename TCallback>
	PLL_DEVHOST void ComputeSecondDerivatives(TCallback cb, Params theta) const
	{
		for (int y = 0; y < Height(); y++)
		{
			for (int x = 0; x < Width(); x++)
			{
				// mu=bg
				// dmu/dbg = 1
				// 
				const T firstOrder[] = { 1 };
				const T secondOrder[] = { 0 };
				cb(y * Width() + x, theta[0], firstOrder, secondOrder);
			}
		}
	}
};

// Computes likelihoods of model and background-only, and stores them in the diagnostics output array
class GLRT_PSF : public cuEstimator {
public:
	GLRT_PSF(cuEstimator* org) : 
		cuEstimator(org->SampleSize(), org->NumConstants(), 3, org->ParamFormat(), org->ParamLimits()),
		psf(org)
	{}

	struct Buffers {
		Buffers(int psfsmpcount, int numspots) :
			expval(psfsmpcount*numspots),
			numspots(numspots) {}
		DeviceArray<float> expval;
		int numspots;
	};
	std::mutex streamDataMutex;
	std::unordered_map<cudaStream_t, Buffers> streamData;

	Buffers* GetBuffers(cudaStream_t stream, int numspots)
	{
		return LockedFunction(streamDataMutex, [&]() {
			auto it = streamData.find(stream);

			if (it != streamData.end() && it->second.numspots < numspots) {
				streamData.erase(it);
				it = streamData.end();
			}

			if (it == streamData.end())
				it = streamData.emplace(stream, Buffers(SampleCount(), numspots)).first;
			return &it->second;
		});
	}

	void Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_params, 
		float* d_diag, int* d_iterations, int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)  override
	{
		psf->Estimate(d_sample, d_const, d_roipos, d_initial, d_params, d_diag, d_iterations, numspots, d_trace, traceBufLen, stream);
		Buffers* b = GetBuffers(stream, numspots);
		Vector3f* llbg = (Vector3f*)d_diag;

		psf->ExpectedValue(b->expval.data(), d_params, d_const, d_roipos, numspots, stream);
		const float* expval = b->expval.data();

		int roisizeX = SampleSize()[1], roisizeY = SampleSize()[0];
		int smpcount = SampleCount();
		LaunchKernel(numspots, [=]__device__(int i) {
			const float* ev = &expval[smpcount*i];
			const float* smp = &d_sample[i*smpcount];
			float mean = 0.0f;
			for (int i = 0; i < roisizeY * roisizeX; i++)
				mean  += d_sample[i];//-readnoise[i]; // verify this with readnoise?
			mean /= roisizeY * roisizeX;

			float ll_on = 0.0f, ll_off=0.0f;
			for (int y = 0; y < roisizeY; y++)
			{
				for (int x = 0; x < roisizeX; x++) {
					float mu = ev[y*roisizeX + x];
					float readnoise = 0.0f;//smpofs.Get({ y,x }, roipos[i]);
					mu += readnoise;
					mu = fmax(mu, 1e-8f);
					float d = smp[y*roisizeX + x] + readnoise;
					d = fmax(d, 1e-8f);
					ll_on += d * log(mu) - mu;
					ll_off += d * log(mean) - mean;
				}
			}

			llbg[i][0] = ll_on;
			llbg[i][1] = ll_off;
			llbg[i][2] = mean;
		}, 0, stream);
	}

	// Uses the provided model
	virtual void ChiSquareAndCRLB(const float* d_params, const float* sample, const float* d_const,
		const int* d_roipos, float* crlb, float* chisq, int numspots, cudaStream_t stream)
	{
		psf->ChiSquareAndCRLB(d_params, sample, d_const, d_roipos, crlb, chisq, numspots, stream);
	}
	virtual void ExpectedValue(float * expectedvalue, const float * d_params, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream)
	{
		psf->ExpectedValue(expectedvalue, d_params, d_const, d_roipos, numspots, stream);
	}
	virtual void Derivatives(float * deriv, float * expectedvalue, const float * theta, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream)
	{
		psf->Derivatives(deriv, expectedvalue, theta, d_const, d_roipos, numspots, stream);
	}
protected:
	cuEstimator* psf;
};







CDLL_EXPORT Estimator * GLRT_CreatePSF(Estimator* model, Context* ctx)
{
	try {
		cuEstimator* psf;
		psf = new GLRT_PSF(model->Unwrap());
		if (ctx) psf->SetContext(ctx);
		return Estimator_WrapCUDA(psf);
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}

}

