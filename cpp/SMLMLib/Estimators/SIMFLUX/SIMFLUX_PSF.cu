// SIMFLUX Estimator model
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "simflux/SIMFLUX.h"
#include "simflux/ExcitationModel.h"
#include "Estimators/Estimator.h"
#include "CudaUtils.h"
#include "Estimators//EstimatorImpl.h"
#include "SIMFLUX_PSF.h"
#include "simflux/SIMFLUX_Models.h"
#include "Estimators/Gaussian/GaussianPSFModels.h"

static std::vector<int> samplesize(std::vector<int> org, int numPatterns)
{
	org.insert(org.begin(), numPatterns);
	return org;
}




class cuSIMFLUXEstimator : public cuEstimator
{
public:

	struct Excitation {
		float exc;
		float deriv[3];
	};

	struct Params {
		float pos[3];
		float I, bg;
	};

	enum {
		XIndex = 0, 
		YIndex = 1, 
		ZIndex = 2, 
		IntensityIndex = 3,
		BGIndex = 4
	};

	cuEstimator* psf;
	bool simfluxFit;
	int numPatterns;

	struct DeviceBuffers {
		DeviceBuffers(int baseSmpCount, int baseSmpDims, int numspots, int K, int numPatterns) :
			summed(baseSmpCount*numspots),
			psf_ev(baseSmpCount*numspots),
			params(numspots),
			excitations(numPatterns*numspots),
			baseroipos(baseSmpDims*numspots),
			psf_deriv(baseSmpCount*numspots*K),
			numspots(numspots) {}
		DeviceArray<float> summed;
		DeviceArray<float> psf_ev;
		DeviceArray<Params> params;
		DeviceArray<int> baseroipos;
		DeviceArray<float> psf_deriv;

		DeviceArray<Excitation> excitations; // q,dqdx,dqdy,dqdz
		int numspots;
	};
	std::mutex streamDataMutex;
	std::unordered_map<cudaStream_t, DeviceBuffers> streamData;

	DeviceBuffers* GetDeviceBuffers(cudaStream_t stream, int numspots)
	{
		return LockedFunction(streamDataMutex, [&]() {
			auto it = streamData.find(stream);

			if (it != streamData.end() && it->second.numspots < numspots) {
				streamData.erase(it);
				it = streamData.end();
			}

			if (it == streamData.end())
				it = streamData.emplace(stream, DeviceBuffers(psf->SampleCount(), psf->SampleIndexDims(), numspots, psf->NumParams(), numPatterns)).first;
			return &it->second;
			});
	}

	cuSIMFLUXEstimator(cuEstimator* psf, int numPatterns, bool simfluxFit) :
		cuEstimator(samplesize(psf->SampleSize(), numPatterns), simfluxFit ? 6 * numPatterns : 0, 2 * numPatterns, psf->ParamFormat(), psf->ParamLimits()),
			psf(psf), simfluxFit(simfluxFit), numPatterns(numPatterns)
	{
		assert(psf->SampleIndexDims() == 2);
		assert(psf->NumParams() == 5);
	}

	void ComputeThetaAndExcitation(cudaStream_t stream, DeviceBuffers& db, const float* d_params, const float* d_const, const int* d_roipos, int numspots)
	{
		const int K = 5;
		// evaluate the actual Estimator
		float* d_psf_ev = db.psf_ev.data();
		Params* d_psf_params = db.params.data();
		Int2* d_psf_roipos = (Int2*)db.baseroipos.data();

		// Parameters must be made local so the 'this' pointer is not captured by CUDA side lambda
		int numPatterns = this->numPatterns;
		const auto d_mod_ = (const SIMFLUX_Modulation*)d_const;
		Excitation* d_exc = db.excitations.data();
		
		LaunchKernel(numspots, [=]__device__(int i) {
			for (int j=0;j<3;j++)
				d_psf_params[i].pos[j] = d_params[i * K + j];
			d_psf_params[i].bg = 0;
			d_psf_params[i].I = 1.0f;
			d_psf_roipos[i] = { d_roipos[i * 3 + 1],d_roipos[i * 3 + 2] };
		}, 0, stream);
		
		//DeviceArray<float3> pos(numspots);
		//float3* pos_ = pos.data();

		LaunchKernel(numPatterns, numspots, [=]__device__(int p, int i) {
			const int * roipos = &d_roipos[i * 3];
			SineWaveExcitation epModel(numPatterns, &d_mod_[i * numPatterns]);
			int e = roipos[0] + p;
			if (e > numPatterns) e -= numPatterns; // cheap % (% is expensive on cuda)
			float q;
			float xyz[3] = {
				d_params[i * K + 0] + roipos[2],
				d_params[i * K + 1] + roipos[1],
				d_params[i * K + 2]
			};
			//pos_[i] = xyz;
			float qderiv[3];
			epModel.ExcitationPattern(q, qderiv, e, xyz);
			d_exc[i * numPatterns + p] = { q, { qderiv[0],qderiv[1],qderiv[2]} };
		}, 0, stream);

		/*
		cudaStreamSynchronize(stream);
		auto chk = db.excitations.ToVector();
		for (int j = 0; j < chk.size(); j++)
		{
			DebugPrintf("%d: ", j);  PrintVector(*(Vector4f*)&chk[j]);
		}
		*/
	}

	virtual void ExpectedValue(float * expectedvalue, const float * d_params, const float * d_const, 
		const int * d_roipos, int numspots, cudaStream_t stream) override
	{
		auto& db = *GetDeviceBuffers(stream,numspots);
		ComputeThetaAndExcitation(stream, db, d_params, d_const, d_roipos, numspots);
		Params* d_psf_params = db.params.data();
		float* d_psf_ev = db.psf_ev.data();
		Int2* d_psf_spotpos = (Int2*)db.baseroipos.data();
		psf->ExpectedValue(d_psf_ev, (const float*)d_psf_params, d_const, (int*)d_psf_spotpos, numspots, stream);

		// compute excitation values and generate the SIM-ed expected values
		int numPatterns = this->numPatterns;
		int smpcount = psf->SampleCount();
		int K = NumParams();
		Excitation* d_exc = db.excitations.data();
		float bg = 1.0f;// 1.0f / numPatterns;

		LaunchKernel(numspots, numPatterns, smpcount, [=]__device__(int i, int p, int smp) {
			Excitation exc = d_exc[i*numPatterns+p];

			float psf = d_psf_ev[smpcount * i + smp];
			float intensity = d_params[i * K + IntensityIndex];
			float background = d_params[i * K + BGIndex];
			expectedvalue[smpcount*numPatterns*i + smpcount * p + smp] =
				 psf * exc.exc * intensity + background * bg;
		}, 0, stream);
	}

	virtual void Derivatives(float * d_deriv, float * d_expectedvalue, const float * d_params, 
		const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream) override
	{
		auto& sd = *GetDeviceBuffers(stream, numspots);
		ComputeThetaAndExcitation(stream, sd, d_params, d_const, d_roipos, numspots);

		Params* d_psf_theta = sd.params.data();
		float* d_psf_ev = sd.psf_ev.data();
		float* d_psf_deriv = sd.psf_deriv.data();
		Int2* d_psf_spotpos = (Int2*)sd.baseroipos.data();
		// Compute base psf psf_deriv with I=1, bg=0
		psf->Derivatives(d_psf_deriv, d_psf_ev, (const float*)d_psf_theta, d_const, (int*)d_psf_spotpos, numspots, stream);

		int numPatterns = this->numPatterns;
		int psfSmpCount = psf->SampleCount();
		int K = NumParams();
		Excitation* d_exc = sd.excitations.data();
		float exc_bg = 1.0f / numPatterns;
		int sfSmpCount = numPatterns * psfSmpCount;

		LaunchKernel(psfSmpCount, numPatterns, numspots, [=]__device__(int smp, int p, int i) {
			float* spot_deriv = &d_deriv[sfSmpCount*K *i];
			float* spot_ev = &d_expectedvalue[sfSmpCount*i];
			const float* spot_psf_deriv = &d_psf_deriv[psfSmpCount*K*i];
			auto exc = d_exc[i*numPatterns + p];

			float spotIntensity = d_params[i*K + IntensityIndex];
			float spotBg = d_params[i*K + BGIndex];
			float psf_ev = d_psf_ev[psfSmpCount*i + smp];
			spot_ev[psfSmpCount*p + smp] = exc.exc * psf_ev * spotIntensity + spotBg * exc_bg;

			for (int k = 0; k < 3; k++) {
				// d(mu,p) = d(psf,p) * I * q + d(I,p) * psf * q + d(q,p) * psf * I
				spot_deriv[sfSmpCount * k + psfSmpCount * p + smp] = spotIntensity * (spot_psf_deriv[psfSmpCount * k + smp] * exc.exc +
					psf_ev * exc.deriv[k]);
			} 
			// Intensity
			spot_deriv[sfSmpCount * 3 + psfSmpCount * p + smp] = psf_ev * exc.exc;
			// Background
			spot_deriv[sfSmpCount * 4 + psfSmpCount * p + smp] = exc_bg;
		}, 0, stream);
	}


	void Estimate(const float * d_sample, const float * d_const, const int* d_roipos, const float * d_initial,
		float * d_params, float* d_diag, int* d_iterations, int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)
	{
		DeviceBuffers* db = GetDeviceBuffers(stream, numspots);

		// Sum samples
		int orgSmpCount = psf->SampleCount();
		int numPatterns = (int)this->numPatterns;
		float* d_sum = db->summed.data();
		LaunchKernel(numspots, orgSmpCount, [=]__device__(int i, int j) {
			const float* smp = &d_sample[orgSmpCount*numPatterns*i];
			float sum = 0.0f;
			for (int k = 0; k < numPatterns; k++)
				sum += smp[orgSmpCount*k + j];
			d_sum[orgSmpCount*i + j] = sum;
		}, 0, stream);

		// Compute roipos[:,1:]
		int* baseroipos = db->baseroipos.data();
		int nsmpdims = psf->SampleIndexDims();
		LaunchKernel(numspots, [=]__device__(int i) {
			for (int j = 0; j < nsmpdims; j++)
				baseroipos[i*nsmpdims + j] = d_roipos[(1 + nsmpdims)*i + j + 1];
		}, 0, stream);

		psf->Estimate(d_sum, d_const, db->baseroipos.data(), d_initial, d_params, 0, d_iterations, numspots, d_trace, traceBufLen, stream);

		// Compute theta with I=1 and bg=0
		int K = this->NumParams();
		Params* adjParams = db->params.data();
		LaunchKernel(numspots, [=]__device__(int i) {
			for (int j=0;j<3;j++)
				adjParams[i].pos[j] = d_params[i*K + j];
			adjParams[i].I = 1.0f;
			adjParams[i].bg = 0.0f;
		}, 0, stream);

		// Compute expected values with I=1,bg=0.
		float* computed_psf = db->psf_ev.data();
		psf->ExpectedValue(computed_psf, (const float*)adjParams, d_const, baseroipos, numspots, stream);

		// Estimate intensities and backgrounds 
		Vector2f* IBg = (Vector2f*)d_diag;
		LaunchKernel(numspots, numPatterns, [=]__device__(int spot, int pat) {
			IntensityBgModel model({ orgSmpCount,1 }, &computed_psf[orgSmpCount*spot]);
			const float* smp = &d_sample[orgSmpCount*numPatterns*spot + orgSmpCount * pat];
			Vector2f initial{ 1.0f, d_params[spot*K + BGIndex] / numPatterns };
			auto r = LevMarOptimize(smp, (const float*)0, initial, model, 15);
			IBg[spot*numPatterns + pat] = r.estimate;
		}, 0, stream);

		if (simfluxFit)
		{
			//TODO: This can be optimized to directly compute derivatives from PSF derivatives and excitations
			cuEstimator::Estimate(d_sample, d_const, d_roipos, d_params, d_params, d_diag, d_iterations,
				numspots, d_trace, traceBufLen, stream);
		}
	}
};


CDLL_EXPORT Estimator* SIMFLUX_CreateEstimator(Estimator* psf, int num_patterns, int simfluxFit, Context* ctx)
{
	try {
		cuEstimator* original_cuda = psf->Unwrap();
		if (!original_cuda) return 0;

		Estimator* sf_estim = new cuEstimatorWrapper(new cuSIMFLUXEstimator(original_cuda, num_patterns, !!simfluxFit));

		if (ctx) sf_estim->SetContext(ctx);
		return sf_estim;
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}



std::vector<int> prependInt(std::vector<int> v, int a) {
	v.insert(v.begin(), a);
	return v;
}


template<typename BaseOffset>
class SampleOffset_Multiply
{
public:
	BaseOffset base;
	float factor;
	PLL_DEVHOST SampleOffset_Multiply(BaseOffset b, float factor) :base(b), factor(factor) {}

	PLL_DEVHOST float Get(Int2 samplepos, Int2 roipos) const {
		return base.Get(samplepos, roipos)*factor;
	}
	PLL_DEVHOST float Get(Int3 samplepos, Int3 roipos) const {
		return base.Get(samplepos, roipos)*factor;
	}
};

class SIMFLUX_Gauss2D_CUDA_PSF : public cuEstimator 
{
public:
	typedef Gauss2d_params Params;
	typedef Int3 SampleIndex;
	typedef SIMFLUX_Calibration TCalibration;
	typedef SIMFLUX_Gauss2D_Model TModel;

	bool simfluxFit; // If false, only per-pattern intensities/backgrounds are estimated and a Gaussian fit is done on the summed frames
	float2 sigma;
	int numframes, roisize;
	int numPatterns;

	struct Buffers {
		Buffers(int psfsmpcount, int numspots) :
			summed(psfsmpcount*numspots),
			numspots(numspots) {}
		DeviceArray<float> summed;
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
				it = streamData.emplace(stream, Buffers(roisize*roisize, numspots)).first;
			return &it->second;
		});
	}

	SIMFLUX_Gauss2D_CUDA_PSF(int roisize, int numframes, int numPatterns, float2 sigma, bool simfluxFit) :
		cuEstimator({ numframes,roisize,roisize },  simfluxFit ? 6 * numPatterns : 0, numframes * 4, Gauss2D_Model_XYIBg::ParamFormat(), 
			{
				{Gauss2D_Border, roisize - 1 - Gauss2D_Border },
				{Gauss2D_Border, roisize - 1 - Gauss2D_Border },
				{0.01f, 1e9f },
				{1e-6f, 1e9f },
			}),
		roisize(roisize), 
		numframes(numframes), 
		simfluxFit(simfluxFit), 
		sigma(sigma), 
		numPatterns(numPatterns)
	{}

	void ExpectedValue(float* d_image, const float* d_params, const float *d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
	{
		const Params* theta = (const Params*)d_params;
		float2 sigma = this->sigma;
		int roisize = this->roisize;
		int numframes = this->numframes;
		int sc = SampleCount();
		const SIMFLUX_Modulation* mod = (const SIMFLUX_Modulation*)d_const;
		int numPatterns = this->numPatterns;
		int numconst = this->NumConstants();
		auto roipos = (const typename TModel::TSampleIndex *)d_roipos;
		if (simfluxFit) {
			LaunchKernel(numspots, [=]__device__(int i) {
				SineWaveExcitation exc (numPatterns, &mod[numPatterns*i]);
				SIMFLUX_Calibration calib = { exc,sigma };
				TModel model(roisize, calib, 0, numframes, numframes, roipos[i]);
				ComputeExpectedValue(theta[i], model, &d_image[sc*i]);
			}, 0, stream);
		}
		else {
			assert(0);
		}
	}

	void Derivatives(float* d_deriv, float *d_expectedvalue, const float* d_params, const float *d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
	{
		const Params* theta = (const Params*)d_params;
		const SIMFLUX_Modulation* mod = (const SIMFLUX_Modulation*)d_const;
		float2 sigma = this->sigma;
		int numPatterns = this->numPatterns;

		int smpcount = SampleCount();
		int roisize = this->roisize;
		int numframes = this->numframes;
		int numconst = this->NumConstants();
		int K = NumParams();
		auto roipos = (const typename TModel::TSampleIndex *)d_roipos;
		if (simfluxFit) {
			LaunchKernel(numspots, [=]__device__(int i) {
				SineWaveExcitation exc(numPatterns, &mod[numPatterns * i]);
				SIMFLUX_Calibration calib = { exc,sigma };
				TModel model(roisize, calib, 0, numframes, numframes, roipos[i]);
				ComputeDerivatives(theta[i], model, &d_deriv[i*smpcount*K], &d_expectedvalue[i*smpcount]);
			}, 0, stream);
		}
		else {
			assert(0);
		}
	}

	// d_sample[numspots, SampleCount()], d_params[numspots, numparams], d_initial[numspots, NumParams()], d_params[numspots, NumParams()], d_iterations[numspots]
	void Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_params, float* d_diag,
		int *iterations,int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)  override
	{
		Buffers* db = GetBuffers(stream, numspots);

		int numPatterns = this->numPatterns;
		Params* theta = (Params*)d_params;
		Params* trace = (Params*)d_trace;
		const SIMFLUX_Modulation* mod = (const SIMFLUX_Modulation*)d_const;
		float2 sigma = this->sigma;
		int smpcount = SampleCount();
		int roisize = this->roisize;
		int numframes = this->numframes;
		auto roipos = (const typename TModel::TSampleIndex *)d_roipos;
		LMParams lmParams = this->lmParams;
	
		float* d_sums = db->summed.data();

		LaunchKernel(numspots, PLL_FN(int i) {
			Gauss2D_Model_XYIBg model({ roisize,roisize }, sigma);

			const float* smp = &d_sample[i*smpcount];

			float* smp_sum = &d_sums[i*roisize*roisize];
			ComputeImageSums(smp, smp_sum, roisize, roisize, numframes);
			Int2 roiposYX{ roipos[i][1], roipos[i][2] };

			Params estim;
			if (d_initial)
			{
				estim = ((Params*)d_initial)[i];
				iterations[i] = 0;
			}
			else{
				//auto com = model.ComputeInitialEstimate(smp_sum);
				auto com = ComputePhasorEstim(smp_sum, roisize, roisize);
				auto initialValue = Vector4f{ com[0],com[1],com[2] * 0.9f, 0.0f };

				auto r = LevMarOptimize((const float*)smp_sum, (const float*)0, initialValue, model, lmParams.iterations,
					&trace[traceBufLen * i], traceBufLen, lmParams.lambda);

				estim = r.estimate;
				estim[3] /= numframes;
				iterations[i] = r.iterations;
			}

			float* psf = smp_sum; // don't need the summed frames anymore at this point.
			IntensityBgModel::ComputeGaussianPSF(sigma, estim[0], estim[1], roisize, psf);
			theta[i] = estim;

			float* spot_diag = &d_diag[i*numframes * 4];
			for (int j = 0; j < numframes; j++) {
				IntensityBgModel ibg_model({ roisize,roisize }, psf);
				auto ibg_r = LevMarOptimize(&smp[roisize*roisize*j], (const float*)0, { 1.0f, 0.0f }, ibg_model, 20);
				auto ibg_crlb = ComputeCRLB(ComputeFisherMatrix(ibg_model, ibg_r.estimate));
				spot_diag[j * 4 + 0] = ibg_r.estimate[0];
				spot_diag[j * 4 + 1] = ibg_r.estimate[1];
				spot_diag[j * 4 + 2] = ibg_crlb[0];
				spot_diag[j * 4 + 3] = ibg_crlb[1];
			}

		},0,stream);

		if (simfluxFit)
		{
			LaunchKernel(numspots, [=]__device__(int i) {
				SineWaveExcitation exc(numPatterns, &mod[numPatterns * i]);
				SIMFLUX_Calibration calib = { exc,sigma };
				TModel model(roisize, calib, 0, numframes, numframes, roipos[i]);
				const float* smp = &d_sample[i*smpcount];

				auto r = LevMarOptimize(smp, (const float*)0, theta[i], model, lmParams.iterations,
					&trace[traceBufLen*i], traceBufLen, lmParams.lambda);
				theta[i] = r.estimate;
				iterations[i] = r.iterations;
			}, 0, stream);
		}
	}
};

CDLL_EXPORT Estimator* SIMFLUX_Gauss2D_CreateEstimator(int num_patterns, float sigmaX ,float sigmaY,
	int roisize, int numframes, bool simfluxFit,  Context* ctx)
{
	try {
		cuEstimator* cpsf;
		float2 sigma{ sigmaX,sigmaY };

		cpsf = new SIMFLUX_Gauss2D_CUDA_PSF(roisize, numframes,
			num_patterns, sigma, simfluxFit);

		Estimator* psf = new cuEstimatorWrapper(cpsf);
		if (ctx) psf->SetContext(ctx);
		return psf;
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}


