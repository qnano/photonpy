// Estimator CUDA wrapper
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include <streambuf>
#include "Estimator.h"
#include "Estimation.h"
#include "ContainerUtils.h"
#include "ThreadUtils.h"


template<typename T, int K>
std::vector<int> makevector(Vector<T, K> v) {
	std::vector<int> r(K);
	for (int i = 0; i < K; i++)
		r[i] = v[i];
	return r;
}

template<typename TModel, typename TCalibration>
class cuPSFImpl : public cuEstimator
{
public:
	typedef typename TCalibration Calibration;
	typedef TModel Model;
	typedef typename Model::Params Params;
	typedef typename TModel::TSampleIndex SampleIndex;

	SampleIndex roisize;

	cuPSFImpl(SampleIndex roisize, std::vector<ParamLimit> limits, int numconst) :
		cuEstimator( makevector(roisize), numconst, 0, TModel::ParamFormat(), limits),
		roisize(roisize)
	{}

	virtual const TCalibration& GetCalibration() = 0;
	/*
	void ChiSquareAndCRLB(const float* d_params, const float* sample, const float* d_const,
		const int* d_roipos, float* crlb, float* chisq, int numspots, cudaStream_t stream)
	{
		const Params* params = (const Params*)d_params;
		TCalibration calib = GetCalibration();
		SampleIndex roisize=this->roisize;
		auto* crlb_ = (typename Params*)crlb;
		int numconst = this->NumConstants(), smpcount=SampleCount();
		LaunchKernel(numspots, [=]__device__(int i) {
			TModel model(roisize, calib, &d_const[i * numconst]);
			const float* smp = sample ? &sample[smpcount * i] : 0;
			auto r = ComputeChiSquareAndCRLB(model, params[i], smp);

			crlb_[i] = r.crlb;
			chisq[i] = r.chisq;
		}, 0, stream);
	}*/

	void ExpectedValue(float* d_image, const float* d_params, const float *d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
	{
		const Params* params = (const Params*)d_params;
		TCalibration calib = GetCalibration();
		auto roisize = this->roisize;
		int sc = SampleCount();
		int numconst = this->NumConstants();
		LaunchKernel(numspots, [=]__device__(int i) {
			TModel model(roisize, calib, &d_const[i * numconst]);
			ComputeExpectedValue(params[i], model, &d_image[sc * i]);
		}, 0, stream);
	}

	void Derivatives(float* d_deriv, float *d_expectedvalue, const float* d_params, const float *d_const, const int* d_roipos, int numspots, cudaStream_t stream) override
	{
		const Params* params = (const Params*)d_params;
		TCalibration calib = GetCalibration();
		int smpcount = SampleCount();
		auto roisize = this->roisize;
		int numconst = this->NumConstants();
		int K = NumParams();
		LaunchKernel(numspots,  [=]__device__(int i) {
			TModel model(roisize, calib, &d_const[i * numconst]);
			ComputeDerivatives(params[i], model, &d_deriv[i * smpcount * K], &d_expectedvalue[i * smpcount]);
		}, 0, stream);
	}
	
	// d_sample[numspots, SampleCount()], d_params[numspots, numparams], d_initial[numspots, NumParams()], d_params[numspots, NumParams()], d_iterations[numspots]
	void Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_params, float* d_diag, int* d_iterations,
		int numspots, float * d_trace, int traceBufLen, cudaStream_t stream)  override
	{
		Params* params = (Params*)d_params;
		Params* trace = (Params*)d_trace;
		const Params* initial = (const Params*)d_initial;
		TCalibration calib = GetCalibration();
		int smpcount = SampleCount();
		auto roisize = this->roisize;
		int numconst = this->NumConstants();
		LMParams lmp = lmParams;
		LaunchKernel(numspots, [=]__device__(int i) {
			TModel model(roisize, calib, &d_const[i*numconst]);
			Params initial_ = initial ? initial[i] : model.ComputeInitialEstimate(&d_sample[i * smpcount]);
			auto r = LevMarOptimize(&d_sample[i * smpcount], (const float*)0, initial_, model, lmp.iterations,
				&trace[traceBufLen * i], traceBufLen, lmp.lambda);
			if(d_iterations) d_iterations[i] = r.iterations;
			params[i] = r.estimate;
		}, 0, stream);
	}
};

template<typename TModel, typename TCalibration>
class PSFImpl : public Estimator
{
public:
	typedef typename TCalibration Calibration;
	typedef typename TModel Model;
	typedef typename TModel::TSampleIndex SampleIndex;

	SampleIndex roisize;
	typedef typename Model::Params Params;

	PSFImpl(SampleIndex roisize, std::vector<ParamLimit> limits, int numconst) :
		Estimator(makevector(roisize), numconst, 1, TModel::ParamFormat(), limits), roisize(roisize)
	{}

	virtual const TCalibration& GetCalibration() = 0;

	// Inherited via Estimator
	void ChiSquareAndCRLB(const float* h_params, const float* sample, const float* h_const,
		const int* h_roipos, float* h_crlb, float* chisq, int numspots)
	{
		const Params* params_ = (const Params*)h_params;
		const TCalibration& calib = GetCalibration();
		auto* crlb_ = (typename Params*)h_crlb;
		int smpcount = SampleCount();
		int numconst = NumConstants();
		ParallelFor(numspots, [&](int i) {
			TModel model(roisize, calib, &h_const[i * numconst]);
			const float* smp = sample ? &sample[smpcount * i] : 0;
			auto r = ComputeChiSquareAndCRLB(model, params_[i], smp);

			crlb_[i] = r.crlb;
			chisq[i] = r.chisq;
		});
	}

	virtual void ExpectedValue(float * expectedvalue, const float * params, const float * h_const, const int* spot_pos, int numspots)  override
	{
		const Params* params_ = (const Params*)params;
		const TCalibration& calib = GetCalibration();
		int numconst = NumConstants();

		ParallelFor(numspots, [&](int i) {
			TModel model(roisize, calib, &h_const[i*numconst]);
			ComputeExpectedValue(params_[i], model, &expectedvalue[model.SampleCount() * i]);
		});
	}
	virtual void Derivatives(float * deriv, float * expectedvalue, const float * params, const float *h_const, const int* spot_pos, int numspots) override
	{
		const Params* params_ = (const Params*)params;
		const TCalibration& calib = GetCalibration();
		int numconst = NumConstants();
		ParallelFor(numspots, [&](int i) {
			TModel model(roisize, calib, &h_const[i*numconst]);
			ComputeDerivatives(params_[i], model, &deriv[model.SampleCount()*Params::K*i], &expectedvalue[model.SampleCount()*i]);
		});
	}
	virtual void Estimate(const float * samples, const float *h_const, const int* spot_pos, const float * initial, float * params,
		float* diag, int* iterations, int numspots, float * trace, int traceBufLen) override
	{
		Params* trace_ = (Params*)trace;
		Params* params_ = (Params*)params;
		const Params* initial_ = (const Params*)initial;
		const TCalibration& calib = GetCalibration();
		int smpcount = SampleCount();
		auto lmp = lmParams;
		int numconst = NumConstants();

		ParallelFor(numspots, [&](int i) {
			TModel model(roisize, calib, &h_const[i*numconst]);
			Params ip = initial_ ? initial_[i] : model.ComputeInitialEstimate(&samples[i * smpcount]);
			auto r = LevMarOptimize(&samples[i * smpcount], (const float*)0, ip, model, lmp.iterations,
				&trace_[traceBufLen * i], traceBufLen, lmp.lambda);
			if(iterations) iterations[i] = r.iterations;
			params_[i] = r.estimate;
		});
	}
};


// Estimator Implementation for models with a plain-old-data (POD) calibration type (no [cuda] memory management needed)
template<typename BasePSFImpl>
class SimpleCalibrationPSF : public BasePSFImpl
{
public:
	typedef typename BasePSFImpl::Calibration Calibration;
	typedef typename BasePSFImpl::SampleIndex SampleIndex;
	Calibration calib;

	const Calibration& GetCalibration() override { return calib; }

	SimpleCalibrationPSF(const Calibration& calib, SampleIndex roisize, std::vector<ParamLimit> limits, int numconst) :
		BasePSFImpl(roisize, limits, numconst), calib(calib)
	{}
};

// Wraps a cuEstimator into a host-memory Estimator
class cuEstimatorWrapper : public Estimator {
	cuEstimator* psf;
public:
	cuEstimatorWrapper(cuEstimator* psf);
	~cuEstimatorWrapper();

	cuEstimator* Unwrap() override { return psf; }

	// Inherited via Estimator
	virtual void ChiSquareAndCRLB(const float* d_params, const float* sample, const float* d_const,
		const int* d_roipos, float* crlb, float* chisq, int numspots) override;

	virtual void ExpectedValue(float * h_expectedvalue, const float * h_theta, const float* h_const, const int* spot_pos, int numspots) override;
	virtual void Derivatives(float * h_deriv, float * h_expectedvalue, const float * h_theta, const float* h_const, const int* spot_pos, int numspots) override;
	virtual void Estimate(const float * h_sample, const float* h_const, const int* spot_pos, const float * h_initial, 
		float * h_theta, float * h_diag, int* iterations, int numspots, float* d_trace, int traceBufLen) override;

	virtual void SetLMParams(LMParams p);
	virtual LMParams GetLMParams();

};



class CenterOfMassEstimator : public cuEstimator {
public:
	CenterOfMassEstimator(int roisize);

	virtual void ChiSquareAndCRLB(const float* d_params, const float* sample, const float* d_const,
		const int* d_roipos, float* crlb, float* chisq, int numspots, cudaStream_t stream) override {}

	// Implement COM
	virtual void Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_params, float* d_diag, int* d_iterations,
		int numspots, float * d_trace, int traceBufLen, cudaStream_t stream) override;

	// All these do nothing
	virtual void ExpectedValue(float * expectedvalue, const float * d_params, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream) {}
	virtual void Derivatives(float * deriv, float * expectedvalue, const float * theta, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream) {}
protected:
	int roisize;
};

// pSMLM
class PhasorEstimator : public cuEstimator {
public:
	PhasorEstimator(int roisize);

	virtual void Estimate(const float* d_sample, const float *d_const, const int* d_roipos, const float* d_initial, float* d_params, float* d_diag, int* d_iterations,
		int numspots, float * d_trace, int traceBufLen, cudaStream_t stream) override;

	virtual void ChiSquareAndCRLB(const float* d_params, const float* sample, const float* d_const,
		const int* d_roipos, float* crlb, float* chisq, int numspots, cudaStream_t stream) override {}

	// All these do nothing
	virtual void ExpectedValue(float * expectedvalue, const float * d_params, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream) {}
	virtual void Derivatives(float * deriv, float * expectedvalue, const float * theta, const float * d_const, const int * d_roipos, int numspots, cudaStream_t stream) {}
protected:
	int roisize;
};




CDLL_EXPORT Estimator* CreatePhasorEstimator(int roisize, Context* ctx);
CDLL_EXPORT Estimator* CreateCenterOfMassEstimator(int roisize, Context* ctx);



