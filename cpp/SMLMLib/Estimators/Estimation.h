// Model-independent Fisher matrix calculation and Levenberg-Marquardt optimizer (https://www.nature.com/articles/nmeth0510-338)
// All estimators here are for Poisson-distributed samples.
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "SolveMatrix.h"
#include "Vector.h"
#include <math_constants.h>


template<typename Params>
struct FisherMatrixType {
	const static int K = Params::K;
	typedef Vector<typename Params::TElem, K*K> type;
};



template<typename TModel, typename Params>
PLL_DEVHOST auto ComputeFisherMatrix(const TModel& model, const Params& theta)
-> typename FisherMatrixType<Params>::type
{
	const int K = model.K;
	typedef typename Params::TElem T;
	typename FisherMatrixType<Params>::type fi;

	for (int i = 0; i < K * K; i++)
		fi[i] = 0;

	model.ComputeDerivatives([&](int idx, T mu, const T(&jacobian)[K]) {
		T readnoiseVarGain2 = 0.0f;
		//if (readnoise) readnoiseVarGain2 = readnoise[smpIndex];

		mu += readnoiseVarGain2;
		mu = fmax(1e-8f, mu);

		float inv_mu_c = 1.0f / mu;
		for (int i = 0; i < K; i++) {
			for (int j = i; j < K; j++) {
				const T fi_ij = jacobian[i] * jacobian[j] * inv_mu_c;
				fi[K * i + j] += fi_ij;
			}
		};
		}, theta);
	// fill below diagonal
	for (int i = 1; i < K; i++)
		for (int j = 0; j < i; j++)
			fi[K * i + j] = fi[K * j + i];

	return fi;
}


template<typename TParams>
struct ChiSquareAndCRLB {
	TParams crlb;
	float chisq;
};


template<typename TModel, typename Params, typename TSampleIndexer>
PLL_DEVHOST auto ComputeChiSquareAndCRLB(const TModel& model, const Params& theta, typename TSampleIndexer sample)
	-> typename ChiSquareAndCRLB<Params>
{
	const int K = model.K;
	typedef typename Params::TElem T;
	typename FisherMatrixType<Params>::type fi;

	float chisq = 0.0f;
 
	for (int i = 0; i < K*K; i++)
		fi[i] = 0;
	
	model.ComputeDerivatives([&](int idx, T mu, const T(&jacobian)[K]) {
		T readnoiseVarGain2 = 0.0f;
		//if (readnoise) readnoiseVarGain2 = readnoise[smpIndex];

		mu += readnoiseVarGain2;
		mu = fmax(1e-8f, mu);

		if (sample) {
			float e = sample[idx] - mu;
			chisq += e * e / mu;
		}

		float inv_mu_c = 1.0f / mu;
		for (int i = 0; i < K; i++) {
			for (int j = i; j < K; j++) {
				const T fi_ij = jacobian[i] * jacobian[j] * inv_mu_c;
				fi[K*i + j] += fi_ij;
			}
		};
	}, theta);
	// fill below diagonal
	for (int i = 1; i < K; i++)
		for (int j = 0; j < i; j++)
			fi[K*i + j] = fi[K*j + i];

	return { InvertMatrix(fi).diagonal().sqrt(), chisq };
}

template<typename T, typename TModel, typename TConstArrayIndexer>
PLL_DEVHOST void ComputeLevMarAlphaBeta(T* __restrict alpha, T* __restrict beta, const typename TModel::Params& theta,
	TConstArrayIndexer sample, TConstArrayIndexer readnoise, const TModel& model)
{
	const int K = model.K;
	model.ComputeDerivatives([&](int smpIndex, T mu, const T* jacobian) {
		T readnoiseVarGain2 = 0.0f;
		if (readnoise) readnoiseVarGain2 = readnoise[smpIndex];
		T sampleValue = sample[smpIndex] + readnoiseVarGain2;
		mu += readnoiseVarGain2;
		if (sampleValue < 1e-6f) sampleValue = 1e-6f;

		float mu_c = mu > 1e-6f ? mu : 1e-6f;
		float invmu = 1.0f / mu_c;
		T x_f2 = sampleValue * invmu*invmu;

		for (int i = 0; i < K; i++)
			for (int j = i; j < K; j++)
				alpha[K*i + j] += jacobian[i] * jacobian[j] * x_f2;

		T beta_factor = 1 - sampleValue * invmu;
		for (int i = 0; i < K; i++) {
			beta[i] -= beta_factor * jacobian[i];
		}
	}, theta);

	// fill below diagonal
	for (int i = 1; i < K; i++)
		for (int j = 0; j < i; j++)
			alpha[K*i + j] = alpha[K*j + i];
}



template<typename TModel>
PLL_DEVHOST float ComputeLogLikelihood(const typename TModel::Params& theta, const float* sample, const float* readnoise, 
	const TModel& model)
{
	typedef float T;
	T LL = 0;
	const int K = model.K;
	model.ComputeDerivatives([&](int idx, T mu, const T (&jacobian)[K]) {
		float readnoise_ = 0.0f;
		if (readnoise) readnoise_ = readnoise[idx];

		T sampleValue = sample[idx] + readnoise_;
		mu += readnoise_;
		mu = fmax(mu, 1e-8f);
		sampleValue = fmax(sampleValue, 1e-8f);
		LL += sampleValue * log(mu) - mu;
	}, theta);
	return LL;
}


template<typename TModel, typename TArrayIndexer>
PLL_DEVHOST void ComputeExpectedValue(const typename TModel::Params& theta, const TModel& model, TArrayIndexer psf_ev)
{
	typedef typename std::remove_reference<decltype(psf_ev[0])>::type T;
	const int K = model.K;
	model.ComputeDerivatives([&](int idx, T mu, const T(&jacobian)[K]) {
		psf_ev[idx] = mu;
	}, theta);
}

template<typename TModel, typename TArrayIndexer>
PLL_DEVHOST void ComputeDerivatives(const typename TModel::Params& theta, const TModel& model, TArrayIndexer psf_deriv, TArrayIndexer psf_ev)
{
	typedef typename std::remove_reference<decltype(psf_ev[0])>::type T;
	const int K = model.K;
	const int img_size = model.SampleCount();
	model.ComputeDerivatives([&](int j, T mu, const T(&jacobian)[K]) {
		for (int i = 0; i < K; i++) 
			psf_deriv[j + i * img_size] = jacobian[i];

		psf_ev[j] = mu;
	}, theta);
}



template<typename TParams >
struct OptimizerResult
{
	TParams estimate;
	int iterations;
	TParams initialValue;
};


template<typename TModel, typename TSampleArrayIndexer, typename TReadnoiseIndexer>
PLL_DEVHOST OptimizerResult <typename TModel::Params > LevMarOptimize(TSampleArrayIndexer sample, TReadnoiseIndexer readnoiseVG2, const typename TModel::Params initialValue,
	const TModel& model, int maxIterations, typename TModel::Params* trace=0, int traceBufLen=0, float lambdaStep=1e-15f)
{
	typedef float T;
	int i;
	typename TModel::Params params = initialValue;
	for (i = 0; i < maxIterations; i++) {
		if (i < traceBufLen)
			trace[i] = params;

		const int K = model.K;
		T alpha[K * K] = {};
		T beta[K] = {};
		T thetaStep[K];
		T LU[K * K] = {};

		ComputeLevMarAlphaBeta(alpha, beta, params, sample, readnoiseVG2, model);

		for (int k = 0; k < K; k++) {
			if (lambdaStep < 0.0f) // non scale invariant
				alpha[k * K + k] -= lambdaStep;
			else {
				// scale invariant
				float s = 0.0f;
				for (int j = 0; j < K; j++)
					s += alpha[j * K + k] * alpha[j * K + k];
				alpha[k * K + k] += s * lambdaStep;
			}
		}

		if (!Cholesky(K, alpha, LU))
			break;
		if (!SolveCholesky(LU, beta, thetaStep))
			break;

		typename TModel::Params last = params;
		bool smallDeltas = true;
		for (int k = 0; k < K; k++) {
			params[k] += thetaStep[k];
		}
		model.CheckLimits(params);

		for (int k = 0; k < K; k++) {
			if (abs(last[k] - params[k]) > model.StopLimit(k)) {
				smallDeltas = false;
				break;
			}
		}
		if (i > 0 && smallDeltas)
			break;
	}
	OptimizerResult<typename TModel::Params> lr;
	lr.iterations = i + 1;
	lr.estimate = params;
	lr.initialValue = initialValue;
	return lr;
}



// TModel needs to implement ComputeSecondDerivatives() for this to work
template<typename TModel>
PLL_DEVHOST OptimizerResult<typename TModel::Params> NewtonRaphson(const float* __restrict sample, const float* readnoiseVG2, typename TModel::Params initialValue,
	TModel& model, int maxIterations, typename TModel::Params* trace = 0, int traceBufLen = 0, float stepFactor=0.8f)
{
	const int K = TModel::Params::K;
	typedef float T;
	typename TModel::Params theta = initialValue, lastTheta;

	T num[K];
	T denom[K];
	int i;
	for (i = 0; i < maxIterations; i++) {
		if (i < traceBufLen)
			trace[i] = theta;

		for (int j = 0; j < K; j++)
			num[j] = denom[j] = 0.0f;

		model.ComputeSecondDerivatives([&](int smpIndex, T mu, const T(&firstOrder)[K], const T(&secondOrder)[K]) {
			T x = sample[smpIndex] += readnoiseVG2[smpIndex];
			mu += readnoiseVG2[smpIndex];
			T mu_clamped = mu > 1e-8f ? mu : 1e-8f;
			T invMu = 1.0f / mu_clamped;
			T m = x * invMu - 1;
			for (int j = 0; j < K; j++) {
				num[j] += firstOrder[j] * m;
				denom[j] += secondOrder[j] * m - firstOrder[j] * firstOrder[j] * x*invMu*invMu;
			}
		}, theta);
		if (i > 0) {
			bool smallDeltas = true;
			for (int k = 0; k < K; k++) {
				T delta = theta[k] - lastTheta[k];
				if (abs(delta) > model.StopLimit(k)) {
					smallDeltas = false;
					break;
				}
			}
			if (smallDeltas)
				break;
		}

		lastTheta = theta;
		typename TModel::Params step;
		for (int j = 0; j < K; j++)
			step[j] = num[j] / denom[j];

#ifndef __CUDA_ARCH__
		//DebugPrintf("iteration %d: ",i);
		//PrintVector(step);
#endif

		theta -= step*stepFactor;
		model.CheckLimits(theta);
	}
	OptimizerResult<typename TModel::Params> lr;
	lr.iterations = i;
	lr.estimate = theta;
	return lr;
}


template<typename T, int KK> PLL_DEVHOST Vector<T, CompileTimeSqrt(KK)> ComputeCRLB(const Vector<T, KK>& fisher)
{
	const int K = CompileTimeSqrt(KK);
	Vector<T,KK> inv;
	Vector<T, K> crlb;
	if (InvertMatrix<T,KK>(fisher, inv))
	{
		for (int i = 0; i < K; i++)
			crlb[i] = sqrt(fabs(inv[i*(K + 1)]));
	}
	else {
		for (int i = 0; i < K; i++)
			crlb[i] = INFINITY;
	}
	return crlb;
}


template<typename T>
PLL_DEVHOST Vector3f ComputeCOM(const T* sample, Int2 roisize)
{
	// Compute initial value
	T sumX = 0, sumY = 0, sum = 0;
	for (int y = 0; y < roisize[0]; y++)
		for (int x = 0; x < roisize[1]; x++) {
			T v = sample[y * roisize[1] + x];
			sumX += x * v;
			sumY += y * v;
			sum += v;
		}

	T comx = sumX / sum;
	T comy = sumY / sum;
	return { comx,comy, sum };
}

template<typename T>
PLL_DEVHOST Vector3f ComputePhasorEstim(const T* smp, int w, int h)
{
	//		fx = np.sum(np.sum(roi, 0)*np.exp(-2j*np.pi*np.arange(roi.shape[1]) / roi.shape[1]))
	//		fy = np.sum(np.sum(roi, 1)*np.exp(-2j*np.pi*np.arange(roi.shape[0]) / roi.shape[0]))
	float fx_re = 0.0f, fx_im = 0.0f;
	float freqx = 2 * CUDART_PI_F / w;
	for (int x = 0; x < w; x++)
	{
		float sum = 0.0f;
		for (int y = 0; y < h; y++)
			sum += smp[y*w + x];
		fx_re += sum * cos(-x * freqx);
		fx_im += sum * sin(-x * freqx);
	}
	//    angX = np.angle(fft_values[0,1])
	float angX = atan2(fx_im, fx_re);
	if (angX > 0) angX -= 2 * CUDART_PI_F;
	float posx = abs(angX) / freqx;

	float fy_re = 0.0f, fy_im = 0.0f;
	float freqy = 2 * CUDART_PI_F / h;
	float total = 0.0f;
	for (int y = 0; y < h; y++)
	{
		float sum = 0.0f;
		for (int x = 0; x < w; x++)
			sum += smp[y*w + x];
		fy_re += sum * cos(-y * freqy);
		fy_im += sum * sin(-y * freqy);
		total += sum;
	}

	float angY = atan2(fy_im, fy_re);
	if (angY > 0) angY -= 2 * CUDART_PI_F;
	float posy = abs(angY) / freqy;

	return { posx,posy,total };
}
