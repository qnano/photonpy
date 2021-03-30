// Hidden markov model that runs in a cuda kernel
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "palala.h"


template<int numstates>
class HMM {
public:
	enum {S=numstates};

	// assuming sizeof(int)=sizeof(float)
	PLL_DEVHOST static int viterbi_temp_size(int numsamples)
	{
		return numsamples * numstates * 2;
	}

	// logTransition[i * N + j] = probability of going from i to j
	template<typename EmissionLogProbFn>
	PLL_DEVHOST static void Viterbi(int numsamples, const float* priors, const float* logTransition, float *temp, EmissionLogProbFn emissionLogProb, int* output_sequence)
	{
		int* temp_choice = (int*)temp;
		float* temp_logmu = &temp[numsamples*numstates];

		//	logmu(1, :) = log(prior) + logemission(samples(1, :), 1:m);
		auto logmu = [&](int smp, int state) -> float& { return temp_logmu[numstates*smp + state]; };
		auto choice = [&](int smp, int state) -> int& { return temp_choice[numstates*smp + state]; };

		for (int z = 0; z < numstates; z++)
			logmu(0, z) = log(priors[z]) + emissionLogProb(0, z);

		for (int k = 1; k < numsamples; k++) {
			for (int z = 0; z < numstates; z++) {
				float a = logTransition[0 * numstates + z] + logmu(k - 1, 0);
				int maxelem = 0;
				for (int j = 1; j < numstates; j++) {
					float b = logTransition[j * numstates + z] + logmu(k - 1, j);
					if (b > a) {
						maxelem = j; a = b;
					}
				}
				choice(k, z) = maxelem;
				logmu(k, z) = a + emissionLogProb(k, z);
			}
		}

		int laststate = 0;
		for (int i = 1; i < numstates; i++)
			if (logmu(numsamples - 1, laststate) < logmu(numsamples - 1, i)) laststate = i;
		output_sequence[numsamples - 1] = laststate;

		for (int k = numsamples - 2; k >= 0; k--) {
			laststate = choice(k + 1, laststate);
			output_sequence[k] = laststate;
		}
	}

	template<typename EmissionProbFn>
	static void Viterbi(int numsamples, const float* priors, const float* logTransition, EmissionProbFn emissionProb, int* output_sequence)
	{
		std::vector<float> temp(numstates*numsamples*2);

		Viterbi(numsamples, priors, logTransition, &temp[0], emissionProb, output_sequence);
	}

	template<int n>
	PLL_DEVHOST __forceinline static float logsum(float (&f)[n])
	{
		float b = f[0];
		for (int i = 1; i < n; i++) {
			float t = f[i];
			if (t > b) b = t;
		}
		if (isinf(b))
			return -INFINITY;
		float sum = 0.0f;
		for (int i = 0; i < n; i++)
			sum += expf(f[i] - b);
		return b + log(sum);
	}

	template<int n>
	PLL_DEVHOST static void lognormalize(float (&v)[n])
	{
		float s = logsum(v);
		for (int i = 0; i < n; i++)
			v[i] -= s;
	}

	PLL_DEVHOST static int ForwardBackwardTempSize(int numsamples)
	{
		return numstates * numsamples * 2;
	}

	template<typename EmissionLogProbFn>
	PLL_DEVHOST static void ForwardBackward(int numsamples, const float* priors, const float* logTransition, float *temp, EmissionLogProbFn emissionLogProb, float* log_posterior)
	{
		auto loga = [&](int smp, int state) -> float& { return temp[numstates*smp + state]; };
		auto logb = [&](int smp, int state) -> float& { return temp[numstates*numsamples + numstates*smp + state]; };

		// a(k, l) is the alpha(k) for the value of z=l
		// alpha(k, l) = p(x(1:k), z(k) | model)

		// Forward algorithm:
		// Goal: compute p(z(k), x(1:k))
		for (int z = 0; z < numstates; z++)
			loga(0, z) = log(priors[z]) + emissionLogProb(0, z);

		for (int k = 1; k < numsamples; k++)
		{
			for (int z = 0; z < numstates; z++) {
				float temp[numstates];
				for (int i = 0; i < numstates; i++)
					temp[i] = loga(k - 1, i) + logTransition[i*numstates + z];
				loga(k, z) = emissionLogProb(k, z) + logsum(temp);
			}
		}

		for (int z = 0; z < numstates; z++)
			logb(numsamples - 1, z) = 0;

		for (int k = numsamples - 2; k >= 0; k--) {
			for (int z = 0; z < numstates; z++) {
				float temp[numstates];
				for (int i = 0; i < numstates; i++)
					temp[i] = logb(k + 1, i) + emissionLogProb(k + 1, i) + logTransition[z*numstates + i];
				logb(k, z) = logsum(temp);
			}
		}

		for (int k = 0; k < numsamples;k++) {
			float tmp[numstates];

			for (int z = 0; z < numstates; z++)
				tmp[z] = loga(k, z) + logb(k, z);
			lognormalize(tmp);

			for (int i = 0; i < numstates; i++)
				log_posterior[k*numstates + i] = tmp[i];
		}
	}

	template<typename EmissionLogProbFn>
	static void ForwardBackward(int numsamples, const float* priors, const float* logTransition, EmissionLogProbFn emissionLogProb, float* log_posterior)
	{
		std::vector<float> temp(ForwardBackwardTempSize(numsamples));
		ForwardBackward(numsamples, priors, logTransition, &temp[0], emissionLogProb, log_posterior);
	}

};


