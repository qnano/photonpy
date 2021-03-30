// Compute continuous closed-form FRC according to in
// "Closed-Form Expression Of The Fourier Ring-Correlation For Single-Molecule Localization Microscopy"
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include <list>
#include "DLLMacros.h"
#include <vector>
#include "Vector.h"
#include "ThreadUtils.h"
#include "KDTree.h"

template<int D>
void _ComputeContinuousFRC(const Vector<float, D>* data, int numspots, std::vector<float> rho, float maxDistance, float* frc, bool useCuda, float cutoffDist, float cutoffSigma)
{
	typedef Vector<float, D> Pt;

	// I'm calling it the FRC inner product for lack of a better name (the paper doesnt seem to name it)
	auto frcInnerProd = [&](const std::vector<Pt>& setF, const std::vector<Pt>& setG) {

		KDTree<float, D> kdtree(setG, 20);

		int nrho = (int)rho.size();

		std::vector<float> sums(rho.size());

		auto processBatch = [&](int startIndex, std::vector<int> indices, std::vector<int> startpos, std::vector<int> counts) {
			int batchsize = (int)startpos.size();

			//float ic = 1.0f / (2 * cutoffSigma * cutoffSigma);

			// Note that all these pointer parameters are automatically copied to CUDA memory by palala if needed
			auto cb = [=]__device__ __host__(int i, int rhoIndex,
				const Pt * setF, const Pt * setG,
				const int* nbList,
				const int* nbcounts,
				const int* nbstartIdx,
				float* sumsPerRho,
				const float* rho)
			{
				// iterate through all neighbors
				float sum = 0.0f;
				float rho_ = rho[rhoIndex];
				int first = nbstartIdx[i];
				for (int n = 0; n < nbcounts[i]; n++) {
					int nb = nbList[first + n];
					float dist = (setF[startIndex + i] - setG[nb]).length();
					//float d = fmaxf(cutoffDist-dist,0.0f);
					float c = 1.0f;// exp(-d * d * ic); // cutoff works nice but not sure how to mathematically explain it nicely
#ifdef __CUDA_ARCH__
					sum += j0(dist * rho_) * c;
#else
					sum += (float)_j0(dist * rho_) * c;
#endif
				}
				sumsPerRho[rhoIndex * batchsize + i] = sum;
			};

			std::vector<float> sumsPerRho(batchsize * rho.size());

			//DebugPrintf("Processing batch with %d spots. #Neighbors=%d\n", batchsize, indices.size());

			palala_for((int)counts.size(), nrho, useCuda, cb,
				const_array(setF.data(), setF.size()),
				const_array(setG.data(), setG.size()),
				indices, counts, startpos,
				sumsPerRho, rho);

			for (int r = 0; r < nrho; r++) {
				for (int i = 0; i < batchsize; i++)
					sums[r] += sumsPerRho[r * batchsize + i];
			}
			return true;
		};

		IterateThroughNeighbors(kdtree, setF, Pt::ones()*maxDistance, 1000000, 0, processBatch);
		return sums;
	};

	// To compute FRC we need to split the dataset in two sets F and G. 
	// For convenience we choose to put all even in F and odd in G
	std::vector<Pt> setA, setB;
	setA.reserve(numspots / 2 + 1);
	setB.reserve(numspots / 2 + 1);
	for (int i = 0; i < numspots; i++) {
		if (i % 2 == 0) setA.push_back(data[i]);
		else setB.push_back(data[i]);
	}

	auto aa = frcInnerProd(setA, setA);
	auto bb = frcInnerProd(setB, setB);
	auto ab = frcInnerProd(setA, setB);

	for (int i = 0; i < rho.size(); i++) {
		frc[i] = ab[i] / sqrt(aa[i] * bb[i]);
	}
}

/*
Compute continuous closed-form FRC according to (14) in
"Closed-Form Expression Of The Fourier Ring-Correlation For Single-Molecule Localization Microscopy"

rho: float[nrho]
frc: result, float[nrho]
data: float[numspots, dims]
*/
CDLL_EXPORT void ComputeContinuousFRC(const float* data, int dims, int numspots, const float* rho, int nrho, float* frc, float maxDistance, bool useCuda, float cutoffDist, float cutoffSigma)
{
	std::vector<float> rho_(rho, rho + nrho);
	if (dims == 2)
		_ComputeContinuousFRC((Vector2f*)data, numspots, rho_, maxDistance, frc, useCuda, cutoffDist, cutoffSigma);
	else if (dims == 3)
		_ComputeContinuousFRC((Vector3f*)data, numspots, rho_, maxDistance, frc, useCuda, cutoffDist, cutoffSigma);
	else
		DebugPrintf("Invalid number of dimensions in ComputeContinuousFRC\n");
}


