// EstimationQueue implementation - runs Estimator::Estimate calls on multiple cuda streams
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "CudaUtils.h"
#include <vector>
#include <list>
#include <memory>
#include <mutex>
#include "Context.h"

#include "Estimators/Estimator.h"

class cuEstimator;
class Estimator;

// Accepts host- or device memory localization jobs

struct EstimatorResult {
	int id;
	float chisq;
	int iterations;
};

class EstimationQueue : public ContextObject {
public:
	DLL_EXPORT EstimationQueue(cuEstimator* psf, int batchSize, int maxQueueLen, bool keepSamples, int numStreams, Context* ctx=0);
	DLL_EXPORT ~EstimationQueue();

	DLL_EXPORT void Schedule(int id, const float* h_samples, const float* h_initial,
					const float* h_constants, const int* h_roipos);

	// Schedule many
	DLL_EXPORT void Schedule(int count, const int *id, const float* h_samples, const float* h_initial,
		const float* h_constants, const int* h_roipos);

	DLL_EXPORT void Flush();
	DLL_EXPORT bool IsIdle();

	DLL_EXPORT int GetResultCount();
	DLL_EXPORT int GetQueueLength(); // in spots

	// Returns the number of actual returned localizations. 
	// Results are removed from the queue after copy to the provided memory
	// All arrays are optional (can be zero) if not needed.
	DLL_EXPORT int GetResults(int maxresults, float* estim, float* diag, float* crlb, int* roipos, float* samples, EstimatorResult* result);


protected:

	void _Flush(bool lockScheduleMutex);

	struct Batch
	{
		Batch(int maxspots, cuEstimator* psf);

		std::vector<int> ids;
		PinnedArray<float> samples, constants;
		PinnedArray<int> roipos;
		PinnedArray<float> estimates, diagnostics;
		PinnedArray<float> fisherInfo, crlb;
		PinnedArray<int> iterations;
		PinnedArray<float> chisq;
		int numspots;
	};


	struct StreamData {
		StreamData() { stream = 0;  }
		std::unique_ptr<Batch> currentBatch;
		cudaStream_t stream;

		DeviceArray<float> estimates, samples, constants, diagnostics, fi, crlb;
		DeviceArray<int> roipos; // [SampleIndexDims * batchsize]
		DeviceArray<int> iterations;
		DeviceArray<float> chisq;
	};

	std::list<std::unique_ptr<Batch>> todo;
	int numActive; // also guarded using todoMutex
	std::mutex todoMutex;

	std::list<std::unique_ptr<Batch>> recycle;
	std::mutex recycleMutex;

	std::list<std::unique_ptr<Batch>> finished;
	std::mutex finishedMutex;

	std::unique_ptr<Batch> next; // currently filling up this batch
	std::mutex scheduleMutex;

	cuEstimator* psf;
	std::vector<StreamData> streams;
	int maxQueueLen, batchSize;

	bool keepSamples;

	std::thread* thread;
	volatile bool stopThread;

	virtual void Launch(std::unique_ptr<Batch> b, StreamData& sd);
	void ThreadMain();

	int numconst, K, smpcount, smpdims;
	bool useInitialValues;
};

// C API wrappers
CDLL_EXPORT EstimationQueue* EstimQueue_Create(Estimator* psf, int batchSize, int maxQueueLen, bool keepSamples, int numStreams, Context* ctx);
CDLL_EXPORT void EstimQueue_Delete(EstimationQueue* queue);

CDLL_EXPORT void EstimQueue_Schedule(EstimationQueue* q, int numspots, const int *ids, const float* h_samples, const float* h_initial,
	const float* h_constants, const int* h_roipos);

CDLL_EXPORT void EstimQueue_Flush(EstimationQueue* q);
CDLL_EXPORT bool EstimQueue_IsIdle(EstimationQueue* q);

CDLL_EXPORT int EstimQueue_GetResultCount(EstimationQueue* q);

// Returns the number of actual returned localizations. 
// Results are removed from the queue after copyInProgress to the provided memory
CDLL_EXPORT int EstimQueue_GetResults(EstimationQueue* q, int maxresults, float* estim, float* diag,
	float* crlb, int* roipos, float* samples, EstimatorResult* result);

CDLL_EXPORT int EstimQueue_GetQueueLength(EstimationQueue* q);
