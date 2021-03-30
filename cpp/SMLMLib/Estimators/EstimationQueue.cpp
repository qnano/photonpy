// EstimationQueue implementation - runs Estimator::Estimate calls on multiple cuda streams
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "EstimationQueue.h"
#include "Estimators/Estimator.h"
#include "ThreadUtils.h"
#include <iostream>


EstimationQueue::EstimationQueue(cuEstimator * psf, int batchSize, int maxQueueLen, bool keepSamples, int numStreams, Context* ctx) 
	: ContextObject(ctx), maxQueueLen(maxQueueLen), batchSize(batchSize), numconst(psf->NumConstants()), 
	K(psf->NumParams()), smpcount(psf->SampleCount()), smpdims(psf->SampleIndexDims()), psf(psf)
{
	if (numStreams < 0)
		numStreams = 3;

	useInitialValues = false;

	this->keepSamples = keepSamples;

	//psf->SetMaxSpots(batchSize);

	streams.resize(numStreams);
	for (int i = 0; i < numStreams; i++) {
		auto& sd = streams[i];
		cudaStreamCreate(&sd.stream);
			
		sd.constants.Init(batchSize*numconst);
		sd.diagnostics.Init(batchSize*psf->DiagSize());
		sd.estimates.Init(batchSize*K);
		sd.roipos.Init(batchSize*smpdims);
		sd.samples.Init(batchSize*smpcount);
		sd.fi.Init(batchSize*K*K);
		sd.crlb.Init(batchSize*K);
		sd.chisq.Init(batchSize);
		sd.iterations.Init(batchSize);
	}

	numActive = 0;
	next = std::make_unique<Batch>(batchSize, psf);

	stopThread = false;
	thread = new std::thread([&]() {
		ThreadMain();
	});
}


EstimationQueue::~EstimationQueue()
{
	stopThread = true;
	thread->join();

	for (auto& s : streams)
		cudaStreamDestroy(s.stream);
}

void EstimationQueue::ThreadMain()
{
	if(context)
		cudaSetDevice(context->GetDeviceIndex());

	while (!stopThread)
	{
		bool idle = true;
		for (auto& s : streams) {
			if (!s.currentBatch)
			{
				std::unique_ptr<Batch> fr;
				LockedAction(todoMutex, [&]()
				{
					if (!todo.empty()) {
						fr = std::move(todo.front());
						todo.pop_front();
						numActive++;
					}
				});
				if (fr) {
					Launch(std::move(fr), s);
					idle = false;
				}
			}

			if (s.currentBatch && cudaStreamQuery(s.stream) == cudaSuccess) {

				if (!keepSamples) {
					s.currentBatch->samples.Free();
				}

				// all done, move to finished
				LockedAction(finishedMutex, [&]() {
					finished.push_back(std::move(s.currentBatch));
				});
				LockedAction(todoMutex, [&]() {numActive--; });
				idle = false;
			}
		}

		if (idle) // dont waste cpu
			std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}

bool EstimationQueue::IsIdle()
{
	return LockedFunction(todoMutex, [&]() {
		return todo.size() + numActive;
	}) == 0;
}

int EstimationQueue::GetResultCount()
{
	return LockedFunction(finishedMutex, [&]() {
		int total = 0;
		for (auto& l : finished)
			total += l->numspots;
		return total;
	});
}

int EstimationQueue::GetResults(int count, float * estim, float * diag, 
	float* crlb, int* roipos, float* samples, EstimatorResult* results)
{
	int copied = 0;

	while (copied < count) {
		int space = count - copied;

		// Is there batch finished and does it fully fit in the remaining space?
		auto b = LockedFunction(finishedMutex, [&]() {
			if (finished.empty()  || finished.back()->numspots > space)
				return std::unique_ptr<Batch>();

			auto b = std::move(finished.back());
			finished.pop_back();
			return b;
		});

		if (!b)
			return copied;

		if (results) {
			for (int i = 0; i < b->numspots; i++) {
				results[copied + i].id = b->ids[i];
				results[copied + i].chisq = b->chisq[i];
				results[copied + i].iterations = b->iterations[i];
			}
		}

		if (estim)
		{
			for (int i = 0; i < K * b->numspots; i++)
				estim[copied * K + i] = b->estimates[i];
		}
		if (crlb) {
			for (int i = 0; i < K*b->numspots; i++)
				crlb[copied*K + i] = b->crlb[i];
		}
		if (roipos) {
			for (int i = 0; i < psf->SampleIndexDims() * b->numspots; i++)
				roipos[copied * psf->SampleIndexDims() + i] = b->roipos[i];
		}
		if (diag) {
			for (int i = 0; i < psf->DiagSize() * b->numspots; i++)
				diag[copied * psf->DiagSize() + i] = b->diagnostics[i];
		}
		if (samples && keepSamples) {
			for (int i = 0; i < psf->SampleCount() * b->numspots; i++)
				samples[copied * psf->SampleCount() + i] = b->samples[i];
		}
		copied += b->numspots;
		b->numspots = 0;
		LockedAction(recycleMutex, [&]() {
			recycle.push_back(std::move(b));
		});
	}
	return copied;
}

int EstimationQueue::GetQueueLength()
{
	int sum = 0;
	LockedAction(todoMutex, [&]() {
		for (auto& b : todo)
			sum += b->numspots;

		sum += numActive;
	});
	return sum;
}

void EstimationQueue::Flush()
{
	_Flush(true);
}

void EstimationQueue::_Flush(bool lockScheduleMutex)
{
	auto fn = [&]() {
		if (next->numspots == 0)
			return;

		LockedAction(todoMutex, [&]() {
			todo.push_back(std::move(next));
		});
		LockedAction(recycleMutex, [&]() {
			if (!recycle.empty()) {
				next = std::move(recycle.front());

				// if keepSamples is false, we have deallocated the memory to allow storing more results
				size_t smpcount = psf->SampleCount() * batchSize;
				if (next->samples.size() != smpcount)
					next->samples.Init(smpcount);

				recycle.pop_front();
			}
		});
		if (!next) next = std::make_unique<Batch>(batchSize, psf);
	};

	if (lockScheduleMutex) 
		LockedAction(scheduleMutex, fn);
	else
		fn();
}

void EstimationQueue::Schedule(int count, const int* ids, const float * h_samples, const float * h_constants, const float* h_initial, const int * h_roipos)
{
	const float* smp = h_samples;
	const float* constants = h_constants;
	const int* roipos = h_roipos;
	const float* initial = h_initial;

	while (GetQueueLength() >= maxQueueLen) {
		std::this_thread::sleep_for(std::chrono::milliseconds(5));
	}

	useInitialValues = !!initial;

	for (int i = 0; i < count; i++) {
		Schedule(ids[i], smp, initial, constants, roipos);
		smp += psf->SampleCount();
		constants += psf->NumConstants();
		roipos += psf->SampleIndexDims();

		if (initial)
			initial += K;
	}
}

void EstimationQueue::Schedule(int id, const float * h_samples, const float* h_initial, const float * h_constants, const int * h_roipos)
{
	LockedAction(scheduleMutex, [&]() {
		int i = next->numspots++;

		next->ids[i] = id;
		for (int j = 0; j < numconst; j++)
			next->constants[i * numconst + j] = h_constants[j];
		for (int j = 0; j < smpcount; j++)
			next->samples[i * smpcount + j] = h_samples[j];
		for (int j = 0; j < smpdims; j++)
			next->roipos[i * smpdims + j] = h_roipos[j];

		if (h_initial)
		{
			for (int j = 0; j < K; j++)
				next->estimates[i * K + j] = h_initial[j];
		}

		if (next->numspots == batchSize) 
			_Flush(false);
	});
}


void EstimationQueue::Launch(std::unique_ptr<Batch> b, EstimationQueue::StreamData& sd)
{
	// Copy to GPU
	sd.samples.CopyToDevice(b->samples.data(), b->numspots*psf->SampleCount(), true, sd.stream);
	sd.constants.CopyToDevice(b->constants.data(), b->numspots*psf->NumConstants(), true, sd.stream);
	sd.roipos.CopyToDevice(b->roipos.data(), b->numspots*psf->SampleIndexDims(), true, sd.stream);
	sd.estimates.CopyToDevice(b->estimates.data(), b->numspots * psf->NumParams(), true, sd.stream);

	// Process
	psf->Estimate(sd.samples.data(), sd.constants.data(), sd.roipos.data(), useInitialValues ? sd.estimates.data() : 0, sd.estimates.data(),
		sd.diagnostics.data(), sd.iterations.data(), b->numspots, 0, 0, sd.stream);
	psf->ChiSquareAndCRLB(sd.estimates.data(), sd.samples.data(), sd.constants.data(), sd.roipos.data(), 
		sd.crlb.data(), sd.chisq.data(), b->numspots, sd.stream);

	// Copy results to host
	sd.estimates.CopyToHost(b->estimates.data(), b->numspots * psf->NumParams(), true, sd.stream);
	sd.diagnostics.CopyToHost(b->diagnostics.data(), b->numspots*psf->DiagSize(), true, sd.stream);
	sd.crlb.CopyToHost(b->crlb.data(), b->numspots*psf->NumParams(), true, sd.stream);
	sd.iterations.CopyToHost(b->iterations.data(), b->numspots, true, sd.stream);
	sd.chisq.CopyToHost(b->chisq.data(), b->numspots, true, sd.stream);

	sd.currentBatch = std::move(b);
}

EstimationQueue::Batch::Batch(int batchsize, cuEstimator * psf)
	:numspots(0)
{
	ids.resize(batchsize);
	roipos.Init(batchsize*psf->SampleIndexDims());
	constants.Init(batchsize * psf->NumConstants());
	estimates.Init(batchsize * psf->NumParams());
	diagnostics.Init(batchsize*psf->DiagSize());
	samples.Init(batchsize * psf->SampleCount());
	fisherInfo.Init(batchsize*psf->NumParams()*psf->NumParams());
	crlb.Init(batchsize*psf->NumParams());
	iterations.Init(batchsize);
	chisq.Init(batchsize);
}


CDLL_EXPORT EstimationQueue* EstimQueue_Create(Estimator* psf, int batchSize, int maxQueueLen, bool keepSamples, int numStreams, Context* ctx)
{
	try {
		cuEstimator* cudapsf = psf->Unwrap();
		if (!cudapsf)
		{
			std::cerr << "EstimationQueue needs a CUDA based Estimator object." << std::endl;
			return 0;
		}

		return new EstimationQueue(cudapsf, batchSize, maxQueueLen, keepSamples, numStreams, ctx);
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
	return 0;
}

CDLL_EXPORT void EstimQueue_Delete(EstimationQueue* queue)
{
	try {
		delete queue;
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}

CDLL_EXPORT void EstimQueue_Schedule(EstimationQueue* q, int numspots, const int* ids,
	const float* h_samples, const float* h_initial, const float* h_constants, const int* h_roipos)
{
	try {
		q->Schedule(numspots, ids, h_samples, h_constants, h_initial, h_roipos);
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}

CDLL_EXPORT void EstimQueue_Flush(EstimationQueue* q)
{
	try {
		q->Flush();
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
	}
}

CDLL_EXPORT bool EstimQueue_IsIdle(EstimationQueue* q)
{
	try {
		return q->IsIdle();
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return true;
	}
}

CDLL_EXPORT int EstimQueue_GetResultCount(EstimationQueue* q)
{
	try {
		return q->GetResultCount();
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}

CDLL_EXPORT int EstimQueue_GetQueueLength(EstimationQueue *q)
{
	try {
		return q->GetQueueLength();
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}

// Returns the number of actual returned localizations. 
// Results are removed from the queue after they are copied to the provided memory
CDLL_EXPORT int EstimQueue_GetResults(EstimationQueue* q, int maxresults, float* estim, float* diag,
										float* crlb, int *roipos, float* samples, EstimatorResult* result)
{
	try {
		return q->GetResults(maxresults, estim, diag, crlb, roipos, samples, result);
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}

