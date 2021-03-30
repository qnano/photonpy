// Image queue that applies calibration correction, gain/offset, uniform filters
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "ImgFilterQueue.h"
#include "ThreadUtils.h"
#include "ImageFilters.h"
#include "Context.h"

#include "ImgFilterQueueImpl.h"

#include "Estimators/EstimationQueue.h"
#include "SpotDetection/SpotDetector.h"
#include "CameraCalibration.h"

#include "CudaMath.h"
#include <thrust/functional.h>

#include "ImageProcessor.h"
#include "ROIQueue.h"

#include "DebugImageCallback.h"



void CastFrame(const DeviceImage<uint16_t>& src, DeviceImage<float>& dst, cudaStream_t stream) 
{
	auto src_ = src.GetConstIndexer();
	auto img_ = dst.GetIndexer();
	LaunchKernel(src.width, src.height, [=]__device__(int x, int y) {
		img_(x, y) = src_(x, y);
	}, 0, stream);
}

RawFrameInputQueue::RawFrameInputQueue(Config config, ICalibrationProcessor* calib, Context*ctx) :
	ImgQueue(config.imageSize, ctx), config(config), calib(calib)
{
	frameNumber = 0;
}

void RawFrameInputQueue::AddHostFrame(const uint16_t * data)
{
	std::unique_ptr<RawInputFrame> f = GetNewInputFrame();
	f->framenum = frameNumber++;
	memcpy(f->rawimg.data(), data, sizeof(uint16_t)*config.imageSize.x *config.imageSize.y);

	AddFrame(std::move(f));
}


void RawFrameInputQueue::ProcessFrame(std::unique_ptr<RawInputFrame> frame)
{
	auto output = GetNewOutputFrame();

	// convert to floats
	for (int i = 0; i < config.imageSize.x * config.imageSize.y; i++) {
		frame->rawimgPhotons[i] = frame->rawimg[i];
	}

	if (calib)
	{
		calib->ProcessImage(ImageIndexer(config.imageSize.x, config.imageSize.y, frame->rawimgPhotons.data()));
	}

	output->d_image.CopyFromHost(frame->rawimgPhotons.data());
	output->framenum = frame->framenum;

	cudaStreamSynchronize(stream);
	RecycleInput(std::move(frame));
	AddToFinished(std::move(output));
}


RemoveBackgroundFilterQueue::RemoveBackgroundFilterQueue(int2 imgsize, Int2 filterSize, Context*ctx) :
	ImgQueue(imgsize, ctx), 
	uniformFilterSize(filterSize), 
	temp1(imgsize), 
	d_xyfiltered1(imgsize), 
	d_xyfiltered2(imgsize)
{
}

void RemoveBackgroundFilterQueue::ProcessFrame(std::unique_ptr<Frame> frame)
{
	auto output = GetNewOutputFrame();
	output->framenum = frame->framenum;
	UniformFilter2D(frame->d_image, temp1, d_xyfiltered1, uniformFilterSize[0], stream);
	UniformFilter2D(frame->d_image, temp1, d_xyfiltered2, uniformFilterSize[1], stream);
	ApplyBinaryOperator2D(d_xyfiltered1, d_xyfiltered2, output->d_filtered, thrust::minus<float>(), stream);
	output->d_image.Swap(frame->d_image);

	cudaStreamSynchronize(stream);
	AddToFinished(std::move(output));
	RecycleInput(std::move(frame));
}


class SpotDetectedFrame
{
public:
	SpotDetectedFrame(int2 imgsize) {}
};



struct Empty { 	
	Empty(int2) {}
};


struct SummedFrame : public Frame {
	SummedFrame(int2 imgsize) :
		Frame(imgsize)
	{
		assert(0); // this one should not be called
	}
	SummedFrame(int2 imgsize, int sumframes) :
		Frame(imgsize), sumframes(sumframes), original({ imgsize.x, imgsize.y*sumframes }) {}

	int sumframes;
	DeviceImage<float> original;
};


// TODO: Come up with better class naming..
class SpotDetectorImgQueue : public ImgQueue<SummedFrame, Empty> {
public:
	SpotDetectorImgQueue(int2 imgsize, ROIQueue* dst, ISpotDetectorFactory* sd) :
		ImgQueue(imgsize,0), h_indices(imgsize.x*imgsize.y), dst(dst) {
		spotDetector = sd->CreateInstance(imgsize.x, imgsize.y);

		Int3 rs = dst->ROIShape();
		roiWidth = rs[2];
		roiHeight = rs[1];
		roiFrames = rs[0];

		h_image.Init(imgsize.x*imgsize.y*roiFrames);
		h_samples.resize(roiWidth*roiHeight*roiFrames);
	}
	~SpotDetectorImgQueue() {
		delete spotDetector;
	}

	int NumFinishedFrames()
	{
		return LockedFunction(framesFinishedMutex, [&]() {return framesFinished; });
	}

protected:
	ROIQueue* dst;
	PinnedArray<float> h_image;
	PinnedArray<IndexWithScore> h_indices;
	ISpotDetector* spotDetector;
	std::vector<float> h_samples; 
	int roiWidth, roiHeight, roiFrames;

	int framesFinished=0;
	std::mutex framesFinishedMutex;

	// Inherited via ImgQueue
	virtual void ProcessFrame(std::unique_ptr<SummedFrame> frame) override
	{
		//ShowDebugImage(frame->original, 1, "original img");
		frame->original.CopyToHost(h_image.data(), stream);

		// Run spot detection on the processed image (after calibration and summing)
		spotDetector->Detect(frame->d_image, stream);

		cudaStreamSynchronize(stream);
		int framenum = frame->framenum;
		RecycleInput(std::move(frame));

		spotDetector->Completed();
		auto results = spotDetector->GetResults();
		h_indices.CopyFromDevice(results.d_indices, results.numSpots, stream);
		cudaStreamSynchronize(stream);

		for (int i = 0; i < results.numSpots; i++)
		{
			int centerPixelIndex = h_indices[i].index;
			int centerY = centerPixelIndex / imgsize.x, centerX = centerPixelIndex % imgsize.x;
			int cornerX = centerX - roiWidth / 2, cornerY = centerY - roiHeight / 2;
			int roiPixels = roiWidth * roiHeight;
			for (int z = 0; z < roiFrames; z++) {
				for (int y = 0; y < roiHeight; y++) {
					int fy = cornerY + y;
					if (fy < 0) fy = 0;
					if (fy >= imgsize.y) fy = imgsize.y - 1;
					for (int x = 0; x < roiWidth; x++) {
						int fx = cornerX + x;
						if (fx < 0) fx = 0;
						if (fx >= imgsize.x) fx = imgsize.x - 1;

						h_samples[z * roiPixels + y * roiWidth + x] = h_image[ z * imgsize.x*imgsize.y + fy* imgsize.x + fx];
					}
				}
			}

			//ShowDebugImage(roiWidth, roiHeight, 1, h_samples.data(), SPrintf("roi%d", i).c_str());

			dst->PushROI(h_samples.data(), framenum, cornerX, cornerY, h_indices[i].zplane, h_indices[i].score);
		}

		LockedAction(framesFinishedMutex, [&]() {
			framesFinished++;
		});
	}
};

class SumFrames : public ImgQueue<Frame, SummedFrame> {
public:
	int count=0;
	int sumframes;
	std::unique_ptr<SummedFrame> current;
	int framenum = 0;

	SumFrames(int2 imgsize, int sumframes, Context* ctx) : ImgQueue(imgsize,ctx), sumframes(sumframes)
	{
		current = GetNewOutputFrame();
	}

	virtual SummedFrame* AllocateNewOutputFrame() override
	{
		return new SummedFrame(imgsize, sumframes);
	}

	// Inherited via ImgQueue
	virtual void ProcessFrame(std::unique_ptr<Frame> frame) override
	{
		// Keep original as well as the summed image
		current->original.CopyFromDevice(frame->d_image.data, frame->d_image.pitch, 0, count*imgsize.y, imgsize.x, imgsize.y, stream);

		auto& sum = current->d_image;
		if (count == 0)
			sum.Swap(frame->d_image);
		else
			sum.Apply(frame->d_image, thrust::plus<float>(), stream);

		count++;

		if (count == sumframes)
		{
			current->framenum = framenum++;

			cudaStreamSynchronize(stream);
			AddToFinished(std::move(current));
			current = GetNewOutputFrame();
			count = 0;
		}
		RecycleInput(std::move(frame));
	}
};

struct SpotLocalizerQueue : public ImageProcessor
{
	class Dispatch : public ImgQueue<SummedFrame,Empty> {
	public:
		Dispatch(int2 imgsize, SpotLocalizerQueue* q, Context* ctx) : ImgQueue(imgsize, ctx), owner(q) {}
 		SpotLocalizerQueue* owner;
		int current=0;

		// Inherited via ImgQueue
		virtual void ProcessFrame(std::unique_ptr<SummedFrame> frame) override
		{
			owner->spotDetectors[current++]->AddFrame(std::move(frame));
			current = current % owner->spotDetectors.size();
		}

		virtual std::unique_ptr<SummedFrame> GetRecycledInputFrame() override
		{
			for (auto& sd : owner->spotDetectors) {
				auto f = sd->GetRecycledInputFrame();
				if (f) return std::move(f);
			}
			return std::unique_ptr<SummedFrame>();
		}
	};

	SpotLocalizerQueue(int2 imgsize, ICalibrationProcessor* calibration, ISpotDetectorFactory* detectorFactory,
		ROIQueue* queue, int numThreads)
	{
		this->target = queue;
		int sumframes = queue->ROIShape()[0];

		for (int i = 0; i < numThreads; i++)
			spotDetectors.push_back(std::make_unique<SpotDetectorImgQueue>(imgsize, queue, detectorFactory));

		// RawFrameInputQueue -> SumFramesQueue -> Dispatch -> SpotDetector
		RawFrameInputQueue::Config config{ imgsize };
		rawInputQueue = std::make_unique<RawFrameInputQueue>(config, calibration, nullptr);
		dispatchQueue = std::make_unique<Dispatch>(imgsize, this, nullptr);

		this->sumFramesQueue = std::make_unique<SumFrames>(imgsize, sumframes, nullptr);
		rawInputQueue->SetTarget(sumFramesQueue.get());
		this->sumFramesQueue->SetTarget(dispatchQueue.get());
	}

	~SpotLocalizerQueue()
	{
		// stop things before they get deleted to prevent invalid access
		if (sumFramesQueue) sumFramesQueue->Stop();
		dispatchQueue->Stop();
		rawInputQueue->Stop();
		for (auto& s : spotDetectors)
			s->Stop();
	}

	ROIQueue* target;

	std::unique_ptr<RawFrameInputQueue> rawInputQueue;
	std::unique_ptr<Dispatch> dispatchQueue;
	std::unique_ptr<SumFrames> sumFramesQueue;
	std::vector<std::unique_ptr<SpotDetectorImgQueue> > spotDetectors;

	// Inherited via ImageProcessor
	virtual void AddFrame(const uint16_t * data) override
	{
		rawInputQueue->AddHostFrame(data);
	}
	virtual void AddFrame(const float* d) override
	{
		throw std::runtime_error("Not implemented");
	}
	virtual int GetQueueLength() override
	{
		return rawInputQueue->GetQueueLength();
	}

	virtual int NumFinishedFrames() override
	{
		int total = 0;
		for (auto& sd : spotDetectors)
			total += sd->NumFinishedFrames();
		return total;
	}

	virtual bool IsIdle() override
	{
		bool idle = rawInputQueue->IsIdle() && dispatchQueue->IsIdle() && sumFramesQueue->IsIdle();
		for (auto& sd : spotDetectors)
			idle = idle && sd->IsIdle();
		return idle;
	}
};

CDLL_EXPORT ImageProcessor * SpotExtractionQueue_Create(int width, int height, ROIQueue* roilist,
	ISpotDetectorFactory* spotDetectorFactory, ICalibrationProcessor* preprocessor, 
	int numDetectionThreads, int sumframes, Context* ctx)
{
	try {
		int2 imgsize{ width,height };
		SpotLocalizerQueue* q = new SpotLocalizerQueue(imgsize, preprocessor, spotDetectorFactory,
			roilist, numDetectionThreads);
		if (ctx) q->SetContext(ctx);
		return q;
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}

//CDLL_EXPORT ImageProcessor* SelectedSpotLocalizerQueue



