// Image queue that applies calibration correction, gain/offset, uniform filters
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once
#include <thread>
#include <list>
#include <mutex>
#include <unordered_map>
#include <deque>
#include "CudaUtils.h"
#include "Context.h"
#include "ThreadUtils.h"
#include "Vector.h"

template<typename T>
class IFrameSource {
public:
	virtual bool HasFinishedFrame() = 0;
	virtual std::unique_ptr<T> GetFinishedFrame() = 0;
	virtual void RecycleOutput(std::unique_ptr<T> frame) = 0;
};

template<typename T>
class IFrameSink {
public:
	virtual std::unique_ptr<T> GetRecycledInputFrame() = 0;
	virtual void AddFrame(std::unique_ptr<T> frame) = 0;
};

template<typename TInputFrame, typename TOutputFrame>
class ImgQueue : public ContextObject, public IFrameSource<TOutputFrame>, public IFrameSink<TInputFrame> {
public:
	virtual void ProcessFrame(std::unique_ptr<TInputFrame> frame) = 0;
	void AddToFinished(std::unique_ptr<TOutputFrame> frame);
	ImgQueue(int2 imgsize, Context* ctx);
	~ImgQueue();
	std::unique_ptr<TOutputFrame> GetFinishedFrame();
	void RecycleInput(std::unique_ptr<TInputFrame> f);
	std::unique_ptr<TInputFrame> GetRecycledInputFrame() override; // return null if none
	void RecycleOutput(std::unique_ptr<TOutputFrame> frame) override;
	int NumFinishedFrames();
	bool HasFinishedFrame() override;
	int GetQueueLength();
	void AddFrame(std::unique_ptr<TInputFrame> frame);
	virtual std::unique_ptr<TOutputFrame> GetNewOutputFrame();
	virtual std::unique_ptr<TInputFrame> GetNewInputFrame();

	virtual TOutputFrame* AllocateNewOutputFrame()
	{
		return new TOutputFrame(imgsize);
	}
	virtual TInputFrame* AllocateNewInputFrame()
	{
		return new TInputFrame(imgsize);
	}

	void Stop();
	void SetSource(IFrameSource<TInputFrame>* src) { source = src; }
	void SetTarget(IFrameSink<TOutputFrame>* dst) { target = dst; }

	bool IsIdle();

protected:
	void ProcessThreadMain();

	int2 imgsize;
	IFrameSink<TOutputFrame>* target = 0;
	IFrameSource<TInputFrame>* source = 0;
	std::unique_ptr<std::thread> processThread;
	std::atomic<bool> processingFrame;

	std::deque<std::unique_ptr<TInputFrame>> todo;
	std::mutex todoMutex;

	std::list<std::unique_ptr<TInputFrame>> todoRecycle;
	std::mutex todoRecycleMutex;

	// finished and read out
	std::list<std::unique_ptr<TOutputFrame>> recycle;
	std::mutex recycleMutex;

	// finished but not read-out
	std::list<std::unique_ptr<TOutputFrame>> finished;
	std::mutex finishedMutex;

	int maxQueueLength = 12;

	volatile bool abortProcessing = false;

	cudaStream_t stream;
};


struct RawInputFrame {
	RawInputFrame(int2 imgsize) : 
		rawimg(imgsize.x*imgsize.y), 
		rawimgPhotons(imgsize.x*imgsize.y), 
		bgimg(imgsize.x*imgsize.y), framenum(0) {}
	int framenum;
	PinnedArray<uint16_t> rawimg;
	PinnedArray<float> rawimgPhotons;
	PinnedArray<float> bgimg;
};


struct Frame {
	Frame(int2 imgsize) : d_image(imgsize) {}
	int framenum = 0;
	DeviceImage<float> d_image; // after calib
};

struct FilteredFrame : public Frame {
	FilteredFrame(int2 imgsize) : Frame(imgsize), d_filtered(imgsize) {}
	DeviceImage<float> d_filtered;
};

class ICalibrationProcessor;

class RawFrameInputQueue : public ImgQueue<RawInputFrame,Frame> {
public:
	struct Config
	{
		int2 imageSize;
	};
	Config config;
	ICalibrationProcessor* calib;
	std::atomic<int> frameNumber;

	RawFrameInputQueue(Config config, ICalibrationProcessor* calib, Context*ctx);
	void AddHostFrame(const uint16_t* data);

	// Inherited via ImgQueue
	virtual void ProcessFrame(std::unique_ptr<RawInputFrame> frame) override;
};

struct HostMemoryFrame 
{
	HostMemoryFrame(int2 imgsize) : h_image(imgsize.x*imgsize.y), h_filtered(imgsize.x*imgsize.y) {}
	int framenum = 0;
	PinnedArray<float> h_image, h_filtered;
};


class RemoveBackgroundFilterQueue : public ImgQueue<Frame, FilteredFrame> {
public:
	DeviceImage<float> d_xyfiltered1, d_xyfiltered2, temp1;
	Int2 uniformFilterSize;

	RemoveBackgroundFilterQueue(int2 imgsize, Int2 filterSize, Context*ctx);
	virtual void ProcessFrame(std::unique_ptr<Frame> frame) override;
};




template<typename TFrame>
class CopyToHostQueue : public ImgQueue<TFrame, HostMemoryFrame> {
public:
	typedef ImgQueue<TFrame, HostMemoryFrame> base;
	CopyToHostQueue(int2 imgsize, Context*ctx) : base(imgsize, ctx) {
		ThrowIfCUDAError(cudaStreamCreate(&stream2));
	}
	~CopyToHostQueue() {
		cudaStreamDestroy(stream2);
	}
	int CopyOutputFrame(float * image, float * filtered);
	virtual void ProcessFrame(std::unique_ptr<TFrame> frame) override;
protected:
	cudaStream_t stream2;
};

class ROIQueue;
class ISpotDetectorFactory;
class ImageProcessor;

CDLL_EXPORT ImageProcessor* SpotExtractionQueue_Create(int width, int height, ROIQueue* roilist,
	ISpotDetectorFactory* spotDetectorFactory, ICalibrationProcessor* preprocessor,
	int numDetectionThreads, int sumframes, Context* ctx);
