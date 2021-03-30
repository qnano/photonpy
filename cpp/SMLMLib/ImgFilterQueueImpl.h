// Image queue that applies calibration correction, gain/offset, uniform filters
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once
#include "ImgFilterQueue.h"

template<typename TInputFrame, typename TOutputFrame>
inline void ImgQueue<TInputFrame, TOutputFrame>::RecycleOutput(std::unique_ptr<TOutputFrame> frame)
{
	LockedAction(recycleMutex, [&]() {
		recycle.push_back(std::move(frame));
	});
}

template<typename TInputFrame, typename TOutputFrame>
inline void ImgQueue<TInputFrame, TOutputFrame>::AddToFinished(std::unique_ptr<TOutputFrame> frame)
{
	if (target)
	{
		target->AddFrame(std::move(frame));
	}
	else {
		LockedAction(finishedMutex, [&]() {
			finished.push_back(std::move(frame));
		});
	}
}

template<typename TInputFrame, typename TOutputFrame>
inline ImgQueue<TInputFrame, TOutputFrame>::ImgQueue(int2 imgsize, Context* ctx) : 
	ContextObject(ctx), imgsize(imgsize), source(0)
{
	cudaStreamCreate(&stream);
	processThread = std::make_unique<std::thread>([&]() {ProcessThreadMain(); });
}

template<typename TInputFrame, typename TOutputFrame>
inline bool ImgQueue<TInputFrame, TOutputFrame>::IsIdle()
{
	return LockedFunction(todoMutex, [&]() {
		return todo.empty() && !processingFrame;
	});
}

template<typename TInputFrame, typename TOutputFrame>
inline void ImgQueue<TInputFrame, TOutputFrame>::ProcessThreadMain()
{
	SetCudaDevice();

	while (!abortProcessing)
	{
		std::unique_ptr<TInputFrame> item;

		if (source)
		{
			item = source->GetFinishedFrame();
		}
		else {
			LockedAction(todoMutex, [&]() {
				if (!todo.empty()) {
					item = std::move(todo.front());
					todo.pop_front();
					processingFrame = true;
				}
			});
		}

		if (item) {
			ProcessFrame(std::move(item));
			processingFrame = false;
		}
		else {
			std::this_thread::sleep_for(std::chrono::microseconds(500));
		}
	}
}

template<typename TInputFrame, typename TOutputFrame>
inline ImgQueue<TInputFrame, TOutputFrame>::~ImgQueue()
{
	Stop();
	cudaStreamDestroy(stream);
}

template<typename TInputFrame, typename TOutputFrame>
inline std::unique_ptr<TOutputFrame> ImgQueue<TInputFrame, TOutputFrame>::GetFinishedFrame()
{
	std::unique_ptr<TOutputFrame> r;
	LockedAction(finishedMutex, [&]() {
		if (!finished.empty()) {
			r = std::move(finished.front());
			finished.pop_front();
		}
	});
	return r;
}

template<typename TInputFrame, typename TOutputFrame>
inline void ImgQueue<TInputFrame, TOutputFrame>::RecycleInput(std::unique_ptr<TInputFrame> f)
{
	LockedAction(todoRecycleMutex, [&]() {
		todoRecycle.push_back(std::move(f));
	});
}

template<typename TInputFrame, typename TOutputFrame>
inline std::unique_ptr<TInputFrame> ImgQueue<TInputFrame, TOutputFrame>::GetRecycledInputFrame()
{
	std::unique_ptr<TInputFrame> fr;
	LockedAction(todoRecycleMutex, [&]() {
		if (!todoRecycle.empty()) {
			fr = std::move(todoRecycle.front());
			todoRecycle.pop_front();
		}
	});
	return fr;
}

template<typename TInputFrame, typename TOutputFrame>
inline int ImgQueue<TInputFrame, TOutputFrame>::NumFinishedFrames()
{
	return LockedFunction(finishedMutex, [&]() {
		return (int) finished.size();
	});
}

template<typename TInputFrame, typename TOutputFrame>
inline bool ImgQueue<TInputFrame, TOutputFrame>::HasFinishedFrame()
{
	return LockedFunction(finishedMutex, [&]() {
		return finished.empty();
	});
}

template<typename TInputFrame, typename TOutputFrame>
inline int ImgQueue<TInputFrame, TOutputFrame>::GetQueueLength()
{
	return LockedFunction(todoMutex, [&]() {
		return (int)todo.size();
	});
}

template<typename TInputFrame, typename TOutputFrame>
inline void ImgQueue<TInputFrame, TOutputFrame>::AddFrame(std::unique_ptr<TInputFrame> frame)
{
	while (GetQueueLength() == maxQueueLength) {
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	LockedAction(todoMutex, [&]() {
		todo.push_back(std::move(frame));
	});
}

template<typename TInputFrame, typename TOutputFrame>
inline std::unique_ptr<TOutputFrame> ImgQueue<TInputFrame, TOutputFrame>::GetNewOutputFrame()
{
	std::unique_ptr<TOutputFrame> fr;
	if (target)
		fr = target->GetRecycledInputFrame();

	if (!fr) {
		LockedAction(recycleMutex, [&]() {
			if (!recycle.empty()) {
				fr = std::move(recycle.front());
				recycle.pop_front();
			}
		});
	}

	if (!fr)
		fr = std::unique_ptr<TOutputFrame>(AllocateNewOutputFrame());

	return std::move(fr);
}

template<typename TInputFrame, typename TOutputFrame>
inline std::unique_ptr<TInputFrame> ImgQueue<TInputFrame, TOutputFrame>::GetNewInputFrame()
{
	std::unique_ptr<TInputFrame> fr = GetRecycledInputFrame();
	if (!fr)
		fr = std::unique_ptr<TInputFrame>(AllocateNewInputFrame());
	return std::move(fr);
}

template<typename TInputFrame, typename TOutputFrame>
inline void ImgQueue<TInputFrame, TOutputFrame>::Stop()
{
	if (processThread) {
		abortProcessing = true;
		processThread->join();
		processThread.reset();
	}
}




template<typename TFrame>
int CopyToHostQueue<TFrame>::CopyOutputFrame(float * image, float * filtered)
{
	auto f = GetFinishedFrame();

	if (!f)
		return -1;

	int n = f->framenum;
	memcpy(image, f->h_image.data(), sizeof(float)*f->h_image.size());
	memcpy(filtered, f->h_filtered.data(), sizeof(float)*f->h_filtered.size());
	RecycleOutput(std::move(f));
	return n;
}

template<typename TFrame>
void CopyToHostQueue<TFrame>::ProcessFrame(std::unique_ptr<TFrame> frame) {
	auto output = GetNewOutputFrame();

	output->framenum = frame->framenum;
	frame->d_filtered.CopyToHost(output->h_filtered.data(), stream);
	frame->d_image.CopyToHost(output->h_image.data(), stream2);

	cudaStreamSynchronize(stream);
	cudaStreamSynchronize(stream2);

	//DebugPrintf("image[0]=%f, filtered[0]=%f\n", output->h_image[0], output->h_filtered[0]);

	AddToFinished(std::move(output));
	RecycleInput(std::move(frame));
}
