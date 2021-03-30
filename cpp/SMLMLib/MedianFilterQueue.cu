// Median filter image queue
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021


#include "palala.h"
#include "MedianFilter/MedianFilter.h"
#include "DLLMacros.h"
#include "ImageProcessor.h"
#include "ThreadUtils.h"
#include "Vector.h"



class MedianFilterImgQueue : public ImageProcessor 
{
public:
	struct Image {
		Image(const float* data, int w, int h) : image(data,data+w*h), result(w*h) {
		}
		std::vector<float> image;
		std::vector<float> result;
	};

	std::unique_ptr<std::thread> processThread;
	std::vector<siy::median_filter<float> > medianFilters;

	std::mutex todoLock, finishedLock;
	std::list < std::unique_ptr<Image> > todo, finished;

	volatile bool abortProcessing, processingFrame = false;
	int width,height,window;
	bool flushed = false;

	MedianFilterImgQueue(int w, int h, int frames) : width(w), height(h), window(frames) {
		abortProcessing = false;
		processThread = std::make_unique<std::thread>([&]() {ProcessThreadMain(); });

		medianFilters.reserve(w * h);
		for (int i = 0; i < w * h; i++) {
			medianFilters.push_back({ frames });
		}
	}

	~MedianFilterImgQueue()
	{
		abortProcessing = true;
		processThread->join();
		processThread.reset();
	}

	virtual void AddFrame(const float* data) {
		std::unique_ptr<Image> img = std::make_unique<Image>(data, width, height);
		LockedAction(todoLock, [&]() {
			todo.push_back(std::move(img));
		});
	}

	virtual int GetQueueLength() {
		return LockedFunction(todoLock, [&]() {
			return todo.size() + processingFrame ? 1 : 0;
			});
	}

	virtual int ReadFinishedFrame(float* original, float* processed) {
		return LockedFunction(finishedLock, [&]() {
			if (finished.empty())
				return 0;

			std::unique_ptr<Image> img = std::move(finished.back());
			finished.pop_back();

			for (int i = 0; i < width * height; i++) {
				original[i] = img->image[i];
				processed[i] = img->result[i];
			}
			return 1;
		});
	}

	virtual int NumFinishedFrames() { 
		return LockedFunction(finishedLock, [&]() {
			return finished.size();
			});
	}

	virtual bool IsIdle()
	{
		return GetQueueLength() == 0;
	}

	void ProcessThreadMain() {
		while (!abortProcessing)
		{
			std::unique_ptr<Image> item;
			std::vector<float> result(width * height);
			//std::list < std::unique_ptr< Image > > inMedian;
			int done = 0;

			LockedAction(todoLock, [&]() {
				if (!todo.empty()) {
					item = std::move(todo.front());
					todo.pop_front();
					processingFrame = true;
				}
			});

			if (item) {
				const float* data = item->image.data();
				ParallelFor(height, [&](int y) {
					for (int x = 0; x < width; x++) {
						item->result[y * width + x] = medianFilters[y * width + x].filter(data[y * width + x]);
					}
				});

				LockedAction(finishedLock, [&]() {
					finished.push_front(std::move(item));
				});
				LockedAction(todoLock, [&]() {
					processingFrame = false;
				});
			}
			else {
				std::this_thread::sleep_for(std::chrono::microseconds(500));
			}
		}

	}

};



CDLL_EXPORT MedianFilterImgQueue* MedianFilterQueue_CreateF32(int w, int h, int nframes, Context* ctx)
{
	auto *r = new MedianFilterImgQueue(w, h, nframes);
	if (ctx)
		r->SetContext(ctx);
	return r;
}

