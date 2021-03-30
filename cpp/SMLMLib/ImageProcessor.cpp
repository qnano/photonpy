// Image pipeline API
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "ImageProcessor.h"
#include "ImgFilterQueue.h"
#include "ImgFilterQueueImpl.h"
#include "Estimators/EstimationQueue.h"

#include <deque>

CDLL_EXPORT void ImgProc_Flush(ImageProcessor* q)
{
	q->Flush();
}

CDLL_EXPORT void ImgProc_AddFrameU16(ImageProcessor* q, const uint16_t* data)
{
	q->AddFrame(data);
}

CDLL_EXPORT void ImgProc_AddFrameF32(ImageProcessor* q, const float* data)
{
	q->AddFrame(data);

}

CDLL_EXPORT int ImgProc_GetQueueLength(ImageProcessor* p)
{
	return p->GetQueueLength();
}

CDLL_EXPORT int ImgProc_ReadFrame(ImageProcessor* q, float* image, float* processed)
{
	return q->ReadFinishedFrame(image, processed);
}

CDLL_EXPORT int ImgProc_NumFinishedFrames(ImageProcessor * q)
{
	return q->NumFinishedFrames();
}

CDLL_EXPORT bool ImgProc_IsIdle(ImageProcessor* q)
{
	return q->IsIdle();
}

CDLL_EXPORT void ImgProc_Destroy(ImageProcessor* q)
{
	delete q;
}



struct Empty {
	Empty(int2) {}
};



struct ExtractionROI
{
	Int2 cornerpos;
	int startframe, numframes;
};

class ROIExtractor : public ImageProcessor {
public:
	struct SpotsPerFrame {
		std::vector<ExtractionROI> starting;
	};

	struct Spot {
		Spot(ExtractionROI roi, int roisize) :
			roi(roi), framedata(roisize*roisize*roi.numframes) {}

		ExtractionROI roi;
		std::vector<float> framedata;
	};

	class ExtractorQueue : public ImgQueue<Frame, Empty> {
	public:
		PinnedArray<float> h_image;
		ExtractorQueue(int2 imgsize, Context* ctx, std::vector<SpotsPerFrame> spotsPerFrame, int roisize, int roiframes) :
			ImgQueue(imgsize, ctx), h_image(imgsize.x*imgsize.y), roisize(roisize), roiframes(roiframes), spotsPerFrame(spotsPerFrame) {
		}
		~ExtractorQueue()
		{
			DeleteAll(results);
			DeleteAll(active);
		}

		int roisize, roiframes;
		std::vector<SpotsPerFrame> spotsPerFrame;
		std::list<Spot*> active;

		std::deque<Spot*> results;
		std::mutex resultsMutex;
		int numFinished = 0;

		int GetResultCount() 
		{
			return LockedFunction(resultsMutex, [&]() { return (int)results.size(); });
		}

		int GetResults(int nspots, ExtractionROI* rois, float* framedata)
		{
			std::vector<Spot*> tocopy;

			LockedAction(resultsMutex, [&]() {
				for (int i = 0; i < nspots; i++) {
					if (results.empty())
						break;
					tocopy.push_back(std::move(results.front()));
					results.pop_front();
				}
			});

			for (int i = 0; i < nspots; i++)
			{
				rois[i] = tocopy[i]->roi;
				float* dst = &framedata[roiframes*roisize*roisize*i];

				int nframes = std::min(rois[i].numframes, roiframes);

				for (int f = 0; f < nframes*roisize*roisize; f++)
					dst[f] = tocopy[i]->framedata[f];
			}
			return (int)tocopy.size();
		}

		// Inherited via ImgQueue
		virtual void ProcessFrame(std::unique_ptr<Frame> frame) override
		{
			frame->d_image.CopyToHost(h_image.data(), stream);
			int framenum = frame->framenum;
			RecycleInput(std::move(frame));
			cudaStreamSynchronize(stream);

			if (framenum < spotsPerFrame.size())
			{
				for (auto roi : spotsPerFrame[framenum].starting)
					active.push_back(new Spot(roi, roisize));
			}

			for (auto &s : active)
			{
				// copy pixels
				int frame = framenum - s->roi.startframe;
				for (int y = 0; y < roisize; y++)
					for (int x = 0; x < roisize; x++)
					{
						int ry = y + s->roi.cornerpos[1];
						int rx = x + s->roi.cornerpos[0];
						s->framedata[roisize*roisize*frame + roisize * y + x] = h_image[ry*imgsize.x + rx];
					}
			}

			std::list< Spot* > next;
			LockedAction(resultsMutex, [&]() {
				for (auto &s : active) {
					if (framenum < s->roi.startframe + s->roi.numframes - 1)
						next.push_back(std::move(s));
					else
						results.push_back(std::move(s));
				}
				numFinished++;
			});

			active = next;

		}
	};



	// Inherited via ImageProcessor
	virtual void AddFrame(const uint16_t * data) override
	{
		rawInputQueue->AddHostFrame(data);
	}
	virtual int GetQueueLength() override
	{
		return rawInputQueue->GetQueueLength();
	}

	virtual int NumFinishedFrames() override
	{
		int total = 0;
		return total;
	}

	virtual bool IsIdle() override
	{
		return rawInputQueue->IsIdle() && extractorQueue->IsIdle();
	}

	ROIExtractor(int2 imgsize, ExtractionROI* rois, int numrois, int roiframes, int roisize, ICalibrationProcessor* calibration, Context* ctx)
	{
		// count frames
		int lastframe = -1;
		for (int i = 0; i < numrois; i++)
		{
			int f = rois[i].numframes + rois[i].startframe;
			if (f > lastframe) lastframe = f;
		}
		std::vector<SpotsPerFrame> perframe(lastframe+1);
		for (int i = 0; i < numrois; i++)
			perframe[rois[i].startframe].starting.push_back(rois[i]);

		RawFrameInputQueue::Config config{ imgsize };
		rawInputQueue = std::make_unique<RawFrameInputQueue>(config, calibration, ctx);
		extractorQueue = std::make_unique<ExtractorQueue>(imgsize, ctx, std::move(perframe),roisize, roiframes);
		rawInputQueue->SetTarget(extractorQueue.get());
	}

	std::unique_ptr<ExtractorQueue> extractorQueue;
	std::unique_ptr<RawFrameInputQueue> rawInputQueue;
};

CDLL_EXPORT ROIExtractor* ROIExtractor_Create(int imgWidth, int imgHeight, ExtractionROI* rois,
	int numrois, int roiframes, int roisize, ICalibrationProcessor* imgCalibration, Context* ctx)
{
	auto* e = new ROIExtractor({ imgWidth,imgHeight }, rois, numrois, roiframes, roisize, imgCalibration, ctx);
	return e;
}

CDLL_EXPORT int ROIExtractor_GetResultCount(ROIExtractor *re) {
	return re->extractorQueue->GetResultCount();
}

CDLL_EXPORT int ROIExtractor_GetResults(ROIExtractor* re, int numrois, ExtractionROI* rois, float* framedata)
{
	return re->extractorQueue->GetResults(numrois, rois, framedata);
}
