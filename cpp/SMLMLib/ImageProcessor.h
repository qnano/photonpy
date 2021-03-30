// Image pipeline API
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "Context.h"

class ImageProcessor : public ContextObject {
public:
	virtual ~ImageProcessor() {}

	virtual void AddFrame(const float* data) {}
	virtual void AddFrame(const uint16_t* data) {}
	virtual int GetQueueLength() = 0;

	// Defaults to no output
	virtual int NumFinishedFrames() { return 0; }
	virtual int ReadFinishedFrame(float* original, float* processed) { return 0; }

	virtual bool IsIdle() = 0;
	virtual void Flush() {}
};

CDLL_EXPORT void ImgProc_Flush(ImageProcessor* q);
CDLL_EXPORT void ImgProc_AddFrameU16(ImageProcessor* q, const uint16_t* data);
CDLL_EXPORT void ImgProc_AddFrameF32(ImageProcessor* q, const float* data);
CDLL_EXPORT int ImgProc_GetQueueLength(ImageProcessor* p);
CDLL_EXPORT int ImgProc_ReadFrame(ImageProcessor* q, float* image, float* processed);
CDLL_EXPORT int ImgProc_NumFinishedFrames(ImageProcessor * q);
CDLL_EXPORT bool ImgProc_IsIdle(ImageProcessor* q);
CDLL_EXPORT void ImgProc_Destroy(ImageProcessor* q);

class ROIExtractor;
struct ExtractionROI;
class ICalibrationProcessor;

CDLL_EXPORT ROIExtractor* ROIExtractor_Create(int imgWidth, int imgHeight, ExtractionROI* rois,
	int numrois, int roiframes, int roisize, ICalibrationProcessor* imgCalibration, Context* ctx);
CDLL_EXPORT int ROIExtractor_GetResultCount(ROIExtractor *re);
CDLL_EXPORT int ROIExtractor_GetResults(ROIExtractor* re, int numrois, ExtractionROI* rois, float* framedata);

