// A queue that stores many small 2d or 3d images 
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include <mutex>
#include <deque>
#include <vector>

#include "Context.h"
#include "Vector.h"

struct ROIInfo
{
	int id;
	float score;
	int x,y,z;
};

//  A thread safe FIFO queue for ROIs
class ROIQueue : public ContextObject
{
public:
	DLL_EXPORT ROIQueue(Int3 roishape);
	DLL_EXPORT ~ROIQueue();
	void DLL_EXPORT PushROI(const float* data, int id, int x,int y, int z, float score);
	void DLL_EXPORT PushROI(const float* data, const ROIInfo& roi);
	void DLL_EXPORT PopROIs(int count, ROIInfo* rois, float* data);
	int DLL_EXPORT Length();

	int SampleCount() { return smpcount; }
	Int3 ROIShape() { return roishape;  }

protected:
	struct ROI
	{
		ROIInfo info;
		std::vector<float> data;
	};

	int smpcount;
	Int3 roishape;
	std::mutex listMutex;
	std::deque<ROI> rois;
};

CDLL_EXPORT ROIQueue* RQ_Create(const Int3& shape, Context*ctx);
CDLL_EXPORT void RQ_Pop(ROIQueue*q, int count, ROIInfo* rois, float* data);
CDLL_EXPORT int RQ_Length(ROIQueue* q);
CDLL_EXPORT void RQ_Push(ROIQueue* q, int count ,const ROIInfo* info, const float* data);
CDLL_EXPORT void RQ_SmpShape(ROIQueue* q, Int3* shape);

