// A queue that stores many small 2d or 3d images 
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "ROIQueue.h"
#include "ThreadUtils.h"

ROIQueue::ROIQueue(Int3 roishape) : roishape(roishape), smpcount(roishape[0]*roishape[1]*roishape[2])
{}

ROIQueue:: ~ROIQueue()
{}

void ROIQueue::PushROI(const float* data, int id, int x, int y, int z, float score)
{
	ROIInfo info;
	info.id = id;
	info.x = x;
	info.y = y;
	info.z = z;
	info.score = score;
	PushROI(data, info);
}

void ROIQueue::PushROI(const float* data, const ROIInfo& roi)
{
	std::vector<float> d(data, data + smpcount);
	LockedAction(listMutex, [&]() {
		rois.push_back({ roi, std::move(d) });
	});
}

void ROIQueue::PopROIs(int count, ROIInfo* dst, float* data)
{
	std::vector<ROI> tocopy;

	if (count > Length())
		return;

	tocopy.reserve(count);
	LockedAction(listMutex, [&]() {
		for (int i = 0; i < count; i++) {
			tocopy.push_back(std::move(rois.front()));
			rois.pop_front();
		}
	});

	for (int i = 0; i < tocopy.size(); i++) {
		auto& src = tocopy[count - 1 - i];
		dst[i] = src.info;
		for (int j = 0; j < smpcount; j++)
			data[i * smpcount + j] = src.data[j];
	}
}

int ROIQueue::Length()
{
	return LockedFunction(listMutex, [&]() {
		return rois.size();
	});
}

CDLL_EXPORT ROIQueue* RQ_Create(const Int3& shape, Context* ctx)
{
	ROIQueue* q=new ROIQueue(shape);
	q->SetContext(ctx);
	return q;
}

CDLL_EXPORT void RQ_Pop(ROIQueue* q, int count, ROIInfo* rois, float* data)
{
	q->PopROIs(count, rois, data);
}

CDLL_EXPORT int RQ_Length(ROIQueue* q)
{
	return q->Length();
}

CDLL_EXPORT void RQ_Push(ROIQueue* q, int count, const ROIInfo* info, const float* data)
{
	for(int i=0;i<count;i++)
		q->PushROI(&data[q->SampleCount()*i], info[i]);
}


CDLL_EXPORT void RQ_SmpShape(ROIQueue* q, Int3* shape)
{
	*shape = q->ROIShape();
}

CDLL_EXPORT void RQ_Delete(ROIQueue* q)
{
	delete q;
}
