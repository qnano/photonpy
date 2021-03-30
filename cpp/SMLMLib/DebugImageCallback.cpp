// API to send images back to python for debugging
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "MemLeakDebug.h"
#include "DebugImageCallback.h"
#include "CudaUtils.h"
#include <mutex>

std::mutex callbackMutex;
void(*debugImageCallback)(int w,int h, int numImages, const float* d, const char *title) = 0;


CDLL_EXPORT void SetDebugImageCallback(void(*cb)(int width, int height, int numImages, const float *data, const char *title))
{
	std::lock_guard<std::mutex> l(callbackMutex);
	debugImageCallback = cb;
}

CDLL_EXPORT void ShowDebugImage(int w, int h, int numImages, const float* data, const char *title)
{
	std::lock_guard<std::mutex> l(callbackMutex);

	if (debugImageCallback)
		debugImageCallback(w, h, numImages, data, title);
}

DLL_EXPORT void ShowDebugImage(const DeviceImage<float>& img, int numImages, const char *title)
{
	auto h_data = img.AsVector();
	ShowDebugImage(img.width, img.height/numImages, numImages, h_data.data(), title);
}


// For float2 or cuComplex (the typical usecase), only the first element is passed through
DLL_EXPORT void ShowDebugImage(const DeviceImage<float2>& img, int numImages, const char* title)
{
	auto h_data = img.AsVector();
	std::vector<float> r(h_data.size());
	for (int i = 0; i < h_data.size(); i++) 
		r[i] = h_data[i].x;
	ShowDebugImage(img.width, img.height / numImages, numImages, r.data(), title);
}



