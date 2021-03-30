// API to send images back to python for debugging
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once
#include "DLLMacros.h"
#include "CudaUtils.h"

CDLL_EXPORT void SetDebugImageCallback(void(*cb)(int width,int height, int numImg, const float* data, const char* title));
CDLL_EXPORT void ShowDebugImage(int w, int h, int numImg, const float* data, const char *title);

template<typename T>
class DeviceImage;

// Actual image height is img.h / numImg
DLL_EXPORT void ShowDebugImage(const DeviceImage<float>& img, int numImg, const char *title);
DLL_EXPORT void ShowDebugImage(const DeviceImage<float2>& img, int numImg, const char* title);
