// Measure cuda memory use
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "CudaUtils.h"
#include "ThreadUtils.h"

void EmptyKernel(cudaStream_t s) {
	LaunchKernel(1, [=]__device__(int i) {}, 0, s);
}

CDLL_EXPORT int CudaGetNumDevices()
{
	int c;
	cudaGetDeviceCount(&c);
	return c;
}

CDLL_EXPORT bool CudaSetDevice(int index)
{
	return cudaSetDevice(index) == cudaSuccess;
}

CDLL_EXPORT bool CudaGetDeviceInfo(int index, int& numMultiprocessors, char* name, int namelen)
{
	cudaDeviceProp prop;
	if (cudaGetDeviceProperties(&prop, index) != cudaSuccess)
		return false;

	numMultiprocessors = prop.multiProcessorCount;
	strcpy_s(name, namelen, prop.name);
	return true;
}

static std::mutex pinnedMemMutex, devicePitchedMemMutex, deviceMemMutex;
static uint64_t pinnedMemAmount=0, devicePitchedMemAmount=0, deviceMemAmount = 0, devicePitchedNumAllocs=0;

CDLL_EXPORT void CudaGetMemoryUse(uint64_t& pinnedBytes, uint64_t& devicePitchedBytes, uint64_t& deviceBytes, uint64_t& pitchedNumAllocs)
{
	pinnedBytes = pinnedMemAmount;
	devicePitchedBytes = devicePitchedMemAmount;
	deviceBytes = deviceMemAmount;
	pitchedNumAllocs = devicePitchedNumAllocs;
}

int CudaMemoryCounter::AddPinnedMemory(int amount)
{
	return LockedFunction(pinnedMemMutex, [&]() {
		pinnedMemAmount += amount;
		return pinnedMemAmount;
	});
}

int CudaMemoryCounter::AddDevicePitchedMemory(int amount)
{
	return LockedFunction(devicePitchedMemMutex, [&]() {
		if (amount > 0) devicePitchedNumAllocs++;
		else devicePitchedNumAllocs--;
		devicePitchedMemAmount += amount;
		return devicePitchedMemAmount;
	});
}

int CudaMemoryCounter::AddDeviceMemory(int amount)
{
	return LockedFunction(deviceMemMutex, [&]() {
		deviceMemAmount += amount;
		return deviceMemAmount;
	});
}
