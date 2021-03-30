// Spot detection algorithm, roughly implementing the method in
// Huang F, Schwartz SL, Byars JM, Lidke KA(2011).Simultaneous multiple emitter
// fitting for single molecule super - resolution imaging.Biomed Opt Express 2, 1377�1393.
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "CudaUtils.h"
#include "Vector.h"
#include <memory>

#include "SpotIndexListGenerator.h"

struct IndexWithScore {
	int index;
	float score;
	int zplane;
};

// Needed to index list generation
struct non_negative
{
	__host__ __device__
		bool operator()(const IndexWithScore& x)
	{
		return x.index >= 0;
	}
};


struct SpotLocationList 
{
	const IndexWithScore* d_indices; // pixel indices into frame
	int numSpots;
};


struct SpotDetectorConfig
{
	// In a nutshell:
	// detectionImage = uniform1 - uniform2
	// Selected spot locations = max(detectionImage, maxFilterSize) == detectionImage
	int uniformFilter1Size, uniformFilter2Size, maxFilterSize;

	// Roisize is used to remove ROIs near the image border
	int roisize;

	// Only spots where detectionImage > intensityThreshold are selected
	float minIntensity, maxIntensity;

	float* backgroundImage = 0;

	SpotDetectorConfig(float sigma, int roisize, float minIntensity,float maxIntensity) {
		uniformFilter1Size = int(1 + sigma * 2);
		uniformFilter2Size = uniformFilter1Size * 2;
		this->roisize = roisize;
		this->minIntensity = minIntensity;
		this->maxIntensity = maxIntensity;
		maxFilterSize = int(sigma * 3+1);
	}
};

class ISpotDetector;

class ISpotDetectorFactory
{
public:
	ISpotDetectorFactory() {}
	virtual ~ISpotDetectorFactory() {}
	virtual ISpotDetector* CreateInstance(int width, int height) = 0;
	virtual const char* GetName() = 0;
};


class ISpotDetector
{
public:
	virtual ~ISpotDetector() {}
	virtual void Detect(const DeviceImage<float>& srcImage, cudaStream_t stream) = 0;
	virtual SpotLocationList GetResults() = 0;
	virtual void Completed() = 0;
};


class SpotDetector : public ISpotDetector
{
public:
	DeviceImage<float> temp, filtered1, filtered2;
	DeviceImage<float> maxFiltered;
	DeviceArray<IndexWithScore> indices;
	DeviceImage<float> backgroundImage, srcImageMinusBg;
	SpotDetectorConfig config;
	SpotIndexListGenerator<IndexWithScore> indexListGenerator;

	SpotDetector(int2 imgsize, const SpotDetectorConfig& cfg);

	// Inherited via ISpotDetector
	SpotLocationList GetResults() override;
	void Detect(const DeviceImage<float>& srcImage, cudaStream_t stream) override;
	void Completed() override {}

	class Factory : public ISpotDetectorFactory
	{
	public:
		Factory(SpotDetectorConfig cfg) : config(cfg) {}
		ISpotDetector* CreateInstance(int width, int height);
		const char* GetName();
		SpotDetectorConfig config;
	};

};


class ISpotImageSource
{
public:
	virtual const int2 * GetROICornerPositions(int streamIndex) = 0;
	virtual const float* GetSpotImages(int& numspots, int streamIndex) = 0; // should return cuda memory
	virtual SpotLocationList GetSpotList(int streamIndex) = 0;
	virtual int ROISize() = 0;
	virtual int MaxSpotCount() = 0;
};





CDLL_EXPORT void SpotDetector_DestroyFactory(ISpotDetectorFactory* factory);
CDLL_EXPORT ISpotDetectorFactory* SpotDetector_Configure(const SpotDetectorConfig& config);

// Standalone usage of spot detector
class ICalibrationProcessor;
CDLL_EXPORT int SpotDetector_ProcessFrame(const float* frame, int width, int height, int roisize,
	int maxSpots, float* spotScores,  int* spotz, Int2* cornerPos, float* rois, ISpotDetectorFactory* sdf, ICalibrationProcessor* calib = 0);

CDLL_EXPORT void ExtractROIs(const float *frames, int width, int height, int depth, int roiX, int roiY, int roiZ, const Int3 * startpos, int numspots, float * rois);

