// Localization spot detection using PSF correlation
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "SpotDetector.h"
#include "FFT.h"


template<typename TIndex>
class SpotIndexListGenerator;


// Spot detection for 3D SR, by performing a convolution with the 3D Estimator
class PSFCorrelationSpotDetector : public ISpotDetector
{
public:
	class Config
	{
	public:
		int imgWidth, imgHeight, maxFilterWindowSizeXY, backgroundFilterSize;
		std::vector<float> bgImage;
		int depth, roisize;
		std::vector<float> psf;
		float minPhotons; // threshold
		bool debugMode;
	};

	PSFCorrelationSpotDetector(Config config);
	Config config;
	DeviceArray<IndexWithScore> indices;
	std::unique_ptr<SpotIndexListGenerator<IndexWithScore>> indexListGenerator;

	std::vector <float> psfPeakIntensities; // [z]
	std::vector<DeviceImage<cufftComplex>> f_psf; // [z, [imgHeight, imgWidth]]
	DeviceImage<cufftComplex> f_img; // Fourier transform of srcImageMinusBg
	DeviceImage<cufftComplex> f_conv;// Temporary complex valued output
	DeviceImage<float> convStack; // The resulting stack of convolutions 
	DeviceImage<float> convStackMaxFilter, maxFilterTemp;
	DeviceImage<cufftComplex> srcImageMinusBg, temp, f_backgroundSmoothKernel;
	DeviceImage<float> backgroundImage;
	FFTPlan2D fft_forw, fft_inv;
	int frame = 0; // just for debugging

	// Inherited via ISpotDetector
	SpotLocationList GetResults() override;
	void Detect(const DeviceImage<float>& srcImage, cudaStream_t stream) override;
	void Completed() override {}

	class Factory : public ISpotDetectorFactory
	{
	public:
		Factory(Config cfg) : config(cfg) {}
		ISpotDetector* CreateInstance(int width, int height);
		const char* GetName();
		Config config;
	};
};

CDLL_EXPORT ISpotDetectorFactory* PSFCorrelationSpotDetector_Configure(const float* bgImage, int imageWidth, int imageHeight,
										const float* psfstack, int roisize, int depth, int maxFilterSizeXY, float minPhotons, int uniformBgFilterSize, int debugMode);

