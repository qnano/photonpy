// Localization spot detection using PSF correlation
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "PSFCorrelationSpotDetector.h"
#include "DebugImageCallback.h"
#include "ImageFilters.h"
#include <thrust/functional.h>
#include <cub/cub.cuh>
#include "SpotIndexListGenerator.h"


CDLL_EXPORT ISpotDetectorFactory* PSFCorrelationSpotDetector_Configure(const float* bgImage, int imageWidth, int imageHeight, const float* psfstack, 
			int roisize, int depth, int maxFilterSizeXY, float minPhotons, int uniformBgFilterSize, int debugMode)
{
	PSFCorrelationSpotDetector::Config cfg;
	if (bgImage)
		cfg.bgImage.assign(bgImage, bgImage + (imageWidth * imageHeight));
	else
		cfg.bgImage.assign(imageWidth * imageHeight, 0.0f);

	cfg.depth = depth;
	cfg.imgHeight = imageHeight;
	cfg.imgWidth = imageWidth;
	cfg.maxFilterWindowSizeXY = maxFilterSizeXY;
	cfg.minPhotons = minPhotons;
	cfg.roisize = roisize;
	cfg.debugMode = debugMode != 0;
	cfg.backgroundFilterSize = uniformBgFilterSize;
	cfg.psf.assign(psfstack, psfstack + (roisize * roisize * depth));

	return new PSFCorrelationSpotDetector::Factory(cfg);
}


ISpotDetector* PSFCorrelationSpotDetector::Factory::CreateInstance(int width, int height)
{
	if (width != config.imgWidth || height != config.imgHeight)
		return 0;

	return new PSFCorrelationSpotDetector(config);
}


const char* PSFCorrelationSpotDetector::Factory::GetName()
{
	return "PSF Correlation Spot Detector";
}



struct one_minus
{
	__host__ __device__ cuComplex operator()(float2 a) const
	{
		return make_cuComplex(1.0f - a.x, -a.y);
	}
};


PSFCorrelationSpotDetector::PSFCorrelationSpotDetector(Config config) : 

	backgroundImage(config.bgImage.data(), config.imgWidth,config.imgHeight) ,
	f_img(config.imgWidth,config.imgHeight),
	fft_forw(1, config.imgWidth, config.imgHeight, srcImageMinusBg.PitchInPixels(), f_img.PitchInPixels(), CUFFT_C2C),
	fft_inv(1, config.imgWidth, config.imgHeight, f_img.PitchInPixels(), srcImageMinusBg.PitchInPixels(), CUFFT_C2C),
	srcImageMinusBg(config.imgWidth,config.imgHeight), psfPeakIntensities(config.depth),
	config(config)
{
	int w = config.imgWidth, h = config.imgHeight;
	int depth = config.depth;
	int roisize = config.roisize;
	convStack.Init(w, h * depth);
	convStackMaxFilter.Init(w, h * depth);
	maxFilterTemp.Init(w, h * depth);
	temp.Init(w, h);
	f_conv.Init(w, h);
	f_psf.resize(depth);

	float sigma = config.backgroundFilterSize;
	const float sqrt2Pi = sqrt(2 * 3.141593f);
	std::vector<cuComplex> bgfilter(w * h);

	for (int y = 0; y < h; y++) {
		for (int x = 0; x < w; x++) {
			float dx = x < w / 2 ? x : w - x;
			float dy = y < h / 2 ? y : h - y;
			bgfilter[y * w + x] = make_cuComplex(exp(-0.5f * dx*dx * dy*dy / sigma * sigma) / (sigma * sqrt2Pi), 0.0f);
		}
	}

	f_backgroundSmoothKernel.Init(w, h);
	temp.CopyFromHost(bgfilter.data());
	fft_forw.Transform(temp, f_backgroundSmoothKernel, true);

	ApplyUnaryOperator2D(f_backgroundSmoothKernel, one_minus());

	std::vector<cuComplex> psfext(w * h);
	for (int z = 0; z < depth; z++) {
		
		const float* psf = &config.psf[z * roisize * roisize];

		// normalize
		float sum = 0.0f;
		for (int i = 0; i < roisize * roisize; i++)
			sum += psf[i];

		for (size_t i = 0; i < psfext.size(); i++)
			psfext[i] = {};

		float f = 1.0f / sum;
		for (int y = 0; y < roisize; y++) {
			for (int x = 0; x < roisize; x++) {
				int dstx = (x - roisize / 2) % w;
				int dsty = (y - roisize / 2) % h;
				if (dstx < 0) dstx += w;
				if (dsty < 0) dsty += h;
				psfext[dsty * w + dstx] = make_cuComplex(f*psf[y * roisize + x], 0.0f);
			}
		}

		psfPeakIntensities[z] = psfext[0].x / sum;
		//DebugPrintf("Peak[%d]: %f, sum=%f\n", z, psfext[0].x, sum);

		DeviceImage<cuComplex> d_psf(psfext.data(), w, h);
		f_psf[z].Init(w, h);
		fft_forw.Transform(d_psf, f_psf[z], true);
	}

	int numpixels = w * h;
	indices.Init(numpixels);

	indexListGenerator = std::make_unique<SpotIndexListGenerator< IndexWithScore>>();
	indexListGenerator->Init(w, h);

	cudaStreamSynchronize(0);
}

SpotLocationList PSFCorrelationSpotDetector::GetResults()
{
	SpotLocationList list;
	list.numSpots = indexListGenerator->numspots[0];
	list.d_indices = indexListGenerator->selectedIndices.ptr();
	return list;
}

struct complex_multiply
{
	__host__ __device__ cuComplex operator()(cuComplex a, cuComplex b) const
	{
		return cuCmulf(a, b);
	}
};


struct subtract_to_complex
{
	__host__ __device__ cuComplex operator()(float a, float b) const
	{
		return make_cuComplex(a - b, 0.0f);
	}
};


namespace std
{
	template<> struct hash<cufftComplex>
	{
		std::size_t operator()(cufftComplex const& s) const noexcept
		{
			std::size_t h1 = std::hash<float>{}(s.x);
			std::size_t h2 = std::hash<float>{}(s.y);
			return h1 ^ (h2 << 1);
		}
	};
}

template<typename T>
static size_t hash(const std::vector<T>& c) {
	size_t h = 0;
	for (auto& v : c)
		h = h ^ (std::hash<T>{}(v) << 1);
	return h;
}


void PSFCorrelationSpotDetector::Detect(const DeviceImage<float>& srcImage, cudaStream_t stream)
{
	ApplyBinaryOperator2D(srcImage, backgroundImage, srcImageMinusBg, subtract_to_complex(), stream);
	fft_forw.Transform(srcImageMinusBg, f_img, true, stream);

	// F-1{psf * F{I-bg}}
	for (int z = 0; z < config.depth; z++)
	{
		// Multiply with PSF in fourier-domain
		ApplyBinaryOperator2D(f_img, f_psf[z], f_conv, complex_multiply(), stream);
		fft_inv.Transform(f_conv, temp, false, stream); // temp=inverse(f_conv)*(1-bgfilter)
		float f = 1.0f / (f_img.width * f_img.height);// *psfPeakIntensities[z]);

		// copy the real part to convstack
		cuComplex* temp_ = temp.data; int temp_pitch = temp.PitchInPixels(), dst_pitch = convStack.PitchInPixels();
		float* dst = convStack.Index(0, z * config.imgHeight);
		LaunchKernel(temp.height, temp.width, [=]__device__(int y, int x) {
			dst[y * dst_pitch + x] = temp_[y * temp_pitch + x].x *f;
		}, 0, stream);
	}

	convStackMaxFilter.Clear(stream);
	ComparisonFilter2DStack(convStack, convStackMaxFilter, maxFilterTemp, config.maxFilterWindowSizeXY, config.depth, thrust::maximum<float>(), stream);

	if (config.debugMode)
	{
		ShowDebugImage(convStack, config.depth, "convstack");
		ShowDebugImage(convStackMaxFilter, config.depth, "convstackmax");

		auto h_maxfilter = convStackMaxFilter.AsVector();
		auto max = *std::max_element(h_maxfilter.begin(), h_maxfilter.end());
		DebugPrintf("max(convstackmax)=%f\n", max);
	}

	// convert to indices
	float* d_convstack = convStack.data;
	float* d_max = convStackMaxFilter.data;
	int pitch = convStack.PitchInPixels();
	IndexWithScore *d_indices = indices.ptr();
	int w = config.imgWidth, h = config.imgHeight, depth = config.depth;
	int roisize = config.roisize;
	float threshold = config.minPhotons;
	bool debugMode = config.debugMode;

	LaunchKernel(w, h, [=]__device__(int x, int y) {
		// run maxfilter in z direction
		float maxVal = d_max[y * pitch + x];
		for (int z = 1; z < depth; z++) {
			float v = d_max[(y + z * h) * pitch + x];
			maxVal = v > maxVal ? v : maxVal;
		}

		int maxZ = -1;
		for (int z = 0; z < depth; z++) {
			float a = d_convstack[(y + z*h)*pitch + x];

			if (a == maxVal)
				maxZ = z;
		}
		bool isMax = maxZ >= 0 && maxVal > threshold
			&& x > roisize / 2
			&& y > roisize / 2
			&& x + (roisize - roisize / 2) < w - 1
			&& y + (roisize - roisize / 2) < h - 1;

		if (debugMode) {
			for (int z = 0; z < depth; z++)
				d_convstack[(y + z * h) * pitch + x] = maxZ == z && maxVal > threshold;
		}
		if (isMax)
			d_indices[y*w + x] = { y * w + x, maxVal, maxZ };
		else
			d_indices[y*w + x] = { -1, 0.0f, 0 };
	}, 0, stream);

	frame++;

	if (config.debugMode) {
		auto h_idx = indices.ToVector();
		int count = 0;
		std::vector<float> idxmap(h_idx.size());
		for (int i = 0; i < h_idx.size(); i++) {
			if (h_idx[i].index >= 0) {
				idxmap[i] = h_idx[i].zplane + 1;
				count++;
			}
			else
				idxmap[i] = 0;
		}
		size_t h_src = hash(srcImage.AsVector());
		size_t h_srcImageMinusBg = hash(srcImageMinusBg.AsVector());
		size_t h_f_img = hash(f_img.AsVector());
		size_t h_convStackMaxFilter = hash(convStackMaxFilter.AsVector());

		DebugPrintf("sd frame %d. src hash %d. src-bg hash %d, f_img hash %d, convStackMaxFilter hash %d. Spot count: %d\n", 
			frame, h_src, h_srcImageMinusBg, h_f_img, h_convStackMaxFilter, count);
		ShowDebugImage(w, h, 1, idxmap.data(), "indexmap");
		ShowDebugImage(convStack, config.depth, "maxpos");
	}
	indexListGenerator->Compute(indices, stream);
}
