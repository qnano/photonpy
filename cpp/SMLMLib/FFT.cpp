// CUDA FFT wrapper
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021

#include "FFT.h"
#include "DebugImageCallback.h"

const char *cufftErrorString(cufftResult_t r)
{
	switch (r) {
	case CUFFT_SUCCESS: return "success";
	case CUFFT_INVALID_PLAN: return "invalid plan";
	case CUFFT_ALLOC_FAILED: return "alloc failed";
	case CUFFT_INVALID_TYPE: return "invalid type";
	case CUFFT_INVALID_VALUE: return "invalid value";
	case CUFFT_INTERNAL_ERROR: return "internal error";
	case CUFFT_EXEC_FAILED: return "exec failed";
	case CUFFT_SETUP_FAILED: return "setup failed";
	case CUFFT_INVALID_SIZE: return "invalid size";
	case CUFFT_UNALIGNED_DATA: return "unaligned data";
	default: return "unknown error";
	}
}

static void ThrowIfError(cufftResult_t r)
{
	if (r != CUFFT_SUCCESS)
		throw std::runtime_error(SPrintf("CUFFT Error: %s", cufftErrorString(r)));
}

FFTPlan2D::FFTPlan2D(int count, int width, int height, int srcPitchInElems, int dstPitchInElems, cufftType type) : 
	srcPitchInElems(srcPitchInElems), dstPitchInElems(dstPitchInElems)
{
	int n[2] = { height,width };
	int rank = 2;
	int inembed[2] = { 0, srcPitchInElems }; // inembed[0] and onembed[0] are ignored
	int idist = height * srcPitchInElems; // size of a single complex image
	int istride = 1;
	int onembed[2] = { 0, dstPitchInElems};
	int ostride = 1;
	int odist = height * dstPitchInElems;

	// cuFFT docs:
	//input[b * idist + (x * inembed[1] + y) * istride]
	//output[b * odist + (x * onembed[1] + y) * ostride]

	cufftResult result = cufftPlanMany(&handle, rank, n, inembed, istride, idist, onembed, ostride, odist, type, count);
	if (result != CUFFT_SUCCESS)
		throw std::runtime_error(SPrintf("cufftPlanMany failed: %s", cufftErrorString(result)));

	size_t workAreaSize;
	cufftEstimateMany(rank, n, inembed, istride, idist, onembed, ostride, odist, type, count, &workAreaSize);
#ifdef _DEBUG
	DebugPrintf("work area size for %d x %d x %d fft: %d bytes\n", count, height, width, workAreaSize);
#endif
}

FFTPlan2D::~FFTPlan2D()
{
	if (handle != 0)
		cufftDestroy(handle);
	handle = 0;
}

void FFTPlan2D::Transform(const cufftComplex* src, cufftComplex* dst, bool forward, cudaStream_t stream)
{
	ThrowIfError(cufftSetStream(handle, stream));
	ThrowIfError(cufftExecC2C(handle, (cufftComplex*)src, dst, forward ? CUFFT_FORWARD : CUFFT_INVERSE));
}

/*
void FFTPlan2D::Transform(const float* src, cufftComplex* dst, cudaStream_t stream)
{
	ThrowIfError(cufftSetStream(handle, stream));
	ThrowIfError(cufftExecR2C(handle, (float*)src, dst));
}

void FFTPlan2D::Transform(const cufftComplex* src, float* dst, cudaStream_t stream)
{
	ThrowIfError(cufftSetStream(handle, stream));
	ThrowIfError(cufftExecC2R(handle, (cufftComplex*)src, dst));
}


void FFTPlan2D::Transform(const DeviceImage<cufftComplex>& src, DeviceImage<float>& dst, cudaStream_t stream)
{
	ThrowIfError(cufftSetStream(handle, stream));
	ThrowIfError(cufftExecC2R(handle, src.data, dst.data));
}
void FFTPlan2D::Transform(const DeviceImage<float>& src, DeviceImage<cufftComplex>& dst, cudaStream_t stream)
{
	ThrowIfError(cufftSetStream(handle, stream));
	ThrowIfError(cufftExecR2C(handle, src.data, dst.data));
}
*/
void FFTPlan2D::Transform(const DeviceImage<cufftComplex>& src, DeviceImage<cufftComplex>& dst, bool forward, cudaStream_t stream)
{
	if(src.PitchInPixels() != srcPitchInElems)
		throw std::runtime_error(SPrintf("Passing invalid pitch to FFT (%d should be %d)", src.PitchInPixels(), srcPitchInElems));
	if (dst.PitchInPixels() != dstPitchInElems)
		throw std::runtime_error(SPrintf("Passing invalid pitch to FFT (%d should be %d)", dst.PitchInPixels(), dstPitchInElems));
	ThrowIfError(cufftSetStream(handle, stream));
	ThrowIfError(cufftExecC2C(handle, src.data, dst.data, forward ? CUFFT_FORWARD : CUFFT_INVERSE));
}

void FFTPlan2D::TransformInPlace(DeviceImage<cufftComplex>& d, bool forward, cudaStream_t stream)
{
	ThrowIfError(cufftSetStream(handle, stream));
	ThrowIfError(cufftExecC2C(handle, d.data, d.data, forward?  CUFFT_FORWARD : CUFFT_INVERSE));
}

FFTPlan1D::~FFTPlan1D()
{
	if (handle != 0)
		cufftDestroy(handle);
	handle = 0;
}

FFTPlan1D::FFTPlan1D(int len, int count)
{	
	// According to 
	//https://docs.nvidia.com/cuda/cufft/index.html
	//  1D layout is:
	// input[b * idist + x * istride]
	// output[b * odist + x * ostride]
	int n[1] = { len };
	int rank = 1;
	int idist = len; // distance between batches
	int istride = 1; // distance between elements
	int ostride = 1;
	int outputlen = len;
	int odist = len; // distance between output batches
	cufftResult result = cufftPlanMany(&handle, rank, n, 0, istride, idist, 0, ostride, odist, CUFFT_C2C, count);
	if (result != CUFFT_SUCCESS)
		throw std::runtime_error(SPrintf("cufftPlanMany failed: %s", cufftErrorString(result)));

	outputlen = len;
	inputlen = len;
}

void FFTPlan1D::Transform(const cuFloatComplex* src, cuFloatComplex* dst, bool forward, cudaStream_t stream)
{
	direction = forward ? CUFFT_FORWARD : CUFFT_INVERSE;
	ThrowIfError(cufftSetStream(handle, stream));
	ThrowIfError(cufftExecC2C(handle, (cufftComplex*)src, dst, direction));
}



CDLL_EXPORT void FFT(const cuFloatComplex* src, cuFloatComplex* dst, int batchsize, int siglen, int forward)
{
	FFTPlan1D plan(siglen, batchsize);

	DeviceArray<cuFloatComplex> d_src(siglen*batchsize, src);
	DeviceArray<cuFloatComplex> d_dst(siglen*batchsize);

	plan.Transform(d_src.ptr(), d_dst.ptr(), forward);

	d_dst.CopyToHost(dst, false);
	if (!forward) {
		// CUDA FFT needs rescaling to match numpy's fft/ifft. 
		float f = 1.0f / siglen;
		for (int i = 0; i < batchsize; i++)
			for (int j = 0; j < siglen; j++) {
				dst[siglen*i + j] *= f;
			}
	}
}


CDLL_EXPORT void FFT2(const cuFloatComplex* src, cuFloatComplex* dst, int batchsize, int sigw, int sigh, int forward)
{
	DeviceImage<cuFloatComplex> d_src(src, sigw, sigh * batchsize);
	DeviceImage<cuFloatComplex> d_dst(sigw, sigh * batchsize);

	FFTPlan2D plan(batchsize, sigw, sigh, d_src.PitchInPixels(), d_dst.PitchInPixels());

	//ShowDebugImage(d_src, 1, "src");

	plan.Transform(d_src.data, d_dst.data, forward);

	//ShowDebugImage(d_dst, 1, "dst");

	//ShowDebugImage(d_dst, 1, "src");
	d_dst.CopyToHost(dst);
	/*
	FFTPlan2D plan(batchsize, sigw, sigh, sigw, sigw);

	DeviceArray<cuFloatComplex> d_src(sigw*sigh*batchsize, src);
	DeviceArray<cuFloatComplex> d_dst(sigw*sigh*batchsize);

	plan.Transform(d_src.ptr(), d_dst.ptr(), forward);

	d_dst.CopyToHost(dst, false);
	*/

	if (!forward) {
		// CUDA FFT needs rescaling to match numpy's fft/ifft. 
		float f = 1.0f / (sigw*sigh);
		for (int i = 0; i < batchsize; i++)
			for (int j = 0; j < sigw*sigh; j++) {
				dst[sigw*sigh*i + j] *= f;
			}
	}
}


