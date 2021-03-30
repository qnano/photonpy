#pragma once

#include "CudaUtils.h"

template<typename T>
void UniformFilter2D(const DeviceImage<T>& input, DeviceImage<T>& temp, DeviceImage<T>& output, int windowSize, cudaStream_t stream=0)
{
	assert(temp.width == input.width && temp.height == input.height);
	assert(output.width == input.width && output.height == input.height);

	int xback = -windowSize / 2;
	int xforw = xback + windowSize;
	T* d_in = input.data, *d_temp = temp.data, *d_out = output.data;

	// assumes pitch is always a multiple of sizeof(T), still fair assumption
	int pitch_in = input.pitch / sizeof(T);
	int pitch_temp = temp.pitch / sizeof(T);
	int pitch_out = output.pitch / sizeof(T);
	int w = input.width, h = input.height;

	// X filter
	LaunchKernel(input.width, input.height, [=]__device__(int x, int y) {
		int xs = x + xback;
		if (xs < 0) xs = 0;

		int xe = x + xforw;
		if (xe > w) xe = w;

		int count = xe - xs;

		T sum{};
		for (int i = xs; i < xe; i++)
			sum += d_in[y*pitch_in+i];
		d_temp[y*pitch_temp+x] = sum / count;
	}, 0, stream);
	// Y filter
	int yback = -windowSize / 2;
	int yforw = yback + windowSize;
	LaunchKernel(input.width, input.height, [=]__device__(int x, int y) {
		int ys = y + yback;
		if (ys < 0) ys = 0;

		int ye = y + yforw;
		if (ye > h) ye = h;

		int count = ye - ys;

		T sum{};
		for (int i = ys; i < ye; i++)
			sum += d_temp[i*pitch_temp+x];
		d_out[y*pitch_out+x] = sum / count;
	}, 0, stream);
}


template<typename T, typename TOperator>
void ComparisonFilter2DStack(const DeviceImage<T>& input, DeviceImage<T>& output, DeviceImage<T>& temp, int windowSize, int depth, TOperator op, cudaStream_t stream = 0)
{
	int w = input.width;

	assert(temp.width == input.width && temp.height == input.height);
	assert(output.width == input.width && output.height == input.height);

	int xback = -windowSize / 2;
	int xforw = xback + windowSize;
	T* d_in = input.data, *d_temp = temp.data, *d_out = output.data;

	// assumes pitch is always a multiple of sizeof(T), still fair assumption
	int pitch_in = input.pitch / sizeof(T);
	int pitch_temp = temp.pitch / sizeof(T);
	int pitch_out = output.pitch / sizeof(T);

	// X filter
	LaunchKernel(input.width, input.height, [=]__device__(int x, int y) {
		int xs = x + xback;
		if (xs < 0) xs = 0;

		int xe = x + xforw;
		if (xe > w) xe = w;

		T value = d_in[y*pitch_in+xs];
		for (int i = xs + 1; i < xe; i++)
			value = op(value, d_in[y*pitch_in + i]);
		d_temp[y*pitch_temp + x] = value;
	}, 0, stream);
	// Y filter
	int yback = -windowSize / 2;
	int yforw = yback + windowSize;
	int slice_h = input.height / depth;
	LaunchKernel(w, slice_h, depth, [=]__device__(int x, int y, int z) {
		int ys = y + yback;
		if (ys < 0) ys = 0;

		int ye = y + yforw;
		if (ye > slice_h) ye = slice_h;

		ys += slice_h * z;
		ye += slice_h * z;

		T value = d_temp[ys*pitch_temp + x];
		for (int i = ys + 1; i < ye; i++)
			value = op(value, d_temp[i*pitch_temp + x]);
		d_out[(y+slice_h*z)*pitch_out + x] = value;
	}, 0, stream);
}

template<typename T>
class ImageNormalizer
{
	DeviceArray<T> accum;
	int width, height, numImg;

public:
	ImageNormalizer(int w, int h, int numImages)
		: accum(h*numImages), width(w), height(h), numImg(numImages)
	{}

	void Normalize(DeviceImage<T>& images, cudaStream_t stream)
	{
		// TODO: Use CUB reduce
		auto indexer = images.GetIndexer();
		T* dst = accum.ptr();
		int w = width, h = height;
		LaunchKernel(h*numImg, [=]__device__(int y) {
			T sum = {};
			for (int x = 0; x < w; x++)
				sum += indexer.pixel(x, y);
			dst[y] = sum;
		}, 0, stream);
		LaunchKernel(numImg, [=]__device__(int img) {
			T sum = {};
			for (int y = 0; y < h; y++)
				sum += dst[img*h + y];
			dst[img*h] = 1.0f / (sum > 0.0f ? sum : 1.0f);
		}, 0, stream);
		LaunchKernel(w, h, numImg, [=]__device__(int x, int y, int img) {
			float factor = dst[img*h];
			indexer.pixel(x, y + img * h) *= factor;
		}, 0, stream);
	}
};


template<typename T1,typename T2, typename T3, typename TOperator>
void ApplyBinaryOperator2D(const DeviceImage<T1>& a, const DeviceImage<T2>& b, DeviceImage<T3>& out, TOperator op, cudaStream_t stream=0)
{
	int pitch_a = a.pitch/sizeof(T1), pitch_b = b.pitch/sizeof(T2), pitch_out = out.pitch/sizeof(T3);
	T1* d_a = a.data;
	T2* d_b = b.data;
	T3 *d_out = out.data;

	LaunchKernel(a.width, a.height, [=]__device__(int x, int y) {
		d_out[y*pitch_out + x] = op(d_a[y*pitch_a + x], d_b[y*pitch_b + x]);
	}, 0, stream);
}	


template<typename T1, typename TOperator>
void ApplyUnaryOperator2D(DeviceImage<T1>& a, TOperator op, cudaStream_t stream = 0)
{
	int pitch = a.pitch / sizeof(T1);
	T1* d_a = a.data;

	LaunchKernel(a.width, a.height, [=]__device__(int x, int y) {
		d_a[y * pitch + x] = op(d_a[y * pitch + x]);
	}, 0, stream);
}





