// Extract all non-negative indices from a cuda array
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include <cub/cub.cuh>


template<typename TIndex>
class SpotIndexListGenerator
{
public:
	DeviceArray<TIndex> selectedIndices;
	PinnedArray<TIndex> h_selected;

	DeviceArray<int> numFoundSpots;
	DeviceArray<uint8_t> partitionTempStorage; // for cub::DevicePartition::If
	PinnedArray<int> numspots;

	void Init(int w, int h)
	{
		int numpixels = w * h;
		selectedIndices.Init(numpixels);
		h_selected.Init(numpixels);
		numFoundSpots.Init(1);
		numspots.Init(1);
	}

	void Compute(const DeviceArray<TIndex>& indices, cudaStream_t stream)
	{
		if (!partitionTempStorage.ptr())
		{
			size_t tempBytes;
			CUDAErrorCheck(cub::DevicePartition::If(0, tempBytes, indices.ptr(), selectedIndices.ptr(),
				numFoundSpots.ptr(), (int)indices.size(), non_negative(), stream));
			partitionTempStorage.Init(tempBytes);
		}

		size_t tmpsize = partitionTempStorage.size();
		CUDAErrorCheck(cub::DevicePartition::If(partitionTempStorage.ptr(), tmpsize, indices.ptr(),
			selectedIndices.ptr(), numFoundSpots.ptr(), (int)indices.size(), non_negative(), stream));

		h_selected.CopyFromDevice(selectedIndices, stream);
		numFoundSpots.CopyToHost(numspots.data(), true, stream);
	}
};

