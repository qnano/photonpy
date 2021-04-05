// SIMFLUX helper functions
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "palala.h"
#include "SIMFLUX.h"
#include "SolveMatrix.h"

#include "Estimators/Estimation.h"
#include "StringUtils.h"
#include "ThreadUtils.h"
#include "ExcitationModel.h"

#pragma warning(disable : 4503) // decorated name length exceeded, name was truncated

#include "SIMFLUX_Models.h"


struct SIMFLUX_SampleIndex
{
	int f, x, y;
};




CDLL_EXPORT void SIMFLUX_ProjectPointData(const Vector3f *xyI, int numpts, int projectionWidth, 
	float scale, int numProjAngles, const float *projectionAngles, float* output, float* shifts)
{
	int pw = projectionWidth;
	ParallelFor(numProjAngles, [&](int p) {
		float* proj = &output[pw*p];
		for (int i = 0; i < pw; i++)
			proj[i] = 0.0f;
		float kx = cos(projectionAngles[p]);
		float ky = sin(projectionAngles[p]);

		double moment = 0.0;
		for (int i = 0; i < numpts; i++)
			moment += kx * xyI[i][0] + ky * xyI[i][1];
		float shift = pw / 2 - scale * float(moment) / numpts;
		if(shifts) shifts[p] = shift;

		for (int i = 0; i < numpts; i++) {
			float coord = kx * xyI[i][0] + ky * xyI[i][1];
			int index = int(scale * coord + shift + 0.5f);
			if (index < 0 || index> pw - 1)
				continue;
			proj[index] += xyI[i][2];
		}
	});
}




CDLL_EXPORT void SIMFLUX_DFT2D_Points(const Vector3f* xyI, int numpts, const Vector2f* k, int numk, Vector2f* output, bool useCuda)
{
	palala_for(numk, useCuda, PLL_FN(int i, const Vector3f* xyI, const Vector2f* k, Vector2f* output) {
		Vector2f k_ = k[i];

		// Use Kahan Sum for reduced errors
		Vector2f sum, c;

		for (int j = 0; j < numpts; j++) {
			float p = xyI[j][0] * k_[0] + xyI[j][1] * k_[1];
			float I = xyI[j][2];
			Vector2f input = { cos(p) * I, sin(p) * I };
			Vector2f y = input - c;
			Vector2f t = sum + y;
			c = (t - sum) - y;
			sum = t;
		}
		output[i] = sum;
	}, const_array(xyI, numpts),
		const_array(k, numk),
		out_array(output, numk));
}


struct SpotToExtract {
	int linkedIndex;
	int numroi;
	int firstframe;
};



// Generate a sorted list of ROIs to extract from a tiff file for simflux localization:
// spotToLinkedIdx: int[numspots]
// startframes: int[numlinked]
// ontime: int[numlinked], number of frames the linked spot is on
// result: SpotToExtract[numspots]
// Returns number of elements in result list
CDLL_EXPORT int SIMFLUX_GenerateROIExtractionList(int *startframes, int *ontime, int maxfits, 
								int numlinked, int numpatterns, SpotToExtract* result)
{
	int fits = 0;
	for (int i = 0; i < numlinked; i++) 
	{
		int frame = 0;
		while (frame < ontime[i] && fits<maxfits) {
			int remaining = ontime[i] - frame;

			// Can't use
			if (remaining < numpatterns)
				break;

			if (remaining < numpatterns * 2)
			{
				result[fits].numroi = remaining;
				result[fits].firstframe = startframes[i] + frame;
				result[fits].linkedIndex = i;
				fits++;
				break; // used up all of the linked spot frames
			}
			else {
				result[fits].firstframe = startframes[i] + frame;
				result[fits].numroi = numpatterns;
				result[fits].linkedIndex = i;
				fits++;
				frame += numpatterns;
			}
		}
	}
	return fits;
}

