// Localization rendering utils
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021

#include "Rendering.h"
#include "MathUtils.h"
#include "Estimators/Gaussian/GaussianPSF.h"


// spotList [ x y sigmaX sigmaY intensity ]
CDLL_EXPORT void Gauss2D_Draw(double * image, int imgw, int imgh, float * spotList, int nspots, float addSigma)
{
	auto squared = [](double x) { return x * x; };

	for (int i = 0; i < nspots; i++) {
		float* spot = &spotList[5 * i];
		// just a nice heuristic that seems to work well
		double sigmaScale = 2 + log(20000.0f) * 0.1f + addSigma;
		double hwx = spot[2] * sigmaScale;
		double hwy = spot[3] * sigmaScale;
		int minx = int(spot[0] - hwx), miny = int(spot[1] - hwy);
		int maxx = int(spot[0] + hwx + 1), maxy = int(spot[1] + hwy + 1);
		if (minx < 0) minx = 0;
		if (miny < 0) miny = 0;
		if (maxx > imgw - 1) maxx = imgw - 1;
		if (maxy > imgh - 1) maxy = imgh - 1;

		double _1o2sxs = 1.0f / (sqrt(2.0f) * spot[2]);
		double _1o2sys = 1.0f / (sqrt(2.0f) * spot[3]);
		for (int y = miny; y <= maxy; y++) {
			for (int x = minx; x <= maxx; x++) {
				double& pixel = image[y*imgw + x];
				pixel += spot[4] * exp(-(squared((x - spot[0])*_1o2sxs) + squared((y - spot[1])*_1o2sys))) / (2 * MATH_PI*spot[2] * spot[3]);
			}
		}
	}
}




CDLL_EXPORT void DrawROIs(float* image, int width, int height, const float* rois, int numrois, int roisize, Int2* roiposYX)
{
	for (int i = 0; i < numrois; i++) {
		int roix = roiposYX[i][1];
		int roiy = roiposYX[i][0];
		const float* roi = &rois[roisize*roisize*i];

		int sx = 0, sy = 0;
		int roiWidth = roisize, roiHeight = roisize;
		if (roix < 0) { sx = -roix;  }
		if (roiy < 0) { sy = -roiy;  }
		if (roix + roiWidth > width) { roiWidth = width - roix; }
		if (roiy + roiHeight > height) { roiHeight = height - roiy; }

		for (int y=sy;y<roiHeight;y++)
			for (int x = sx;x<roiWidth;x++)
				image[(y + roiy)*width + x + roix] += roi[y*roisize + x];
	}
}


