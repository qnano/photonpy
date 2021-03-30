// sCMOS camera calibration implemented based on 
// Video-rate nanoscopy using sCMOS camera�specific single-molecule localization algorithms
// https://www.nature.com/articles/nmeth.2488
#pragma once

#include <vector>
#include "Estimators/Estimation.h"
#include "Context.h"
#include "CudaUtils.h"



struct ImageIndexer {
	int w, h;
	float* data;

	ImageIndexer(int w, int h, float* data) : w(w), h(h), data(data) {}

	float& operator()(int x, int y) const { return data[w * y + x]; }
};

class ICalibrationProcessor : public ContextObject
{
public:
	virtual void ProcessImage(ImageIndexer image) = 0;
	virtual ~ICalibrationProcessor() {}
};


class sCMOS_Calibration : public ICalibrationProcessor
{
public:
	sCMOS_Calibration(int2 imageSize, const float* offset, const float* gain, const float * variance);

	virtual void ProcessImage(ImageIndexer image) override;

protected:
	int2 imgsize;
	std::vector<float> h_offset, h_gain, h_vargain2;
};

class GainOffsetCalibration : public ICalibrationProcessor {
public:
	GainOffsetCalibration(float gain, float offset) : 
		gain(gain), offset(offset) {}
	// Inherited via ImageProcessor
	virtual void ProcessImage(ImageIndexer image) override;

	float gain, offset;
};


class GainOffsetImageCalibration : public ICalibrationProcessor {
public:
	GainOffsetImageCalibration(int2 imgsize, const float* gain, const float* offset);
	// Inherited via ImageProcessor
	virtual void ProcessImage(ImageIndexer image) override;

	std::vector<float> gain, offset;
	int2 imgsize;
};


CDLL_EXPORT sCMOS_Calibration* sCMOS_Calib_Create(int w,int h, const float* offset, const float* gain, const float *variance, Context* ctx);
CDLL_EXPORT GainOffsetCalibration* GainOffsetCalib_Create(float gain, float offset, Context* ctx);

