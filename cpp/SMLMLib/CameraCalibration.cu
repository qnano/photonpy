// Camera Calibration
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#include "MemLeakDebug.h"
#include "CameraCalibration.h"


sCMOS_Calibration::sCMOS_Calibration(int2 imageSize, const float * offset, const float * gain, const float * variance)
	: imgsize(imageSize),
	h_offset(imageSize.x*imageSize.y),
	h_gain(imageSize.x* imageSize.y),
	h_vargain2(imageSize.x* imageSize.y)
{
	for (int i = 0; i < h_vargain2.size(); i++)
		h_vargain2[i] = variance[i] / (gain[i] * gain[i]);
}

void sCMOS_Calibration::ProcessImage(ImageIndexer image)
{
	for (int y = 0; y < imgsize.y; y++) {
		for (int x = 0; x < imgsize.x; x++) {
			image(x,y) = (image(x,y) - h_offset[y * imgsize.x + x]) * h_gain[y * imgsize.x + x];// +vargain2(x, y);  var/gain^2 is now added during optimization
		}
	}
}



void GainOffsetCalibration::ProcessImage(ImageIndexer image)
{
	for (int y = 0; y < image.h; y++) {
		for (int x = 0; x < image.w; x++) {
			float v = (image(x,y) - offset) * gain;
			if (v < 0.0f) v = 0.0f;
			image(x,y) = v;
		}
	}
}



GainOffsetImageCalibration::GainOffsetImageCalibration(int2 imgsize, const float * gain, const float * offset) :
	imgsize(imgsize),
	gain(gain, gain + imgsize.x*imgsize.y), 
	offset(offset, offset + imgsize.x*imgsize.y)
{
}

void GainOffsetImageCalibration::ProcessImage(ImageIndexer image)
{
	for (int y = 0; y < imgsize.y; y++) {
		for (int x = 0; x < imgsize.x; x++) {
			float v = (image(x,y) - offset[y * imgsize.x + x]) * gain[y * imgsize.x + x];
			if (v < 0.0f) v = 0.0f;
			image(x,y) = v;
		}
	}
}



CDLL_EXPORT sCMOS_Calibration * sCMOS_Calib_Create(int w, int h, const float * offset, const float * gain, const float * variance, Context* ctx)
{
	auto* r = new sCMOS_Calibration({ w,h }, offset, gain, variance);
	if (ctx) r->SetContext(ctx);
	return r;
}

CDLL_EXPORT GainOffsetCalibration * GainOffsetCalib_Create(float gain, float offset, Context* ctx)
{
	auto* r = new GainOffsetCalibration(gain, offset);
	if (ctx) r->SetContext(ctx);
	return r;
}


CDLL_EXPORT GainOffsetImageCalibration * GainOffsetImageCalib_Create(int width,int height, const float *gain, const float* offset, Context* ctx)
{
	auto* r = new GainOffsetImageCalibration({ width,height },gain, offset);
	if (ctx) r->SetContext(ctx);
	return r;
}
