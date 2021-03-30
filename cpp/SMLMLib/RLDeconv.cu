


#include "Context.h"
#include "CudaUtils.h"
#include "palala.h"


#include "FFT.h"

class Deconv3D : public ContextObject
{
public:
	FFTPlan2D fft, fftInv;

	Deconv3D(int width, int height, int depth) : 
		fft(depth, width, height, width, CUFFT_R2C),
		fftInv(depth, width, height, width, CUFFT_C2R)
	{

	}
};

// src[depth,h,w]
// dst[h,w]
// psfStack[depth,roisize,roisize]
CDLL_EXPORT void Convolv3D(float* dst, int w, int h, float* src, float* psfStack, int roisize, int depth, bool cuda)
{
	for (int y=0;y<h;y++)
		for (int x = 0; x < w; x++)
		{
			float sum = 0.0f;


		}

}

void RLDeconvolve3D(float* srcImage, float* output, int w, int h, int outputLayers, float* psfStack, int roisize, int roidepth, bool cuda)
{
/*	palala_for(w, h numk, useCuda, PALALA(int i, const Vector3f * xyI, const Vector2f * k, Vector2f * output) {
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
		out_array(output, numk));*/
}




