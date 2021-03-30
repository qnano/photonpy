// Zernike coefficients
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once

#include "DLLMacros.h"
#include "palala.h"
#include "MathUtils.h"
#include "CudaUtils.h"


template<typename T>
PLL_DEVHOST T ZernikeRadial(T rho, int n, int m)
{
	if ((n - m) % 2 != 0) return 0;
	T sum = {};
	for (int k = 0; k<int((n - m) / 2 + 1); k++) {
		sum += (T)pow(rho, n - T(2)*k) * (T)pow(T(-1.0), k) * Factorial(n - k) / T(Factorial(k) * Factorial((n + m) / 2 - k) * Factorial((n - m) / 2 - k));
	}
	return sum;
}

template<typename T>
PLL_DEVHOST T ZernikePolynomial(T rho, T phi, int n, int m)
{
	if (m < 0) return ZernikeRadial(rho, n, -m) * sin(-m * phi);
	return ZernikeRadial(rho, n, m) * cos(m*phi);
}

struct ZernikeIndices {
	int n, m;

	static ZernikeIndices FromNoll(int j)
	{
		int n = 0;
		int j1 = j - 1;
		while (j1 > n) {
			n++;
			j1 -= n;
		}
		int m = IntegerPower(-1, j) * ((n % 2) + 2 * int((j1 + ((n + 1) % 2)) / 2));
		return ZernikeIndices{ n,m };
	}
};



class ZernikePolynomialBuffer
{
public:
	ZernikePolynomialBuffer(int maxNoll, int imgsize, float zernikeUnitCircleRadius) : maxNoll(maxNoll),
		h_zernikeModes(maxNoll)
		// noll indices starts at one so this works out
	{
		std::vector<float> h_mode(imgsize*imgsize); // host side

		d_zernikeModes.Init(imgsize, imgsize*maxNoll);
		for (int j = 1; j <= maxNoll; j++) {
			auto nm = ZernikeIndices::FromNoll(j);

			for (int y = 0; y < imgsize; y++) {
				for (int x = 0; x < imgsize; x++) {
					float& value = h_mode[y*imgsize + x];
					int mx = x - imgsize / 2, my = y - imgsize / 2;
					float r = sqrt((float)(mx*mx + my*my));
					float rho = r / zernikeUnitCircleRadius;
					float theta = atan2f((float)my,(float) mx);
					if (rho > 1.0f)
						value = 0.0f;
					else
						value = ZernikePolynomial(rho, theta, nm.n, nm.m);
				}
			}

			h_zernikeModes[j - 1] = h_mode;
			d_zernikeModes.CopyFromHost(&h_mode[0], 0, imgsize*(j - 1), imgsize, imgsize);
		}
	}

	int maxNoll;
	DeviceImage<float> d_zernikeModes;
	std::vector<std::vector<float>> h_zernikeModes;
};


