// Quadratic least squares that can be compiled in a cuda kernel
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021

#pragma once


template<typename T>
class LsqSqQuadFit
{
public:
	T a,b,c,d;
	float xoffset;

	struct Coeff {
		T s40, s30, s20, s10, s21, s11, s01, s00;

        PLL_DEVHOST void abc(T& a, T& b, T& c, T& d) {
			d = s40 * (s20 * s00 - s10 * s10) - s30 * (s30 * s00 - s10 * s20) + s20 * (s30 * s10 - s20 * s20);

			a = (s21*(s20 * s00 - s10 * s10) - s11*(s30 * s00 - s10 * s20) + s01*(s30 * s10 - s20 * s20)) / d;
			b = (s40*(s11 * s00 - s01 * s10) - s30*(s21 * s00 - s01 * s20) + s20*(s21 * s10 - s11 * s20)) / d;
			c = (s40*(s20 * s01 - s10 * s11) - s30*(s30 * s01 - s10 * s21) + s20*(s30 * s11 - s20 * s21)) / d;
		}
	};

	PLL_DEVHOST LsqSqQuadFit(uint numPts, const T* xval, const T* yval, const T* weights=0)
	{
		calculate(numPts, xval, yval, weights);
		xoffset =0;
	}

	PLL_DEVHOST LsqSqQuadFit()
	{
		a=b=c=d=0;
		xoffset =0;
	}

	PLL_DEVHOST void calculate(uint numPts, const T* X, const T* Y, const T* weights)
	{
		Coeff co = computeSums(X, Y, weights, numPts);
		co.abc(a,b,c,d);
	}
    
	PLL_DEVHOST T compute(T pos)
	{
		pos -= xoffset;
		return a*pos*pos + b*pos + c;
	}

	PLL_DEVHOST T computeDeriv(T pos)
	{
		pos -= xoffset;
		return 2*a*pos + b;
	}

	PLL_DEVHOST T maxPos()
	{
		return -b/(2*a);
	}
   
private:

    PLL_DEVHOST Coeff computeSums(const T* X, const T* Y, const T* weights, uint numPts) // get sum of x
    {
        //notation sjk to mean the sum of x_i^j*y_i^k. 
    /*    s40 = getSx4(); //sum of x^4
        s30 = getSx3(); //sum of x^3
        s20 = getSx2(); //sum of x^2
        s10 = getSx();  //sum of x
        

        s21 = getSx2y(); //sum of x^2*y
        s11 = getSxy();  //sum of x*y
        s01 = getSy();   //sum of y
		*/

		if (weights) {
			T Sx = 0, Sy = 0;
			T Sx2 = 0, Sx3 = 0;
			T Sxy = 0, Sx4=0, Sx2y=0;
			T Sw = 0;
			for (uint i=0;i<numPts;i++)
			{
				T x = X[i];
				T y = Y[i];
				Sx += x*weights[i];
				Sy += y*weights[i];
				T sq = x*x;
				Sx2 += x*x*weights[i];
				Sx3 += sq*x*weights[i];
				Sx4 += sq*sq*weights[i];
				Sxy += x*y*weights[i];
				Sx2y += sq*y*weights[i];
				Sw += weights[i];
			}

			Coeff co;
			co.s10 = Sx; co.s20 = Sx2; co.s30 = Sx3; co.s40 = Sx4;
			co.s01 = Sy; co.s11 = Sxy; co.s21 = Sx2y;
			co.s00 = Sw;
			return co;
		} else {
			T Sx = 0, Sy = 0;
			T Sx2 = 0, Sx3 = 0;
			T Sxy = 0, Sx4=0, Sx2y=0;
			for (uint i=0;i<numPts;i++)
			{
				T x = X[i];
				T y = Y[i];
				Sx += x;
				Sy += y;
				T sq = x*x;
				Sx2 += x*x;
				Sx3 += sq*x;
				Sx4 += sq*sq;
				Sxy += x*y;
				Sx2y += sq*y;
			}

			Coeff co;
			co.s10 = Sx; co.s20 = Sx2; co.s30 = Sx3; co.s40 = Sx4;
			co.s01 = Sy; co.s11 = Sxy; co.s21 = Sx2y;
			co.s00 = numPts;
			return co;
		}
    }

};

// Computes the interpolated maximum position
template<typename T, int numPts=3>
class ComputeMaxInterp {
public:
	static PLL_DEVHOST T max_(T a, T b) { return a>b ? a : b; }
	static PLL_DEVHOST T min_(T a, T b) { return a<b ? a : b; }

	static PLL_DEVHOST T Compute(T* data, int len, const  T* weights, LsqSqQuadFit<T>* fit=0)
	{
		int iMax=0;
		T vMax=data[0];
		for (int k=1;k<len;k++) {
			if (data[k]>vMax) {
				vMax = data[k];
				iMax = k;
			}
		}
		T xs[numPts];
		int startPos = max_(iMax-numPts/2, 0);
		int endPos = min_(iMax+(numPts-numPts/2), len);
		int numpoints = endPos - startPos;

		if (numpoints<3)
			return iMax;
		else {
			for(int i=startPos;i<endPos;i++)
				xs[i-startPos] = i-iMax;

			LsqSqQuadFit<T> qfit(numpoints, xs, &data[startPos], weights);
			if (fit) *fit = qfit;
			//printf("iMax: %d. qfit: data[%d]=%f\n", iMax, startPos, data[startPos]);
			//for (int k=0;k<numpoints;k++) {
		//		printf("data[%d]=%f\n", startPos+k, data[startPos]);
			//}

			if (fabs(qfit.a)<1e-9f)
				return (T)iMax;
			else {
				T interpMax = qfit.maxPos();
				return (T)iMax + interpMax;
			}
		}
	}


};
