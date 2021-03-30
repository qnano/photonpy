// Square matrix solves that compile in cuda kernels
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
#pragma once
#include "Vector.h"

// http://www.sci.utah.edu/~wallstedt/LU.htm

// Crout uses unit diagonals for the upper triangle
template<typename T, int d>
PLL_DEVHOST void Crout(T *S, T*D) {
	for (int k = 0; k<d; ++k) {
		for (int i = k; i<d; ++i) {
			T sum{};
			for (int p = 0; p<k; ++p)
				sum += D[i*d + p] * D[p*d + k];
			D[i*d + k] = S[i*d + k] - sum; // not dividing by diagonals
		}
		for (int j = k + 1; j<d; ++j) {
			T sum{};
			for (int p = 0; p<k; ++p)
				sum += D[k*d + p] * D[p*d + j];
			D[k*d + j] = (S[k*d + j] - sum) / D[k*d + k];
		}
	}
}
template<typename T, int d>
PLL_DEVHOST void SolveCrout(T *LU, T*b, T*x) {
	T y[d];
	for (int i = 0; i<d; ++i) {
		T sum{};
		for (int k = 0; k<i; ++k)
			sum += LU[i*d + k] * y[k];
		y[i] = (b[i] - sum) / LU[i*d + i];
	}
	for (int i = d - 1; i >= 0; --i) {
		T sum{};
		for (int k = i + 1; k<d; ++k)
			sum += LU[i*d + k] * x[k];
		x[i] = (y[i] - sum); // not dividing by diagonals
	}
}



// Doolittle uses unit diagonals for the lower triangle
template<typename T, int d>
PLL_DEVHOST  void Doolittle(double*S, double*D) {
	for (int k = 0; k<d; ++k) {
		for (int j = k; j<d; ++j) {
			T sum{};
			for (int p = 0; p<k; ++p)
				sum += D[k*d + p] * D[p*d + j];
			D[k*d + j] = (S[k*d + j] - sum); // not dividing by diagonals
		}
		for (int i = k + 1; i<d; ++i) {
			T sum{};
			for (int p = 0; p<k; ++p)
				sum += D[i*d + p] * D[p*d + k];
			D[i*d + k] = (S[i*d + k] - sum) / D[k*d + k];
		}
	}
}
template<typename T, int d>
PLL_DEVHOST void SolveDoolittle(double*LU, double*b, double*x) {
	T y[d];
	for (int i = 0; i<d; ++i) {
		T sum{};
		for (int k = 0; k<i; ++k)
			sum += LU[i*d + k] * y[k];
		y[i] = (b[i] - sum); // not dividing by diagonals
	}
	for (int i = d - 1; i >= 0; --i) {
		T sum{};
		for (int k = i + 1; k<d; ++k)
			sum += LU[i*d + k] * x[k];
		x[i] = (y[i] - sum) / LU[i*d + i];
	}
}

// Cholesky requires the matrix to be symmetric positive-definite
template<typename T, int d>
PLL_DEVHOST bool Cholesky(T* S, T* D) {
	for (int k = 0; k < d; ++k) {
		T sum{};
		for (int p = 0; p < k; ++p)
			sum += D[k * d + p] * D[k * d + p];
		T v = S[k * d + k] - sum;
		if (v == 0.0f) return false;
		D[k * d + k] = sqrt(v);
		for (int i = k + 1; i < d; ++i) {
			T sum{};
			for (int p = 0; p < k; ++p)
				sum += D[i * d + p] * D[k * d + p];
			T q = D[k * d + k];
			if (q == 0.0f) return false;
			D[i * d + k] = (S[i * d + k] - sum) / q;
		}
	}
	return true;
}


// Cholesky requires the matrix to be symmetric positive-definite
template<typename T>
PLL_DEVHOST bool Cholesky(int d, T*S, T*D) {
	for (int k = 0; k<d; ++k) {
		T sum{};
		for (int p = 0; p<k; ++p)
			sum += D[k*d + p] * D[k*d + p];
		T v = S[k*d + k] - sum;
		if (v == 0.0f) return false;
		D[k*d + k] = sqrt(v);
		for (int i = k + 1; i<d; ++i) {
			T sum{};
			for (int p = 0; p<k; ++p)
				sum += D[i*d + p] * D[k*d + p];
			T q = D[k*d + k];
			if (q == 0.0f) return false;
			D[i*d + k] = (S[i*d + k] - sum) / q;
		}
	}
	return true;
}

// This version could be more efficient on some architectures
// Use solveCholesky for both Cholesky decompositions
template<typename T, int K>
PLL_DEVHOST void CholeskyRow(int d, T S[K], T*D) {
	for (int k = 0; k < d; ++k) {
		for (int j = 0; j < d; ++j) {
			T sum{};
			for (int p = 0; p < j; ++p) 
				sum += D[k*d + p] * D[j*d + p];
			D[k*d + j] = (S[k*d + j] - sum) / D[j*d + j];
		}
		T sum{};
		for (int p = 0; p < k; ++p) 
			sum += D[k*d + p] * D[k*d + p];
		D[k*d + k] = sqrt(S[k*d + k] - sum);
	}
}

template<typename T,int d>
PLL_DEVHOST bool SolveCholesky(const T(&LU)[d*d], const T(&b)[d], T(&x)[d]) {
	T y[d];
	for (int i = 0; i < d; ++i) {
		T sum{};
		for (int k = 0; k < i; ++k)
			sum += LU[i*d + k] * y[k];
		T v = LU[i*d + i];
		if (v == 0.0f) return false;
		y[i] = (b[i] - sum) / v;
	}
	for (int i = d - 1; i >= 0; --i) {
		T sum{};
		for (int k = i + 1; k < d; ++k)
			sum += LU[k*d + i] * x[k];
		x[i] = (y[i] - sum) / LU[i*d + i];
	}
	return true;
}


template<typename T>
PLL_DEVHOST bool SolveCholesky(int d, const T *LU, const T *b, T *x, T* temp) {
	for (int i = 0; i < d; ++i) {
		T sum{};
		for (int k = 0; k < i; ++k)
			sum += LU[i*d + k] * temp[k];
		T v = LU[i*d + i];
		if (v == 0.0f) return false;
		temp[i] = (b[i] - sum) / v;
	}
	for (int i = d - 1; i >= 0; --i) {
		T sum{};
		for (int k = i + 1; k < d; ++k)
			sum += LU[k*d + i] * x[k];
		x[i] = (temp[i] - sum) / LU[i*d + i];
	}
	return true;
}


template<typename T, int d>
PLL_DEVHOST void ApplyMatrix(const T (&m)[d*d], const T (&x)[d], T (&y)[d]) {
	for (int i = 0; i < d; i++)
	{
		T sum{};
		for (int j = 0; j < d; j++)
			sum += m[i*d + j] * x[j];
		y[i] = sum;
	}
}

template<typename T, int d>
PLL_DEVHOST typename Vector<T,d> ApplyMatrix(const T(&m)[d*d], const Vector<T,d>&x)
{
	Vector<T, d> y;
	for (int i = 0; i < d; i++)
	{
		T sum{};
		for (int j = 0; j < d; j++)
			sum += m[i*d + j] * x[j];
		y[i] = sum;
	}
	return y;
}

// Modified version of https://en.wikipedia.org/wiki/LU_decomposition#C_code_examples
/* INPUT: A - array of pointers to rows of a square matrix having dimension N
*        Tol - small tolerance number to detect failure when the matrix is near degenerate
* OUTPUT: Matrix A is changed, it contains both matrices L-E and U as A=(L-E)+U such that P*A=L*U.
*        The permutation matrix is not stored as a matrix, but in an integer vector P of size N+1
*        containing column indexes where the permutation matrix has "1". The last element P[N]=S+N,
*        where S is the number of row exchanges needed for determinant computation, det(P)=(-1)^S
*/
template<typename T, int N>
PLL_DEVHOST bool LUPDecompose(T(&A)[N*N], T Tol, int(&P)[N + 1]) {
	for (int i = 0; i <= N; i++)
		P[i] = i; //Unit permutation matrix, P[N] initialized with N

	for (int i = 0; i < N; i++) {
		T maxA = 0.0, absA;
		int imax = i;

		for (int k = i; k < N; k++)
			if ((absA = fabs(A[k*N + i])) > maxA) {
				maxA = absA;
				imax = k;
			}

		if (maxA < Tol)
			return false; //failure, matrix is degenerate

		if (imax != i) {
			//pivoting P
			int j = P[i];
			P[i] = P[imax];
			P[imax] = j;

			//pivoting rows of A
			for (int j = 0; j < N; j++) {
				T tmp = A[i*N + j];
				A[i*N + j] = A[imax*N + j];
				A[imax*N + j] = tmp;
			}

			//counting pivots starting from N (for determinant)
			P[N]++;
		}

		for (int j = i + 1; j < N; j++) {
			A[j*N + i] /= A[i*N + i];

			for (int k = i + 1; k < N; k++)
				A[j*N + k] -= A[j*N + i] * A[i*N + k];
		}
	}

	return true;  //decomposition done 
}


// Modified version of https://en.wikipedia.org/wiki/LU_decomposition#C_code_examples
/* INPUT: A - array of pointers to rows of a square matrix having dimension N
*        Tol - small tolerance number to detect failure when the matrix is near degenerate
* OUTPUT: Matrix A is changed, it contains both matrices L-E and U as A=(L-E)+U such that P*A=L*U.
*        The permutation matrix is not stored as a matrix, but in an integer vector P of size N+1
*        containing column indexes where the permutation matrix has "1". The last element P[N]=S+N,
*        where S is the number of row exchanges needed for determinant computation, det(P)=(-1)^S
*/
template<typename T>
PLL_DEVHOST bool LUPDecompose(int N, T *A, T Tol, int *P) {
	for (int i = 0; i <= N; i++)
		P[i] = i; //Unit permutation matrix, P[N] initialized with N

	for (int i = 0; i < N; i++) {
		T maxA = 0.0, absA;
		int imax = i;

		for (int k = i; k < N; k++)
			if ((absA = fabs(A[k*N + i])) > maxA) {
				maxA = absA;
				imax = k;
			}

		if (maxA < Tol)
			return false; //failure, matrix is degenerate

		if (imax != i) {
			//pivoting P
			int j = P[i];
			P[i] = P[imax];
			P[imax] = j;

			//pivoting rows of A
			for (int j = 0; j < N; j++) {
				T tmp = A[i*N + j];
				A[i*N + j] = A[imax*N + j];
				A[imax*N + j] = tmp;
			}

			//counting pivots starting from N (for determinant)
			P[N]++;
		}

		for (int j = i + 1; j < N; j++) {
			A[j*N + i] /= A[i*N + i];

			for (int k = i + 1; k < N; k++)
				A[j*N + k] -= A[j*N + i] * A[i*N + k];
		}
	}

	return true;  //decomposition done 
}

/* INPUT: A,P filled in LUPDecompose; b - rhs vector; N - dimension
* OUTPUT: x - solution vector of A*x=b
*/
template<typename T, int N>
PLL_DEVHOST void LUPSolve(const T* A, const int *P, T *b, T* x) {
	for (int i = 0; i < N; i++) {
		x[i] = b[P[i]];

		for (int k = 0; k < i; k++)
			x[i] -= A[i*N + k] * x[k];
	}

	for (int i = N - 1; i >= 0; i--) {
		for (int k = i + 1; k < N; k++)
			x[i] -= A[i*N + k] * x[k];

		x[i] = x[i] / A[i*N + i];
	}
}


/* INPUT: A,P filled in LUPDecompose; N - dimension
* OUTPUT: IA is the inverse of the initial matrix
*/
template<typename T, int N>
PLL_DEVHOST void LUPInvert(T(&A)[N*N], int(&P)[N + 1], T(&IA)[N*N]) {

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			if (P[i] == j)
				IA[i*N + j] = 1.0;
			else
				IA[i*N + j] = 0.0;

			for (int k = 0; k < i; k++)
				IA[i*N + j] -= A[i*N + k] * IA[k*N + j];
		}

		for (int i = N - 1; i >= 0; i--) {
			for (int k = i + 1; k < N; k++)
				IA[i*N + j] -= A[i*N + k] * IA[k*N + j];

			IA[i*N + j] = IA[i*N + j] / A[i*N + i];
		}
	}
}


/* INPUT: A,P filled in LUPDecompose; N - dimension
* OUTPUT: IA is the inverse of the initial matrix. P needs N+1 space
*/
template<typename T>
PLL_DEVHOST void LUPInvert(int N, T* A, int *P, T *IA) {

	for (int j = 0; j < N; j++) {
		for (int i = 0; i < N; i++) {
			if (P[i] == j)
				IA[i*N + j] = 1.0;
			else
				IA[i*N + j] = 0.0;

			for (int k = 0; k < i; k++)
				IA[i*N + j] -= A[i*N + k] * IA[k*N + j];
		}

		for (int i = N - 1; i >= 0; i--) {
			for (int k = i + 1; k < N; k++)
				IA[i*N + j] -= A[i*N + k] * IA[k*N + j];

			IA[i*N + j] = IA[i*N + j] / A[i*N + i];
		}
	}
}

/* INPUT: A,P filled in LUPDecompose; N - dimension.
* OUTPUT: Function returns the determinant of the initial matrix
*/
template<typename T>
T LUPDeterminant(T *A, int *P, int N) {

	T det = A[0];

	for (int i = 1; i < N; i++)
		det *= A[i*N + i];

	if ((P[N] - N) % 2 == 0)
		return det;
	else
		return -det;
}

template<typename T, int N2>
PLL_DEVHOST bool InvertMatrix(const T(&A)[N2], T(&out)[N2], T tol = 1e-9f) {
	const size_t N = CompileTimeSqrt(N2);
	int P[N + 1];

	T copy[N2];
	for (size_t i = 0; i < N2; i++)
		copy[i] = A[i];

	if (!LUPDecompose<T, N>(copy, tol, P))
		return false;

	LUPInvert<T, N>(copy, P, out);
	return true;
}

// P needs N+1 space
// A is modified in place
template<typename T>
PLL_DEVHOST bool InvertMatrix(int N, T* A, int* P, T  *out, T tol = 1e-9f)
{
	if (!LUPDecompose(N, A , tol, P))
		return false;

	LUPInvert<T>(N, A, P, out);
	return true;
}



template<typename T, int N2>
PLL_DEVHOST bool InvertMatrix(const Vector<T, N2>& in, Vector<T, N2>& out, T tol = 1e-9f) {
	return InvertMatrix<T, N2>(in.elem, out.elem, tol);
}

template<typename T, int N2>
PLL_DEVHOST Vector<T,N2> InvertMatrix(const Vector<T, N2>& in, T tol = 1e-9f) {
	Vector<T, N2> r;
	if (!InvertMatrix<T, N2>(in.elem, r.elem, tol))
		r.setInf();
	return r;
}


template<typename T, int InSize, int OutSize>
PLL_DEVHOST void CopySubMatrix(const Vector<T, InSize>& in, int startRow, int startCol, Vector<T, OutSize>& out)
{
	constexpr int w_out = CompileTimeSqrt(OutSize);
	constexpr int w_in = CompileTimeSqrt(InSize);

	for (int row=0; row<w_out; row++)
		for (int col = 0; col < w_out; col++)
			out[row * w_out + col] = in[(row + startRow) * w_in + (col + startCol)];
}


template<typename T1,typename T2, int N2>
PLL_DEVHOST auto MultiplyMatrix(const Vector<T1, N2>& a, const Vector<T2, N2>& b) -> Vector< decltype(T1()*T2()), N2 > {
	constexpr int N = CompileTimeSqrt(N2);
	typedef decltype(T1()*T2()) RT;
	Vector< RT, N2 > r;

	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
		{
			RT sum = {};
			for (int k = 0; k < N; k++)
				sum += a[i*N + k] * b[k*N + j];
			r[i*N + j] = sum;
		}
	return r;
}

template<typename T, int N2>
PLL_DEVHOST Vector<T, N2> TransposeMatrix(const Vector<T, N2>& a) {
	Vector<T, N2> r;
	constexpr int N = CompileTimeSqrt(N2);
	for (int i = 0; i < N; i++)
		for (int j = 0; j < N; j++)
			r[i*N + j] = a[j*N + i];
	return r;
}
