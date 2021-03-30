// SIMFLUX Estimator model
// 
// photonpy - Single molecule localization microscopy library
// © Jelmer Cnossen 2018-2021
// todo: rewrite everything because its awfully slow
#include "simflux/SIMFLUX.h"
#include "simflux/ExcitationModel.h"
#include "Estimators/Estimator.h"
#include "CudaUtils.h"
#include "Estimators/EstimatorImpl.h"
#include "SIMFLUX_PSF.h"
#include "simflux/SIMFLUX_Models.h"
#include "Estimators/Gaussian/GaussianPSFModels.h"
#include "Vector.h"
#include "DebugImageCallback.h"

// SIMFLUX-Single-Frame
class cuCFSFEstimator : public cuEstimator
{
public:
	struct Excitation {
		float exc;
		float deriv[3];
	};

	DeviceArray<Vector3f> d_offsets;
	cuEstimator* psf;
	Int2 roishape;
	bool simfluxMode;
	int patternsPerFrame, numFrames;

	struct DeviceBuffers {
		DeviceBuffers(int baseSmpCount, int baseSmpDims, int numspots, int psfParams, int numPatterns) :
			psf_ev(baseSmpCount* numspots*numPatterns), // [numPatterns, numspots, smpcount ]
			spot_params(psfParams*numspots*numPatterns), /// [ numPatterns, numspots, psfParams ]
			excitations(numPatterns* numspots),
			psf_deriv(baseSmpCount* numspots* psfParams*numPatterns), // [numPatterns, numspots, psfParams, smpcount] 
			numspots(numspots) {}
		DeviceArray<float> psf_ev;
		DeviceArray<float> spot_params;
		DeviceArray<float> psf_deriv;
		DeviceArray<Excitation> excitations; // q,dqdx,dqdy,dqdz
		int numspots;
	};
	std::mutex streamDataMutex;
	std::unordered_map<cudaStream_t, DeviceBuffers> streamData;

	DeviceBuffers* GetDeviceBuffers(cudaStream_t stream, int numspots)
	{
		return LockedFunction(streamDataMutex, [&]() {
			auto it = streamData.find(stream);

			if (it != streamData.end() && it->second.numspots < numspots) {
				streamData.erase(it);
				it = streamData.end();
			}

			if (it == streamData.end())
				it = streamData.emplace(stream, 
					DeviceBuffers(psf->SampleCount(), psf->SampleIndexDims(), numspots, psf->NumParams(), d_offsets.size())).first;
			return &it->second;
			});
	}

	cuCFSFEstimator(cuEstimator* psf, std::vector<Vector3f> offsets, int patternsPerFrame ,bool simfluxFit);

	virtual void ExpectedValue(float* expectedvalue, const float* d_params, const float* d_const,
		const int* d_roipos, int numspots, cudaStream_t stream) override;

	virtual void Derivatives(float* d_deriv, float* d_expectedvalue, const float* d_params,
		const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream) override;

	void UpdateBuffers(DeviceBuffers& db, const float* d_params, const float* d_const, const int* d_roipos, int numspots, bool deriv, cudaStream_t stream);
};


std::pair<std::string, std::vector<ParamLimit>> ParamList(cuEstimator* psf, bool simfluxFit, int npatterns)
{
	if (simfluxFit) {
		return { psf->ParamFormat(), psf->ParamLimits() };
	}
	else {
		std::string fmt = "x,y,";
		auto psflim = psf->ParamLimits();
		std::vector<ParamLimit> limits = { psflim[0], psflim[1] };

		// Add 3D?
		if (psflim.size() == 5) {
			fmt += "z,";
			limits.push_back(psflim[2]);
		}

		ParamLimit Ilimit = psflim[psf->ParamIndex("I")];
		for (int i = 0; i < npatterns; i++) {
			limits.push_back(Ilimit);
			fmt += SPrintf("I%d,", i);
		}

		fmt += "bg";
		limits.push_back(psflim[psf->ParamIndex("bg")]);
		return { fmt,limits };
	}
}

static  std::vector<int> makeSampleSize(std::vector<int> samplesize, int numframes) {
	samplesize.insert(samplesize.begin(), numframes);
	return samplesize;
}

cuCFSFEstimator::cuCFSFEstimator(cuEstimator* psf, std::vector<Vector3f> offsets, int patternsPerFrame, bool simfluxFit)
 : cuEstimator( makeSampleSize(psf->SampleSize(), offsets.size()/patternsPerFrame), simfluxFit ? 6*offsets.size() : 0, 0,
	ParamList(psf, simfluxFit,(int) offsets.size()).first.c_str(),
	ParamList(psf, simfluxFit, (int)offsets.size()).second),
	psf(psf), simfluxMode(simfluxFit), d_offsets(offsets),
	roishape(psf->SampleSize()),patternsPerFrame(patternsPerFrame), numFrames(offsets.size()/patternsPerFrame)
{
	assert(psf->SampleIndexDims() == 2);
	assert(psf->NumParams() == 4 || psf->NumParams()==5);
}


void cuCFSFEstimator::UpdateBuffers(DeviceBuffers& db, const float* d_params, const float* d_const, 
	const int* d_roipos, int numspots, bool deriv, cudaStream_t stream)
{
	// evaluate the actual Estimator
	float* d_psf_ev = db.psf_ev.data();
	float* d_psf_params = db.spot_params.data();

	int numPatterns = (int)d_offsets.size();
	int K = NumParams();
	const Vector3f* d_offset = this->d_offsets.data();
	int numcoords = psf->NumParams() - 2;
	int K_psf = psf->NumParams();
	bool simfluxMode = this->simfluxMode;

	if (simfluxMode)
	{
		SIMFLUX_Modulation* d_mod = (SIMFLUX_Modulation*)d_const;
		Excitation* d_exc = db.excitations.data();

		LaunchKernel(numPatterns, numspots, [=]__device__(int p, int i) {
			float* dst = &d_psf_params[p * (numspots * K_psf) + i * K_psf];

			for (int j = 0; j < numcoords; j++)
				dst[j] = d_params[i * K + j] + d_offset[p][j]; // xyz

			const int* roipos = &d_roipos[i * 3];
			SineWaveExcitation epModel(numPatterns, &d_mod[numPatterns*i]);
			float q;
			float xyz[3] = {
				d_params[i * K + 0] + roipos[2],
				d_params[i * K + 1] + roipos[1],
				numcoords == 3 ? d_params[i * K + 2] : 0.0f
			};
			//pos_[i] = xyz;
			float qderiv[3];
			epModel.ExcitationPattern(q, qderiv, p, xyz);
			d_exc[i * numPatterns + p] = { q, { qderiv[0],qderiv[1],qderiv[2]} };

			dst[numcoords + 0] = 1.0f; // I
			dst[numcoords + 1] = 0.0f; // bg
		}, 0, stream);
	}
	else
	{
		LaunchKernel(numPatterns, numspots, [=]__device__(int p, int i) {
			float* dst = &d_psf_params[p * (numspots * K_psf) + i * K_psf];

			for (int j = 0; j < numcoords; j++)
				dst[j] = d_params[i * K + j] + d_offset[p][j]; // xyz

			dst[numcoords + 0] = 1.0f; // I
			dst[numcoords + 1] = 0.0f; // bg
		}, 0, stream);
	}

	/*
	layout of
	d_psf_deriv: [numPatterns][numspots][numparams][roisize][roisize]
	d_psf_ev: [numPatterns][numspots][roisize][roisize]
	d_psf_params [numPatterns][numspots][numparams]
	*/

	// we could call this in one batch but then the constants and roipos would have to be duplicated as well
	int numpixels = psf->SampleCount();
	for (int i = 0; i < numPatterns; i++) {

		if (deriv) {
			psf->Derivatives(
				&db.psf_deriv.data()[numspots * K_psf * numpixels * i],
				&d_psf_ev[i * numspots * numpixels],
				&d_psf_params[K_psf * numspots * i], d_const, d_roipos, numspots, stream);
		}
		else {
			psf->ExpectedValue(
				&d_psf_ev[i * numspots * numpixels],
				&d_psf_params[K_psf * numspots * i], d_const, d_roipos, numspots, stream);
		}
	}


	/*
	auto psf_ev = db.psf_ev.ToVector();
	ShowDebugImage(psf->SampleSize()[1], psf->SampleSize()[0] * numspots, numPatterns, psf_ev.data(), "PSF EV");

	auto psf_deriv = db.psf_deriv.ToVector();
	ShowDebugImage(psf->SampleSize()[1], psf->SampleSize()[1] * K_psf, numspots * numPatterns, psf_deriv.data(), "PSF Deriv");
	*/
}

void cuCFSFEstimator::ExpectedValue(float* expectedvalue, const float* d_params, const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream)
{
	auto& db = *GetDeviceBuffers(stream, numspots);
	
	UpdateBuffers(db, d_params, d_const, d_roipos, numspots, false, stream);

	const float* psf_ev = db.psf_ev.data();
	int K = NumParams();
	int I_idx = psf->NumParams() - 2;
	int smpcount = SampleCount();
	int numPatterns = (int) d_offsets.size();
	bool simfluxMode = this->simfluxMode;
	Excitation* exc = db.excitations.data();
	int numpixels = roishape.prod();
	int nppf = patternsPerFrame;
	LaunchKernel(numpixels, numspots, numFrames, [=]__device__(int pixel, int roi, int frameIdx) {
		const float* roi_param = &d_params[K*roi];
		int smp = frameIdx * numpixels + pixel;
		if (simfluxMode) {
			float sum = 0.0f;
			for (int j = 0; j < nppf; j++)
				sum += psf_ev[numspots * numpixels * (frameIdx * nppf + j) + roi * numpixels + pixel] * exc[frameIdx * nppf + roi*numPatterns+j].exc; // PSF*I*Q(theta)
			expectedvalue[roi * smpcount + smp] = sum * roi_param[I_idx] + roi_param[K-1];
		}
		else {
			float sum = roi_param[K-1]; // bg
			for (int j = 0; j < nppf; j++)
				sum += psf_ev[numspots * numpixels * (frameIdx * nppf +j) + roi * numpixels + pixel] * roi_param[I_idx+j+ frameIdx * nppf]; // PSF*I_k
			expectedvalue[roi * smpcount + smp] = sum;
		}
	}, 0, stream);
}

void cuCFSFEstimator::Derivatives(float* d_deriv, float* d_expectedvalue, const float* d_params, const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream)
{
	auto& db = *GetDeviceBuffers(stream, numspots);

	UpdateBuffers(db, d_params, d_const, d_roipos, numspots, true, stream);

	const float* psf_ev = db.psf_ev.data();
	const float* psf_deriv = db.psf_deriv.data();
	int np = NumParams();
	int smpcount = SampleCount();
	int numPatterns = (int)d_offsets.size();
	int numcoords = psf->NumParams() - 2;
	const int I_idx = numcoords;
	int np_psf = psf->NumParams();
	bool simfluxMode = this->simfluxMode;

	Excitation* d_exc = db.excitations.data();

	int numpixels = roishape.prod();
	int nppf = patternsPerFrame;

	if (simfluxMode) {
		LaunchKernel(numpixels, numspots, numFrames, [=]__device__(int pixel, int spot, int frameIdx) {
			const float* roi_param = &d_params[np * spot];
			float sum_ev = 0.0f;

			int smp = frameIdx * numpixels + pixel;

			// Compute derivatives for X,Y, and optionally Z
			for (int j = 0; j < numcoords; j++) {
				float s = 0.0f;
				/*
				layout of
				d_psf_deriv: [numPatterns][numspots][numparams][roisize][roisize]
				d_psf_ev: [numPatterns][numspots][roisize][roisize]
				d_psf_params [numPatterns][numspots][numparams]
				*/
				for (int p = 0; p < nppf; p++) {
					float p_ev = psf_ev[numspots * numpixels * (p+frameIdx*nppf) + spot * numpixels + pixel];
					float sd = psf_deriv[numspots * np_psf * numpixels * (p+frameIdx*nppf) + spot * np_psf * numpixels + j * numpixels + pixel];

					Excitation e = d_exc[spot * numPatterns + (p+frameIdx*nppf)];
					s += (sd * e.exc + p_ev * e.deriv[j]) * roi_param[I_idx]; // 

					if (j == 0)
						sum_ev += p_ev * e.exc;// roi_param[I_idx];
				}
				d_deriv[spot * smpcount * np + j * smpcount + smp] = s;
			}

			// Compute derivatives for intensities
			d_deriv[spot * smpcount * np + I_idx * smpcount + smp] = sum_ev;
			sum_ev *= roi_param[I_idx];

			int bg_idx = np - 1;
			d_deriv[spot * smpcount * np + bg_idx * smpcount + smp] = 1.0f;// bg
			d_expectedvalue[spot * smpcount + smp] = sum_ev + roi_param[bg_idx];
		}, 0, stream);
	}
	else {
		LaunchKernel(numpixels, numspots, numFrames, [=]__device__(int pixel, int spot, int frameIdx) {
			const float* roi_param = &d_params[np * spot];
			float sum_ev = 0.0f;
			int smp = frameIdx * numpixels + pixel;

			// Compute derivatives for X,Y, and optionally Z
			for (int j = 0; j < numcoords; j++) {
				float s = 0.0f;
				for (int p = 0; p < nppf; p++) {
					float p_ev = psf_ev[numspots * numpixels * (p+frameIdx*nppf) + spot * numpixels + pixel];

					float sd = psf_deriv[numspots * np_psf * numpixels * (p + frameIdx * nppf) + spot * np_psf * numpixels + j * numpixels + pixel];
					s += sd * roi_param[I_idx + (p + frameIdx * nppf)]; // psf * I_k
					if (j == 0)
						sum_ev += p_ev * roi_param[I_idx + (p + frameIdx * nppf)];
				}
				d_deriv[spot * smpcount * np + j * smpcount + smp] = s;
			}

			// Compute derivatives for intensities
			for (int p = 0; p < nppf; p++)
				d_deriv[spot * smpcount * np + (I_idx + (p + frameIdx * nppf)) * smpcount + smp] = 
					psf_ev[numspots * numpixels * (p + frameIdx * nppf) + spot * numpixels + pixel];

			int bg_idx = np - 1;
			d_deriv[spot * smpcount * np + bg_idx * smpcount + smp] = 1.0f;// bg
			d_expectedvalue[spot * smpcount + smp] = sum_ev + roi_param[bg_idx];
		}, 0, stream);

	}
}


// SIMFLUX-Single-Frame
class cuSFSFEstimatorOld : public cuEstimator
{
public:
	struct Excitation {
		float exc;
		float deriv[3];
	};

	DeviceArray<Vector3f> d_offsets;
	cuEstimator* psf;
	bool simfluxMode;

	struct DeviceBuffers {
		DeviceBuffers(int baseSmpCount, int baseSmpDims, int numspots, int psfParams, int numPatterns) :
			psf_ev(baseSmpCount* numspots* numPatterns), // [numPatterns, numspots, smpcount ]
			spot_params(psfParams* numspots* numPatterns), /// [ numPatterns, numspots, psfParams ]
			excitations(numPatterns* numspots),
			psf_deriv(baseSmpCount* numspots* psfParams* numPatterns), // [numPatterns, numspots, psfParams, smpcount] 
			numspots(numspots) {}
		DeviceArray<float> psf_ev;
		DeviceArray<float> spot_params;
		DeviceArray<float> psf_deriv;
		DeviceArray<Excitation> excitations; // q,dqdx,dqdy,dqdz
		int numspots;
	};
	std::mutex streamDataMutex;
	std::unordered_map<cudaStream_t, DeviceBuffers> streamData;

	DeviceBuffers* GetDeviceBuffers(cudaStream_t stream, int numspots)
	{
		return LockedFunction(streamDataMutex, [&]() {
			auto it = streamData.find(stream);

			if (it != streamData.end() && it->second.numspots < numspots) {
				streamData.erase(it);
				it = streamData.end();
			}

			if (it == streamData.end())
				it = streamData.emplace(stream,
					DeviceBuffers(psf->SampleCount(), psf->SampleIndexDims(), numspots, psf->NumParams(), d_offsets.size())).first;
			return &it->second;
			});
	}

	cuSFSFEstimatorOld(cuEstimator* psf, std::vector<Vector3f> offsets, bool simfluxFit);

	virtual void ExpectedValue(float* expectedvalue, const float* d_params, const float* d_const,
		const int* d_roipos, int numspots, cudaStream_t stream) override;

	virtual void Derivatives(float* d_deriv, float* d_expectedvalue, const float* d_params,
		const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream) override;

	void UpdateBuffers(DeviceBuffers& db, const float* d_params, const float* d_const, const int* d_roipos, int numspots, bool deriv, cudaStream_t stream);
};



cuSFSFEstimatorOld::cuSFSFEstimatorOld(cuEstimator* psf, std::vector<Vector3f> offsets, bool simfluxFit)
	: cuEstimator(psf->SampleSize(), simfluxFit ? 6 * offsets.size() : 0, 0,
		ParamList(psf, simfluxFit, (int)offsets.size()).first.c_str(),
		ParamList(psf, simfluxFit, (int)offsets.size()).second),
	psf(psf), simfluxMode(simfluxFit), d_offsets(offsets)
{
	assert(psf->SampleIndexDims() == 2);
	assert(psf->NumParams() == 4 || psf->NumParams() == 5);
}


void cuSFSFEstimatorOld::UpdateBuffers(DeviceBuffers& db, const float* d_params, const float* d_const, const int* d_roipos, int numspots, bool deriv, cudaStream_t stream)
{
	// evaluate the actual Estimator
	float* d_psf_ev = db.psf_ev.data();
	float* d_psf_params = db.spot_params.data();

	int numPatterns = (int)d_offsets.size();
	int K = NumParams();
	const Vector3f* d_offset = this->d_offsets.data();
	int numcoords = psf->NumParams() - 2;
	int K_psf = psf->NumParams();
	bool simfluxMode = this->simfluxMode;

	if (simfluxMode)
	{
		SIMFLUX_Modulation* d_mod = (SIMFLUX_Modulation*)d_const;
		Excitation* d_exc = db.excitations.data();

		LaunchKernel(numPatterns, numspots, [=]__device__(int p, int i) {
			float* dst = &d_psf_params[p * (numspots * K_psf) + i * K_psf];

			for (int j = 0; j < numcoords; j++)
				dst[j] = d_params[i * K + j] + d_offset[p][j]; // xyz

			const int* roipos = &d_roipos[i * 3];
			SineWaveExcitation epModel(numPatterns, &d_mod[numPatterns * i]);
			float q;
			float xyz[3] = {
				d_params[i * K + 0] + roipos[2],
				d_params[i * K + 1] + roipos[1],
				numcoords == 3 ? d_params[i * K + 2] : 0.0f
			};
			//pos_[i] = xyz;
			float qderiv[3];
			epModel.ExcitationPattern(q, qderiv, p, xyz);
			d_exc[i * numPatterns + p] = { q, { qderiv[0],qderiv[1],qderiv[2]} };

			dst[numcoords + 0] = 1.0f; // I
			dst[numcoords + 1] = 0.0f; // bg
		}, 0, stream);
	}
	else
	{
		LaunchKernel(numPatterns, numspots, [=]__device__(int p, int i) {
			float* dst = &d_psf_params[p * (numspots * K_psf) + i * K_psf];

			for (int j = 0; j < numcoords; j++)
				dst[j] = d_params[i * K + j] + d_offset[p][j]; // xyz

			dst[numcoords + 0] = 1.0f; // I
			dst[numcoords + 1] = 0.0f; // bg
		}, 0, stream);
	}

	int smpcount = psf->SampleCount();
	for (int i = 0; i < numPatterns; i++) {
		if (deriv) {
			psf->Derivatives(
				&db.psf_deriv.data()[numspots * K_psf * smpcount * i],
				&d_psf_ev[i * numspots * smpcount],
				&d_psf_params[K_psf * numspots * i], d_const, d_roipos, numspots, stream);
		}
		else {
			psf->ExpectedValue(
				&d_psf_ev[i * numspots * smpcount],
				&d_psf_params[K_psf * numspots * i], d_const, d_roipos, numspots, stream);
		}
	}

	/*
	auto psf_ev = db.psf_ev.ToVector();
	ShowDebugImage(psf->SampleSize()[1], psf->SampleSize()[0] * numspots, numPatterns, psf_ev.data(), "PSF EV");

	auto psf_deriv = db.psf_deriv.ToVector();
	ShowDebugImage(psf->SampleSize()[1], psf->SampleSize()[1] * K_psf, numspots * numPatterns, psf_deriv.data(), "PSF Deriv");
	*/
}

void cuSFSFEstimatorOld::ExpectedValue(float* expectedvalue, const float* d_params, const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream)
{
	auto& db = *GetDeviceBuffers(stream, numspots);

	UpdateBuffers(db, d_params, d_const, d_roipos, numspots, false, stream);

	const float* psf_ev = db.psf_ev.data();
	int K = NumParams();
	int I_idx = psf->NumParams() - 2;
	int smpcount = SampleCount();
	int numPatterns = (int)d_offsets.size();
	bool simfluxMode = this->simfluxMode;
	Excitation* exc = db.excitations.data();
	LaunchKernel(smpcount, numspots, [=]__device__(int smp, int roi) {
		const float* roi_param = &d_params[K * roi];
		if (simfluxMode) {
			float sum = 0.0f;
			for (int j = 0; j < numPatterns; j++)
				sum += psf_ev[numspots * smpcount * j + roi * smpcount + smp] * exc[roi * numPatterns + j].exc; // PSF*I*Q(theta)
			expectedvalue[roi * smpcount + smp] = sum * roi_param[I_idx] + roi_param[K - 1];
		}
		else {
			float sum = roi_param[K - 1]; // bg
			for (int j = 0; j < numPatterns; j++)
				sum += psf_ev[numspots * smpcount * j + roi * smpcount + smp] * roi_param[I_idx + j]; // PSF*I_k
			expectedvalue[roi * smpcount + smp] = sum;
		}
	}, 0, stream);
}

void cuSFSFEstimatorOld::Derivatives(float* d_deriv, float* d_expectedvalue, const float* d_params, const float* d_const, const int* d_roipos, int numspots, cudaStream_t stream)
{
	auto& db = *GetDeviceBuffers(stream, numspots);

	UpdateBuffers(db, d_params, d_const, d_roipos, numspots, true, stream);

	const float* psf_ev = db.psf_ev.data();
	const float* psf_deriv = db.psf_deriv.data();
	int np = NumParams();
	int smpcount = SampleCount();
	int numPatterns = (int)d_offsets.size();
	int numcoords = psf->NumParams() - 2;
	const int I_idx = numcoords;
	int np_psf = psf->NumParams();
	bool simfluxMode = this->simfluxMode;

	Excitation* d_exc = db.excitations.data();

	if (simfluxMode) {
		LaunchKernel(smpcount, numspots, [=]__device__(int smp, int spot) {
			const float* roi_param = &d_params[np * spot];
			float sum_ev = 0.0f;

			// Compute derivatives for X,Y, and optionally Z
			for (int j = 0; j < numcoords; j++) {
				float s = 0.0f;
				for (int p = 0; p < numPatterns; p++) {
					float p_ev = psf_ev[numspots * smpcount * p + spot * smpcount + smp];
					float sd = psf_deriv[numspots * np_psf * smpcount * p + spot * np_psf * smpcount + j * smpcount + smp];

					Excitation e = d_exc[spot * numPatterns + p];
					s += (sd * e.exc + p_ev * e.deriv[j]) * roi_param[I_idx]; // 

					if (j == 0)
						sum_ev += p_ev * e.exc;// roi_param[I_idx];
				}
				d_deriv[spot * smpcount * np + j * smpcount + smp] = s;
			}

			// Compute derivatives for intensities
			d_deriv[spot * smpcount * np + I_idx * smpcount + smp] = sum_ev;
			sum_ev *= roi_param[I_idx];

			int bg_idx = np - 1;
			d_deriv[spot * smpcount * np + bg_idx * smpcount + smp] = 1.0f;// bg
			d_expectedvalue[spot * smpcount + smp] = sum_ev + roi_param[bg_idx];
		}, 0, stream);
	}
	else {
		LaunchKernel(smpcount, numspots, [=]__device__(int smp, int spot) {
			const float* roi_param = &d_params[np * spot];
			float sum_ev = 0.0f;

			// Compute derivatives for X,Y, and optionally Z
			for (int j = 0; j < numcoords; j++) {
				float s = 0.0f;
				for (int p = 0; p < numPatterns; p++) {
					float p_ev = psf_ev[numspots * smpcount * p + spot * smpcount + smp];

					float sd = psf_deriv[numspots * np_psf * smpcount * p + spot * np_psf * smpcount + j * smpcount + smp];
					s += sd * roi_param[I_idx + p]; // psf * I_k
					if (j == 0)
						sum_ev += p_ev * roi_param[I_idx + p];
				}
				d_deriv[spot * smpcount * np + j * smpcount + smp] = s;
			}

			// Compute derivatives for intensities
			for (int p = 0; p < numPatterns; p++)
				d_deriv[spot * smpcount * np + (I_idx + p) * smpcount + smp] = psf_ev[numspots * smpcount * p + spot * smpcount + smp];

			int bg_idx = np - 1;
			d_deriv[spot * smpcount * np + bg_idx * smpcount + smp] = 1.0f;// bg
			d_expectedvalue[spot * smpcount + smp] = sum_ev + roi_param[bg_idx];
		}, 0, stream);

	}
}

// offsets[num_patterns]
CDLL_EXPORT Estimator* CFSF_CreateEstimator(Estimator* psf,
	const Vector3f* offsets, int patternsPerFrame, int numPatterns, bool simfluxFit, Context* ctx)
{
	try {
		cuEstimator* original_cuda = psf->Unwrap();
		if (!original_cuda) return 0;

		std::vector<Vector3f> offsets_(offsets, offsets + numPatterns);

		cuEstimator* e;
		
		if (patternsPerFrame == -1)
			e= new cuSFSFEstimatorOld(original_cuda, offsets_, !!simfluxFit);
		else
			e=new cuCFSFEstimator(original_cuda, offsets_, patternsPerFrame, !!simfluxFit);

		Estimator* sf_estim = new cuEstimatorWrapper(e);

		if (ctx) sf_estim->SetContext(ctx);
		return sf_estim;
	}
	catch (const std::runtime_error & e) {
		DebugPrintf("%s\n", e.what());
		return 0;
	}
}

