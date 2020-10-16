#ifndef AVGPROCESS_H
#define AVGPROCESS_H

#include <map>
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
#include <hipfft.h>
#else
#include <cufft.h>
#endif

#include "../default.h"
#include "../hip/HipVariables.h"
#include "EMFile.h"
//#include "../hip/HipKernel.h"
//#include "../hip/HipArrays.h"
//#include "../hip/HipTextures.h"
//#include "../hip/HipReducer.h"
//#include "../hip/HipRot.h"
//#include "../hip/HipBasicKernel.h"
#include "../hip/HipContext.h"
#include "Kernels.h"

using namespace std;

struct maxVals_t {
  float ccVal;
  int index;
  float rphi;
  float rpsi;
  float rthe;

  void getXYZ(int size, int &x, int &y, int &z);
  static int getIndex(int size, int x, int y, int z);
  size_t ref;
};

class AvgProcess {
private:
public:
  // virtual ~AvgProcessC2C();
  virtual maxVals_t execute(float *_data, float *wedge, float *filter,
                            float oldphi, float oldpsi, float oldtheta,
                            float rDown, float rUp, float smooth,
                            float3 oldShift, bool couplePhiToPsi,
                            bool computeCCValOnly, int oldIndex) {
    maxVals_t t;
    return t;
  };
};

class KernelModuls {
private:
  bool compilerOutput;
  bool infoOutput;

public:
  KernelModuls(Hip::HipContext *aHipCtx);
  hipModule_t modbasicKernels;
  hipModule_t modkernel;
};

class AvgProcessOriginalBinaryKernels : public AvgProcess {
private:
	KernelModuls modules;

	bool binarizeMask;
	bool rotateMaskCC;
	bool useFilterVolume;
	float phi, psi, theta;
	float shiftX, shiftY, shiftZ;
	float maxCC;
	float phi_angiter;
	float phi_angincr;
	float angiter;
	float angincr;

	size_t sizeVol, sizeTot;

	float* mask;
	float* ref;
	float* ccMask;
	float* sum_h;
	int*   index;
	float2* sumCplx;

	hipStream_t stream;
	Hip::HipContext* ctx;

	/* AS Deprecated before HIP BinaryKernels
	HipRot rot;
	HipRot rotMask;
	HipRot rotMaskCC;
	HipReducer reduce;
	HipSub sub;
	HipMakeCplxWithSub makecplx;
	HipBinarize binarize;
	HipMul mul;
	HipFFT fft;
	HipMax max;
	*/

	HipDeviceVariable d_ffttemp;
	HipDeviceVariable d_particle, d_particleCplx, d_particleSqrCplx, d_particleCplx_orig, d_particleSqrCplx_orig, d_filter;
	HipDeviceVariable d_wedge;
	HipDeviceVariable d_reference, d_reference_orig, d_referenceCplx;
	HipDeviceVariable d_mask, d_mask_orig, d_maskCplx;
	HipDeviceVariable d_ccMask, d_ccMask_Orig;
	HipDeviceVariable d_buffer;
	HipDeviceVariable d_index;
	HipDeviceVariable nVox;
	HipDeviceVariable sum;
	HipDeviceVariable sumSqr;
	HipDeviceVariable maxVals;

	SubCplxKernel aSubCplxKernel;
	FFTShiftRealKernel aFFTShiftRealKernel;
	MakeCplxWithSubKernel aMakeCplxWithSubKernel;
	MulVolKernel aMulVolKernel;
	BandpassFFTShiftKernel aBandpassFFTShiftKernel;
	MakeRealKernel aMakeRealKernel;
	MulKernel aMulKernel;
	MakeCplxWithSqrSubKernel aMakeCplxWithSqrSubKernel;
	CorrelKernel aCorrelKernel;
	ConvKernel aConvKernel;
	EnergyNormKernel aEnergyNormKernel;
	FFTShift2Kernel aFFTShift2Kernel;
	BinarizeKernel aBinarizeKernel;
	SubKernel aSubKernel;
	MaxKernel aMaxKernel;

	Reducer aReduceKernel;

	RotateKernel aRotateKernelROT;
	RotateKernel aRotateKernelMASK;
	RotateKernel aRotateKernelMASKCC;

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
	hipfftHandle ffthandle;
#else
	cufftHandle ffthandle;
#endif

public:
  AvgProcessOriginalBinaryKernels(size_t _sizeVol, hipStream_t _stream,
                     Hip::HipContext *_ctx, float *_mask, float *_ref,
                     float *_ccMask, float aPhiAngIter, float aPhiAngInc,
                     float aAngIter, float aAngIncr, bool aBinarizeMask,
                     bool aRotateMaskCC, bool aUseFilterVolume,
                     bool linearInterpolation, KernelModuls modules);
  ~AvgProcessOriginalBinaryKernels();

  maxVals_t execute(float *_data, float *wedge, float *filter, float oldphi,
                    float oldpsi, float oldtheta, float rDown, float rUp,
                    float smooth, float3 oldShift, bool couplePhiToPsi,
                    bool computeCCValOnly, int oldIndex);
  // maxVals_t executeMaxAll(float* _data, float oldphi, float oldpsi, float
  // oldtheta, float rDown, float rUp, float smooth, vector<float>& allCCs);
};

class AvgProcessOriginal : public AvgProcess {
private:
  bool binarizeMask;
  bool rotateMaskCC;
  bool useFilterVolume;
  float phi, psi, theta;
  float shiftX, shiftY, shiftZ;
  float maxCC;
  float phi_angiter;
  float phi_angincr;
  float angiter;
  float angincr;

  size_t sizeVol, sizeTot;

  float *mask;
  float *ref;
  float *ccMask;
  float *sum_h;
  int *index;
  float2 *sumCplx;

  hipStream_t stream;
  Hip::HipContext *ctx;

  HipDeviceVariable d_ffttemp;
  HipDeviceVariable d_particle, d_particleCplx, d_particleSqrCplx,
      d_particleCplx_orig, d_particleSqrCplx_orig, d_filter;
  HipDeviceVariable d_wedge;
  HipDeviceVariable d_reference, d_reference_orig, d_referenceCplx;
  HipDeviceVariable d_mask, d_mask_orig, d_maskCplx;
  HipDeviceVariable d_ccMask, d_ccMask_Orig;
  HipDeviceVariable d_buffer;
  HipDeviceVariable d_index;
  HipDeviceVariable nVox;
  HipDeviceVariable sum;
  HipDeviceVariable sumSqr;
  HipDeviceVariable maxVals;

  Reducer aReduceKernel;
  RotateKernel aRotateKernelROT;
  RotateKernel aRotateKernelMASK;
  RotateKernel aRotateKernelMASKCC;

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftHandle ffthandle;
#else
  cufftHandle ffthandle;
#endif

public:
  AvgProcessOriginal(size_t _sizeVol, hipStream_t _stream,
                     Hip::HipContext *_ctx, float *_mask, float *_ref,
                     float *_ccMask, float aPhiAngIter, float aPhiAngInc,
                     float aAngIter, float aAngIncr, bool aBinarizeMask,
                     bool aRotateMaskCC, bool aUseFilterVolume,
                     bool linearInterpolation, KernelModuls modules);
  ~AvgProcessOriginal();

  maxVals_t execute(float *_data, float *wedge, float *filter, float oldphi,
                    float oldpsi, float oldtheta, float rDown, float rUp,
                    float smooth, float3 oldShift, bool couplePhiToPsi,
                    bool computeCCValOnly, int oldIndex);
  // maxVals_t executeMaxAll(float* _data, float oldphi, float oldpsi, float
  // oldtheta, float rDown, float rUp, float smooth, vector<float>& allCCs);
};

class AvgProcessC2C : public AvgProcess {
private:
  // KernelModuls modules;

  bool binarizeMask;
  bool rotateMaskCC;
  bool useFilterVolume;
  float phi, psi, theta;
  float shiftX, shiftY, shiftZ;
  float maxCC;
  float phi_angiter;
  float phi_angincr;
  float angiter;
  float angincr;

  size_t sizeVol, sizeTot;

  float *mask;
  float *ref;
  float *ccMask;
  float *sum_h;
  float *sum_mask;
  float *sum_ref;
  int *index;
  float2 *sumCplx;

  hipStream_t stream;
  Hip::HipContext *ctx;

  HipDeviceVariable d_ffttemp;
  HipDeviceVariable d_particle, d_particleCplx, d_particleSqrCplx,
      d_particleCplx_orig, d_particleSqrCplx_orig, d_filter;
  HipDeviceVariable d_wedge, d_manipulation, d_manipulation_ref;
  HipDeviceVariable d_reference, d_reference_orig, d_referenceCplx;
  HipDeviceVariable d_mask, d_mask_orig, d_maskCplx;
  HipDeviceVariable d_ccMask, d_ccMask_Orig;
  HipDeviceVariable d_buffer;
  HipDeviceVariable d_index;
  HipDeviceVariable nVox;
  HipDeviceVariable sum;
  HipDeviceVariable sumSqr;
  HipDeviceVariable maxVals;

  Reducer aReduceKernel;
  RotateKernel aRotateKernelROT;
  RotateKernel aRotateKernelMASK;
  RotateKernel aRotateKernelMASKCC;

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftHandle ffthandle;
#else
  cufftHandle ffthandle;
#endif

public:
  AvgProcessC2C(size_t _sizeVol, hipStream_t _stream, Hip::HipContext *_ctx,
                float *_mask, float *_ref, float *_ccMask, float aPhiAngIter,
                float aPhiAngInc, float aAngIter, float aAngIncr,
                bool aBinarizeMask, bool aRotateMaskCC, bool aUseFilterVolume,
                bool linearInterpolation, KernelModuls modules);
  ~AvgProcessC2C();

  maxVals_t execute(float *_data, float *wedge, float *filter, float oldphi,
                    float oldpsi, float oldtheta, float rDown, float rUp,
                    float smooth, float3 oldShift, bool couplePhiToPsi,
                    bool computeCCValOnly, int oldIndex);
  // maxVals_t executeMaxAll(float* _data, float oldphi, float oldpsi, float
  // oldtheta, float rDown, float rUp, float smooth, vector<float>& allCCs);
};

class AvgProcessReal2Complex : public AvgProcess {
private:
  // KernelModuls modules;

  bool binarizeMask;
  bool rotateMaskCC;
  bool useFilterVolume;
  float phi, psi, theta;
  float shiftX, shiftY, shiftZ;
  float maxCC;
  float phi_angiter;
  float phi_angincr;
  float angiter;
  float angincr;

  size_t sizeVol, sizeTot;

  float *mask;
  float *ref;
  float *ccMask;
  float *sum_h;
  float *sum_mask;
  float *sum_ref;
  int *index;
  float2 *sumCplx;

  hipStream_t stream;
  Hip::HipContext *ctx;

  HipDeviceVariable d_ffttemp;
  HipDeviceVariable d_particle, d_particle_sqr, d_particleCplx, d_particleSqrCplx, d_particleCplx_orig_RC,
      d_particleCplx_orig, d_particleSqrCplx_orig, d_filter;
  HipDeviceVariable d_wedge, d_manipulation, d_manipulation_ref;
  HipDeviceVariable d_reference, d_reference_orig, d_referenceCplx, d_referenceCplx_RC; 
  HipDeviceVariable d_mask, d_mask_orig, d_maskCplx;
  HipDeviceVariable d_ccMask, d_ccMask_Orig;
  HipDeviceVariable d_buffer;
  HipDeviceVariable d_index;
  HipDeviceVariable nVox;
  HipDeviceVariable sum;
  HipDeviceVariable sumSqr;
  HipDeviceVariable maxVals;

  Reducer aReduceKernel;
  RotateKernel aRotateKernelROT;
  RotateKernel aRotateKernelMASK;
  RotateKernel aRotateKernelMASKCC;

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftHandle ffthandle;
#else
  cufftHandle ffthandle;
#endif

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftHandle ffthandle_R2C;
  hipfftHandle ffthandle_C2R;
#else
  cufftHandle ffthandle_R2C;
  cufftHandle ffthandle_C2R;
#endif

public:
  AvgProcessReal2Complex(size_t _sizeVol, hipStream_t _stream, Hip::HipContext *_ctx,
                float *_mask, float *_ref, float *_ccMask, float aPhiAngIter,
                float aPhiAngInc, float aAngIter, float aAngIncr,
                bool aBinarizeMask, bool aRotateMaskCC, bool aUseFilterVolume,
                bool linearInterpolation, KernelModuls modules);
  ~AvgProcessReal2Complex();

  maxVals_t execute(float *_data, float *wedge, float *filter, float oldphi,
                    float oldpsi, float oldtheta, float rDown, float rUp,
                    float smooth, float3 oldShift, bool couplePhiToPsi,
                    bool computeCCValOnly, int oldIndex);
  // maxVals_t executeMaxAll(float* _data, float oldphi, float oldpsi, float
  // oldtheta, float rDown, float rUp, float smooth, vector<float>& allCCs);
};

class AvgProcessR2C : public AvgProcess {
private:
  // KernelModuls modules;

  bool binarizeMask;
  bool rotateMaskCC;
  bool useFilterVolume;
  float phi, psi, theta;
  float shiftX, shiftY, shiftZ;
  float maxCC;
  float phi_angiter;
  float phi_angincr;
  float angiter;
  float angincr;

  size_t sizeVol, sizeTot;

  float *mask;
  float *ref;
  float *ccMask;
  float *sum_h;
  float *sum_mask;
  float *sum_ref;
  int *index;
  float2 *sumCplx;

  hipStream_t stream;
  Hip::HipContext *ctx;

  HipDeviceVariable d_ffttemp;
  HipDeviceVariable d_particle, d_particle_sqr, d_particleCplx_RC,
      d_particleSqrCplx_RC, d_particleCplx_orig_RC, d_particleSqrCplx_orig_RC,
      d_filter;
  HipDeviceVariable d_wedge;
  HipDeviceVariable d_reference, d_reference_orig, d_referenceCplx_RC;
  HipDeviceVariable d_mask, d_mask_orig, d_maskCplx, d_maskCplx_RC;
  HipDeviceVariable d_ccMask, d_ccMask_Orig;
  HipDeviceVariable d_buffer;
  HipDeviceVariable d_index;
  HipDeviceVariable nVox;
  HipDeviceVariable sum;
  HipDeviceVariable sumSqr;
  HipDeviceVariable maxVals;

  Reducer aReduceKernel;
  RotateKernel aRotateKernelROT;
  RotateKernel aRotateKernelMASK;
  RotateKernel aRotateKernelMASKCC;

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftHandle ffthandle_R2C;
  hipfftHandle ffthandle_C2R;
#else
  cufftHandle ffthandle_R2C;
  cufftHandle ffthandle_C2R;
#endif

public:
  AvgProcessR2C(size_t _sizeVol, hipStream_t _stream, Hip::HipContext *_ctx,
                float *_mask, float *_ref, float *_ccMask, float aPhiAngIter,
                float aPhiAngInc, float aAngIter, float aAngIncr,
                bool aBinarizeMask, bool aRotateMaskCC, bool aUseFilterVolume,
                bool linearInterpolation, KernelModuls modules);
  ~AvgProcessR2C();

  maxVals_t execute(float *_data, float *wedge, float *filter, float oldphi,
                    float oldpsi, float oldtheta, float rDown, float rUp,
                    float smooth, float3 oldShift, bool couplePhiToPsi,
                    bool computeCCValOnly, int oldIndex);

  // maxVals_t executeMaxAll(float* _data, float oldphi, float oldpsi, float
  // oldtheta, float rDown, float rUp, float smooth, vector<float>& allCCs);
};


class AvgProcessPhaseCorrelation : public AvgProcess {
private:
  // KernelModuls modules;

  bool binarizeMask;
  bool rotateMaskCC;
  bool useFilterVolume;
  float phi, psi, theta;
  float shiftX, shiftY, shiftZ;
  float maxCC;
  float phi_angiter;
  float phi_angincr;
  float angiter;
  float angincr;

  size_t sizeVol, sizeTot;

  float *mask;
  float *ref;
  float *ccMask;
  float *sum_h;
  float *sum_mask;
  float *sum_ref;
  int *index;
  float2 *sumCplx;

  hipStream_t stream;
  Hip::HipContext *ctx;

  HipDeviceVariable d_ffttemp;
  HipDeviceVariable d_particle, d_particle_sqr, d_particleCplx_RC,
      d_particleSqrCplx_RC, d_particleCplx_orig_RC, d_particleSqrCplx_orig_RC,
      d_filter;
  HipDeviceVariable d_wedge;
  HipDeviceVariable d_reference, d_reference_orig, d_referenceCplx_RC;
  HipDeviceVariable d_mask, d_mask_orig, d_maskCplx, d_maskCplx_RC;
  HipDeviceVariable d_ccMask, d_ccMask_Orig;
  HipDeviceVariable d_buffer;
  HipDeviceVariable d_index;
  HipDeviceVariable nVox;
  HipDeviceVariable sum;
  HipDeviceVariable sumSqr;
  HipDeviceVariable maxVals;

  Reducer aReduceKernel;
  RotateKernel aRotateKernelROT;
  RotateKernel aRotateKernelMASK;
  RotateKernel aRotateKernelMASKCC;

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftHandle ffthandle_R2C;
  hipfftHandle ffthandle_C2R;
#else
  cufftHandle ffthandle_R2C;
  cufftHandle ffthandle_C2R;
#endif

public:
  AvgProcessPhaseCorrelation(size_t _sizeVol, hipStream_t _stream, Hip::HipContext *_ctx,
                float *_mask, float *_ref, float *_ccMask, float aPhiAngIter,
                float aPhiAngInc, float aAngIter, float aAngIncr,
                bool aBinarizeMask, bool aRotateMaskCC, bool aUseFilterVolume,
                bool linearInterpolation, KernelModuls modules);
  ~AvgProcessPhaseCorrelation();

  maxVals_t execute(float *_data, float *wedge, float *filter, float oldphi,
                    float oldpsi, float oldtheta, float rDown, float rUp,
                    float smooth, float3 oldShift, bool couplePhiToPsi,
                    bool computeCCValOnly, int oldIndex);

  // maxVals_t executeMaxAll(float* _data, float oldphi, float oldpsi, float
  // oldtheta, float rDown, float rUp, float smooth, vector<float>& allCCs);
};

class AvgProcessR2C_Stream : public AvgProcess {
private:
  // KernelModuls modules;

  bool binarizeMask;
  bool rotateMaskCC;
  bool useFilterVolume;
  float phi, psi, theta;
  float shiftX, shiftY, shiftZ;
  float maxCC;
  float phi_angiter;
  float phi_angincr;
  float angiter;
  float angincr;

  size_t sizeVol, sizeTot;

  float *mask;
  float *ref;
  float *ccMask;
  float *sum_h;
  int *index;
  float2 *sumCplx;

  hipStream_t stream;
  Hip::HipContext *ctx;

  /*
  HipRot rot;
  HipRot rotMask;
  HipRot rotMaskCC;
  HipReducer reduce;
  HipMakeCplxWithSub makecplx;
  HipBinarize binarize;

  HipSub sub;
  HipSub_RC rc_sub;
  HipMul mul;
  HipMul_RC rc_mul;
  HipFFT fft;
  HipFFT_RC rc_fft;
  HipMax max;
  */

  HipDeviceVariable d_ffttemp;
  HipDeviceVariable d_particle, d_particle_sqr, d_particleCplx_RC,
      d_particleSqrCplx_RC, d_particleCplx_orig_RC, d_particleSqrCplx_orig_RC,
      d_filter;
  HipDeviceVariable d_wedge;
  HipDeviceVariable d_reference, d_reference_orig, d_referenceCplx_RC;
  HipDeviceVariable d_mask, d_mask_orig, d_maskCplx, d_maskCplx_RC;
  HipDeviceVariable d_ccMask, d_ccMask_Orig;
  HipDeviceVariable d_buffer;
  HipDeviceVariable d_index;
  HipDeviceVariable nVox;
  HipDeviceVariable sum;
  HipDeviceVariable sumSqr;
  HipDeviceVariable maxVals;

  /*
          FFTShiftRealKernel aFFTShiftRealKernel;
          MulVolKernel_RC aMulVolKernel_RC;
          MulVolKernel_RR aMulVolKernel_RR;
          BandpassFFTShiftKernel_RC aBandpassFFTShiftKernel_RC;
          MulKernel_Real aMulKernel_Real;
          CorrelKernel_RC aCorrelKernel_RC;
          ConvKernel_RC aConvKernel_RC;
          EnergyNormKernel_RC aEnergyNormKernel_RC;
          SubKernel_RC aSubKernel_RC;
          SubKernel aSubKernel;
          SqrSubKernel_RC aSqrSubKernel_RC;

          MaxKernel aMaxKernel;
          BinarizeKernel aBinarizeKernel;
          SubCplxKernel_RC aSubCplxKernel_RC;
  */
  Reducer aReduceKernel;

  RotateKernel aRotateKernelROT;
  RotateKernel aRotateKernelMASK;
  RotateKernel aRotateKernelMASKCC;

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftHandle ffthandle_R2C;
  hipfftHandle ffthandle_C2R;
#else
  cufftHandle ffthandle_R2C;
  cufftHandle ffthandle_C2R;
#endif

public:
  AvgProcessR2C_Stream(size_t _sizeVol, hipStream_t _stream,
                       Hip::HipContext *_ctx, float *_mask, float *_ref,
                       float *_ccMask, float aPhiAngIter, float aPhiAngInc,
                       float aAngIter, float aAngIncr, bool aBinarizeMask,
                       bool aRotateMaskCC, bool aUseFilterVolume,
                       bool linearInterpolation, KernelModuls modules);
  ~AvgProcessR2C_Stream();

  maxVals_t execute(float *_data, float *wedge, float *filter, float oldphi,
                    float oldpsi, float oldtheta, float rDown, float rUp,
                    float smooth, float3 oldShift, bool couplePhiToPsi,
                    bool computeCCValOnly, int oldIndex);

  // maxVals_t executeMaxAll(float* _data, float oldphi, float oldpsi, float
  // oldtheta, float rDown, float rUp, float smooth, vector<float>& allCCs);
};

// inline void __cufftSafeCall(hipfftResult_t err, const char *file, const int
// line)
// {
// 	if( HIPFFT_SUCCESS != err)
// 	{
// 		std::string errMsg;
// 		switch(err)
// 		{
// 		case HIPFFT_INVALID_PLAN:
// 			errMsg = "Invalid plan";
// 			break;
// 		case HIPFFT_ALLOC_FAILED:
// 			errMsg = "HIPFFT_ALLOC_FAILED";
// 			break;
// 		case HIPFFT_INVALID_TYPE:
// 			errMsg = "HIPFFT_INVALID_TYPE";
// 			break;
// 		case HIPFFT_INVALID_VALUE:
// 			errMsg = "HIPFFT_INVALID_VALUE";
// 			break;
// 		case HIPFFT_INTERNAL_ERROR:
// 			errMsg = "HIPFFT_INTERNAL_ERROR";
// 			break;
// 		case HIPFFT_EXEC_FAILED:
// 			errMsg = "HIPFFT_EXEC_FAILED";
// 			break;
// 		case HIPFFT_SETUP_FAILED:
// 			errMsg = "HIPFFT_SETUP_FAILED";
// 			break;
// 		case HIPFFT_INVALID_SIZE:
// 			errMsg = "HIPFFT_INVALID_SIZE";
// 			break;
// 		case HIPFFT_UNALIGNED_DATA:
// 			errMsg = "HIPFFT_UNALIGNED_DATA";
// 			break;
// 		case HIPFFT_INCOMPLETE_PARAMETER_LIST:
// 			errMsg = "HIPFFT_INCOMPLETE_PARAMETER_LIST";
// 			break;
// 		case HIPFFT_INVALID_DEVICE:
// 			errMsg = "HIPFFT_INVALID_DEVICE";
// 			break;
// 		case HIPFFT_PARSE_ERROR:
// 			errMsg = "HIPFFT_PARSE_ERROR";
// 			break;
// 		case HIPFFT_NO_WORKSPACE:
// 			errMsg = "HIPFFT_NO_WORKSPACE";
// 			break;
// 		}

// 		HipException ex(file, line, errMsg, (hipError_t)err);
// 		throw ex;
// 	} //if CUDA_SUCCESS
// }

#endif // AVGPROCESS_H
