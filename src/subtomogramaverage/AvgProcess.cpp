#include <bitset>
#include <chrono>
#include <fstream>
#include <iostream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "AvgProcess.h"
#include "../hip_kernels/kernels.cpp"

/* AS
 * AvgProcess.cpp now holds different versions of the Averaging Process.
 * In the past there was one process for a single particle and we added
 * different functions doing a simular task either in a different way (for 
 * speed, accuracy, ...). This allowed for easy testing and benchmarking of
 * different versions during my thesis.
 */ 

typedef std::chrono::high_resolution_clock Clock;

/* AS 
 * We use different Grid and Blocksize depending on the device (AMD or Nvidia) 
 * we want to run on.
 */
#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)
#define grid dim3(sizeVol / 32, sizeVol/16, sizeVol)
#define block dim3(32, 16, 1)
#define grid_RC dim3((sizeVol / 2 + 1) / (sizeVol / 2 + 1), sizeVol/16, sizeVol)
#define block_RC dim3(sizeVol / 2 + 1, 16, 1)
#endif

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
#define grid dim3(sizeVol / 32, sizeVol / 16, sizeVol)
#define block dim3(32, 16, 1)
#define grid_RC dim3((sizeVol / 2 + 1) / (sizeVol / 2 + 1), sizeVol/16, sizeVol)
#define block_RC dim3(sizeVol / 2 + 1, 16, 1)
#endif

#define EPS (0.000001f)

/* AS
 * In the original code most kernels had its own class which added no additional
 * functionality. This code has also been portet to hip but for ease of
 * developement in all new versions of the code we removed the classes and now 
 * simply inline the kernels. The old variant we refer to as a binary kernels 
 * since they are stored as binary. It's quite possible that this functionality 
 * wasn't available at the time of the original code but now they seem 
 * unneccessary.
 * 
 * To differentiate between Kernels in the basickernel and kernel file we call
 * kernels name those kernels needed in this file with a trailing underscore.
 * This should enable us to reduce the size in the future if we decide to remove
 * all the different kernel classes seen in the binary kernel version.
 */

extern "C" __global__ void mul_Real_(int size, float in, float *outVol) 
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  outVol[z * size * size + y * size + x] =
      outVol[z * size * size + y * size + x] * in;
}


extern "C" __global__ void correl_RC_(int size, float2 *inVol, float2 *outVol) 
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  float2 o = outVol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x];
  float2 i = inVol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x];
  float2 erg;
  erg.x = (o.x * i.x) + (o.y * i.y);
  erg.y = (o.x * i.y) - (o.y * i.x);
  outVol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x] = erg;
}


extern "C" __global__ void conv_RC_(int size, float2 *inVol, float2 *outVol) 
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  float2 o = outVol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x];
  float2 i = inVol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x];
  float2 erg;
  erg.x = (o.x * i.x) - (o.y * i.y);
  erg.y = (o.x * i.y) + (o.y * i.x);
  outVol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x] = erg;
}


extern "C" __global__ void energynorm_RC_(int size, float *particle,
                                           float *partSqr, float *cccMap,
                                           float *energyRef, float *nVox,
                                           float2 *temp, float *ccMask) 
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  float part = particle[z * size * size + y * size + x];
  float energyLocal = partSqr[z * size * size + y * size + x];

  float erg = 0;

  energyLocal -= part * part / nVox[0];
  energyLocal = sqrt(energyLocal) * sqrt(energyRef[0]);

  if (energyLocal > EPS) {
    erg = cccMap[z * size * size + y * size + x] / energyLocal;
  }

  int i = (x + size / 2) % size;
  int j = (y + size / 2) % size;
  int k = (z + size / 2) % size;

  cccMap[z * size * size + y * size + x] = erg;
  erg *= ccMask[k * size * size + j * size + i];
  temp[k * size * size + j * size + i].x = erg;
}


extern "C" __global__ void energynormMulMulMUl_RC(int size, float *particle,
                                           float *partSqr, float *cccMap,
                                           float *energyRef, float *nVox,
                                           float2 *temp, float *ccMask, 
                                           float multiply) 
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  float part = particle[z * size * size + y * size + x] * multiply;
  float energyLocal = partSqr[z * size * size + y * size + x] * multiply;

  float erg = 0;

  energyLocal -= part * part / nVox[0];
  energyLocal = sqrt(energyLocal) * sqrt(energyRef[0]);

  if (energyLocal > EPS) {
    erg = cccMap[z * size * size + y * size + x] * multiply / energyLocal;
  }

  int i = (x + size / 2) % size;
  int j = (y + size / 2) % size;
  int k = (z + size / 2) % size;

  cccMap[z * size * size + y * size + x] = erg;
  erg *= ccMask[k * size * size + j * size + i];
  temp[k * size * size + j * size + i].x = erg;
}


extern "C" __global__ void sub_(int size, float *inVol, float *outVol, 
                                  float val) 
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  outVol[z * size * size + y * size + x] =
      inVol[z * size * size + y * size + x] - val;
}


extern "C" __global__ void sub_RC_(int size, float *inVol, float *outVol,
                                    float *h_sum, float val) 
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  outVol[z * size * size + y * size + x] =
      inVol[z * size * size + y * size + x] - (h_sum[0] / val);
}


extern "C" __global__ void sqrsub_RC_(int size, float *inVol, float *outVol,
                                       float *h_sum, float val) 
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  outVol[z * size * size + y * size + x] =
      inVol[z * size * size + y * size + x] *
          inVol[z * size * size + y * size + x] -
      (h_sum[0] / val);
}


extern "C" __global__ void binarize_(int size, float *inVol, float *outVol) 
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  outVol[z * size * size + y * size + x] =
      inVol[z * size * size + y * size + x] > 0.5f ? 1.0f : 0.0f;
}


extern "C" __global__ void subCplx_RC_(int size, float *inVol, float *outVol,
                                        float *subval, float *divVal) 
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  outVol[z * size * (size) + y * (size) + x] =
      inVol[z * size * size + y * (size) + x] - subval[0] / divVal[0];
}


extern "C" __global__ void fftshiftReal_(int size, float *volIn, float *volOut) 
{
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  const int z = blockIdx.z * blockDim.z + threadIdx.z;

  int i = (x + size / 2) % size;
  int j = (y + size / 2) % size;
  int k = (z + size / 2) % size;

  float temp = volIn[k * size * size + j * size + i];
  volOut[z * size * size + y * size + x] = temp;
}


extern "C" __global__ void mulVol_RC_(int size, float *inVol, float2 *outVol) 
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  float2 temp = outVol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x];
  temp.x *= inVol[z * size * size + y * size + x];
  temp.y *= inVol[z * size * size + y * size + x];
  outVol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x] = temp;
}


extern "C" __global__ void mulVol_RR_(int size, float *inVol, float *outVol) 
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  float temp = outVol[z * size * size + y * size + x];
  temp *= inVol[z * size * size + y * size + x];
  outVol[z * size * size + y * size + x] = temp;
}


extern "C" __global__ void mulVolRRMul(int size, float *inVol, float *outVol, float val) 
{
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  float temp = outVol[z * size * size + y * size + x] * val;
  temp *= inVol[z * size * size + y * size + x];
  outVol[z * size * size + y * size + x] = temp;
}


extern "C" __global__ void bandpassFFTShift_RC_(int size, float2 *vol,
                                                 float rDown, float rUp,
                                                 float smooth) 
{
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  int z = blockIdx.z * blockDim.z + threadIdx.z;

  int i = (x + size / 2) % size;
  int j = (y + size / 2) % size;
  int k = (z + size / 2) % size;

  float2 temp = vol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x];

  // use squared smooth for Gaussian
  smooth = smooth * smooth;

  float center = size / 2;
  float3 vox = make_float3(i - center, j - center, k - center);

  float dist = sqrt(vox.x * vox.x + vox.y * vox.y + vox.z * vox.z);
  float scf = (dist - rUp) * (dist - rUp);
  smooth > 0 ? scf = exp(-scf / smooth) : scf = 0;

  if (dist > rUp) {
    temp.x *= scf;
    temp.y *= scf;
  }

  scf = (dist - rDown) * (dist - rDown);
  smooth > 0 ? scf = exp(-scf / smooth) : scf = 0;

  if (dist < rDown) {
    temp.x *= scf;
    temp.y *= scf;
  }

  vol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x] = temp;
}


extern "C" __global__ void findmax_(float *maxVals, float *index, float *val,
                                     float rphi, float rpsi, float rthe) 
{
  float oldmax = maxVals[0];
  if (val[0] > oldmax) {
    maxVals[0] = val[0];
    maxVals[1] = index[0];
    maxVals[2] = rphi;
    maxVals[3] = rpsi;
    maxVals[4] = rthe;
  }
}


extern "C"
__global__ void phaseCorrel_RC(int size, float2 * inVol, float2 * outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  float2 o = outVol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x];
	float2 i = inVol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x];
	float2 erg;
	erg.x = (o.x * i.x) + (o.y * i.y);
	erg.y = (o.x * i.y) - (o.y * i.x);
	float amplitude = sqrtf(erg.x * erg.x + erg.y * erg.y);
	if (amplitude != 0)
	{
		erg.x /= amplitude;
		erg.y /= amplitude;
	}
	else
	{
		erg.x = erg.y = 0;
	}
	outVol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x] = erg;
}


void maxVals_t::getXYZ(int size, int &x, int &y, int &z) 
{
  z = index / size / size;
  y = (index - z * size * size) / size;
  x = index - z * size * size - y * size;

  x -= size / 2;
  y -= size / 2;
  z -= size / 2;
}


int maxVals_t::getIndex(int size, int x, int y, int z) 
{
  x += size / 2;
  y += size / 2;
  z += size / 2;

  return z * size * size + y * size + x;
}


/* AS
 * Average Process Original Binary Kernels migrated to HIP
 * No Performance / No Quality Improvements
 */
AvgProcessOriginalBinaryKernels::AvgProcessOriginalBinaryKernels(
    size_t _sizeVol, hipStream_t _stream, Hip::HipContext *_ctx, float *_mask,
    float *_ref, float *_ccMask, float aPhiAngIter, float aPhiAngInc,
    float aAngIter, float aAngIncr, bool aBinarizeMask, bool aRotateMaskCC,
    bool aUseFilterVolume, bool linearInterpolation, KernelModuls modules)
    : sizeVol(_sizeVol),
	sizeTot(_sizeVol * _sizeVol * _sizeVol),
	stream(_stream),
	binarizeMask(aBinarizeMask),
	rotateMaskCC(aRotateMaskCC),
	useFilterVolume(aUseFilterVolume),
	ctx(_ctx),
	/* AS Deprecated used before HIP BinaryKernels
	, //rot((int)_sizeVol, _stream, _ctx, linearInterpolation),
	, //rotMask((int)_sizeVol, _stream, _ctx, linearInterpolation),
	, //rotMaskCC((int)_sizeVol, _stream, _ctx, linearInterpolation),
	, //reduce((int)_sizeVol * (int)_sizeVol * (int)_sizeVol, _stream, _ctx),
	, //sub((int)_sizeVol, _stream, _ctx),
	, //makecplx((int)_sizeVol, _stream, _ctx),
	, //binarize((int)_sizeVol, _stream, _ctx),
	, //mul((int)_sizeVol, _stream, _ctx),
	, //fft((int)_sizeVol, _stream, _ctx),
	, //max(_stream, _ctx),
	*/
	mask(_mask),
	ref(_ref),
	ccMask(_ccMask),
	/* AS 
   * Done we declare a massive amount of variables which when we increase the 
   * Volume size need massive amounts of storage. Check if we can reduce the
   * variables needed or reduce the size of some variables (sum, sumsqr ...) 
   */
	d_ffttemp(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
	d_particle(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	d_wedge(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	d_filter(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	d_reference_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	d_mask_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	d_reference(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	d_mask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	d_ccMask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	d_ccMask_Orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
	d_particleCplx(_sizeVol * _sizeVol * (_sizeVol) * sizeof(float2)),
	d_particleSqrCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
	d_particleCplx_orig(_sizeVol * _sizeVol * (_sizeVol) * sizeof(float2)),
	d_particleSqrCplx_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
	d_referenceCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
	d_maskCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
	d_buffer(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)), //should be sufficient for everything...
	d_index(_sizeVol * _sizeVol * _sizeVol * sizeof(int)), //should be sufficient for everything...
	nVox(_sizeVol * _sizeVol * _sizeVol * sizeof(float)), //should be sufficient for everything...
	sum(_sizeVol * _sizeVol * _sizeVol * sizeof(float)), //should be sufficient for everything...
	sumSqr(_sizeVol * _sizeVol * _sizeVol * sizeof(float)), //should be sufficient for everything...
	maxVals(sizeof(maxVals_t)), //should be sufficient for everything...
	phi_angiter(aPhiAngIter),
	phi_angincr(aPhiAngInc),
	angiter(aAngIter),
	angincr(aAngIncr),
	modules(ctx),
	/* AS (Done for later iterations of the AvgProcess) Gird- and Blockdimension
   * are set to the sizes most beneficial for Nvidia GPUs. This should either,	
   * be changed to dynamically decide on the correct size given the GPU's 
   * capabilities or for now take two variants and use them wether 
   * AMD or NV GPUs are used. 
   */
	aSubCplxKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aFFTShiftRealKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aMakeCplxWithSubKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aMulVolKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aBandpassFFTShiftKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aMulKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aMakeRealKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aMakeCplxWithSqrSubKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aCorrelKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aConvKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aEnergyNormKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aFFTShift2Kernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aReduceKernel(modules.modkernel, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aSubKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aMaxKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1)),
	aRotateKernelROT(modules.modbasicKernels, make_dim3(sizeVol, sizeVol , sizeVol), make_dim3(1, 1, 1), sizeVol, linearInterpolation),
	aRotateKernelMASK(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1), sizeVol, linearInterpolation),
	aRotateKernelMASKCC(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1), sizeVol, linearInterpolation),
	aBinarizeKernel(modules.modbasicKernels, make_dim3(sizeVol / 32, sizeVol / 16, sizeVol), make_dim3(32, 16, 1))
{
  hipSafeCall(hipMallocHost((void **)&index, sizeof(int)));
  hipSafeCall(hipMallocHost((void **)&sum_h, sizeof(float)));
  hipSafeCall(hipMallocHost((void **)&sumCplx, sizeof(float2)));

/* AS
 * The hipFFT library currently is not plug and play sadly. Basically you have
 * to manually set it up. But the following issue states that they are moving 
 * to a sort of wrapper interface to easily switch.
 * 
 * https://github.com/ROCmSoftwarePlatform/rocFFT/issues/276
 */
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  int n[] = {(int)sizeVol, (int)sizeVol, (int)sizeVol};
  hipfftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, HIPFFT_C2C, 1);
  hipfftSetStream(ffthandle, stream);
#else
  int n[] = {(int)sizeVol, (int)sizeVol, (int)sizeVol};
  cufftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, 1);
  cufftSetStream(ffthandle, stream);
#endif

	d_reference.CopyHostToDevice(ref);

	aReduceKernel.sum(d_reference, d_buffer, sizeTot);
	float sum = 0;

	d_buffer.CopyDeviceToHost(&sum, sizeof(float));
	sum = sum / sizeTot;
	aSubKernel(sizeVol, d_reference, d_reference_orig, sum);

	d_mask_orig.CopyHostToDevice(mask);

	d_ccMask_Orig.CopyHostToDevice(ccMask);
	d_ccMask.CopyHostToDevice(ccMask);
}

AvgProcessOriginalBinaryKernels::~AvgProcessOriginalBinaryKernels() {
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftDestroy(ffthandle);
#else
  cufftDestroy(ffthandle);
#endif
}

maxVals_t AvgProcessOriginalBinaryKernels::execute(float *_data, 
                                      float* wedge,float *filter,
                                      float oldphi, float oldpsi,
                                      float oldtheta, float rDown, float rUp,
                                      float smooth, float3 oldShift,
                                      bool couplePhiToPsi,
                                      bool computeCCValOnly, int oldIndex) {
  /* AS
   * The Average Process rotates the reference for different angles defined in
   * the configuration file and compares it to the particle. We return the best
   * fit and now have a approx. delta of the rotation angles and shifts.
   *
   * C2C defines the Fourier Transform Type Complex to Complex which needs
   * more time than the Real 2 Complex version (R2C)
   */

	int oldWedge = -1;
	maxVals_t m;
	m.index = 0;
	m.ccVal = -10000;
	m.rphi = 0;
	m.rpsi = 0;
	m.rthe = 0;

	hipSafeCall(hipStreamSynchronize(stream));
	maxVals.CopyHostToDeviceAsync(stream, &m);
	d_particle.CopyHostToDevice(wedge);

	aFFTShiftRealKernel(sizeVol, d_particle, d_wedge);

	if (useFilterVolume) {
		d_particle.CopyHostToDevice(filter);
		aFFTShiftRealKernel(sizeVol, d_particle, d_filter);
	}

	d_particle.CopyHostToDeviceAsync(stream, _data);
	// AS ToDo check if hipStream_tSynchronize(stream) is needed

	aMakeCplxWithSubKernel(sizeVol, d_particle, d_particleCplx_orig, 0.);


#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
	hipfftExecC2C(ffthandle, (hipfftComplex*)d_particleCplx_orig.GetDevicePtr(), (hipfftComplex*)d_particleCplx_orig.GetDevicePtr(), HIPFFT_FORWARD);
#else
	cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), CUFFT_FORWARD);
#endif

	aMulVolKernel(sizeVol, d_wedge, d_particleCplx_orig);

	if (useFilterVolume)
	{
		aMulVolKernel(sizeVol, d_filter, d_particleCplx_orig);
	}
	else
	{
		aBandpassFFTShiftKernel(sizeVol, d_particleCplx_orig, rDown, rUp, smooth);
	}

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
	//rocfft_execute(rocffthandle, (void**)d_particleCplx_orig.GetDevicePtr(), (void**)d_particleCplx_orig.GetDevicePtr(), forwardinfo);
	//status = rocfft_execute(rocffthandle, (void**)d_particleCplx_orig.GetDevicePtr(), (void**)d_particleCplx_orig.GetDevicePtr(), forwardinfo);
	hipfftExecC2C(ffthandle, (hipfftComplex*)d_particleCplx_orig.GetDevicePtr(), (hipfftComplex*)d_particleCplx_orig.GetDevicePtr(), HIPFFT_BACKWARD);
#else
	cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), CUFFT_INVERSE);
#endif

	aMulKernel(sizeVol, 1.0f / sizeTot, d_particleCplx_orig);
	aMakeRealKernel(sizeVol, d_particleCplx_orig, d_particle);

	aReduceKernel.sum(d_particle, d_buffer, sizeTot);

	//d_buffer.CopyDeviceToHostAsync(stream, sum_h, sizeof(float));
	d_buffer.CopyDeviceToHost(sum_h, sizeof(float));

	//hipDeviceSynchronize();
	//hipStreamSynchronize(stream);

	// hipSafeCall(hipDeviceSynchronize());
	//hipSafeCall(hipCtxSynchronize());

	aMakeCplxWithSubKernel(sizeVol, d_particle, d_particleCplx_orig, *sum_h / sizeTot);
	aMakeCplxWithSqrSubKernel(sizeVol, d_particle, d_particleSqrCplx_orig, *sum_h / sizeTot);

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
	hipfftExecC2C(ffthandle, (hipfftComplex*)d_particleCplx_orig.GetDevicePtr(), (hipfftComplex*)d_particleCplx_orig.GetDevicePtr(), HIPFFT_FORWARD);
	hipfftExecC2C(ffthandle, (hipfftComplex*)d_particleSqrCplx_orig.GetDevicePtr(), (hipfftComplex*)d_particleSqrCplx_orig.GetDevicePtr(), HIPFFT_FORWARD);
#else
	cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleCplx_orig.GetDevicePtr(), CUFFT_FORWARD);
	cufftExecC2C(ffthandle, (cufftComplex*)d_particleSqrCplx_orig.GetDevicePtr(), (cufftComplex*)d_particleSqrCplx_orig.GetDevicePtr(), CUFFT_FORWARD);
#endif

	aMulVolKernel(sizeVol, d_wedge, d_particleCplx_orig);

	if (useFilterVolume)
	{
		aMulVolKernel(sizeVol, d_filter, d_particleCplx_orig);
	}
	else
	{
		aBandpassFFTShiftKernel(sizeVol, d_particleCplx_orig, rDown, rUp, smooth);
	}

	aRotateKernelROT.SetTexture(d_reference_orig);
	aRotateKernelMASK.SetTexture(d_mask_orig);

	if (rotateMaskCC)
	{
		aRotateKernelMASKCC.SetTexture(d_ccMask_Orig);
		aRotateKernelMASKCC.SetOldAngles(oldphi, oldpsi, oldtheta);
	}

	aRotateKernelROT.SetOldAngles(oldphi, oldpsi, oldtheta);
	aRotateKernelMASK.SetOldAngles(oldphi, oldpsi, oldtheta);

	float rthe = 0;
	float rpsi = 0;
	float rphi = 0;
	float maxthe = 0;
	float maxpsi = 0;
	float maxphi = 0;
	float npsi, dpsi;

	int angles = 0;
	double time = 0;

	for (int iterPhi = 0; iterPhi < 2 * phi_angiter + 1; ++iterPhi)
	{
		rphi = phi_angincr * (iterPhi - phi_angiter);
		for (int iterThe = (int)0; iterThe < angiter + 1; ++iterThe)
		{
			if (iterThe == 0)
			{
				npsi = 1;
				dpsi = 360;
			}
			else
			{
				dpsi = angincr / sinf(iterThe * angincr * (float)M_PI / 180.0f);
				npsi = ceilf(360.0f / dpsi);
			}
			rthe = iterThe * angincr;
			for (int iterPsi = 0; iterPsi < npsi; ++iterPsi)
			{
				rpsi = iterPsi * dpsi;

				if (couplePhiToPsi)
				{
					rphi = phi_angincr * (iterPhi - phi_angiter) - rpsi;
				}
				else
				{
					rphi = phi_angincr * (iterPhi - phi_angiter);
				}


				d_particleCplx.CopyDeviceToDeviceAsync(stream, d_particleCplx_orig);
				d_particleSqrCplx.CopyDeviceToDeviceAsync(stream, d_particleSqrCplx_orig);

				aRotateKernelROT.do_rotate(sizeVol, d_reference, rphi, rpsi, rthe);
				aRotateKernelMASK.do_rotate(sizeVol, d_mask, rphi, rpsi, rthe);

				if (rotateMaskCC)
				{
					d_ccMask.Memset(0);
					aRotateKernelMASKCC.do_rotate(sizeVol, d_ccMask, rphi, rpsi, rthe);
				}

				if (binarizeMask)
				{
					aBinarizeKernel(sizeVol, d_mask, d_mask);
				}

				aReduceKernel.sum(d_mask, nVox, sizeTot);

				aMakeCplxWithSubKernel(sizeVol, d_reference, d_referenceCplx, 0);
				aMakeCplxWithSubKernel(sizeVol, d_mask, d_maskCplx, 0);

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
				// rocfft_execute(rocffthandle, (void**)d_particleCplx_orig.GetDevicePtr(), (void**)d_particleCplx_orig.GetDevicePtr(), NULL);
				hipfftExecC2C(ffthandle, (hipfftComplex*)d_referenceCplx.GetDevicePtr(), (hipfftComplex*)d_referenceCplx.GetDevicePtr(), HIPFFT_FORWARD);
#else
				cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_FORWARD);
#endif

				aMulVolKernel(sizeVol, d_wedge, d_referenceCplx);

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
				// rocfft_execute(rocffthandle, (void**)d_particleCplx_orig.GetDevicePtr(), (void**)d_particleCplx_orig.GetDevicePtr(), NULL);
				hipfftExecC2C(ffthandle, (hipfftComplex*)d_maskCplx.GetDevicePtr(), (hipfftComplex*)d_maskCplx.GetDevicePtr(), HIPFFT_FORWARD);
#else
				cufftExecC2C(ffthandle, (cufftComplex*)d_maskCplx.GetDevicePtr(), (cufftComplex*)d_maskCplx.GetDevicePtr(), CUFFT_FORWARD);
#endif

				if (useFilterVolume)
				{
					aMulVolKernel(sizeVol, d_filter, d_referenceCplx);
				}
				else
				{
					aBandpassFFTShiftKernel(sizeVol, d_referenceCplx, rDown, rUp, smooth);
				}

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
				// rocfft_execute(rocffthandle, (void**)d_particleCplx_orig.GetDevicePtr(), (void**)d_particleCplx_orig.GetDevicePtr(), NULL);
				hipfftExecC2C(ffthandle, (hipfftComplex*)d_referenceCplx.GetDevicePtr(), (hipfftComplex*)d_referenceCplx.GetDevicePtr(), HIPFFT_BACKWARD);
#else
				cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_INVERSE);
#endif

				aMulKernel(sizeVol, 1.0f / sizeTot, d_referenceCplx);
				aMulVolKernel(sizeVol, d_mask, d_referenceCplx);

				aReduceKernel.sumcplx(d_referenceCplx, sum, sizeTot);

				aSubCplxKernel(sizeVol, d_referenceCplx, d_referenceCplx, sum, nVox);
				aMulVolKernel(sizeVol, d_mask, d_referenceCplx);

				aReduceKernel.sumsqrcplx(d_referenceCplx, sumSqr, sizeTot);

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
				// rocfft_execute(rocffthandle, (void**)d_particleCplx_orig.GetDevicePtr(), (void**)d_particleCplx_orig.GetDevicePtr(), NULL);
				hipfftExecC2C(ffthandle, (hipfftComplex*)d_referenceCplx.GetDevicePtr(), (hipfftComplex*)d_referenceCplx.GetDevicePtr(), HIPFFT_FORWARD);
#else
				cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_FORWARD);
#endif

				aCorrelKernel(sizeVol, d_particleCplx, d_referenceCplx);

				aConvKernel(sizeVol, d_maskCplx, d_particleCplx);
				aConvKernel(sizeVol, d_maskCplx, d_particleSqrCplx);

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
				// rocfft_execute(rocffthandle, (void**)d_particleCplx_orig.GetDevicePtr(), (void**)d_particleCplx_orig.GetDevicePtr(), NULL);
				hipfftExecC2C(ffthandle, (hipfftComplex*)d_referenceCplx.GetDevicePtr(), (hipfftComplex*)d_referenceCplx.GetDevicePtr(), HIPFFT_BACKWARD);
#else
				cufftExecC2C(ffthandle, (cufftComplex*)d_referenceCplx.GetDevicePtr(), (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_INVERSE);
#endif
				aMulKernel(sizeVol, 1.0f / sizeTot, d_referenceCplx);


#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
				hipfftExecC2C(ffthandle, (hipfftComplex*)d_particleCplx.GetDevicePtr(), (hipfftComplex*)d_particleCplx.GetDevicePtr(), HIPFFT_BACKWARD);
#else
				cufftExecC2C(ffthandle, (cufftComplex*)d_particleCplx.GetDevicePtr(), (cufftComplex*)d_particleCplx.GetDevicePtr(), CUFFT_INVERSE);
#endif
				aMulKernel(sizeVol, 1.0f / sizeTot, d_particleCplx);

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
				// rocfft_execute(rocffthandle, (void**)d_particleCplx_orig.GetDevicePtr(), (void**)d_particleCplx_orig.GetDevicePtr(), NULL);
				hipfftExecC2C(ffthandle, (hipfftComplex*)d_particleSqrCplx.GetDevicePtr(), (hipfftComplex*)d_particleSqrCplx.GetDevicePtr(), HIPFFT_BACKWARD);
#else
				cufftExecC2C(ffthandle, (cufftComplex*)d_particleSqrCplx.GetDevicePtr(), (cufftComplex*)d_particleSqrCplx.GetDevicePtr(), CUFFT_INVERSE);
#endif

				aMulKernel(sizeVol, 1.0f / sizeTot, d_particleSqrCplx);

				aEnergyNormKernel(sizeVol, d_particleCplx, d_particleSqrCplx, d_referenceCplx, sumSqr, nVox);

				aFFTShift2Kernel(sizeVol, d_referenceCplx, d_ffttemp);

				aMulVolKernel(sizeVol, d_ccMask, d_ffttemp);

				if (computeCCValOnly)
				{
					//only read out the CC value at the old shift position and store it in d_buffer
					d_index.CopyHostToDevice(&oldIndex);

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
					hipSafeCall(hipMemcpyDtoD(d_buffer.GetDevicePtr(), (float2*)d_ffttemp.GetDevicePtr() + oldIndex, sizeof(float)));
#else
					hipSafeCall(hipMemcpyDtoD(d_buffer.GetDevicePtr(), d_ffttemp.GetDevicePtr() + oldIndex, sizeof(float)));
#endif
				}
				else
				{
					//find new Maximum value and store position and value
					aReduceKernel.maxindexcplx(d_ffttemp, d_buffer, d_index, sizeTot);
				}

				aMaxKernel(maxVals, d_index, d_buffer, rphi, rpsi, rthe);
			}
		}
	}

	hipSafeCall(hipStreamSynchronize(stream));
	maxVals.CopyDeviceToHost(&m);
	hipSafeCall(hipStreamSynchronize(stream));

	return m;
}


/* AS
 * Average Process Kernels migrated to HIP but without the use of Binary Kernels
 * (much easier to write and test code using hipLaunchKernelGGL as alternative
 * to the <<< >>> method from Nvidia) 
 * No Performance / No Quality Improvements
 */
AvgProcessOriginal::AvgProcessOriginal(
    size_t _sizeVol, hipStream_t _stream, Hip::HipContext *_ctx, float *_mask,
    float *_ref, float *_ccMask, float aPhiAngIter, float aPhiAngInc,
    float aAngIter, float aAngIncr, bool aBinarizeMask, bool aRotateMaskCC,
    bool aUseFilterVolume, bool linearInterpolation, KernelModuls modules)
    : sizeVol(_sizeVol), sizeTot(_sizeVol * _sizeVol * _sizeVol),
      stream(_stream), binarizeMask(aBinarizeMask), rotateMaskCC(aRotateMaskCC),
      useFilterVolume(aUseFilterVolume), ctx(_ctx),
      /* AS Deprecated used before HIP BinaryKernels
      // rot((int)_sizeVol, _stream, _ctx, linearInterpolation),
      // rotMask((int)_sizeVol, _stream, _ctx, linearInterpolation),
      // rotMaskCC((int)_sizeVol, _stream, _ctx, linearInterpolation),
      // reduce((int)_sizeVol * (int)_sizeVol * (int)_sizeVol, _stream, _ctx),
      // sub((int)_sizeVol, _stream, _ctx),
      // makecplx((int)_sizeVol, _stream, _ctx),
      // binarize((int)_sizeVol, _stream, _ctx),
      // mul((int)_sizeVol, _stream, _ctx),
      // fft((int)_sizeVol, _stream, _ctx),
      // max(_stream, _ctx),
      */
      mask(_mask), ref(_ref), ccMask(_ccMask),
      d_ffttemp(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_particle(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_wedge(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_filter(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_reference_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_mask_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_reference(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_mask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_ccMask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_ccMask_Orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_particleCplx(_sizeVol * _sizeVol * (_sizeVol) * sizeof(float2)),
      d_particleSqrCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_particleCplx_orig(_sizeVol * _sizeVol * (_sizeVol) * sizeof(float2)),
      d_particleSqrCplx_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_referenceCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_maskCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_buffer(_sizeVol * _sizeVol * _sizeVol *
               sizeof(float2)), // should be sufficient for everything...
      d_index(_sizeVol * _sizeVol * _sizeVol *
              sizeof(int)), // should be sufficient for everything...
      nVox(_sizeVol * _sizeVol * _sizeVol *
           sizeof(float)), // should be sufficient for everything...
      sum(_sizeVol * _sizeVol * _sizeVol *
          sizeof(float)), // should be sufficient for everything...
      sumSqr(_sizeVol * _sizeVol * _sizeVol *
             sizeof(float)),      // should be sufficient for everything...
      maxVals(sizeof(maxVals_t)), // should be sufficient for everything...
      phi_angiter(aPhiAngIter), phi_angincr(aPhiAngInc), angiter(aAngIter),
      angincr(aAngIncr),
      aRotateKernelROT(modules.modbasicKernels,
                       dim3(sizeVol / 32, sizeVol / 16, sizeVol),
                       dim3(32, 16, 1), sizeVol, linearInterpolation),
      aRotateKernelMASK(modules.modbasicKernels,
                        dim3(sizeVol / 32, sizeVol / 16, sizeVol),
                        dim3(32, 16, 1), sizeVol, linearInterpolation),
      aRotateKernelMASKCC(modules.modbasicKernels,
                          dim3(sizeVol / 32, sizeVol / 16, sizeVol),
                          dim3(32, 16, 1), sizeVol, linearInterpolation),
      aReduceKernel(modules.modkernel,
                    dim3(sizeVol / 32, sizeVol / 16, sizeVol),
                    dim3(32, 16, 1)) 
{
  hipSafeCall(hipMallocHost((void **)&index, sizeof(int)));
  hipSafeCall(hipMallocHost((void **)&sum_h, sizeof(float)));
  hipSafeCall(hipMallocHost((void **)&sumCplx, sizeof(float2)));

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  int n[] = {(int)sizeVol, (int)sizeVol, (int)sizeVol};
  hipfftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, HIPFFT_C2C, 1);
  hipfftSetStream(ffthandle, stream);
#else
  int n[] = {(int)sizeVol, (int)sizeVol, (int)sizeVol};
  cufftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, 1);
  cufftSetStream(ffthandle, stream);
#endif

  d_reference.CopyHostToDevice(ref);

  aReduceKernel.sum(d_reference, d_buffer, sizeTot);
  float sum = 0;
  d_buffer.CopyDeviceToHost(&sum, sizeof(float));
  sum = sum / sizeTot;
  // aSubKernel(sizeVol, d_reference, d_reference_orig, sum); 
  // AS Substract Average
  /* AS We now launch all kernels with this kernel launch command 
   * https://rocmdocs.amd.com/en/latest/Programming_Guides/HIP-GUIDE.html
   */
  hipLaunchKernelGGL(Sub, grid, block, 0, stream, sizeVol,
                     (float *)d_reference.GetDevicePtr(),
                     (float *)d_reference_orig.GetDevicePtr(), sum);

  d_mask_orig.CopyHostToDevice(mask);
  d_ccMask_Orig.CopyHostToDevice(ccMask);
  d_ccMask.CopyHostToDevice(ccMask);
}

AvgProcessOriginal::~AvgProcessOriginal() {
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftDestroy(ffthandle);
#else
  cufftDestroy(ffthandle);
#endif
}

maxVals_t AvgProcessOriginal::execute(float *_data, float *wedge, float *filter,
                                      float oldphi, float oldpsi,
                                      float oldtheta, float rDown, float rUp,
                                      float smooth, float3 oldShift,
                                      bool couplePhiToPsi,
                                      bool computeCCValOnly, int oldIndex) {
  /* AS
   * The Average Process rotates the reference for different angles defined in
   * the configuration file and compares it to the particle. We return the best
   * fit and now have a approx. delta of the rotation angles and shifts.
   *
   * C2C defines the Fourier Transform Type Complex to Complex which needs
   * more time than the Real 2 Complex version (R2C)
   */

  int oldWedge = -1;
  maxVals_t m;
  m.index = 0;
  m.ccVal = -10000;
  m.rphi = 0;
  m.rpsi = 0;
  m.rthe = 0;

  hipSafeCall(hipStreamSynchronize(stream));
  maxVals.CopyHostToDeviceAsync(stream, &m);

  d_particle.CopyHostToDevice(wedge);
  // aFFTShiftRealKernel(sizeVol, d_particle, d_wedge);
  hipLaunchKernelGGL(FFTShiftReal, grid, block, 0, stream, sizeVol,
                     (float *)d_particle.GetDevicePtr(),
                     (float *)d_wedge.GetDevicePtr());

  // makecplx.MakeCplxWithSub(d_reference, d_referenceCplx, 0);
  // cufftSafeCall(cufftExecC2C(ffthandle,
  // (cufftComplex*)d_referenceCplx.GetDevicePtr(),
  // (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_FORWARD));
  // fft.BandpassFFTShift(d_referenceCplx, rDown, rUp, smooth);
  // cufftSafeCall(cufftExecC2C(ffthandle,
  // (cufftComplex*)d_referenceCplx.GetDevicePtr(),
  // (cufftComplex*)d_referenceCplx.GetDevicePtr(), CUFFT_INVERSE));
  // mul.Mul(1.0f / sizeTot, d_referenceCplx);
  // makecplx.MakeReal(d_referenceCplx, d_reference_orig);

  if (useFilterVolume) {
    d_particle.CopyHostToDevice(filter);
    // aFFTShiftRealKernel(sizeVol, d_particle, d_filter);
    hipLaunchKernelGGL(FFTShiftReal, grid, block, 0, stream, sizeVol,
                       (float *)d_particle.GetDevicePtr(),
                       (float *)d_filter.GetDevicePtr());
  }

  d_particle.CopyHostToDeviceAsync(stream, _data);

  // aMakeCplxWithSubKernel(sizeVol, d_particle, d_particleCplx_orig, 0.);
  hipLaunchKernelGGL(MakeCplxWithSub, grid, block, 0, stream, sizeVol,
                     (float *)d_particle.GetDevicePtr(),
                     (float2 *)d_particleCplx_orig.GetDevicePtr(), 0.f);

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftExecC2C(ffthandle, (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
                (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
                HIPFFT_FORWARD);
#else
  cufftExecC2C(ffthandle, (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
               (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
               CUFFT_FORWARD);
#endif

  // aMulVolKernel(sizeVol, d_wedge, d_particleCplx_orig);
  hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol,
                     (float *)d_wedge.GetDevicePtr(),
                     (float2 *)d_particleCplx_orig.GetDevicePtr());

  if (useFilterVolume) {
    // aMulVolKernel(sizeVol, d_filter, d_particleCplx_orig);
    hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol,
                       (float *)d_filter.GetDevicePtr(),
                       (float2 *)d_particleCplx_orig.GetDevicePtr());
  } else {
    // aBandpassFFTShiftKernel(sizeVol, d_particleCplx_orig, rDown, rUp,
    // smooth);
    hipLaunchKernelGGL(BandpassFFTShift, grid, block, 0, stream, sizeVol,
                       (float2 *)d_particleCplx_orig.GetDevicePtr(), rDown, rUp,
                       smooth);
  }

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftExecC2C(ffthandle, (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
                (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
                HIPFFT_BACKWARD);
#else
  cufftExecC2C(ffthandle, (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
               (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
               CUFFT_INVERSE);
#endif

  // aMulKernel(sizeVol, 1.0f / sizeTot, d_particleCplx_orig);
  hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
                     (float2 *)d_particleCplx_orig.GetDevicePtr());

  // aMakeRealKernel(sizeVol, d_particleCplx_orig, d_particle);
  hipLaunchKernelGGL(MakeReal, grid, block, 0, stream, sizeVol,
                     (float2 *)d_particleCplx_orig.GetDevicePtr(),
                     (float *)d_particle.GetDevicePtr());

  aReduceKernel.sum(d_particle, d_buffer, sizeTot);

  /*
  AS Todo after calculating the sum there is no reason to copy it back to the
  host also doing the copy asyncronous makes no sense and if its done async it
  needs to be synchronized afterwards. Replaced by none async copy

  d_buffer.CopyDeviceToHost(sum_h, sizeof(float));
  */

  /* AS deprecated replaced by none asynchronis copy */
  d_buffer.CopyDeviceToHostAsync(stream, sum_h, sizeof(float));
  hipStreamSynchronize(stream);

  // aMakeCplxWithSubKernel(sizeVol, d_particle, d_particleCplx_orig, *sum_h /
  // sizeTot);
  hipLaunchKernelGGL(MakeCplxWithSub, grid, block, 0, stream, sizeVol,
                     (float *)d_particle.GetDevicePtr(),
                     (float2 *)d_particleCplx_orig.GetDevicePtr(),
                     *sum_h / sizeTot);

  // aMakeCplxWithSqrSubKernel(sizeVol, d_particle, d_particleSqrCplx_orig,
  // *sum_h / sizeTot);
  hipLaunchKernelGGL(MakeCplxSqrWithSub, grid, block, 0, stream, sizeVol,
                     (float *)d_particle.GetDevicePtr(),
                     (float2 *)d_particleSqrCplx_orig.GetDevicePtr(),
                     *sum_h / sizeTot);

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftExecC2C(ffthandle, (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
                (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
                HIPFFT_FORWARD);
  hipfftExecC2C(
      ffthandle, (hipfftComplex *)d_particleSqrCplx_orig.GetDevicePtr(),
      (hipfftComplex *)d_particleSqrCplx_orig.GetDevicePtr(), HIPFFT_FORWARD);
#else
  cufftExecC2C(ffthandle, (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
               (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
               CUFFT_FORWARD);
  cufftExecC2C(ffthandle, (cufftComplex *)d_particleSqrCplx_orig.GetDevicePtr(),
               (cufftComplex *)d_particleSqrCplx_orig.GetDevicePtr(),
               CUFFT_FORWARD);
#endif

  /* AS Fixed The wedge and filter are being applied twice I think this should
   * not happen */
  /* AS After talking to Michael this was done because of numerical uncertainty
   * in the fft and this way 0 is 0 and not some value closed to it. Maybe this 
   * can patched in the future for now this will be removed.
   */
  /* AS Utz and I realized that this also breaks the substraction of the mean
   * ... now the 0 values are less than 0
   */
  /* AS All these issues have been fixed in the quality improved version */

  // aMulVolKernel(sizeVol, d_wedge, d_particleCplx_orig);
  hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol,
                     (float *)d_wedge.GetDevicePtr(),
                     (float2 *)d_particleCplx_orig.GetDevicePtr());

  if (useFilterVolume) {
    // aMulVolKernel(sizeVol, d_filter, d_particleCplx_orig);
    hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol,
                       (float *)d_filter.GetDevicePtr(),
                       (float2 *)d_particleCplx_orig.GetDevicePtr());
  } else {
    // aBandpassFFTShiftKernel(sizeVol, d_particleCplx_orig, rDown, rUp,
    // smooth);
    hipLaunchKernelGGL(BandpassFFTShift, grid, block, 0, stream, sizeVol,
                       (float2 *)d_particleCplx_orig.GetDevicePtr(), rDown, rUp,
                       smooth);
  }

  // float* check = new float[128*64*64];
  // d_particleCplx_orig.CopyDeviceToHost(check);
  // emwrite("c:\\users\\kunz\\Desktop\\check.em", check, 128, 64, 64);
  // exit(0);

  aRotateKernelROT.SetTexture(d_reference_orig);
  aRotateKernelMASK.SetTexture(d_mask_orig);

  if (rotateMaskCC) {
    aRotateKernelMASKCC.SetTexture(d_ccMask_Orig);
    aRotateKernelMASKCC.SetOldAngles(oldphi, oldpsi, oldtheta);
  }
  // rotMaskCC.SetTextureShift(d_ccMask_Orig);

  aRotateKernelROT.SetOldAngles(oldphi, oldpsi, oldtheta);
  aRotateKernelMASK.SetOldAngles(oldphi, oldpsi, oldtheta);
  // rotMaskCC.SetOldAngles(oldphi, oldpsi, oldtheta);
  // rotMaskCC.SetOldAngles(0, 0, 0);

  // for angle...
  float rthe = 0;
  float rpsi = 0;
  float rphi = 0;
  float maxthe = 0;
  float maxpsi = 0;
  float maxphi = 0;
  float npsi, dpsi;

  int angles = 0;
  double time = 0;

  int counter = 0;

  double diff1 = 0;
  double diff2 = 0;
  double diff3 = 0;
  double diff4 = 0;

  float maxTest = -1000;
  int maxindex = -1000;

  for (int iterPhi = 0; iterPhi < 2 * phi_angiter + 1; ++iterPhi) {
    rphi = phi_angincr * (iterPhi - phi_angiter);
    for (int iterThe = (int)0; iterThe < angiter + 1; ++iterThe) {
      if (iterThe == 0) {
        npsi = 1;
        dpsi = 360;
      } else {
        dpsi = angincr / sinf(iterThe * angincr * (float)M_PI / 180.0f);
        npsi = ceilf(360.0f / dpsi);
      }
      rthe = iterThe * angincr;
      for (int iterPsi = 0; iterPsi < npsi; ++iterPsi) {
        rpsi = iterPsi * dpsi;

        if (couplePhiToPsi) {
          rphi = phi_angincr * (iterPhi - phi_angiter) - rpsi;
        } else {
          rphi = phi_angincr * (iterPhi - phi_angiter);
        }

        d_particleCplx.CopyDeviceToDeviceAsync(stream, d_particleCplx_orig);
        d_particleSqrCplx.CopyDeviceToDeviceAsync(stream,
                                                  d_particleSqrCplx_orig);

        aRotateKernelROT.do_rotate(sizeVol, d_reference, rphi, rpsi, rthe);
        aRotateKernelMASK.do_rotate(sizeVol, d_mask, rphi, rpsi, rthe);

        if (rotateMaskCC) {
          d_ccMask.Memset(0);
          aRotateKernelMASKCC.do_rotate(sizeVol, d_ccMask, rphi, rpsi, rthe);
        }

        // rotMaskCC.Rot(d_ccMask, 0, 0, 0);
        // rotMaskCC.Shift(d_ccMask, make_float3(oldShift.x, oldShift.y,
        // oldShift.z) );

        // cout << rotateMaskCC << endl;
        // float* check = new float[64*64*64];
        // d_ccMask.CopyDeviceToHost(check);
        // emwrite("check.em", check, 64, 64, 64);
        // exit(0);

        if (binarizeMask) {
          // aBinarizeKernel(sizeVol, d_mask, d_mask);
          hipLaunchKernelGGL(Binarize, grid, block, 0, stream, sizeVol,
                             (float *)d_mask.GetDevicePtr(),
                             (float *)d_mask.GetDevicePtr());
        }

        aReduceKernel.sum(d_mask, nVox, sizeTot);

        // aMakeCplxWithSubKernel(sizeVol, d_reference, d_referenceCplx, 0);
        // aMakeCplxWithSubKernel(sizeVol, d_mask, d_maskCplx, 0);
        hipLaunchKernelGGL(MakeCplxWithSub, grid, block, 0, stream, sizeVol,
                           (float *)d_reference.GetDevicePtr(),
                           (float2 *)d_referenceCplx.GetDevicePtr(), 0);
        hipLaunchKernelGGL(MakeCplxWithSub, grid, block, 0, stream, sizeVol,
                           (float *)d_mask.GetDevicePtr(),
                           (float2 *)d_maskCplx.GetDevicePtr(), 0);

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(
            ffthandle, (hipfftComplex *)d_referenceCplx.GetDevicePtr(),
            (hipfftComplex *)d_referenceCplx.GetDevicePtr(), HIPFFT_FORWARD);
#else
        cufftExecC2C(ffthandle, (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     CUFFT_FORWARD);
#endif

        // aMulVolKernel(sizeVol, d_wedge, d_referenceCplx);
        hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol,
                           (float *)d_wedge.GetDevicePtr(),
                           (float2 *)d_referenceCplx.GetDevicePtr());

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(ffthandle, (hipfftComplex *)d_maskCplx.GetDevicePtr(),
                      (hipfftComplex *)d_maskCplx.GetDevicePtr(),
                      HIPFFT_FORWARD);
#else
        cufftExecC2C(ffthandle, (cufftComplex *)d_maskCplx.GetDevicePtr(),
                     (cufftComplex *)d_maskCplx.GetDevicePtr(), CUFFT_FORWARD);
#endif

        if (useFilterVolume) {
          // aMulVolKernel(sizeVol, d_filter, d_referenceCplx);
          hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol,
                             (float *)d_filter.GetDevicePtr(),
                             (float2 *)d_referenceCplx.GetDevicePtr());
        } else {
          // aBandpassFFTShiftKernel(sizeVol, d_referenceCplx, rDown, rUp,
          // smooth);
          hipLaunchKernelGGL(BandpassFFTShift, grid, block, 0, stream, sizeVol,
                             (float2 *)d_referenceCplx.GetDevicePtr(), rDown,
                             rUp, smooth);
        }

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(
            ffthandle, (hipfftComplex *)d_referenceCplx.GetDevicePtr(),
            (hipfftComplex *)d_referenceCplx.GetDevicePtr(), HIPFFT_BACKWARD);
#else
        cufftExecC2C(ffthandle, (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     CUFFT_INVERSE);
#endif

        // aMulKernel(sizeVol, 1.0f / sizeTot, d_referenceCplx);
        // aMulVolKernel(sizeVol, d_mask, d_referenceCplx);
        hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
                           (float2 *)d_referenceCplx.GetDevicePtr());
        hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol,
                           (float *)d_mask.GetDevicePtr(),
                           (float2 *)d_referenceCplx.GetDevicePtr());
        aReduceKernel.sumcplx(d_referenceCplx, sum, sizeTot);

        // aSubCplxKernel(sizeVol, d_referenceCplx, d_referenceCplx, sum, nVox);
        // aMulVolKernel(sizeVol, d_mask, d_referenceCplx);
        hipLaunchKernelGGL(SubCplx, grid, block, 0, stream, sizeVol,
                           (float2 *)d_referenceCplx.GetDevicePtr(),
                           (float2 *)d_referenceCplx.GetDevicePtr(),
                           (float *)sum.GetDevicePtr(),
                           (float *)nVox.GetDevicePtr());
        hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol,
                           (float *)d_mask.GetDevicePtr(),
                           (float2 *)d_referenceCplx.GetDevicePtr());
        aReduceKernel.sumsqrcplx(d_referenceCplx, sumSqr, sizeTot);

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(
            ffthandle, (hipfftComplex *)d_referenceCplx.GetDevicePtr(),
            (hipfftComplex *)d_referenceCplx.GetDevicePtr(), HIPFFT_FORWARD);
#else
        cufftExecC2C(ffthandle, (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     CUFFT_FORWARD);
#endif

        // aCorrelKernel(sizeVol, d_particleCplx, d_referenceCplx);
        // aConvKernel(sizeVol, d_maskCplx, d_particleCplx);
        // aConvKernel(sizeVol, d_maskCplx, d_particleSqrCplx);
        hipLaunchKernelGGL(Correl, grid, block, 0, stream, sizeVol,
                           (float2 *)d_particleCplx.GetDevicePtr(),
                           (float2 *)d_referenceCplx.GetDevicePtr());
        hipLaunchKernelGGL(Conv, grid, block, 0, stream, sizeVol,
                           (float2 *)d_maskCplx.GetDevicePtr(),
                           (float2 *)d_particleCplx.GetDevicePtr());
        hipLaunchKernelGGL(Conv, grid, block, 0, stream, sizeVol,
                           (float2 *)d_maskCplx.GetDevicePtr(),
                           (float2 *)d_particleSqrCplx.GetDevicePtr());

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(
            ffthandle, (hipfftComplex *)d_referenceCplx.GetDevicePtr(),
            (hipfftComplex *)d_referenceCplx.GetDevicePtr(), HIPFFT_BACKWARD);
#else
        cufftExecC2C(ffthandle, (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     CUFFT_INVERSE);
#endif
        // aMulKernel(sizeVol, 1.0f / sizeTot, d_referenceCplx);
        hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
                           (float2 *)d_referenceCplx.GetDevicePtr());

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(ffthandle, (hipfftComplex *)d_particleCplx.GetDevicePtr(),
                      (hipfftComplex *)d_particleCplx.GetDevicePtr(),
                      HIPFFT_BACKWARD);
#else
        cufftExecC2C(ffthandle, (cufftComplex *)d_particleCplx.GetDevicePtr(),
                     (cufftComplex *)d_particleCplx.GetDevicePtr(),
                     CUFFT_INVERSE);
#endif

        // aMulKernel(sizeVol, 1.0f / sizeTot, d_particleCplx);
        hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
                           (float2 *)d_particleCplx.GetDevicePtr());

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(
            ffthandle, (hipfftComplex *)d_particleSqrCplx.GetDevicePtr(),
            (hipfftComplex *)d_particleSqrCplx.GetDevicePtr(), HIPFFT_BACKWARD);
#else
        cufftExecC2C(
            ffthandle, (cufftComplex *)d_particleSqrCplx.GetDevicePtr(),
            (cufftComplex *)d_particleSqrCplx.GetDevicePtr(), CUFFT_INVERSE);
#endif

        // aMulKernel(sizeVol, 1.0f / sizeTot, d_particleSqrCplx);
        hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
                           (float2 *)d_particleSqrCplx.GetDevicePtr());

        // aEnergyNormKernel(sizeVol, d_particleCplx, d_particleSqrCplx,
        // d_referenceCplx, sumSqr, nVox); aFFTShift2Kernel(sizeVol,
        // d_referenceCplx, d_ffttemp); aMulVolKernel(sizeVol, d_ccMask,
        // d_ffttemp);

        hipLaunchKernelGGL(Energynorm, grid, block, 0, stream, sizeVol,
                           (float2 *)d_particleCplx.GetDevicePtr(),
                           (float2 *)d_particleSqrCplx.GetDevicePtr(),
                           (float2 *)d_referenceCplx.GetDevicePtr(),
                           (float *)sumSqr.GetDevicePtr(),
                           (float *)nVox.GetDevicePtr());
        hipLaunchKernelGGL(FFTShift, grid, block, 0, stream, sizeVol,
                           (float2 *)d_referenceCplx.GetDevicePtr(),
                           (float2 *)d_ffttemp.GetDevicePtr());

        // float* check = new float[d_ffttemp.GetSize()];
        // d_ffttemp.CopyDeviceToHost(check);
        // emwrite("c:\\users\\kunz\\Desktop\\check.em", check, 400, 200, 200);
        // exit(0);

        hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol,
                           (float *)d_ccMask.GetDevicePtr(),
                           (float2 *)d_ffttemp.GetDevicePtr());
        counter++;

        if (computeCCValOnly) {
          // only read out the CC value at the old shift position and store it
          // in d_buffer
          d_index.CopyHostToDevice(&oldIndex);

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
          hipSafeCall(hipMemcpyDtoD(
              d_buffer.GetDevicePtr(),
              (float2 *)d_ffttemp.GetDevicePtr() + oldIndex, sizeof(float)));
#else
          hipSafeCall(hipMemcpyDtoD(d_buffer.GetDevicePtr(),
                                    d_ffttemp.GetDevicePtr() + oldIndex,
                                    sizeof(float)));
#endif
        } else {
          // find new Maximum value and store position and value
          aReduceKernel.maxindexcplx(d_ffttemp, d_buffer, d_index, sizeTot);
        }

        // aMaxKernel(maxVals, d_index, d_buffer, rphi, rpsi, rthe);
        hipLaunchKernelGGL(FindMax, grid, block, 0, stream,
                           (float *)maxVals.GetDevicePtr(),
                           (float *)d_index.GetDevicePtr(),
                           (float *)d_buffer.GetDevicePtr(), rphi, rpsi, rthe);
      }
    }
  }

  hipSafeCall(hipStreamSynchronize(stream));
  maxVals.CopyDeviceToHost(&m);
  hipSafeCall(hipStreamSynchronize(stream));

  return m;
}


/* AS
 * Average Process with quality improvements
 * (much easier to write and test code using hipLaunchKernelGGL as alternative
 * to the <<< >>> method from Nvidia) 
 * No Performance Improvements
 * 
 * Removed Errors:  - double application of particle preprocessing
 *                  - mean-free reference
 *                  - improved rotation interpolation kernel
 */
AvgProcessC2C::AvgProcessC2C(size_t _sizeVol, hipStream_t _stream,
                             Hip::HipContext *_ctx, float *_mask, float *_ref,
                             float *_ccMask, float aPhiAngIter,
                             float aPhiAngInc, float aAngIter, float aAngIncr,
                             bool aBinarizeMask, bool aRotateMaskCC,
                             bool aUseFilterVolume, bool linearInterpolation,
                             KernelModuls modules)
    : sizeVol(_sizeVol), sizeTot(_sizeVol * _sizeVol * _sizeVol),
      stream(_stream), binarizeMask(aBinarizeMask), rotateMaskCC(aRotateMaskCC),
      useFilterVolume(aUseFilterVolume), ctx(_ctx), mask(_mask), ref(_ref),
      ccMask(_ccMask),

      d_ffttemp(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_particle(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_wedge(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_filter(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_manipulation(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_manipulation_ref(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_reference_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_mask_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_reference(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_mask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_ccMask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_ccMask_Orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_particleCplx(_sizeVol * _sizeVol * (_sizeVol) * sizeof(float2)),
      d_particleSqrCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_particleCplx_orig(_sizeVol * _sizeVol * (_sizeVol) * sizeof(float2)),
      d_particleSqrCplx_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_referenceCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_maskCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_buffer(_sizeVol * _sizeVol * _sizeVol *
               sizeof(float2)), // should be sufficient for everything...
      d_index(_sizeVol * _sizeVol * _sizeVol *
              sizeof(int)), // should be sufficient for everything...
      nVox(_sizeVol * _sizeVol * _sizeVol *
           sizeof(float)), // should be sufficient for everything...
      sum(_sizeVol * _sizeVol * _sizeVol *
          sizeof(float)), // should be sufficient for everything...
      sumSqr(_sizeVol * _sizeVol * _sizeVol *
             sizeof(float)),      // should be sufficient for everything...
      maxVals(sizeof(maxVals_t)), // should be sufficient for everything...
      phi_angiter(aPhiAngIter), phi_angincr(aPhiAngInc), angiter(aAngIter),
      angincr(aAngIncr),

      aRotateKernelROT(modules.modbasicKernels,
                       dim3(sizeVol / 32, sizeVol / 16, sizeVol),
                       dim3(32, 16, 1), sizeVol, linearInterpolation),
      aRotateKernelMASK(modules.modbasicKernels,
                        dim3(sizeVol / 32, sizeVol / 16, sizeVol),
                        dim3(32, 16, 1), sizeVol, linearInterpolation),
      aRotateKernelMASKCC(modules.modbasicKernels,
                          dim3(sizeVol / 32, sizeVol / 16, sizeVol),
                          dim3(32, 16, 1), sizeVol, linearInterpolation),
      aReduceKernel(modules.modkernel,
                    dim3(sizeVol / 32, sizeVol / 16, sizeVol),
                    dim3(32, 16, 1)) {
  hipSafeCall(hipMallocHost((void **)&index, sizeof(int)));
  hipSafeCall(hipMallocHost((void **)&sum_h, sizeof(float)));
  hipSafeCall(hipMallocHost((void **)&sum_ref, sizeof(float)));
  hipSafeCall(hipMallocHost((void **)&sum_mask, sizeof(float)));
  hipSafeCall(hipMallocHost((void **)&sumCplx, sizeof(float2)));

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  int n[] = {(int)sizeVol, (int)sizeVol, (int)sizeVol};
  hipfftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, HIPFFT_C2C, 1);
  hipfftSetStream(ffthandle, stream);
#else
  int n[] = {(int)sizeVol, (int)sizeVol, (int)sizeVol};
  cufftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, 1);
  cufftSetStream(ffthandle, stream);
#endif

  d_reference.CopyHostToDevice(ref);

  aReduceKernel.sum(d_reference, d_buffer, sizeTot);
  float sum = 0;
  d_buffer.CopyDeviceToHost(&sum, sizeof(float));
  sum = sum / sizeTot;
  hipLaunchKernelGGL(Sub, grid, block, 0, stream, sizeVol,
                     (float *)d_reference.GetDevicePtr(),
                     (float *)d_reference_orig.GetDevicePtr(), sum);

  d_mask_orig.CopyHostToDevice(mask);
  d_ccMask_Orig.CopyHostToDevice(ccMask);
  d_ccMask.CopyHostToDevice(ccMask);
  printf("Quality improved and NO Performance");

}

AvgProcessC2C::~AvgProcessC2C() {
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftDestroy(ffthandle);
#else
  cufftDestroy(ffthandle);
#endif
}

maxVals_t AvgProcessC2C::execute(float *_data, float *wedge, float *filter,
                                 float oldphi, float oldpsi, float oldtheta,
                                 float rDown, float rUp, float smooth,
                                 float3 oldShift, bool couplePhiToPsi,
                                 bool computeCCValOnly, int oldIndex) {
  /* AS
   * The Average Process rotates the reference for different angles defined in
   * the configuration file and compares it to the particle. We return the best
   * fit and now know have a approx. delta of the rotation angles and shifts.
   *
   * C2C defines the Fourier Transform Type Complex to Complex which is needs
   * more time than the Real 2 Complex version (R2C)
   */
  
  int oldWedge = -1;
  maxVals_t m;
  m.index = 0;
  m.ccVal = -10000;
  m.rphi = 0;
  m.rpsi = 0;
  m.rthe = 0;

  hipSafeCall(hipStreamSynchronize(stream));
  maxVals.CopyHostToDeviceAsync(stream, &m);

  d_particle.CopyHostToDevice(wedge);
  hipLaunchKernelGGL(FFTShiftReal, grid, block, 0, stream, sizeVol,
                     (float *)d_particle.GetDevicePtr(),
                     (float *)d_wedge.GetDevicePtr());

  if (useFilterVolume) {
    d_particle.CopyHostToDevice(filter);
    hipLaunchKernelGGL(FFTShiftReal, grid, block, 0, stream, sizeVol,
                       (float *)d_particle.GetDevicePtr(),
                       (float *)d_filter.GetDevicePtr());
  }

  d_particle.CopyHostToDeviceAsync(stream, _data);
  // AS Added a hipStreamSynchronize here
  hipStreamSynchronize(stream);

  // aMakeCplxWithSubKernel(sizeVol, d_particle, d_particleCplx_orig, 0.);
  hipLaunchKernelGGL(MakeCplxWithSub, grid, block, 0, stream, sizeVol,
                     (float *)d_particle.GetDevicePtr(),
                     (float2 *)d_particleCplx_orig.GetDevicePtr(), 0.f);

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftExecC2C(ffthandle, (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
                (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
                HIPFFT_FORWARD);
#else
  cufftExecC2C(ffthandle, (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
               (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
               CUFFT_FORWARD);
#endif

  /* AS Can be replaced by ApplyWedgeFilterBandpass Kernel
   * aMulVolKernel(sizeVol, d_wedge, d_particleCplx_orig);
   * For less Memory access per computation - will be used in the performance
   * improved version of the avg process
   */
  hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol, 
              (float *)d_wedge.GetDevicePtr(), 
              (float2 *)d_particleCplx_orig.GetDevicePtr()); 

  if (useFilterVolume)
  {
          // aMulVolKernel(sizeVol, d_filter, d_particleCplx_orig);
          hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol, 
            (float *)d_filter.GetDevicePtr(), 
            (float2 *)d_particleCplx_orig.GetDevicePtr());
  }
  else
  {
          // aBandpassFFTShiftKernel(sizeVol, d_particleCplx_orig, rDown, rUp, smooth); 
          hipLaunchKernelGGL(BandpassFFTShift, grid, block, 0, stream, sizeVol,
            (float2 *)d_particleCplx_orig.GetDevicePtr(), rDown, rUp, smooth);
  }
  
  /*
  hipLaunchKernelGGL(ApplyWedgeFilterBandpass, grid, block, 0, stream, sizeVol,
                     (float *)d_manipulation.GetDevicePtr(),
                     (float *)d_wedge.GetDevicePtr(),
                     (float *)d_filter.GetDevicePtr(),
                     (float2 *)d_particleCplx_orig.GetDevicePtr(), rDown, rUp,
                     smooth, useFilterVolume);
  */
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftExecC2C(ffthandle, (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
                (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
                HIPFFT_BACKWARD);
#else
  cufftExecC2C(ffthandle, (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
               (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
               CUFFT_INVERSE);
#endif

  // aMulKernel(sizeVol, 1.0f / sizeTot, d_particleCplx_orig);
  hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
                     (float2 *)d_particleCplx_orig.GetDevicePtr());

  // aMakeRealKernel(sizeVol, d_particleCplx_orig, d_particle);
  hipLaunchKernelGGL(MakeReal, grid, block, 0, stream, sizeVol,
                     (float2 *)d_particleCplx_orig.GetDevicePtr(),
                     (float *)d_particle.GetDevicePtr());

  aReduceKernel.sum(d_particle, d_buffer, sizeTot);

  d_buffer.CopyDeviceToHost(sum_h, sizeof(float));
  hipStreamSynchronize(stream);

  // aMakeCplxWithSubKernel(sizeVol, d_particle, d_particleCplx_orig, *sum_h /
  // sizeTot);
  hipLaunchKernelGGL(MakeCplxWithSub, grid, block, 0, stream, sizeVol,
                     (float *)d_particle.GetDevicePtr(),
                     (float2 *)d_particleCplx_orig.GetDevicePtr(),
                     *sum_h / sizeTot);

  // aMakeCplxWithSqrSubKernel(sizeVol, d_particle, d_particleSqrCplx_orig,
  // *sum_h / sizeTot);
  hipLaunchKernelGGL(MakeCplxSqrWithSub, grid, block, 0, stream, sizeVol,
                     (float *)d_particle.GetDevicePtr(),
                     (float2 *)d_particleSqrCplx_orig.GetDevicePtr(),
                     *sum_h / sizeTot);

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftExecC2C(ffthandle, (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
                (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
                HIPFFT_FORWARD);
  hipfftExecC2C(
      ffthandle, (hipfftComplex *)d_particleSqrCplx_orig.GetDevicePtr(),
      (hipfftComplex *)d_particleSqrCplx_orig.GetDevicePtr(), HIPFFT_FORWARD);
#else
  cufftExecC2C(ffthandle, (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
               (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
               CUFFT_FORWARD);
  cufftExecC2C(ffthandle, (cufftComplex *)d_particleSqrCplx_orig.GetDevicePtr(),
               (cufftComplex *)d_particleSqrCplx_orig.GetDevicePtr(),
               CUFFT_FORWARD);
#endif

  /* AS Fixed The wedge and filter are being applied twice I think this should
   * not happen */
  /* AS After talking to Michael this was done because of numerical uncertainty
   * in the fft and this way 0 is 0 and not some value closed to it. Maybe this 
   * can patched in the future for now this will be removed.
   */
  /* AS Utz and I realized that this also breaks the substraction of the mean
   * ... now the 0 values are less than 0
   */
  /* AS All these issues have been fixed in the quality improved version */

  /* AS Deprecated
  // aMulVolKernel(sizeVol, d_wedge, d_particleCplx_orig);
  hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol, (float
  *)d_wedge.GetDevicePtr(), (float2 *)d_particleCplx_orig.GetDevicePtr());

  if (useFilterVolume)
  {
          // aMulVolKernel(sizeVol, d_filter, d_particleCplx_orig);
          hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol, (float
  *)d_filter.GetDevicePtr(), (float2 *)d_particleCplx_orig.GetDevicePtr());
  }
  else
  {
          // aBandpassFFTShiftKernel(sizeVol, d_particleCplx_orig, rDown, rUp,
  smooth); hipLaunchKernelGGL(BandpassFFTShift, grid, block, 0, stream, sizeVol,
  (float2 *)d_particleCplx_orig.GetDevicePtr(), rDown, rUp, smooth);
  }
  */

  aRotateKernelROT.SetTexture(d_reference_orig);
  aRotateKernelMASK.SetTexture(d_mask_orig);

  if (rotateMaskCC) {
    aRotateKernelMASKCC.SetTexture(d_ccMask_Orig);
    aRotateKernelMASKCC.SetOldAngles(oldphi, oldpsi, oldtheta);
  }

  aRotateKernelROT.SetOldAngles(oldphi, oldpsi, oldtheta);
  aRotateKernelMASK.SetOldAngles(oldphi, oldpsi, oldtheta);

  // for angle...
  float rthe = 0;
  float rpsi = 0;
  float rphi = 0;
  float maxthe = 0;
  float maxpsi = 0;
  float maxphi = 0;
  float npsi, dpsi;

  int angles = 0;
  double time = 0;

  int counter = 0;

  double diff1 = 0;
  double diff2 = 0;
  double diff3 = 0;
  double diff4 = 0;

  float maxTest = -1000;
  int maxindex = -1000;

  for (int iterPhi = 0; iterPhi < 2 * phi_angiter + 1; ++iterPhi) {
    rphi = phi_angincr * (iterPhi - phi_angiter);
    for (int iterThe = (int)0; iterThe < angiter + 1; ++iterThe) {
      if (iterThe == 0) {
        npsi = 1;
        dpsi = 360;
      } else {
        dpsi = angincr / sinf(iterThe * angincr * (float)M_PI / 180.0f);
        npsi = ceilf(360.0f / dpsi);
      }
      rthe = iterThe * angincr;
      for (int iterPsi = 0; iterPsi < npsi; ++iterPsi) {
        rpsi = iterPsi * dpsi;

        if (couplePhiToPsi) {
          rphi = phi_angincr * (iterPhi - phi_angiter) - rpsi;
        } else {
          rphi = phi_angincr * (iterPhi - phi_angiter);
        }

        d_particleCplx.CopyDeviceToDeviceAsync(stream, d_particleCplx_orig);
        d_particleSqrCplx.CopyDeviceToDeviceAsync(stream,
                                                  d_particleSqrCplx_orig);

        aRotateKernelROT.do_rotate_improved(sizeVol, d_reference, rphi, rpsi, rthe);
        aRotateKernelMASK.do_rotate_improved(sizeVol, d_mask, rphi, rpsi, rthe);

        if (rotateMaskCC) {
          d_ccMask.Memset(0);
          aRotateKernelMASKCC.do_rotate_improved(sizeVol, d_ccMask, rphi, rpsi, rthe);
        }

        if (binarizeMask) {
          // aBinarizeKernel(sizeVol, d_mask, d_mask);
          hipLaunchKernelGGL(Binarize, grid, block, 0, stream, sizeVol,
                             (float *)d_mask.GetDevicePtr(),
                             (float *)d_mask.GetDevicePtr());
        }

        aReduceKernel.sum(d_mask, nVox, sizeTot);

        // aMakeCplxWithSubKernel(sizeVol, d_reference, d_referenceCplx, 0);
        // aMakeCplxWithSubKernel(sizeVol, d_mask, d_maskCplx, 0);
        hipLaunchKernelGGL(MakeCplxWithSub, grid, block, 0, stream, sizeVol,
                           (float *)d_reference.GetDevicePtr(),
                           (float2 *)d_referenceCplx.GetDevicePtr(), 0);
        hipLaunchKernelGGL(MakeCplxWithSub, grid, block, 0, stream, sizeVol,
                           (float *)d_mask.GetDevicePtr(),
                           (float2 *)d_maskCplx.GetDevicePtr(), 0);

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(
            ffthandle, (hipfftComplex *)d_referenceCplx.GetDevicePtr(),
            (hipfftComplex *)d_referenceCplx.GetDevicePtr(), HIPFFT_FORWARD);
#else
        cufftExecC2C(ffthandle, (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     CUFFT_FORWARD);
#endif

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(ffthandle, (hipfftComplex *)d_maskCplx.GetDevicePtr(),
                      (hipfftComplex *)d_maskCplx.GetDevicePtr(),
                      HIPFFT_FORWARD);
#else
        cufftExecC2C(ffthandle, (cufftComplex *)d_maskCplx.GetDevicePtr(),
                     (cufftComplex *)d_maskCplx.GetDevicePtr(), CUFFT_FORWARD);
#endif
        // aMulVolKernel(sizeVol, d_wedge, d_referenceCplx);
        hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol, 
                            (float*)d_wedge.GetDevicePtr(), 
                            (float2 *)d_referenceCplx.GetDevicePtr());

        if (useFilterVolume)
        {
          // aMulVolKernel(sizeVol, d_filter, d_referenceCplx);
          hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol,
                      (float *)d_filter.GetDevicePtr(), 
                      (float2*)d_referenceCplx.GetDevicePtr());
        }
        else
        {
          // aBandpassFFTShiftKernel(sizeVol, d_referenceCplx, rDown, rUp,smooth); 
          hipLaunchKernelGGL(BandpassFFTShift, grid, block, 0, stream,
          sizeVol, (float2 *)d_referenceCplx.GetDevicePtr(), rDown, rUp, smooth);
        }
        
        /*
        hipLaunchKernelGGL(ApplyWedgeFilterBandpass, grid, block, 0, stream,
                           sizeVol, (float *)d_manipulation_ref.GetDevicePtr(),
                           (float *)d_wedge.GetDevicePtr(),
                           (float *)d_filter.GetDevicePtr(),
                           (float2 *)d_referenceCplx.GetDevicePtr(), rDown, rUp,
                           smooth, useFilterVolume);
        */

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(
            ffthandle, (hipfftComplex *)d_referenceCplx.GetDevicePtr(),
            (hipfftComplex *)d_referenceCplx.GetDevicePtr(), HIPFFT_BACKWARD);
#else
        cufftExecC2C(ffthandle, (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     CUFFT_INVERSE);
#endif

        // aMulKernel(sizeVol, 1.0f / sizeTot, d_referenceCplx);
        hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
                           (float2 *)d_referenceCplx.GetDevicePtr());

        /* AS Utz and I realized that this also breaks the substraction of the
         * mean ... now the 0 values are less than 0
         */
        hipLaunchKernelGGL(MakeReal, grid, block, 0, stream, sizeVol,
                           (float2 *)d_referenceCplx.GetDevicePtr(),
                           (float *)d_reference.GetDevicePtr());

        // aMulVolKernel(sizeVol, d_mask, d_referenceCplx);
        hipLaunchKernelGGL(MulVolReal, grid, block, 0, stream, sizeVol,
                           (float *)d_mask.GetDevicePtr(),
                           (float *)d_reference.GetDevicePtr());
        aReduceKernel.sum(d_reference, d_buffer, sizeTot);
        d_buffer.CopyDeviceToHost(sum_ref, sizeof(float));

        aReduceKernel.sum(d_mask, d_buffer, sizeTot);
        d_buffer.CopyDeviceToHost(sum_mask, sizeof(float));

        hipLaunchKernelGGL(
            MeanFree, grid, block, 0, stream, sizeVol,
            (float *)d_mask.GetDevicePtr(), (float *)d_reference.GetDevicePtr(),
            (float2 *)d_referenceCplx.GetDevicePtr(), *sum_ref, *sum_mask);
        
        /* AS Repaced by MeanFree
         * Exact calculation can be read in my thesis
        // aSubCplxKernel(sizeVol, d_referenceCplx, d_referenceCplx, sum, nVox);
        // aMulVolKernel(sizeVol, d_mask, d_referenceCplx);

        hipLaunchKernelGGL(SubCplx, grid, block, 0, stream, sizeVol, (float2
        *)d_referenceCplx.GetDevicePtr(), (float2
        *)d_referenceCplx.GetDevicePtr(), (float *)sum.GetDevicePtr(), (float
        *)nVox.GetDevicePtr()); hipLaunchKernelGGL(MulVol, grid, block, 0,
        stream, sizeVol, (float *)d_mask.GetDevicePtr(), (float2
        *)d_referenceCplx.GetDevicePtr());
        */

        aReduceKernel.sumcplx(d_referenceCplx, sum, sizeTot);
        aReduceKernel.sumsqrcplx(d_referenceCplx, sumSqr, sizeTot);

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(
            ffthandle, (hipfftComplex *)d_referenceCplx.GetDevicePtr(),
            (hipfftComplex *)d_referenceCplx.GetDevicePtr(), HIPFFT_FORWARD);
#else
        cufftExecC2C(ffthandle, (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     CUFFT_FORWARD);
#endif

        // aCorrelKernel(sizeVol, d_particleCplx, d_referenceCplx);
        // aConvKernel(sizeVol, d_maskCplx, d_particleCplx);
        // aConvKernel(sizeVol, d_maskCplx, d_particleSqrCplx);
        hipLaunchKernelGGL(Correl, grid, block, 0, stream, sizeVol,
                           (float2 *)d_particleCplx.GetDevicePtr(),
                           (float2 *)d_referenceCplx.GetDevicePtr());
        hipLaunchKernelGGL(Conv, grid, block, 0, stream, sizeVol,
                           (float2 *)d_maskCplx.GetDevicePtr(),
                           (float2 *)d_particleCplx.GetDevicePtr());
        hipLaunchKernelGGL(Conv, grid, block, 0, stream, sizeVol,
                           (float2 *)d_maskCplx.GetDevicePtr(),
                           (float2 *)d_particleSqrCplx.GetDevicePtr());

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(
            ffthandle, (hipfftComplex *)d_referenceCplx.GetDevicePtr(),
            (hipfftComplex *)d_referenceCplx.GetDevicePtr(), HIPFFT_BACKWARD);
#else
        cufftExecC2C(ffthandle, (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     (cufftComplex *)d_referenceCplx.GetDevicePtr(),
                     CUFFT_INVERSE);
#endif
        // aMulKernel(sizeVol, 1.0f / sizeTot, d_referenceCplx);
        hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
                           (float2 *)d_referenceCplx.GetDevicePtr());

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(ffthandle, (hipfftComplex *)d_particleCplx.GetDevicePtr(),
                      (hipfftComplex *)d_particleCplx.GetDevicePtr(),
                      HIPFFT_BACKWARD);
#else
        cufftExecC2C(ffthandle, (cufftComplex *)d_particleCplx.GetDevicePtr(),
                     (cufftComplex *)d_particleCplx.GetDevicePtr(),
                     CUFFT_INVERSE);
#endif

        // aMulKernel(sizeVol, 1.0f / sizeTot, d_particleCplx);
        hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
                           (float2 *)d_particleCplx.GetDevicePtr());

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
        hipfftExecC2C(
            ffthandle, (hipfftComplex *)d_particleSqrCplx.GetDevicePtr(),
            (hipfftComplex *)d_particleSqrCplx.GetDevicePtr(), HIPFFT_BACKWARD);
#else
        cufftExecC2C(
            ffthandle, (cufftComplex *)d_particleSqrCplx.GetDevicePtr(),
            (cufftComplex *)d_particleSqrCplx.GetDevicePtr(), CUFFT_INVERSE);
#endif

        // aMulKernel(sizeVol, 1.0f / sizeTot, d_particleSqrCplx);
        hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
                           (float2 *)d_particleSqrCplx.GetDevicePtr());

        // aEnergyNormKernel(sizeVol, d_particleCplx, d_particleSqrCplx,
        // d_referenceCplx, sumSqr, nVox); aFFTShift2Kernel(sizeVol,
        // d_referenceCplx, d_ffttemp); aMulVolKernel(sizeVol, d_ccMask,
        // d_ffttemp);

        hipLaunchKernelGGL(Energynorm, grid, block, 0, stream, sizeVol,
                           (float2 *)d_particleCplx.GetDevicePtr(),
                           (float2 *)d_particleSqrCplx.GetDevicePtr(),
                           (float2 *)d_referenceCplx.GetDevicePtr(),
                           (float *)sumSqr.GetDevicePtr(),
                           (float *)nVox.GetDevicePtr());
        hipLaunchKernelGGL(FFTShift, grid, block, 0, stream, sizeVol,
                           (float2 *)d_referenceCplx.GetDevicePtr(),
                           (float2 *)d_ffttemp.GetDevicePtr());
        hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol,
                           (float *)d_ccMask.GetDevicePtr(),
                           (float2 *)d_ffttemp.GetDevicePtr());
        counter++;

        if (computeCCValOnly) {
          // only read out the CC value at the old shift position and store it
          // in d_buffer
          d_index.CopyHostToDevice(&oldIndex);

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
          hipSafeCall(hipMemcpyDtoD(
              d_buffer.GetDevicePtr(),
              (float2 *)d_ffttemp.GetDevicePtr() + oldIndex, sizeof(float)));
#else
          hipSafeCall(hipMemcpyDtoD(d_buffer.GetDevicePtr(),
                                    d_ffttemp.GetDevicePtr() + oldIndex,
                                    sizeof(float)));
#endif
        } else {
          // find new Maximum value and store position and value
          aReduceKernel.maxindexcplx(d_ffttemp, d_buffer, d_index, sizeTot);
        }

        // aMaxKernel(maxVals, d_index, d_buffer, rphi, rpsi, rthe);
        hipLaunchKernelGGL(FindMax, grid, block, 0, stream,
                           (float *)maxVals.GetDevicePtr(),
                           (float *)d_index.GetDevicePtr(),
                           (float *)d_buffer.GetDevicePtr(), rphi, rpsi, rthe);
      }
    }
  }

  hipSafeCall(hipStreamSynchronize(stream));
  maxVals.CopyDeviceToHost(&m);
  hipSafeCall(hipStreamSynchronize(stream));

  return m;
}


// /* AS
//  * Test implementation of the real to complex fft
//  * 
//  * Removed Errors:  - double application of particle preprocessing
//  *                  - mean-free reference
//  *                  - improved rotation interpolation kernel
//  */
// AvgProcessReal2Complex::AvgProcessReal2Complex(size_t _sizeVol, hipStream_t _stream,
//                              Hip::HipContext *_ctx, float *_mask, float *_ref,
//                              float *_ccMask, float aPhiAngIter,
//                              float aPhiAngInc, float aAngIter, float aAngIncr,
//                              bool aBinarizeMask, bool aRotateMaskCC,
//                              bool aUseFilterVolume, bool linearInterpolation,
//                              KernelModuls modules)
//     : sizeVol(_sizeVol), sizeTot(_sizeVol * _sizeVol * _sizeVol),
//       stream(_stream), binarizeMask(aBinarizeMask), rotateMaskCC(aRotateMaskCC),
//       useFilterVolume(aUseFilterVolume), ctx(_ctx), mask(_mask), ref(_ref),
//       ccMask(_ccMask),

//       d_ffttemp(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
//       d_particle(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
//       d_particle_sqr(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
//       d_wedge(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
//       d_filter(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
//       d_manipulation(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
//       d_manipulation_ref(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
//       d_reference_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
//       d_mask_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
//       d_reference(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
//       d_mask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
//       d_ccMask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
//       d_ccMask_Orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
//       d_particleCplx(_sizeVol * _sizeVol * (_sizeVol) * sizeof(float2)),
//       d_particleSqrCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
//       d_particleCplx_orig(_sizeVol * _sizeVol * (_sizeVol) * sizeof(float2)),
//       d_particleCplx_orig_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) * sizeof(float2)),
//       d_referenceCplx_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) * sizeof(float2)),
//       d_particleSqrCplx_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
//       d_referenceCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
//       d_maskCplx(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
//       d_buffer(_sizeVol * _sizeVol * _sizeVol *
//                sizeof(float2)), // should be sufficient for everything...
//       d_index(_sizeVol * _sizeVol * _sizeVol *
//               sizeof(int)), // should be sufficient for everything...
//       nVox(_sizeVol * _sizeVol * _sizeVol *
//            sizeof(float)), // should be sufficient for everything...
//       sum(_sizeVol * _sizeVol * _sizeVol *
//           sizeof(float)), // should be sufficient for everything...
//       sumSqr(_sizeVol * _sizeVol * _sizeVol *
//              sizeof(float)),      // should be sufficient for everything...
//       maxVals(sizeof(maxVals_t)), // should be sufficient for everything...
//       phi_angiter(aPhiAngIter), phi_angincr(aPhiAngInc), angiter(aAngIter),
//       angincr(aAngIncr),

//       aRotateKernelROT(modules.modbasicKernels,
//                        dim3(sizeVol / 32, sizeVol / 16, sizeVol),
//                        dim3(32, 16, 1), sizeVol, linearInterpolation),
//       aRotateKernelMASK(modules.modbasicKernels,
//                         dim3(sizeVol / 32, sizeVol / 16, sizeVol),
//                         dim3(32, 16, 1), sizeVol, linearInterpolation),
//       aRotateKernelMASKCC(modules.modbasicKernels,
//                           dim3(sizeVol / 32, sizeVol / 16, sizeVol),
//                           dim3(32, 16, 1), sizeVol, linearInterpolation),
//       aReduceKernel(modules.modkernel,
//                     dim3(sizeVol / 32, sizeVol / 16, sizeVol),
//                     dim3(32, 16, 1)) {
//   hipSafeCall(hipMallocHost((void **)&index, sizeof(int)));
//   hipSafeCall(hipMallocHost((void **)&sum_h, sizeof(float)));
//   hipSafeCall(hipMallocHost((void **)&sum_ref, sizeof(float)));
//   hipSafeCall(hipMallocHost((void **)&sum_mask, sizeof(float)));
//   hipSafeCall(hipMallocHost((void **)&sumCplx, sizeof(float2)));

// #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
//   int n[] = {(int)sizeVol, (int)sizeVol, (int)sizeVol};
//   hipfftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, HIPFFT_C2C, 1);
//   hipfftSetStream(ffthandle, stream);

//   hipfftPlanMany(&ffthandle_R2C, 3, n, NULL, 0, 0, NULL, 0, 0, HIPFFT_R2C, 1);
//   hipfftSetStream(ffthandle_R2C, stream);

//   hipfftPlanMany(&ffthandle_C2R, 3, n, NULL, 0, 0, NULL, 0, 0, HIPFFT_C2R, 1);
//   hipfftSetStream(ffthandle_C2R, stream);
// #else
//   int n[] = {(int)sizeVol, (int)sizeVol, (int)sizeVol};
//   cufftPlanMany(&ffthandle, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2C, 1);
//   cufftSetStream(ffthandle, stream);

//   cufftPlanMany(&ffthandle_R2C, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, 1);
//   cufftSetStream(ffthandle_R2C, stream);

//   cufftPlanMany(&ffthandle_C2R, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2R, 1);
//   cufftSetStream(ffthandle_C2R, stream);
// #endif

//   d_reference.CopyHostToDevice(ref);

//   aReduceKernel.sum(d_reference, d_buffer, sizeTot);
//   float sum = 0;
//   d_buffer.CopyDeviceToHost(&sum, sizeof(float));
//   sum = sum / sizeTot;
//   hipLaunchKernelGGL(Sub, grid, block, 0, stream, sizeVol,
//                      (float *)d_reference.GetDevicePtr(),
//                      (float *)d_reference_orig.GetDevicePtr(), sum);

//   d_mask_orig.CopyHostToDevice(mask);
//   d_ccMask_Orig.CopyHostToDevice(ccMask);
//   d_ccMask.CopyHostToDevice(ccMask);
//   printf("Transformation from C2C to R2C\n");

// }

// AvgProcessReal2Complex::~AvgProcessReal2Complex() {
// #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
//   hipfftDestroy(ffthandle);
//   hipfftDestroy(ffthandle_R2C);
//   hipfftDestroy(ffthandle_C2R);
// #else
//   cufftDestroy(ffthandle);
//   cufftDestroy(ffthandle_R2C);
//   cufftDestroy(ffthandle_C2R);
// #endif
// }

// maxVals_t AvgProcessReal2Complex::execute(float *_data, float *wedge, float *filter,
//                                  float oldphi, float oldpsi, float oldtheta,
//                                  float rDown, float rUp, float smooth,
//                                  float3 oldShift, bool couplePhiToPsi,
//                                  bool computeCCValOnly, int oldIndex) {
//   /* AS
//    * The Average Process rotates the reference for different angles defined in
//    * the configuration file and compares it to the particle. We return the best
//    * fit and now know have a approx. delta of the rotation angles and shifts.
//    *
//    * C2C defines the Fourier Transform Type Complex to Complex which is needs
//    * more time than the Real 2 Complex version (R2C)
//    */
  
//   int oldWedge = -1;
//   maxVals_t m;
//   m.index = 0;
//   m.ccVal = -10000;
//   m.rphi = 0;
//   m.rpsi = 0;
//   m.rthe = 0;

//   hipSafeCall(hipStreamSynchronize(stream));
//   maxVals.CopyHostToDeviceAsync(stream, &m);

//   d_particle.CopyHostToDevice(wedge);
//   hipLaunchKernelGGL(FFTShiftReal, grid, block, 0, stream, sizeVol,
//                      (float *)d_particle.GetDevicePtr(),
//                      (float *)d_wedge.GetDevicePtr());

//   if (useFilterVolume) {
//     d_particle.CopyHostToDevice(filter);
//     hipLaunchKernelGGL(FFTShiftReal, grid, block, 0, stream, sizeVol,
//                        (float *)d_particle.GetDevicePtr(),
//                        (float *)d_filter.GetDevicePtr());
//   }

//   d_particle.CopyHostToDeviceAsync(stream, _data);
//   hipStreamSynchronize(stream);

// // aMakeCplxWithSubKernel(sizeVol, d_particle, d_particleCplx_orig, 0.);
// hipLaunchKernelGGL(MakeCplxWithSub, grid, block, 0, stream, sizeVol,
//                    (float *)d_particle.GetDevicePtr(),
//                    (float2 *)d_particleCplx_orig.GetDevicePtr(), 0.f);

// #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
//   hipfftExecC2C(ffthandle, (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
//                 (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
//                 HIPFFT_FORWARD);
// #else
//   cufftExecC2C(ffthandle, (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
//                (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
//                CUFFT_FORWARD);
// #endif

//   // AS Can be replaced by ApplyWedgeFilterBandpass Kernel
//   // aMulVolKernel(sizeVol, d_wedge, d_particleCplx_orig);
//   hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol, (float
//   *)d_wedge.GetDevicePtr(), (float2 *)d_particleCplx_orig.GetDevicePtr()); 
//   if (useFilterVolume)
//   {
//           // aMulVolKernel(sizeVol, d_filter, d_particleCplx_orig);
//           hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol, (float
//   *)d_filter.GetDevicePtr(), (float2 *)d_particleCplx_orig.GetDevicePtr());
//   }
//   else
//   {
//           // aBandpassFFTShiftKernel(sizeVol, d_particleCplx_orig, rDown, rUp, smooth); 
//           hipLaunchKernelGGL(BandpassFFTShift, grid, block, 0, stream, sizeVol,
//   (float2 *)d_particleCplx_orig.GetDevicePtr(), rDown, rUp, smooth);
//   }
  
//   /*
//   hipLaunchKernelGGL(ApplyWedgeFilterBandpass, grid, block, 0, stream, sizeVol,
//                      (float *)d_manipulation.GetDevicePtr(),
//                      (float *)d_wedge.GetDevicePtr(),
//                      (float *)d_filter.GetDevicePtr(),
//                      (float2 *)d_particleCplx_orig.GetDevicePtr(), rDown, rUp,
//                      smooth, useFilterVolume);
//   */
// #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
//   hipfftExecC2C(ffthandle, (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
//                 (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
//                 HIPFFT_BACKWARD);
// #else
//   cufftExecC2C(ffthandle, (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
//                (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
//                CUFFT_INVERSE);
// #endif

//   // aMulKernel(sizeVol, 1.0f / sizeTot, d_particleCplx_orig);
//   hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
//                      (float2 *)d_particleCplx_orig.GetDevicePtr());

//   // aMakeRealKernel(sizeVol, d_particleCplx_orig, d_particle);
//   hipLaunchKernelGGL(MakeReal, grid, block, 0, stream, sizeVol,
//                      (float2 *)d_particleCplx_orig.GetDevicePtr(),
//                      (float *)d_particle.GetDevicePtr());

// #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
//   hipfftExecR2C(ffthandle_R2C, (hipfftReal *)d_particle.GetDevicePtr(),
//                 (hipfftComplex *)d_particleCplx_orig_RC.GetDevicePtr());
// #else
//   cufftExecR2C(ffthandle_R2C, (cufftReal *)d_particle.GetDevicePtr(),
//                (cufftComplex *)d_particleCplx_orig_RC.GetDevicePtr());
// #endif

//   // aMulVolKernel_RC(sizeVol, d_wedge, d_particleCplx_orig_RC);
//   hipLaunchKernelGGL(mulVol_RC_, grid_RC, block_RC, 0, stream, sizeVol,
//                      (float *)d_wedge.GetDevicePtr(),
//                      (float2 *)d_particleCplx_orig_RC.GetDevicePtr());

//   if (useFilterVolume) {
//     // aMulVolKernel_RC(sizeVol, d_filter, d_particleCplx_orig_RC);
//     hipLaunchKernelGGL(mulVol_RC_, grid_RC, block_RC, 0, stream, sizeVol,
//                        (float *)d_filter.GetDevicePtr(),
//                        (float2 *)d_particleCplx_orig_RC.GetDevicePtr());
//   } else {
//     // aBandpassFFTShiftKernel_RC(sizeVol, d_particleCplx_orig_RC, rDown, rUp,
//     // smooth);
//     hipLaunchKernelGGL(bandpassFFTShift_RC_, grid_RC, block_RC, 0, stream,
//                        sizeVol, (float2 *)d_particleCplx_orig_RC.GetDevicePtr(),
//                        rDown, rUp, smooth);
//   }
    
//   /*
//   hipLaunchKernelGGL(ApplyWedgeFilterBandpass_RC, grid, block, 0, stream, sizeVol,
//                      (float *)d_wedge.GetDevicePtr(),
//                      (float *)d_filter.GetDevicePtr(),
//                      (float2 *)d_particleCplx_orig_RC.GetDevicePtr(), rDown, rUp,
//                      smooth, useFilterVolume);
//   */

// #if defined(__HIP_PLATFORM_HCC__)
//   hipfftExecC2R(ffthandle_C2R,
//                 (hipfftComplex *)d_particleCplx_orig_RC.GetDevicePtr(),
//                 (hipfftReal *)d_particle.GetDevicePtr());
// #else
//   cufftExecC2R(ffthandle_C2R,
//                (cufftComplex *)d_particleCplx_orig_RC.GetDevicePtr(),
//                (cufftReal *)d_particle.GetDevicePtr());
// #endif

//   hipLaunchKernelGGL(mul_Real_, grid, block, 0, stream, sizeVol,
//                      1.0f / sizeTot, (float *)d_particle.GetDevicePtr());


//   printf("Conversion until here\n");

//   aReduceKernel.sum(d_particle, d_buffer, sizeTot);

//   /*
//   AS Todo after calculating the sum there is no reason to copy it back to the
//   host also doing the copy asyncronous makes no sense. Replaced by none async
//   copy
//   */

//   d_buffer.CopyDeviceToHost(sum_h, sizeof(float));
//   hipStreamSynchronize(stream);

//   /* AS deprecated replaced by none asynchronis copy
//   d_buffer.CopyDeviceToHostAsync(stream, sum_h, sizeof(float));
//   hipDeviceSynchronize();
//   hipStreamSynchronize(stream);

//   hipSafeCall(hipDeviceSynchronize());
//   hipSafeCall(hipCtxSynchronize());
//   */

// /* */
//   // aMakeCplxWithSubKernel(sizeVol, d_particle, d_particleCplx_orig, *sum_h /
//   // sizeTot);
//   hipLaunchKernelGGL(MakeCplxWithSub, grid, block, 0, stream, sizeVol,
//                      (float *)d_particle.GetDevicePtr(),
//                      (float2 *)d_particleCplx_orig.GetDevicePtr(),
//                      *sum_h / sizeTot);

//   // aMakeCplxWithSqrSubKernel(sizeVol, d_particle, d_particleSqrCplx_orig,
//   // *sum_h / sizeTot);
//   hipLaunchKernelGGL(MakeCplxSqrWithSub, grid, block, 0, stream, sizeVol,
//                      (float *)d_particle.GetDevicePtr(),
//                      (float2 *)d_particleSqrCplx_orig.GetDevicePtr(),
//                      *sum_h / sizeTot);
// /* */
// /* 
//   // aSqrSubKernel_RC(sizeVol, d_particle, d_particle_sqr, d_buffer,
//   // (float)sizeTot);
//   hipLaunchKernelGGL(sqrsub_RC_, grid, block, 0, stream, sizeVol,
//                      (float *)d_particle.GetDevicePtr(),
//                      (float *)d_particle_sqr.GetDevicePtr(),
//                      (float *)d_buffer.GetDevicePtr(), (float)sizeTot);

//   // aSubKernel_RC(sizeVol, d_particle, d_particle, d_buffer, (float)sizeTot);
//   hipLaunchKernelGGL(sub_RC_, grid, block, 0, stream, sizeVol,
//                      (float *)d_particle.GetDevicePtr(),
//                      (float *)d_particle.GetDevicePtr(),
//                      (float *)d_buffer.GetDevicePtr(), (float)sizeTot);
// /* */

// #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
//   hipfftExecC2C(ffthandle, (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
//                 (hipfftComplex *)d_particleCplx_orig.GetDevicePtr(),
//                 HIPFFT_FORWARD);
//   hipfftExecC2C(
//       ffthandle, (hipfftComplex *)d_particleSqrCplx_orig.GetDevicePtr(),
//       (hipfftComplex *)d_particleSqrCplx_orig.GetDevicePtr(), HIPFFT_FORWARD);
// #else
//   cufftExecC2C(ffthandle, (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
//                (cufftComplex *)d_particleCplx_orig.GetDevicePtr(),
//                CUFFT_FORWARD);
//   cufftExecC2C(ffthandle, (cufftComplex *)d_particleSqrCplx_orig.GetDevicePtr(),
//                (cufftComplex *)d_particleSqrCplx_orig.GetDevicePtr(),
//                CUFFT_FORWARD);
// #endif

// // #if defined(__HIP_PLATFORM_HCC__)
// //   hipfftExecR2C(ffthandle_R2C, (hipfftReal *)d_particle.GetDevicePtr(),
// //                 (hipfftComplex *)d_particleCplx_orig_RC.GetDevicePtr());
// //   hipfftExecR2C(ffthandle_R2C, (hipfftReal *)d_particle_sqr.GetDevicePtr(),
// //                 (hipfftComplex *)d_particleSqrCplx_orig_RC.GetDevicePtr());
// // #else
// //   cufftExecR2C(ffthandle_R2C, (cufftReal *)d_particle.GetDevicePtr(),
// //                (cufftComplex *)d_particleCplx_orig_RC.GetDevicePtr());
// //   cufftExecR2C(ffthandle_R2C, (cufftReal *)d_particle_sqr.GetDevicePtr(),
// //                (cufftComplex *)d_particleSqrCplx_orig_RC.GetDevicePtr());
// // #endif

//   /* AS Fixed The wedge and filter are being applied twice I think this should
//    * not happen */
//   /* AS After talking to Michael this was done because of numerical uncertainty
//   in the fft and this way 0 is 0 and not some value closed to it. Maybe this can
//   patched in the future for now this will be removed.*/

//   /* AS Deprecated
//   // aMulVolKernel(sizeVol, d_wedge, d_particleCplx_orig);
//   hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol, (float
//   *)d_wedge.GetDevicePtr(), (float2 *)d_particleCplx_orig.GetDevicePtr());

//   if (useFilterVolume)
//   {
//           // aMulVolKernel(sizeVol, d_filter, d_particleCplx_orig);
//           hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol, (float
//   *)d_filter.GetDevicePtr(), (float2 *)d_particleCplx_orig.GetDevicePtr());
//   }
//   else
//   {
//           // aBandpassFFTShiftKernel(sizeVol, d_particleCplx_orig, rDown, rUp,
//   smooth); hipLaunchKernelGGL(BandpassFFTShift, grid, block, 0, stream, sizeVol,
//   (float2 *)d_particleCplx_orig.GetDevicePtr(), rDown, rUp, smooth);
//   }
//   */

//   aRotateKernelROT.SetTexture(d_reference_orig);
//   aRotateKernelMASK.SetTexture(d_mask_orig);

//   if (rotateMaskCC) {
//     aRotateKernelMASKCC.SetTexture(d_ccMask_Orig);
//     aRotateKernelMASKCC.SetOldAngles(oldphi, oldpsi, oldtheta);
//   }

//   aRotateKernelROT.SetOldAngles(oldphi, oldpsi, oldtheta);
//   aRotateKernelMASK.SetOldAngles(oldphi, oldpsi, oldtheta);

//   // for angle...
//   float rthe = 0;
//   float rpsi = 0;
//   float rphi = 0;
//   float maxthe = 0;
//   float maxpsi = 0;
//   float maxphi = 0;
//   float npsi, dpsi;

//   int angles = 0;
//   double time = 0;

//   int counter = 0;

//   double diff1 = 0;
//   double diff2 = 0;
//   double diff3 = 0;
//   double diff4 = 0;

//   float maxTest = -1000;
//   int maxindex = -1000;

//   for (int iterPhi = 0; iterPhi < 2 * phi_angiter + 1; ++iterPhi) {
//     rphi = phi_angincr * (iterPhi - phi_angiter);
//     for (int iterThe = (int)0; iterThe < angiter + 1; ++iterThe) {
//       if (iterThe == 0) {
//         npsi = 1;
//         dpsi = 360;
//       } else {
//         dpsi = angincr / sinf(iterThe * angincr * (float)M_PI / 180.0f);
//         npsi = ceilf(360.0f / dpsi);
//       }
//       rthe = iterThe * angincr;
//       for (int iterPsi = 0; iterPsi < npsi; ++iterPsi) {
//         rpsi = iterPsi * dpsi;

//         if (couplePhiToPsi) {
//           rphi = phi_angincr * (iterPhi - phi_angiter) - rpsi;
//         } else {
//           rphi = phi_angincr * (iterPhi - phi_angiter);
//         }

//         d_particleCplx.CopyDeviceToDeviceAsync(stream, d_particleCplx_orig);
//         d_particleSqrCplx.CopyDeviceToDeviceAsync(stream,
//                                                   d_particleSqrCplx_orig);

//         aRotateKernelROT.do_rotate_improved(sizeVol, d_reference, rphi, rpsi, rthe);
//         aRotateKernelMASK.do_rotate_improved(sizeVol, d_mask, rphi, rpsi, rthe);

//         if (rotateMaskCC) {
//           d_ccMask.Memset(0);
//           aRotateKernelMASKCC.do_rotate_improved(sizeVol, d_ccMask, rphi, rpsi, rthe);
//         }

//         if (binarizeMask) {
//           // aBinarizeKernel(sizeVol, d_mask, d_mask);
//           hipLaunchKernelGGL(Binarize, grid, block, 0, stream, sizeVol,
//                              (float *)d_mask.GetDevicePtr(),
//                              (float *)d_mask.GetDevicePtr());
//         }

//         aReduceKernel.sum(d_mask, nVox, sizeTot);

//         // aMakeCplxWithSubKernel(sizeVol, d_reference, d_referenceCplx, 0);
//         // aMakeCplxWithSubKernel(sizeVol, d_mask, d_maskCplx, 0);

//         hipLaunchKernelGGL(MakeCplxWithSub, grid, block, 0, stream, sizeVol,
//                            (float *)d_mask.GetDevicePtr(),
//                            (float2 *)d_maskCplx.GetDevicePtr(), 0);

// #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
//         hipfftExecC2C(ffthandle, (hipfftComplex *)d_maskCplx.GetDevicePtr(),
//                       (hipfftComplex *)d_maskCplx.GetDevicePtr(),
//                       HIPFFT_FORWARD);
// #else
//         cufftExecC2C(ffthandle, (cufftComplex *)d_maskCplx.GetDevicePtr(),
//                      (cufftComplex *)d_maskCplx.GetDevicePtr(), CUFFT_FORWARD);
// #endif

// //          hipLaunchKernelGGL(MakeCplxWithSub, grid, block, 0, stream, sizeVol,
// //                    (float *)d_reference.GetDevicePtr(),
// //                    (float2 *)d_referenceCplx.GetDevicePtr(), 0);

// // #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
// //         hipfftExecC2C(ffthandle, (hipfftComplex *)d_referenceCplx.GetDevicePtr(),
// //                       (hipfftComplex *)d_referenceCplx.GetDevicePtr(), 
// //                       HIPFFT_FORWARD);
// // #else
// //         cufftExecC2C(ffthandle, (cufftComplex *)d_referenceCplx.GetDevicePtr(),
// //                      (cufftComplex *)d_referenceCplx.GetDevicePtr(),
// //                      CUFFT_FORWARD);
// // #endif

// //         // aMulVolKernel(sizeVol, d_wedge, d_referenceCplx);
// //         hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol, 
// //                             (float*)d_wedge.GetDevicePtr(), 
// //                             (float2 *)d_referenceCplx.GetDevicePtr());

// //         if (useFilterVolume)
// //         {
// //           // aMulVolKernel(sizeVol, d_filter, d_referenceCplx);
// //           hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol,
// //                       (float *)d_filter.GetDevicePtr(), 
// //                       (float2*)d_referenceCplx.GetDevicePtr());
// //         }
// //         else
// //         {
// //           // aBandpassFFTShiftKernel(sizeVol, d_referenceCplx, rDown, rUp,smooth); 
// //           hipLaunchKernelGGL(BandpassFFTShift, grid, block, 0, stream,
// //           sizeVol, (float2 *)d_referenceCplx.GetDevicePtr(), rDown, rUp, smooth);
// //         }
        
// //         /*
// //         hipLaunchKernelGGL(ApplyWedgeFilterBandpass, grid, block, 0, stream,
// //                            sizeVol, (float *)d_manipulation_ref.GetDevicePtr(),
// //                            (float *)d_wedge.GetDevicePtr(),
// //                            (float *)d_filter.GetDevicePtr(),
// //                            (float2 *)d_referenceCplx.GetDevicePtr(), rDown, rUp,
// //                            smooth, useFilterVolume);
// //         */

// // #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
// //         hipfftExecC2C(ffthandle, (hipfftComplex *)d_referenceCplx.GetDevicePtr(),
// //                       (hipfftComplex *)d_referenceCplx.GetDevicePtr(), 
// //                       HIPFFT_BACKWARD);
// // #else
// //         cufftExecC2C(ffthandle, (cufftComplex *)d_referenceCplx.GetDevicePtr(),
// //                      (cufftComplex *)d_referenceCplx.GetDevicePtr(),
// //                      CUFFT_INVERSE);
// // #endif

//       #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
//         hipfftExecR2C(ffthandle_R2C, (hipfftReal *)d_reference.GetDevicePtr(),
//                       (hipfftComplex *)d_referenceCplx_RC.GetDevicePtr());
//       #else
//         cufftExecR2C(ffthandle_R2C, (cufftReal *)d_reference.GetDevicePtr(),
//                     (cufftComplex *)d_referenceCplx_RC.GetDevicePtr());
//       #endif

//         // aMulVolKernel_RC(sizeVol, d_wedge, d_particleCplx_orig_RC);
//         hipLaunchKernelGGL(mulVol_RC_, grid_RC, block_RC, 0, stream, sizeVol,
//                           (float *)d_wedge.GetDevicePtr(),
//                           (float2 *)d_referenceCplx_RC.GetDevicePtr());

//         if (useFilterVolume) {
//           // aMulVolKernel_RC(sizeVol, d_filter, d_particleCplx_orig_RC);
//           hipLaunchKernelGGL(mulVol_RC_, grid_RC, block_RC, 0, stream, sizeVol,
//                             (float *)d_filter.GetDevicePtr(),
//                             (float2 *)d_referenceCplx_RC.GetDevicePtr());
//         } else {
//           // aBandpassFFTShiftKernel_RC(sizeVol, d_particleCplx_orig_RC, rDown, rUp,
//           // smooth);
//           hipLaunchKernelGGL(bandpassFFTShift_RC_, grid_RC, block_RC, 0, stream,
//                             sizeVol, (float2 *)d_referenceCplx_RC.GetDevicePtr(),
//                             rDown, rUp, smooth);
//         }
          
//         /*
//         hipLaunchKernelGGL(ApplyWedgeFilterBandpass_RC, grid, block, 0, stream, sizeVol,
//                           (float *)d_wedge.GetDevicePtr(),
//                           (float *)d_filter.GetDevicePtr(),
//                           (float2 *)d_particleCplx_orig_RC.GetDevicePtr(), rDown, rUp,
//                           smooth, useFilterVolume);
//         */

//       #if defined(__HIP_PLATFORM_HCC__)
//         hipfftExecC2R(ffthandle_C2R,
//                       (hipfftComplex *)d_referenceCplx_RC.GetDevicePtr(),
//                       (hipfftReal *)d_reference.GetDevicePtr());
//       #else
//         cufftExecC2R(ffthandle_C2R,
//                     (cufftComplex *)d_referenceCplx_RC.GetDevicePtr(),
//                     (cufftReal *)d_reference.GetDevicePtr());
//       #endif

//         hipLaunchKernelGGL(mul_Real_, grid, block, 0, stream, sizeVol,
//                           1.0f / sizeTot, (float *)d_reference.GetDevicePtr());
        
//         // // aMulKernel(sizeVol, 1.0f / sizeTot, d_referenceCplx);
//         // hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
//         //                    (float2 *)d_referenceCplx.GetDevicePtr());

//         /* AS Utz and I realized that this also breaks the substraction of the
//          * mean ... now the 0 values are less than 0
//          */
//         // hipLaunchKernelGGL(MakeReal, grid, block, 0, stream, sizeVol,
//         //                    (float2 *)d_referenceCplx.GetDevicePtr(),
//         //                    (float *)d_reference.GetDevicePtr());

//         // aMulVolKernel(sizeVol, d_mask, d_referenceCplx);
//         hipLaunchKernelGGL(MulVolReal, grid, block, 0, stream, sizeVol,
//                            (float *)d_mask.GetDevicePtr(),
//                            (float *)d_reference.GetDevicePtr());

//         aReduceKernel.sum(d_reference, d_buffer, sizeTot);
//         d_buffer.CopyDeviceToHost(sum_ref, sizeof(float));

//         aReduceKernel.sum(d_mask, d_buffer, sizeTot);
//         d_buffer.CopyDeviceToHost(sum_mask, sizeof(float));

//         hipLaunchKernelGGL(
//             MeanFree, grid, block, 0, stream, sizeVol,
//             (float *)d_mask.GetDevicePtr(), (float *)d_reference.GetDevicePtr(),
//             (float2 *)d_referenceCplx.GetDevicePtr(), *sum_ref, *sum_mask);

//         aReduceKernel.sumcplx(d_referenceCplx, sum, sizeTot);
//         aReduceKernel.sumsqrcplx(d_referenceCplx, sumSqr, sizeTot);

// #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
//         hipfftExecC2C(
//             ffthandle, (hipfftComplex *)d_referenceCplx.GetDevicePtr(),
//             (hipfftComplex *)d_referenceCplx.GetDevicePtr(), HIPFFT_FORWARD);
// #else
//         cufftExecC2C(ffthandle, (cufftComplex *)d_referenceCplx.GetDevicePtr(),
//                      (cufftComplex *)d_referenceCplx.GetDevicePtr(),
//                      CUFFT_FORWARD);
// #endif


//         // aCorrelKernel(sizeVol, d_particleCplx, d_referenceCplx);
//         // aConvKernel(sizeVol, d_maskCplx, d_particleCplx);
//         // aConvKernel(sizeVol, d_maskCplx, d_particleSqrCplx);
//         hipLaunchKernelGGL(Correl, grid, block, 0, stream, sizeVol,
//                            (float2 *)d_particleCplx.GetDevicePtr(),
//                            (float2 *)d_referenceCplx.GetDevicePtr());
//         hipLaunchKernelGGL(Conv, grid, block, 0, stream, sizeVol,
//                            (float2 *)d_maskCplx.GetDevicePtr(),
//                            (float2 *)d_particleCplx.GetDevicePtr());
//         hipLaunchKernelGGL(Conv, grid, block, 0, stream, sizeVol,
//                            (float2 *)d_maskCplx.GetDevicePtr(),
//                            (float2 *)d_particleSqrCplx.GetDevicePtr());

// #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
//         hipfftExecC2C(
//             ffthandle, (hipfftComplex *)d_referenceCplx.GetDevicePtr(),
//             (hipfftComplex *)d_referenceCplx.GetDevicePtr(), HIPFFT_BACKWARD);
// #else
//         cufftExecC2C(ffthandle, (cufftComplex *)d_referenceCplx.GetDevicePtr(),
//                      (cufftComplex *)d_referenceCplx.GetDevicePtr(),
//                      CUFFT_INVERSE);
// #endif
//         // aMulKernel(sizeVol, 1.0f / sizeTot, d_referenceCplx);
//         hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
//                            (float2 *)d_referenceCplx.GetDevicePtr());

// #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
//         hipfftExecC2C(ffthandle, (hipfftComplex *)d_particleCplx.GetDevicePtr(),
//                       (hipfftComplex *)d_particleCplx.GetDevicePtr(),
//                       HIPFFT_BACKWARD);
// #else
//         cufftExecC2C(ffthandle, (cufftComplex *)d_particleCplx.GetDevicePtr(),
//                      (cufftComplex *)d_particleCplx.GetDevicePtr(),
//                      CUFFT_INVERSE);
// #endif

//         // aMulKernel(sizeVol, 1.0f / sizeTot, d_particleCplx);
//         hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
//                            (float2 *)d_particleCplx.GetDevicePtr());

// #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
//         hipfftExecC2C(
//             ffthandle, (hipfftComplex *)d_particleSqrCplx.GetDevicePtr(),
//             (hipfftComplex *)d_particleSqrCplx.GetDevicePtr(), HIPFFT_BACKWARD);
// #else
//         cufftExecC2C(
//             ffthandle, (cufftComplex *)d_particleSqrCplx.GetDevicePtr(),
//             (cufftComplex *)d_particleSqrCplx.GetDevicePtr(), CUFFT_INVERSE);
// #endif

//         // aMulKernel(sizeVol, 1.0f / sizeTot, d_particleSqrCplx);
//         hipLaunchKernelGGL(Mul, grid, block, 0, stream, sizeVol, 1.0f / sizeTot,
//                            (float2 *)d_particleSqrCplx.GetDevicePtr());

//         // aEnergyNormKernel(sizeVol, d_particleCplx, d_particleSqrCplx,
//         // d_referenceCplx, sumSqr, nVox); aFFTShift2Kernel(sizeVol,
//         // d_referenceCplx, d_ffttemp); aMulVolKernel(sizeVol, d_ccMask,
//         // d_ffttemp);
//         hipLaunchKernelGGL(MakeReal, grid, block, 0, stream, sizeVol,
//                     (float2 *)d_particleCplx.GetDevicePtr(),
//                     (float *)d_particle.GetDevicePtr());
//         hipLaunchKernelGGL(MakeReal, grid, block, 0, stream, sizeVol,
//                     (float2 *)d_particleSqrCplx.GetDevicePtr(),
//                     (float *)d_particle_sqr.GetDevicePtr());
//         hipLaunchKernelGGL(MakeReal, grid, block, 0, stream, sizeVol,
//                     (float2 *)d_referenceCplx.GetDevicePtr(),
//                     (float *)d_reference.GetDevicePtr());

//         hipLaunchKernelGGL(
//             energynorm_RC_, grid, block, 0, stream, sizeVol,
//             (float *)d_particle.GetDevicePtr(),
//             (float *)d_particle_sqr.GetDevicePtr(),
//             (float *)d_reference.GetDevicePtr(), (float *)sumSqr.GetDevicePtr(),
//             (float *)nVox.GetDevicePtr(), (float2 *)d_ffttemp.GetDevicePtr(),
//             (float *)d_ccMask.GetDevicePtr());

//         // hipLaunchKernelGGL(Energynorm, grid, block, 0, stream, sizeVol,
//         //                    (float2 *)d_particleCplx.GetDevicePtr(),
//         //                    (float2 *)d_particleSqrCplx.GetDevicePtr(),
//         //                    (float2 *)d_referenceCplx.GetDevicePtr(),
//         //                    (float *)sumSqr.GetDevicePtr(),
//         //                    (float *)nVox.GetDevicePtr());
//         // hipLaunchKernelGGL(FFTShift, grid, block, 0, stream, sizeVol,
//         //                    (float2 *)d_referenceCplx.GetDevicePtr(),
//         //                    (float2 *)d_ffttemp.GetDevicePtr());
//         // hipLaunchKernelGGL(MulVol, grid, block, 0, stream, sizeVol,
//         //                    (float *)d_ccMask.GetDevicePtr(),
//         //                    (float2 *)d_ffttemp.GetDevicePtr());

//         counter++;

//         if (computeCCValOnly) {
//           // only read out the CC value at the old shift position and store it
//           // in d_buffer
//           d_index.CopyHostToDevice(&oldIndex);

// #if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
//           hipSafeCall(hipMemcpyDtoD(
//               d_buffer.GetDevicePtr(),
//               (float2 *)d_ffttemp.GetDevicePtr() + oldIndex, sizeof(float)));
// #else
//           hipSafeCall(hipMemcpyDtoD(d_buffer.GetDevicePtr(),
//                                     d_ffttemp.GetDevicePtr() + oldIndex,
//                                     sizeof(float)));
// #endif
//         } else {
//           // find new Maximum value and store position and value
//           aReduceKernel.maxindexcplx(d_ffttemp, d_buffer, d_index, sizeTot);
//         }

//         // aMaxKernel(maxVals, d_index, d_buffer, rphi, rpsi, rthe);
//         hipLaunchKernelGGL(FindMax, grid, block, 0, stream,
//                            (float *)maxVals.GetDevicePtr(),
//                            (float *)d_index.GetDevicePtr(),
//                            (float *)d_buffer.GetDevicePtr(), rphi, rpsi, rthe);
//       }
//     }
//   }

//   hipSafeCall(hipStreamSynchronize(stream));
//   maxVals.CopyDeviceToHost(&m);
//   hipSafeCall(hipStreamSynchronize(stream));

//   return m;
// }


/* AS
 * Average Process with quality and performance improvements
 * 
 * Performance Imp. - Real to Complex FFT
 *                  - Compute per Memory Access
 *                            Many Kernels are combined into a single kernel 
 *                            to increase performance. The original kernels
 *                            should still be visible as comments
 * Removed Errors:  - double application of particle preprocessing
 *                  - mean-free reference
 *                  - improved rotation interpolation kernel
 */
AvgProcessR2C::AvgProcessR2C(size_t _sizeVol, hipStream_t _stream,
                             Hip::HipContext *_ctx, float *_mask, float *_ref,
                             float *_ccMask, float aPhiAngIter,
                             float aPhiAngInc, float aAngIter, float aAngIncr,
                             bool aBinarizeMask, bool aRotateMaskCC,
                             bool aUseFilterVolume, bool linearInterpolation,
                             KernelModuls modules)
    : sizeVol(_sizeVol), sizeTot(_sizeVol * _sizeVol * _sizeVol),
      stream(_stream), binarizeMask(aBinarizeMask), rotateMaskCC(aRotateMaskCC),
      useFilterVolume(aUseFilterVolume), ctx(_ctx), mask(_mask), ref(_ref),
      ccMask(_ccMask),
      d_ffttemp(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_particle(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_particle_sqr(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_wedge(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_filter(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_reference_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_mask_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_reference(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_mask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_ccMask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_ccMask_Orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_particleCplx_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) *
                        sizeof(float2)),
      d_particleSqrCplx_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) *
                           sizeof(float2)),
      d_particleCplx_orig_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) *
                             sizeof(float2)),
      d_particleSqrCplx_orig_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) *
                                sizeof(float2)),
      d_referenceCplx_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) *
                         sizeof(float2)),
      d_maskCplx_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) * sizeof(float2)),

      d_buffer(_sizeVol * _sizeVol * _sizeVol *
               sizeof(float)), // should be sufficient for everything...
      d_index(_sizeVol * _sizeVol * _sizeVol *
              sizeof(int)), // should be sufficient for everything...
      nVox(_sizeVol * _sizeVol * _sizeVol *
           sizeof(float)), // should be sufficient for everything...
      sum(_sizeVol * _sizeVol * _sizeVol *
          sizeof(float)), // should be sufficient for everything...
      sumSqr(_sizeVol * _sizeVol * _sizeVol *
             sizeof(float)),      // should be sufficient for everything...
      maxVals(sizeof(maxVals_t)), // should be sufficient for everything...
      phi_angiter(aPhiAngIter), phi_angincr(aPhiAngInc), angiter(aAngIter),
      angincr(aAngIncr), aRotateKernelROT(modules.modbasicKernels, grid, block,
                                          sizeVol, linearInterpolation),
      aRotateKernelMASK(modules.modbasicKernels, grid, block, sizeVol,
                        linearInterpolation),
      aRotateKernelMASKCC(modules.modbasicKernels, grid, block, sizeVol,
                          linearInterpolation),
      aReduceKernel(modules.modkernel, grid, block)
{
  hipSafeCall(hipMallocHost((void **)&index, sizeof(int)));
  hipSafeCall(hipMallocHost((void **)&sum_h, sizeof(float)));
  hipSafeCall(hipMallocHost((void **)&sumCplx, sizeof(float2)));
  hipSafeCall(hipMallocHost((void **)&sum_ref, sizeof(float)));
  hipSafeCall(hipMallocHost((void **)&sum_mask, sizeof(float)));

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  int n[] = {(int)sizeVol, (int)sizeVol, (int)sizeVol};

  hipfftPlanMany(&ffthandle_R2C, 3, n, NULL, 0, 0, NULL, 0, 0, HIPFFT_R2C, 1);
  hipfftSetStream(ffthandle_R2C, stream);

  hipfftPlanMany(&ffthandle_C2R, 3, n, NULL, 0, 0, NULL, 0, 0, HIPFFT_C2R, 1);
  hipfftSetStream(ffthandle_C2R, stream);
#else
  int n[] = {(int)sizeVol, (int)sizeVol, (int)sizeVol};

  cufftPlanMany(&ffthandle_R2C, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, 1);
  cufftSetStream(ffthandle_R2C, stream);

  cufftPlanMany(&ffthandle_C2R, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2R, 1);
  cufftSetStream(ffthandle_C2R, stream);
#endif

  d_reference.CopyHostToDevice(ref);

  aReduceKernel.sum(d_reference, d_buffer, sizeTot);
  float sum = 0;
  d_buffer.CopyDeviceToHost(&sum, sizeof(float));
  sum = sum / sizeTot;

  // aSubKernel(sizeVol, d_reference, d_reference_orig, sum);
  hipLaunchKernelGGL(sub_, grid, block, 0, stream, sizeVol,
                     (float *)d_reference.GetDevicePtr(),
                     (float *)d_reference_orig.GetDevicePtr(), sum);

  d_mask_orig.CopyHostToDevice(mask);
  d_ccMask_Orig.CopyHostToDevice(ccMask);
  d_ccMask.CopyHostToDevice(ccMask);
}

AvgProcessR2C::~AvgProcessR2C() {
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftDestroy(ffthandle_R2C);
  hipfftDestroy(ffthandle_C2R);
#else
  cufftDestroy(ffthandle_R2C);
  cufftDestroy(ffthandle_C2R);
#endif
}

maxVals_t AvgProcessR2C::execute(float *_data, float *wedge, float *filter,
                                 float oldphi, float oldpsi, float oldtheta,
                                 float rDown, float rUp, float smooth,
                                 float3 oldShift, bool couplePhiToPsi,
                                 bool computeCCValOnly, int oldIndex) 
{
  /* AS
   * The Average Process rotates the reference for different angles defined in 
   * the configuration file and compares it to the particle. We return the best 
   * fit and now know have a approx. delta of the rotation angles and shifts.
   *
   * R2C defines the Fourier Transform Type Real to Complex which is faster than
   * the Complex to Complex version (C2C)
	 */
  int oldWedge = -1;
  maxVals_t m;
  m.index = 0;
  m.ccVal = -10000;
  m.rphi = 0;
  m.rpsi = 0;
  m.rthe = 0;
  hipSafeCall(hipStreamSynchronize(stream));
  maxVals.CopyHostToDeviceAsync(stream, &m);

  d_particle.CopyHostToDevice(wedge);
  // aFFTShiftRealKernel(sizeVol, d_particle, d_wedge);
  hipLaunchKernelGGL(fftshiftReal_, grid, block, 0, stream, sizeVol,
                     (float *)d_particle.GetDevicePtr(),
                     (float *)d_wedge.GetDevicePtr());

  if (useFilterVolume) {
    d_particle.CopyHostToDevice(filter);
    // aFFTShiftRealKernel(sizeVol, d_particle, d_filter);
    hipLaunchKernelGGL(fftshiftReal_, grid, block, 0, stream, sizeVol,
                       (float *)d_particle.GetDevicePtr(),
                       (float *)d_filter.GetDevicePtr());
  }

  d_particle.CopyHostToDeviceAsync(stream, _data);
  hipStreamSynchronize(stream);

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftExecR2C(ffthandle_R2C, (hipfftReal *)d_particle.GetDevicePtr(),
                (hipfftComplex *)d_particleCplx_orig_RC.GetDevicePtr());
#else
  cufftExecR2C(ffthandle_R2C, (cufftReal *)d_particle.GetDevicePtr(),
               (cufftComplex *)d_particleCplx_orig_RC.GetDevicePtr());
#endif

  /* AS Replaced by Applyfilter Kernel 
  // aMulVolKernel_RC(sizeVol, d_wedge, d_particleCplx_orig_RC);
  hipLaunchKernelGGL(mulVol_RC_, grid_RC, block_RC, 0, stream, sizeVol,
                     (float *)d_wedge.GetDevicePtr(),
                     (float2 *)d_particleCplx_orig_RC.GetDevicePtr());

  if (useFilterVolume) {
    // aMulVolKernel_RC(sizeVol, d_filter, d_particleCplx_orig_RC);
    hipLaunchKernelGGL(mulVol_RC_, grid_RC, block_RC, 0, stream, sizeVol,
                       (float *)d_filter.GetDevicePtr(),
                       (float2 *)d_particleCplx_orig_RC.GetDevicePtr());
  } else {
    // aBandpassFFTShiftKernel_RC(sizeVol, d_particleCplx_orig_RC, rDown, rUp,
    // smooth);
    hipLaunchKernelGGL(bandpassFFTShift_RC_, grid_RC, block_RC, 0, stream,
                       sizeVol, (float2 *)d_particleCplx_orig_RC.GetDevicePtr(),
                       rDown, rUp, smooth);
  }
  */

  hipLaunchKernelGGL(ApplyWedgeFilterBandpass_RC, grid_RC, block_RC, 0, stream, sizeVol,
                     (float *)d_wedge.GetDevicePtr(),
                     (float *)d_filter.GetDevicePtr(),
                     (float2 *)d_particleCplx_orig_RC.GetDevicePtr(), rDown, rUp,
                     smooth, useFilterVolume);

#if defined(__HIP_PLATFORM_HCC__)
  hipfftExecC2R(ffthandle_C2R,
                (hipfftComplex *)d_particleCplx_orig_RC.GetDevicePtr(),
                (hipfftReal *)d_particle.GetDevicePtr());
#else
  cufftExecC2R(ffthandle_C2R,
               (cufftComplex *)d_particleCplx_orig_RC.GetDevicePtr(),
               (cufftReal *)d_particle.GetDevicePtr());
#endif

  // aMulKernel_Real(sizeVol, 1.0f / sizeTot, d_particle);
  hipLaunchKernelGGL(mul_Real_, grid, block, 0, stream, sizeVol,
                     1.0f / sizeTot, (float *)d_particle.GetDevicePtr());
  
  aReduceKernel.sum(d_particle, d_buffer, sizeTot);
  
  /* AS Replaces by SubAndSqrSub
  // aSqrSubKernel_RC(sizeVol, d_particle, d_particle_sqr, d_buffer,
  // (float)sizeTot);
  hipLaunchKernelGGL(sqrsub_RC_, grid, block, 0, stream, sizeVol,
                     (float *)d_particle.GetDevicePtr(),
                     (float *)d_particle_sqr.GetDevicePtr(),
                     (float *)d_buffer.GetDevicePtr(), (float)sizeTot);

  // aSubKernel_RC(sizeVol, d_particle, d_particle, d_buffer, (float)sizeTot);
  hipLaunchKernelGGL(sub_RC_, grid, block, 0, stream, sizeVol,
                     (float *)d_particle.GetDevicePtr(),
                     (float *)d_particle.GetDevicePtr(),
                     (float *)d_buffer.GetDevicePtr(), (float)sizeTot);
  */

  /* TODO AS Create combined Kernel for SqrSub and Sub */
  hipLaunchKernelGGL(SubAndSqrSub, grid, block, 0, stream, sizeVol,
                    (float *)d_particle.GetDevicePtr(),
                    (float *)d_particle_sqr.GetDevicePtr(),
                    (float *)d_buffer.GetDevicePtr(), (float)sizeTot);
  
#if defined(__HIP_PLATFORM_HCC__)
  hipfftExecR2C(ffthandle_R2C, (hipfftReal *)d_particle.GetDevicePtr(),
                (hipfftComplex *)d_particleCplx_orig_RC.GetDevicePtr());
  hipfftExecR2C(ffthandle_R2C, (hipfftReal *)d_particle_sqr.GetDevicePtr(),
                (hipfftComplex *)d_particleSqrCplx_orig_RC.GetDevicePtr());
#else
  cufftExecR2C(ffthandle_R2C, (cufftReal *)d_particle.GetDevicePtr(),
               (cufftComplex *)d_particleCplx_orig_RC.GetDevicePtr());
  cufftExecR2C(ffthandle_R2C, (cufftReal *)d_particle_sqr.GetDevicePtr(),
               (cufftComplex *)d_particleSqrCplx_orig_RC.GetDevicePtr());
#endif

  aRotateKernelROT.SetTexture(d_reference_orig);
  aRotateKernelMASK.SetTexture(d_mask_orig);

  if (rotateMaskCC) {
    aRotateKernelMASKCC.SetTexture(d_ccMask_Orig);
    aRotateKernelMASKCC.SetOldAngles(oldphi, oldpsi, oldtheta);
  }

  aRotateKernelROT.SetOldAngles(oldphi, oldpsi, oldtheta);
  aRotateKernelMASK.SetOldAngles(oldphi, oldpsi, oldtheta);

  float rthe = 0;
  float rpsi = 0;
  float rphi = 0;
  float maxthe = 0;
  float maxpsi = 0;
  float maxphi = 0;
  float npsi, dpsi;

  for (int iterPhi = 0; iterPhi < 2 * phi_angiter + 1; ++iterPhi) {
    rphi = phi_angincr * (iterPhi - phi_angiter);
    for (int iterThe = (int)0; iterThe < angiter + 1; ++iterThe) {
      if (iterThe == 0) {
        npsi = 1;
        dpsi = 360;
      } else {
        dpsi = angincr / sinf(iterThe * angincr * (float)M_PI / 180.0f);
        npsi = ceilf(360.0f / dpsi);
      }
      rthe = iterThe * angincr;
      for (int iterPsi = 0; iterPsi < npsi; ++iterPsi) {
        rpsi = iterPsi * dpsi;

        if (couplePhiToPsi) {
          rphi = phi_angincr * (iterPhi - phi_angiter) - rpsi;
        } else {
          rphi = phi_angincr * (iterPhi - phi_angiter);
        }

        d_particleCplx_RC.CopyDeviceToDeviceAsync(stream,
                                                  d_particleCplx_orig_RC);
        d_particleSqrCplx_RC.CopyDeviceToDeviceAsync(stream,
                                                     d_particleSqrCplx_orig_RC);

        aRotateKernelROT.do_rotate(sizeVol, d_reference, rphi, rpsi, rthe);
        aRotateKernelMASK.do_rotate(sizeVol, d_mask, rphi, rpsi, rthe);

        if (rotateMaskCC) {
          d_ccMask.Memset(0);
          aRotateKernelMASKCC.do_rotate(sizeVol, d_ccMask, rphi, rpsi, rthe);
        }

        if (binarizeMask) {
          // aBinarizeKernel(sizeVol, d_mask, d_mask);
          hipLaunchKernelGGL(binarize_, grid, block, 0, stream, sizeVol,
                             (float *)d_mask.GetDevicePtr(),
                             (float *)d_mask.GetDevicePtr());
        }

        aReduceKernel.sum(d_mask, nVox, sizeTot);

#if defined(__HIP_PLATFORM_HCC__)
        hipfftExecR2C(ffthandle_R2C, (hipfftReal *)d_reference.GetDevicePtr(),
                      (hipfftComplex *)d_referenceCplx_RC.GetDevicePtr());
#else
        cufftExecR2C(ffthandle_R2C, (cufftReal *)d_reference.GetDevicePtr(),
                     (cufftComplex *)d_referenceCplx_RC.GetDevicePtr());
#endif

#if defined(__HIP_PLATFORM_HCC__)
        hipfftExecR2C(ffthandle_R2C, (hipfftReal *)d_mask.GetDevicePtr(),
                      (hipfftComplex *)d_maskCplx_RC.GetDevicePtr());
#else
        cufftExecR2C(ffthandle_R2C, (cufftReal *)d_mask.GetDevicePtr(),
                     (cufftComplex *)d_maskCplx_RC.GetDevicePtr());
#endif

        /* AS Replaced by ApplyWedgeFilterBandpass_RC 
        // aMulVolKernel_RC(sizeVol, d_wedge, d_referenceCplx_RC);
        hipLaunchKernelGGL(mulVol_RC_, grid_RC, block_RC, 0, stream, sizeVol,
                           (float *)d_wedge.GetDevicePtr(),
                           (float2 *)d_referenceCplx_RC.GetDevicePtr());

        if (useFilterVolume) {
          // aMulVolKernel_RC(sizeVol, d_filter, d_referenceCplx_RC);
          hipLaunchKernelGGL(mulVol_RC_, grid_RC, block_RC, 0, stream, sizeVol,
                             (float *)d_filter.GetDevicePtr(),
                             (float2 *)d_referenceCplx_RC.GetDevicePtr());
        } else {
          // aBandpassFFTShiftKernel_RC(sizeVol, d_referenceCplx_RC, rDown, rUp,
          // smooth);
          hipLaunchKernelGGL(
              bandpassFFTShift_RC_, grid_RC, block_RC, 0, stream, sizeVol,
              (float2 *)d_referenceCplx_RC.GetDevicePtr(), rDown, rUp, smooth);
        }
        */
        hipLaunchKernelGGL(ApplyWedgeFilterBandpass_RC, grid_RC, block_RC, 0, stream, sizeVol,
                          (float *)d_wedge.GetDevicePtr(),
                          (float *)d_filter.GetDevicePtr(),
                          (float2 *)d_referenceCplx_RC.GetDevicePtr(), rDown, rUp,
                          smooth, useFilterVolume);

#if defined(__HIP_PLATFORM_HCC__)
        hipfftExecC2R(ffthandle_C2R,
                      (hipfftComplex *)d_referenceCplx_RC.GetDevicePtr(),
                      (hipfftReal *)d_reference.GetDevicePtr());
#else
        cufftExecC2R(ffthandle_C2R,
                     (cufftComplex *)d_referenceCplx_RC.GetDevicePtr(),
                     (cufftReal *)d_reference.GetDevicePtr());
#endif

        /* AS Replaced by mulVolRRMul
        // aMulKernel_Real(sizeVol, 1.0f / sizeTot, d_reference);
        hipLaunchKernelGGL(mul_Real_, grid, block, 0, stream, sizeVol,
                           1.0f / sizeTot, (float *)d_reference.GetDevicePtr());

        // aMulVolKernel_RR(sizeVol, d_mask, d_reference);
        hipLaunchKernelGGL(mulVol_RR_, grid, block, 0, stream, sizeVol,
                           (float *)d_mask.GetDevicePtr(),
                           (float *)d_reference.GetDevicePtr());
        */
        hipLaunchKernelGGL(mulVolRRMul, grid, block, 0, stream, sizeVol,
                           (float *)d_mask.GetDevicePtr(),
                           (float *)d_reference.GetDevicePtr(),
                           1.0f / sizeTot);

        aReduceKernel.sum(d_reference, d_buffer, sizeTot);
        d_buffer.CopyDeviceToHost(sum_ref, sizeof(float));

        aReduceKernel.sum(d_mask, d_buffer, sizeTot);
        d_buffer.CopyDeviceToHost(sum_mask, sizeof(float));

        hipLaunchKernelGGL(
            MeanFree_RC, grid, block, 0, stream, sizeVol,
            (float *)d_mask.GetDevicePtr(), (float *)d_reference.GetDevicePtr(),
            *sum_ref, *sum_mask);
        
        aReduceKernel.sum(d_reference, sum, sizeTot);
        aReduceKernel.sumsqr(d_reference, sumSqr, sizeTot);

#if defined(__HIP_PLATFORM_HCC__)
        hipfftExecR2C(ffthandle_R2C, (hipfftReal *)d_reference.GetDevicePtr(),
                      (hipfftComplex *)d_referenceCplx_RC.GetDevicePtr());
#else
        cufftExecR2C(ffthandle_R2C, (cufftReal *)d_reference.GetDevicePtr(),
                     (cufftComplex *)d_referenceCplx_RC.GetDevicePtr());
#endif

        hipStreamSynchronize(stream); 
        
        /* AS Replaced by correlconvconv_rc
        // aCorrelKernel_RC(sizeVol, d_particleCplx_RC, d_referenceCplx_RC);
        hipLaunchKernelGGL(correl_RC_, grid_RC, block_RC, 0, stream, sizeVol,
                           (float2 *)d_particleCplx_RC.GetDevicePtr(),
                           (float2 *)d_referenceCplx_RC.GetDevicePtr());

        // aConvKernel_RC(sizeVol, d_maskCplx_RC, d_particleCplx_RC);
        hipLaunchKernelGGL(conv_RC_, grid_RC, block_RC, 0, stream, sizeVol,
                           (float2 *)d_maskCplx_RC.GetDevicePtr(),
                           (float2 *)d_particleCplx_RC.GetDevicePtr());

        // aConvKernel_RC(sizeVol, d_maskCplx_RC, d_particleSqrCplx_RC);
        hipLaunchKernelGGL(conv_RC_, grid_RC, block_RC, 0, stream, sizeVol,
                           (float2 *)d_maskCplx_RC.GetDevicePtr(),
                           (float2 *)d_particleSqrCplx_RC.GetDevicePtr());
         */
        hipLaunchKernelGGL(correlConvConv_RC, grid_RC, block_RC, 0, stream, sizeVol,
                           (float2 *)d_particleCplx_RC.GetDevicePtr(),
                           (float2 *)d_referenceCplx_RC.GetDevicePtr(),
                           (float2 *)d_maskCplx_RC.GetDevicePtr(),
                           (float2 *)d_particleSqrCplx_RC.GetDevicePtr());
        
#if defined(__HIP_PLATFORM_HCC__)
        hipfftExecC2R(ffthandle_C2R,
                      (hipfftComplex *)d_referenceCplx_RC.GetDevicePtr(),
                      (hipfftReal *)d_reference.GetDevicePtr());
        hipfftExecC2R(ffthandle_C2R,
                      (hipfftComplex *)d_particleCplx_RC.GetDevicePtr(),
                      (hipfftReal *)d_particle.GetDevicePtr());
        hipfftExecC2R(ffthandle_C2R,
                      (hipfftComplex *)d_particleSqrCplx_RC.GetDevicePtr(),
                      (hipfftReal *)d_particle_sqr.GetDevicePtr());
#else
        cufftExecC2R(ffthandle_C2R,
                     (cufftComplex *)d_referenceCplx_RC.GetDevicePtr(),
                     (cufftReal *)d_reference.GetDevicePtr());
        cufftExecC2R(ffthandle_C2R,
                     (cufftComplex *)d_particleCplx_RC.GetDevicePtr(),
                     (cufftReal *)d_particle.GetDevicePtr());
        cufftExecC2R(ffthandle_C2R,
                     (cufftComplex *)d_particleSqrCplx_RC.GetDevicePtr(),
                     (cufftReal *)d_particle_sqr.GetDevicePtr());
#endif

        /* Replaced by energynormMulMulMul 
        // aMulKernel_Real(sizeVol, 1.0f / sizeTot, d_particle);
        hipLaunchKernelGGL(mul_Real_, grid, block, 0, stream, sizeVol,
                           1.0f / sizeTot, (float *)d_particle.GetDevicePtr());
        // aMulKernel_Real(sizeVol, 1.0f / sizeTot, d_particle_sqr);
        hipLaunchKernelGGL(mul_Real_, grid, block, 0, stream, sizeVol,
                           1.0f / sizeTot,
                           (float *)d_particle_sqr.GetDevicePtr());
        // aMulKernel_Real(sizeVol, 1.0f / sizeTot, d_reference);
        hipLaunchKernelGGL(mul_Real_, grid, block, 0, stream, sizeVol,
                           1.0f / sizeTot, (float *)d_reference.GetDevicePtr());

        // aEnergyNormKernel_RC(sizeVol, d_particle, d_particle_sqr,
        // d_reference, sumSqr, nVox, d_ffttemp, d_ccMask);
        hipLaunchKernelGGL(
            energynorm_RC_, grid, block, 0, stream, sizeVol,
            (float *)d_particle.GetDevicePtr(),
            (float *)d_particle_sqr.GetDevicePtr(),
            (float *)d_reference.GetDevicePtr(), (float *)sumSqr.GetDevicePtr(),
            (float *)nVox.GetDevicePtr(), (float2 *)d_ffttemp.GetDevicePtr(),
            (float *)d_ccMask.GetDevicePtr());
         */
          hipLaunchKernelGGL(
            energynormMulMulMUl_RC, grid, block, 0, stream, sizeVol,
            (float *)d_particle.GetDevicePtr(),
            (float *)d_particle_sqr.GetDevicePtr(),
            (float *)d_reference.GetDevicePtr(), (float *)sumSqr.GetDevicePtr(),
            (float *)nVox.GetDevicePtr(), (float2 *)d_ffttemp.GetDevicePtr(),
            (float *)d_ccMask.GetDevicePtr(), 1.0f / sizeTot);

        if (computeCCValOnly) {
          // only read out the CC value at the old shift position and store it
          // in d_buffer
          d_index.CopyHostToDevice(&oldIndex);
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
          hipSafeCall(hipMemcpyDtoD(
              d_buffer.GetDevicePtr(),
              (float2 *)d_ffttemp.GetDevicePtr() + oldIndex, sizeof(float)));
#else
          hipSafeCall(hipMemcpyDtoD(d_buffer.GetDevicePtr(),
                                    d_ffttemp.GetDevicePtr() + oldIndex,
                                    sizeof(float)));
#endif
        } else {
          // find new Maximum value and store position and value
          aReduceKernel.maxindexcplx(d_ffttemp, d_buffer, d_index, sizeTot);
        }

        // aMaxKernel(maxVals, d_index, d_buffer, rphi, rpsi, rthe);
        hipLaunchKernelGGL(findmax_, grid_RC, block_RC, 0, stream,
                           (float *)maxVals.GetDevicePtr(),
                           (float *)d_index.GetDevicePtr(),
                           (float *)d_buffer.GetDevicePtr(), rphi, rpsi, rthe);
      }
    }
  }

  hipSafeCall(hipStreamSynchronize(stream));
  maxVals.CopyDeviceToHost(&m);
  hipSafeCall(hipStreamSynchronize(stream));

  return m;
}


/* AS
 * Average Process with PhaseCorrelation
 */
AvgProcessPhaseCorrelation::AvgProcessPhaseCorrelation(size_t _sizeVol, hipStream_t _stream,
                             Hip::HipContext *_ctx, float *_mask, float *_ref,
                             float *_ccMask, float aPhiAngIter,
                             float aPhiAngInc, float aAngIter, float aAngIncr,
                             bool aBinarizeMask, bool aRotateMaskCC,
                             bool aUseFilterVolume, bool linearInterpolation,
                             KernelModuls modules)
    : sizeVol(_sizeVol), sizeTot(_sizeVol * _sizeVol * _sizeVol),
      stream(_stream), binarizeMask(aBinarizeMask), rotateMaskCC(aRotateMaskCC),
      useFilterVolume(aUseFilterVolume), ctx(_ctx), mask(_mask), ref(_ref),
      ccMask(_ccMask),
      d_ffttemp(_sizeVol * _sizeVol * _sizeVol * sizeof(float2)),
      d_particle(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_particle_sqr(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_wedge(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_filter(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_reference_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_mask_orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_reference(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_mask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_ccMask(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_ccMask_Orig(_sizeVol * _sizeVol * _sizeVol * sizeof(float)),
      d_particleCplx_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) *
                        sizeof(float2)),
      d_particleSqrCplx_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) *
                           sizeof(float2)),
      d_particleCplx_orig_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) *
                             sizeof(float2)),
      d_particleSqrCplx_orig_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) *
                                sizeof(float2)),
      d_referenceCplx_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) *
                         sizeof(float2)),
      d_maskCplx_RC(_sizeVol * _sizeVol * (_sizeVol / 2 + 1) * sizeof(float2)),

      d_buffer(_sizeVol * _sizeVol * _sizeVol *
               sizeof(float)), // should be sufficient for everything...
      d_index(_sizeVol * _sizeVol * _sizeVol *
              sizeof(int)), // should be sufficient for everything...
      nVox(_sizeVol * _sizeVol * _sizeVol *
           sizeof(float)), // should be sufficient for everything...
      sum(_sizeVol * _sizeVol * _sizeVol *
          sizeof(float)), // should be sufficient for everything...
      sumSqr(_sizeVol * _sizeVol * _sizeVol *
             sizeof(float)),      // should be sufficient for everything...
      maxVals(sizeof(maxVals_t)), // should be sufficient for everything...
      phi_angiter(aPhiAngIter), phi_angincr(aPhiAngInc), angiter(aAngIter),
      angincr(aAngIncr), aRotateKernelROT(modules.modbasicKernels, grid, block,
                                          sizeVol, linearInterpolation),
      aRotateKernelMASK(modules.modbasicKernels, grid, block, sizeVol,
                        linearInterpolation),
      aRotateKernelMASKCC(modules.modbasicKernels, grid, block, sizeVol,
                          linearInterpolation),
      aReduceKernel(modules.modkernel, grid, block)

{
  printf("Quality and Performance improved\n");
  hipSafeCall(hipMallocHost((void **)&index, sizeof(int)));
  hipSafeCall(hipMallocHost((void **)&sum_h, sizeof(float)));
  hipSafeCall(hipMallocHost((void **)&sumCplx, sizeof(float2)));
  hipSafeCall(hipMallocHost((void **)&sum_ref, sizeof(float)));
  hipSafeCall(hipMallocHost((void **)&sum_mask, sizeof(float)));

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  int n[] = {(int)sizeVol, (int)sizeVol, (int)sizeVol};

  hipfftPlanMany(&ffthandle_R2C, 3, n, NULL, 0, 0, NULL, 0, 0, HIPFFT_R2C, 1);
  hipfftSetStream(ffthandle_R2C, stream);

  hipfftPlanMany(&ffthandle_C2R, 3, n, NULL, 0, 0, NULL, 0, 0, HIPFFT_C2R, 1);
  hipfftSetStream(ffthandle_C2R, stream);
#else
  int n[] = {(int)sizeVol, (int)sizeVol, (int)sizeVol};

  cufftPlanMany(&ffthandle_R2C, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_R2C, 1);
  cufftSetStream(ffthandle_R2C, stream);

  cufftPlanMany(&ffthandle_C2R, 3, n, NULL, 0, 0, NULL, 0, 0, CUFFT_C2R, 1);
  cufftSetStream(ffthandle_C2R, stream);
#endif

  d_reference.CopyHostToDevice(ref);

  aReduceKernel.sum(d_reference, d_buffer, sizeTot);
  float sum = 0;
  d_buffer.CopyDeviceToHost(&sum, sizeof(float));
  sum = sum / sizeTot;
  // aSubKernel(sizeVol, d_reference, d_reference_orig, sum);
  hipLaunchKernelGGL(sub_, grid, block, 0, stream, sizeVol,
                     (float *)d_reference.GetDevicePtr(),
                     (float *)d_reference_orig.GetDevicePtr(), sum);

  d_mask_orig.CopyHostToDevice(mask);

  d_ccMask_Orig.CopyHostToDevice(ccMask);
  d_ccMask.CopyHostToDevice(ccMask);
}

AvgProcessPhaseCorrelation::~AvgProcessPhaseCorrelation() {
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftDestroy(ffthandle_R2C);
  hipfftDestroy(ffthandle_C2R);
#else
  cufftDestroy(ffthandle_R2C);
  cufftDestroy(ffthandle_C2R);
#endif
}

maxVals_t AvgProcessPhaseCorrelation::execute(float *_data, float *wedge, float *filter,
                                 float oldphi, float oldpsi, float oldtheta,
                                 float rDown, float rUp, float smooth,
                                 float3 oldShift, bool couplePhiToPsi,
                                 bool computeCCValOnly, int oldIndex) 
{
  /* AS
   * new feature introduced by Michael Kunz
   * ported by me to HIP
	 */
  int oldWedge = -1;
  maxVals_t m;
  m.index = 0;
  m.ccVal = -10000;
  m.rphi = 0;
  m.rpsi = 0;
  m.rthe = 0;
  hipSafeCall(hipStreamSynchronize(stream));
  maxVals.CopyHostToDeviceAsync(stream, &m);

  d_particle.CopyHostToDevice(wedge);
  hipLaunchKernelGGL(fftshiftReal_, grid, block, 0, stream, sizeVol,
                     (float *)d_particle.GetDevicePtr(),
                     (float *)d_wedge.GetDevicePtr());

  if (useFilterVolume) {
    d_particle.CopyHostToDevice(filter);
    hipLaunchKernelGGL(fftshiftReal_, grid, block, 0, stream, sizeVol,
                       (float *)d_particle.GetDevicePtr(),
                       (float *)d_filter.GetDevicePtr());
  }

  d_particle.CopyHostToDeviceAsync(stream, _data);
  hipStreamSynchronize(stream);

#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
  hipfftExecR2C(ffthandle_R2C, (hipfftReal *)d_particle.GetDevicePtr(),
                (hipfftComplex *)d_particleCplx_orig_RC.GetDevicePtr());
#else
  cufftExecR2C(ffthandle_R2C, (cufftReal *)d_particle.GetDevicePtr(),
               (cufftComplex *)d_particleCplx_orig_RC.GetDevicePtr());
#endif

  hipLaunchKernelGGL(ApplyWedgeFilterBandpass_RC, grid_RC, block_RC, 0, stream, sizeVol,
                     (float *)d_wedge.GetDevicePtr(),
                     (float *)d_filter.GetDevicePtr(),
                     (float2 *)d_particleCplx_orig_RC.GetDevicePtr(), rDown, rUp,
                     smooth, useFilterVolume);

#if defined(__HIP_PLATFORM_HCC__)
  hipfftExecC2R(ffthandle_C2R,
                (hipfftComplex *)d_particleCplx_orig_RC.GetDevicePtr(),
                (hipfftReal *)d_particle.GetDevicePtr());
#else
  cufftExecC2R(ffthandle_C2R,
               (cufftComplex *)d_particleCplx_orig_RC.GetDevicePtr(),
               (cufftReal *)d_particle.GetDevicePtr());
#endif


  hipLaunchKernelGGL(mul_Real_, grid, block, 0, stream, sizeVol,
                     1.0f / sizeTot, (float *)d_particle.GetDevicePtr());
  
  hipLaunchKernelGGL(fftshiftReal_, grid, block, 0, stream, sizeVol,
                           (float *)d_particle.GetDevicePtr(),
                           (float *)d_ffttemp.GetDevicePtr());

  aReduceKernel.maxindexcplx(d_ffttemp, d_buffer, d_index, sizeTot);
  float maxPCCValue;
	d_buffer.CopyDeviceToHost(&maxPCCValue, sizeof(float));
	hipStreamSynchronize(stream);

  aRotateKernelROT.SetTexture(d_reference_orig);
  aRotateKernelMASK.SetTexture(d_mask_orig);

  if (rotateMaskCC) {
    aRotateKernelMASKCC.SetTexture(d_ccMask_Orig);
    aRotateKernelMASKCC.SetOldAngles(oldphi, oldpsi, oldtheta);
  }

  aRotateKernelROT.SetOldAngles(oldphi, oldpsi, oldtheta);
  aRotateKernelMASK.SetOldAngles(oldphi, oldpsi, oldtheta);

  float rthe = 0;
  float rpsi = 0;
  float rphi = 0;
  float maxthe = 0;
  float maxpsi = 0;
  float maxphi = 0;
  float npsi, dpsi;

  for (int iterPhi = 0; iterPhi < 2 * phi_angiter + 1; ++iterPhi) {
    rphi = phi_angincr * (iterPhi - phi_angiter);
    for (int iterThe = (int)0; iterThe < angiter + 1; ++iterThe) {
      if (iterThe == 0) {
        npsi = 1;
        dpsi = 360;
      } else {
        dpsi = angincr / sinf(iterThe * angincr * (float)M_PI / 180.0f);
        npsi = ceilf(360.0f / dpsi);
      }
      rthe = iterThe * angincr;
      for (int iterPsi = 0; iterPsi < npsi; ++iterPsi) {
        rpsi = iterPsi * dpsi;

        if (couplePhiToPsi) {
          rphi = phi_angincr * (iterPhi - phi_angiter) - rpsi;
        } else {
          rphi = phi_angincr * (iterPhi - phi_angiter);
        }

        d_particleCplx_RC.CopyDeviceToDeviceAsync(stream,
                                                  d_particleCplx_orig_RC);
        
        aRotateKernelROT.do_rotate(sizeVol, d_reference, rphi, rpsi, rthe);
        aRotateKernelMASK.do_rotate(sizeVol, d_mask, rphi, rpsi, rthe);

        if (rotateMaskCC) {
          d_ccMask.Memset(0);
          aRotateKernelMASKCC.do_rotate(sizeVol, d_ccMask, rphi, rpsi, rthe);
        }

        if (binarizeMask) {
          hipLaunchKernelGGL(binarize_, grid, block, 0, stream, sizeVol,
                             (float *)d_mask.GetDevicePtr(),
                             (float *)d_mask.GetDevicePtr());
        }

        hipLaunchKernelGGL(mulVol_RR_, grid, block, 0, stream, sizeVol,
                             (float *)d_mask.GetDevicePtr(),
                             (float *)d_reference.GetDevicePtr());

        hipLaunchKernelGGL(mulVol_RR_, grid, block, 0, stream, sizeVol,
                             (float *)d_mask.GetDevicePtr(),
                             (float *)d_particle.GetDevicePtr()); 

#if defined(__HIP_PLATFORM_HCC__)
        hipfftExecR2C(ffthandle_R2C, (hipfftReal *)d_reference.GetDevicePtr(),
                    (hipfftComplex *)d_referenceCplx_RC.GetDevicePtr());
        hipfftExecR2C(ffthandle_R2C, (hipfftReal *)d_particle.GetDevicePtr(),
                    (hipfftComplex *)d_particleCplx_orig_RC.GetDevicePtr());
#else
        cufftExecR2C(ffthandle_R2C, (cufftReal *)d_reference.GetDevicePtr(),
                    (cufftComplex *)d_referenceCplx_RC.GetDevicePtr());
                     
        cufftExecR2C(ffthandle_R2C, (cufftReal *)d_particle.GetDevicePtr(),
                    (cufftComplex *)d_particleCplx_orig_RC.GetDevicePtr());
#endif

        hipLaunchKernelGGL(phaseCorrel_RC, grid, block, 0, stream, sizeVol,
                    (float2 *)d_particleCplx_orig_RC.GetDevicePtr(),
                    (float2 *)d_referenceCplx_RC.GetDevicePtr()); 

        hipLaunchKernelGGL(ApplyWedgeFilterBandpass_RC, grid_RC, block_RC, 0, stream, sizeVol,
                          (float *)d_wedge.GetDevicePtr(),
                          (float *)d_filter.GetDevicePtr(),
                          (float2 *)d_referenceCplx_RC.GetDevicePtr(), rDown, rUp,
                          smooth, useFilterVolume);

#if defined(__HIP_PLATFORM_HCC__)
        hipfftExecC2R(ffthandle_C2R,
                      (hipfftComplex *)d_referenceCplx_RC.GetDevicePtr(),
                      (hipfftReal *)d_reference.GetDevicePtr());
#else
        cufftExecC2R(ffthandle_C2R,
                     (cufftComplex *)d_referenceCplx_RC.GetDevicePtr(),
                     (cufftReal *)d_reference.GetDevicePtr());
#endif

        hipLaunchKernelGGL(mul_Real_, grid, block, 0, stream, sizeVol,
                           1.0f / sizeTot / maxPCCValue,
                           (float *)d_reference.GetDevicePtr());

        hipLaunchKernelGGL(fftshiftReal_, grid, block, 0, stream, sizeVol,
                           (float *)d_reference.GetDevicePtr(),
                           (float *)d_ffttemp.GetDevicePtr());
        
        hipLaunchKernelGGL(mulVol_RR_, grid, block, 0, stream, sizeVol,
                             (float *)d_ccMask.GetDevicePtr(),
                             (float *)d_ffttemp.GetDevicePtr()); 

        if (computeCCValOnly) {
          // only read out the CC value at the old shift position and store it
          // in d_buffer
          d_index.CopyHostToDevice(&oldIndex);
#if defined(__HIP_PLATFORM_HCC__) && !defined(__HIP_PLATFORM_NVCC__)
          hipSafeCall(hipMemcpyDtoD(
              d_buffer.GetDevicePtr(),
              (float2 *)d_ffttemp.GetDevicePtr() + oldIndex, sizeof(float)));
#else
          hipSafeCall(hipMemcpyDtoD(d_buffer.GetDevicePtr(),
                                    d_ffttemp.GetDevicePtr() + oldIndex,
                                    sizeof(float)));
#endif
        } else {
          // find new Maximum value and store position and value
          aReduceKernel.maxindexcplx(d_ffttemp, d_buffer, d_index, sizeTot);
        }

        // aMaxKernel(maxVals, d_index, d_buffer, rphi, rpsi, rthe);
        hipLaunchKernelGGL(findmax_, grid_RC, block_RC, 0, stream,
                           (float *)maxVals.GetDevicePtr(),
                           (float *)d_index.GetDevicePtr(),
                           (float *)d_buffer.GetDevicePtr(), rphi, rpsi, rthe);
      }
    }
  }

  hipSafeCall(hipStreamSynchronize(stream));
  maxVals.CopyDeviceToHost(&m);
  hipSafeCall(hipStreamSynchronize(stream));

  return m;
}

