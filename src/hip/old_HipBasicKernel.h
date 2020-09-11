#ifndef HIPBASICKERNEL_H
#define HIPBASICKERNEL_H

#include "../default.h"
#include <hip/hip_runtime.h>
#include "HipVariables.h"
#include "HipKernel.h"
#include "HipContext.h"

class HipSub
{
private:
	HipKernel* sub;
	HipKernel* subCplx;
	HipKernel* subCplxMulVol;
	HipKernel* subCplx2;
	HipKernel* add;

	Hip::HipContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	hipStream_t stream;

	void runAddKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runSubKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val);
	void runSubCplxKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, float divVal);
	void runSubCplxKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, HipDeviceVariable& divVal);
	void runSubCplxMulVolKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, HipDeviceVariable& divVal);

public:
	HipSub(int aVolSize, hipStream_t aStream, Hip::HipContext* context);

	void Add(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void Sub(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val);
	void SubCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, float divVal);
	void SubCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, HipDeviceVariable& divVal);
	void SubCplxMulVol(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, HipDeviceVariable& divVal);
};

class HipMakeCplxWithSub
{
private:
	HipKernel* makeReal;
	HipKernel* makeCplxWithSub;
	HipKernel* makeCplxWithSubSqr;

	Hip::HipContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	hipStream_t stream;

	void runRealKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runCplxWithSubKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val);
	void runCplxWithSqrSubKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val);

public:
	HipMakeCplxWithSub(int aVolSize, hipStream_t aStream, Hip::HipContext* context);

	void MakeReal(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void MakeCplxWithSub(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val);
	void MakeCplxWithSqrSub(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val);
};

class HipBinarize
{
private:
	HipKernel* binarize;

	Hip::HipContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	hipStream_t stream;

	void runBinarizeKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);

public:
	HipBinarize(int aVolSize, hipStream_t aStream, Hip::HipContext* context);

	void Binarize(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
};

class HipWedgeNorm
{
private:
	HipKernel* wedge;

	Hip::HipContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	hipStream_t stream;

	void runWedgeNormKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_partdata, HipDeviceVariable& d_odata, int newMethod);

public:
	HipWedgeNorm(int aVolSize, hipStream_t aStream, Hip::HipContext* context);

	void WedgeNorm(HipDeviceVariable& d_idata, HipDeviceVariable& d_partdata, HipDeviceVariable& d_odata, int newMethod);
};



class HipMul
{
private:
	HipKernel* mulVol;
	HipKernel* mulmulVol;
	HipKernel* mulVolCplx;
	HipKernel* mul;

	Hip::HipContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	hipStream_t stream;

	void runMulVolKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runMulVolCplxKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runMulKernel(float val, HipDeviceVariable& d_odata);
	void runMulMulVolKernel(float val, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);

public:
	HipMul(int aVolSize, hipStream_t aStream, Hip::HipContext* context);

	void MulVol(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void MulMulVol(float val, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void MulVolCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void Mul(float val, HipDeviceVariable& d_odata);
};



class HipFFT
{
private:
	HipKernel* conv;
	HipKernel* correl;
	HipKernel* bandpass;
	HipKernel* bandpassFFTShift;
	HipKernel* fftshift;
	HipKernel* fftshiftReal;
	HipKernel* fftshift2;
	HipKernel* energynorm;
	HipKernel* correlConvConv;

	Hip::HipContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	hipStream_t stream;

	void runConvKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runCorrelKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runCorrelConvConvKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& d_idata2, HipDeviceVariable& d_odata2, float val);
	void runBandpassKernel(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	void runBandpassFFTShiftKernel(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	void runFFTShiftKernel(HipDeviceVariable& d_vol);
	void runFFTShiftRealKernel(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut);
	void runFFTShiftKernel2(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut);
	void runEnergyNormKernel(HipDeviceVariable& d_particle, HipDeviceVariable& d_partSqr, HipDeviceVariable& d_cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox, HipDeviceVariable& temp, HipDeviceVariable& ccMask);

public:
	HipFFT(int aVolSize, hipStream_t aStream, Hip::HipContext* context);

	void Conv(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void Correl(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void CorrelConvConv(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& d_idata2, HipDeviceVariable& d_odata2, float val);
	void Bandpass(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	void BandpassFFTShift(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	void FFTShift(HipDeviceVariable& d_vol);
	void FFTShiftReal(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut);
	void FFTShift2(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut);
	void EnergyNorm(HipDeviceVariable& d_particle, HipDeviceVariable& d_partSqr, HipDeviceVariable& d_cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox, HipDeviceVariable& temp, HipDeviceVariable& ccMask);

};

class HipMax
{
private:
	HipKernel* max;

	Hip::HipContext* ctx;
	dim3 blockSize;
	dim3 gridSize;

	hipStream_t stream;

	void runMaxKernel(HipDeviceVariable& maxVals, HipDeviceVariable& index, HipDeviceVariable& val, float rphi, float rpsi, float rthe);

public:
	HipMax(hipStream_t aStream, Hip::HipContext* context);

	void Max(HipDeviceVariable& maxVals, HipDeviceVariable& index, HipDeviceVariable& val, float rphi, float rpsi, float rthe);
};








#endif //BASICKERNEL_H
