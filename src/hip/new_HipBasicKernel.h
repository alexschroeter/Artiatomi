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

public:
	HipSub(int aVolSize, hipStream_t aStream, Hip::HipContext* context);

	void Add(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void Sub(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val);
	void SubCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, float divVal);
	void SubCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, HipDeviceVariable& divVal);
};


class HipMakeCplxWithSub
{
private:
	HipKernel* makeReal;
	HipKernel* makeCplxWithSub;
	HipKernel* makeCplxWithSubSqr;
	HipKernel* makeCplxZero;


	Hip::HipContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	hipStream_t stream;

	void runRealKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runCplxWithSubKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val);
	void runCplxWithSqrSubKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val);
	void runZeroKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);

public:
	HipMakeCplxWithSub(int aVolSize, hipStream_t aStream, Hip::HipContext* context);

	void MakeReal(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void MakeCplxZero(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
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
	HipKernel* mulVol_RR;
	HipKernel* mulVolCplx;
	HipKernel* mul;
	HipKernel* mul_RR;

	Hip::HipContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	hipStream_t stream;

	void runMulVolKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runMulVol_RRKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runMulVolCplxKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runMulKernel(float val, HipDeviceVariable& d_odata);
	void runMul_RRKernel(float val, HipDeviceVariable& d_odata);

public:
	HipMul(int aVolSize, hipStream_t aStream, Hip::HipContext* context);

	void MulVol(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void MulVol_RR(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void MulVolCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void Mul(float val, HipDeviceVariable& d_odata);
	void Mul_RR(float val, HipDeviceVariable& d_odata);
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

	Hip::HipContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	hipStream_t stream;

	void runConvKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runCorrelKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runBandpassKernel(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	void runBandpassFFTShiftKernel(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	void runFFTShiftKernel(HipDeviceVariable& d_vol);
	void runFFTShiftRealKernel(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut);
	void runFFTShiftKernel2(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut);
	void runEnergyNormKernel(HipDeviceVariable& d_particle, HipDeviceVariable& d_partSqr, HipDeviceVariable& d_cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox);

public:
	HipFFT(int aVolSize, hipStream_t aStream, Hip::HipContext* context);

	void Conv(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void Correl(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void Bandpass(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	void BandpassFFTShift(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	void FFTShift(HipDeviceVariable& d_vol);
	void FFTShiftReal(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut);
	void FFTShift2(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut);
	void EnergyNorm(HipDeviceVariable& d_particle, HipDeviceVariable& d_partSqr, HipDeviceVariable& d_cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox);

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


class HipSub_RC
{
private:

	HipKernel* sub_RC;
	HipKernel* sqrsub_RC;
	// HipKernel* subCplx;
	// HipKernel* subCplx2;
	// HipKernel* add;


	Hip::HipContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	hipStream_t stream;


	// void runAddKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runSub_RCKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val);
	void runSqrSub_RCKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val);
	// void runSubCplxKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, float divVal);
	// void runSubCplxKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, HipDeviceVariable& divVal);

public:
	HipSub_RC(int aVolSize, hipStream_t aStream, Hip::HipContext* context);

	
	//void Add(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void Sub_RC(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val);
	void SqrSub_RC(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val);
	//void SubCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, float divVal);
	//void SubCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, HipDeviceVariable& divVal);
};


class HipFFT_RC
{
private:
	//HipKernel* conv_RC;
	//HipKernel* correl_RC;
	/*
	HipKernel* bandpass;
	*/
	HipKernel* bandpassFFTShift_RC;
	/*
	HipKernel* fftshift;
	HipKernel* fftshiftReal;
	*/
	HipKernel* fftshift2_RC;
	//HipKernel* energynorm_RC;


	Hip::HipContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	hipStream_t stream;

	//void runConv_RCKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	//void runCorrel_RCKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	/*
	void runBandpassKernel(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	*/
	void runBandpassFFTShift_RCKernel(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	/*
	void runFFTShiftKernel(HipDeviceVariable& d_vol);
	void runFFTShiftRealKernel(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut);
	*/
	void runFFTShift_RCKernel2(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut);
	//void runEnergyNorm_RCKernel(HipDeviceVariable& d_particle, HipDeviceVariable& d_partSqr, HipDeviceVariable& d_cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox);


public:
	HipFFT_RC(int aVolSize, hipStream_t aStream, Hip::HipContext* context);

	//void Conv_RC(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	//void Correl_RC(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	/*
	void Bandpass(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	*/
	void BandpassFFTShift_RC(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth);
	/*
	void FFTShift(HipDeviceVariable& d_vol);
	void FFTShiftReal(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut);
	*/
	void FFTShift2_RC(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut);
	//void EnergyNorm_RC(HipDeviceVariable& d_particle, HipDeviceVariable& d_partSqr, HipDeviceVariable& d_cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox);
	
};


class HipMul_RC
{
private:
	
	HipKernel* mulVol_RC;
	//HipKernel* mulVolCplx;
	HipKernel* mul_RC;
	

	Hip::HipContext* ctx;
	int volSize;
	dim3 blockSize;
	dim3 gridSize;

	hipStream_t stream;


	void runMulVol_RCKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
//	void runMulVolCplxKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runMul_RCKernel(float val, HipDeviceVariable& d_odata);

public:
	HipMul_RC(int aVolSize, hipStream_t aStream, Hip::HipContext* context);


	void MulVol_RC(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
//	void MulVolCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void Mul_RC(float val, HipDeviceVariable& d_odata);
};

#endif //BASICKERNEL_H
