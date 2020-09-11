#ifndef KERNELS_H
#define KERNELS_H

#include "../hip/HipKernel.h"
#include "../hip/HipVariables.h"

#include "../hip_kernels/DeviceReconstructionParameters.h"
#include "../hip/HipTextures.h"
#include "../hip/HipMissedStuff.h"


class SubCplxKernel : public Hip::HipKernel
{
public:
	SubCplxKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	SubCplxKernel(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& sum, HipDeviceVariable& divVal);
};


class SubCplxKernel_RC : public Hip::HipKernel
{
public:
	SubCplxKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	SubCplxKernel_RC(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& subval, HipDeviceVariable& divVal);
};



class FFTShiftRealKernel : public Hip::HipKernel
{
public:
	FFTShiftRealKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	FFTShiftRealKernel(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output);
};



class MakeCplxWithSubKernel : public Hip::HipKernel
{
public:
	MakeCplxWithSubKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	MakeCplxWithSubKernel(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, float val);
};


class MulVolKernel : public Hip::HipKernel
{
public:
	MulVolKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	MulVolKernel(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output);
};

class MulVolKernel_RR : public Hip::HipKernel
{
public:
	MulVolKernel_RR(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	MulVolKernel_RR(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output);
};

class MulVolKernel_RC : public Hip::HipKernel
{
public:
	MulVolKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	MulVolKernel_RC(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output);
};

class BandpassFFTShiftKernel : public Hip::HipKernel
{
public:
	BandpassFFTShiftKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	BandpassFFTShiftKernel(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, float rDown, float rUp, float smooth);
};

class BandpassFFTShiftKernel_RC : public Hip::HipKernel
{
public:
	BandpassFFTShiftKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	BandpassFFTShiftKernel_RC(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, float rDown, float rUp, float smooth);
};

class MakeRealKernel : public Hip::HipKernel
{
public:
	MakeRealKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	MakeRealKernel(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output);
};


class MulKernel : public Hip::HipKernel
{
public:
	MulKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	MulKernel(hipModule_t aModule);

	float operator()(int size, float val, HipDeviceVariable& output);
};

class MulKernel_RC : public Hip::HipKernel
{
public:
	MulKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	MulKernel_RC(hipModule_t aModule);

	float operator()(int size, float val, HipDeviceVariable& output);
};

class MulKernel_Real : public Hip::HipKernel
{
public:
	MulKernel_Real(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	MulKernel_Real(hipModule_t aModule);

	float operator()(int size, float val, HipDeviceVariable& output);
};

class MakeCplxWithSqrSubKernel : public Hip::HipKernel
{
public:
	MakeCplxWithSqrSubKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	MakeCplxWithSqrSubKernel(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, float val);
};


class CorrelKernel : public Hip::HipKernel
{
public:
	CorrelKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	CorrelKernel(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output);
};

class CorrelKernel_RC : public Hip::HipKernel
{
public:
	CorrelKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	CorrelKernel_RC(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output);
};

class ConvKernel : public Hip::HipKernel
{
public:
	ConvKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	ConvKernel(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output);
};

class ConvKernel_RC : public Hip::HipKernel
{
public:
	ConvKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	ConvKernel_RC(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output);
};


class SubKernel : public Hip::HipKernel
{
public:
	SubKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	SubKernel(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, float val);
};

class SubKernel_RC : public Hip::HipKernel
{
public:
	SubKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	SubKernel_RC(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& h_sum, float val);
};


class SqrSubKernel_RC : public Hip::HipKernel
{
public:
	SqrSubKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	SqrSubKernel_RC(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& h_sum, float val);
};


class EnergyNormKernel : public Hip::HipKernel
{
public:
	EnergyNormKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	EnergyNormKernel(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& particle, HipDeviceVariable& partSqr, HipDeviceVariable& cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox);
};


class EnergyNormKernel_RC : public Hip::HipKernel
{
public:
	EnergyNormKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	EnergyNormKernel_RC(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& particle, HipDeviceVariable& partSqr, HipDeviceVariable& cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox, HipDeviceVariable& temp, HipDeviceVariable& ccMask);
};


class FFTShift2Kernel : public Hip::HipKernel
{
public:
	FFTShift2Kernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	FFTShift2Kernel(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output);
};


class BinarizeKernel : public Hip::HipKernel
{
public:
	BinarizeKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	BinarizeKernel(hipModule_t aModule);

	float operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output);
};

class MaxKernel : public Hip::HipKernel
{
public:
	MaxKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	MaxKernel(hipModule_t aModule);

	float operator()(HipDeviceVariable& maxVal, HipDeviceVariable& index, HipDeviceVariable& val, float rphi, float rpsi, float rthe);
};


class Reducer : public Hip::HipKernel
{
private:
	hipFunction_t sum512;
	hipFunction_t sum256;
	hipFunction_t sum128;
	hipFunction_t sum64;
	hipFunction_t sum32;
	hipFunction_t sum16;
	hipFunction_t sum8;
	hipFunction_t sum4;
	hipFunction_t sum2;
	hipFunction_t sum1;

	hipFunction_t sumSqrCplx512;
	hipFunction_t sumSqrCplx256;
	hipFunction_t sumSqrCplx128;
	hipFunction_t sumSqrCplx64;
	hipFunction_t sumSqrCplx32;
	hipFunction_t sumSqrCplx16;
	hipFunction_t sumSqrCplx8;
	hipFunction_t sumSqrCplx4;
	hipFunction_t sumSqrCplx2;
	hipFunction_t sumSqrCplx1;

	hipFunction_t sumSqr512;
	hipFunction_t sumSqr256;
	hipFunction_t sumSqr128;
	hipFunction_t sumSqr64;
	hipFunction_t sumSqr32;
	hipFunction_t sumSqr16;
	hipFunction_t sumSqr8;
	hipFunction_t sumSqr4;
	hipFunction_t sumSqr2;
	hipFunction_t sumSqr1;

	hipFunction_t sumCplx512;
	hipFunction_t sumCplx256;
	hipFunction_t sumCplx128;
	hipFunction_t sumCplx64;
	hipFunction_t sumCplx32;
	hipFunction_t sumCplx16;
	hipFunction_t sumCplx8;
	hipFunction_t sumCplx4;
	hipFunction_t sumCplx2;
	hipFunction_t sumCplx1;

	hipFunction_t maxIndex512;
	hipFunction_t maxIndex256;
	hipFunction_t maxIndex128;
	hipFunction_t maxIndex64;
	hipFunction_t maxIndex32;
	hipFunction_t maxIndex16;
	hipFunction_t maxIndex8;
	hipFunction_t maxIndex4;
	hipFunction_t maxIndex2;
	hipFunction_t maxIndex1;

	hipFunction_t maxIndexCplx512;
	hipFunction_t maxIndexCplx256;
	hipFunction_t maxIndexCplx128;
	hipFunction_t maxIndexCplx64;
	hipFunction_t maxIndexCplx32;
	hipFunction_t maxIndexCplx16;
	hipFunction_t maxIndexCplx8;
	hipFunction_t maxIndexCplx4;
	hipFunction_t maxIndexCplx2;
	hipFunction_t maxIndexCplx1;

public:
	Reducer(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	Reducer(hipModule_t aModule);

	float sum(HipDeviceVariable& input, HipDeviceVariable& output, int sizeTot);
	void runSumKernel(int size, int blocks, int threads, HipDeviceVariable& input, HipDeviceVariable& output);

	float sumsqrcplx(HipDeviceVariable& input, HipDeviceVariable& output, int sizeTot);
	void runSumSqrCplxKernel(int size, int blocks, int threads, HipDeviceVariable& input, HipDeviceVariable& output);

	float sumsqr(HipDeviceVariable& input, HipDeviceVariable& output, int sizeTot);
	void runSumSqrKernel(int size, int blocks, int threads, HipDeviceVariable& input, HipDeviceVariable& output);

	float sumcplx(HipDeviceVariable& input, HipDeviceVariable& output, int sizeTot);
	void runSumCplxKernel(int size, int blocks, int threads, HipDeviceVariable& input, HipDeviceVariable& output);
	
	float maxindex(HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& d_index, int sizeTot);
	void runMaxIndexKernel(int size, int blocks, int threads, HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& index, bool readIndex);

	float maxindexcplx(HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& d_index, int sizeTot);
	void runMaxIndexCplxKernel(int size, int blocks, int threads, HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& index, bool readIndex);
};

void getNumBlocksAndThreads(int n, int &blocks, int &threads);
unsigned int nextPow2(unsigned int x);


class RotateKernel : public Hip::HipKernel
{

private:
	hipTextureFilterMode mlinearInterpolation;
	int volSize;
	float oldphi, oldpsi, oldtheta;

	HipArray3D shiftTex;
	HipArray3D dataTex;
	HipArray3D dataTexCplx;

	HipTextureObject3D shiftTexObj;
	HipTextureObject3D dataTexObj;
	HipTextureObject3D dataTexCplxObj;

	hipFunction_t rotVol;
	hipFunction_t shift;
	hipFunction_t rotVol_improved;
	hipFunction_t rotVolCplx;
	hipFunction_t shiftrot3d;


public:
	RotateKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim, int volSize, bool linearInterpolation);
	RotateKernel(hipModule_t aModule, int volSize);

	void computeRotMat(float phi, float psi, float theta, float rotMat[3][3]);
	void multiplyRotMatrix(float m1[3][3], float m2[3][3], float out[3][3]);

	void SetTextureShift(HipDeviceVariable& d_idata);
	void SetTexture(HipDeviceVariable& d_idata);
	void SetTextureCplx(HipDeviceVariable& d_idata);

	void SetOldAngles(float aPhi, float aPsi, float aTheta);

	float do_rotate(int size, HipDeviceVariable& input, float phi, float psi, float theta);
	float do_rotate_improved(int size, HipDeviceVariable& input, float phi, float psi, float theta);
	float do_rotateCplx(int size, HipDeviceVariable& d_odata, float phi, float psi, float theta);
	float do_shift(int size, HipDeviceVariable& d_odata, float3 shiftVal);
	float do_shiftrot3d(int size, HipDeviceVariable& d_odata, float phi, float psi, float theta, float3 shiftVal);


};


#endif