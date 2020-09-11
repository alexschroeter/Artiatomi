#ifndef HIPREDUCER_H
#define HIPREDUCER_H

#include "../default.h"
#include <hip/hip_runtime.h>
#include "HipVariables.h"
#include "HipKernel.h"
#include "HipContext.h"

using namespace Hip;

class HipReducer
{
private:
	HipKernel* sum512;
	HipKernel* sum256;
	HipKernel* sum128;
	HipKernel* sum64;
	HipKernel* sum32;
	HipKernel* sum16;
	HipKernel* sum8;
	HipKernel* sum4;
	HipKernel* sum2;
	HipKernel* sum1;

	HipKernel* sumCplx512;
	HipKernel* sumCplx256;
	HipKernel* sumCplx128;
	HipKernel* sumCplx64;
	HipKernel* sumCplx32;
	HipKernel* sumCplx16;
	HipKernel* sumCplx8;
	HipKernel* sumCplx4;
	HipKernel* sumCplx2;
	HipKernel* sumCplx1;

	HipKernel* sumSqrCplx512;
	HipKernel* sumSqrCplx256;
	HipKernel* sumSqrCplx128;
	HipKernel* sumSqrCplx64;
	HipKernel* sumSqrCplx32;
	HipKernel* sumSqrCplx16;
	HipKernel* sumSqrCplx8;
	HipKernel* sumSqrCplx4;
	HipKernel* sumSqrCplx2;
	HipKernel* sumSqrCplx1;

	HipKernel* maxIndex512;
	HipKernel* maxIndex256;
	HipKernel* maxIndex128;
	HipKernel* maxIndex64;
	HipKernel* maxIndex32;
	HipKernel* maxIndex16;
	HipKernel* maxIndex8;
	HipKernel* maxIndex4;
	HipKernel* maxIndex2;
	HipKernel* maxIndex1;

	HipKernel* maxIndexCplx512;
	HipKernel* maxIndexCplx256;
	HipKernel* maxIndexCplx128;
	HipKernel* maxIndexCplx64;
	HipKernel* maxIndexCplx32;
	HipKernel* maxIndexCplx16;
	HipKernel* maxIndexCplx8;
	HipKernel* maxIndexCplx4;
	HipKernel* maxIndexCplx2;
	HipKernel* maxIndexCplx1;

	HipKernel* sumSqr512;
	HipKernel* sumSqr256;
	HipKernel* sumSqr128;
	HipKernel* sumSqr64;
	HipKernel* sumSqr32;
	HipKernel* sumSqr16;
	HipKernel* sumSqr8;
	HipKernel* sumSqr4;
	HipKernel* sumSqr2;
	HipKernel* sumSqr1;

	Hip::HipContext* ctx;
	int voxelCount;

	static const int maxBlocks = 64;
	static const int maxThreads = 256;

	hipStream_t stream;

	void getNumBlocksAndThreads(int n, int &blocks, int &threads);
	uint nextPow2(uint x);

	void runSumKernel(int size, int blocks, int threads, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runMaxIndexKernel(int size, int blocks, int threads, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& d_index, bool readIndex);
	void runSumCplxKernel(int size, int blocks, int threads, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runSumSqrCplxKernel(int size, int blocks, int threads, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runSumSqrKernel(int size, int blocks, int threads, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void runMaxIndexCplxKernel(int size, int blocks, int threads, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& d_index, bool readIndex);

public:

	HipReducer(int aVoxelCount, hipStream_t aStream, Hip::HipContext* context);

	int GetOutBufferSize();

	void Sum(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void MaxIndex(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& d_index);
	void SumCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void SumSqrCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void SumSqr(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata);
	void MaxIndexCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& d_index);
};

#endif 
