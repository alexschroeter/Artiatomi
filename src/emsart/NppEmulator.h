#ifndef NPPEMULATOR_H
#define NPPEMULATOR_H

#include <hip/hip_runtime.h>
#include "hip_kernels/NppEmulatorKernel.h"

class NppEmulator
{
 private:

  hipModule_t mModule;
  dim3 mBlockDim;
  dim3 mGridDim;

  void*mArglist[10];

  DevParamNppEmulator mArgs;

  hipFunction_t mFunctionNppiAdd_32f_C1IR;

  hipFunction_t mFunctionNppiSubC_32f_C1R;

  hipFunction_t mFunctionNppiDivC_32f_C1R;
  hipFunction_t mFunctionNppiDivC_32f_C1IR;

  hipFunction_t mFunctionNppiMean_32f_C1R_F1;
  hipFunction_t mFunctionNppiMean_32f_C1R_F2;

  hipFunction_t mFunctionNppiSum_32f_C1R_F1;
  hipFunction_t mFunctionNppiSum_32f_C1R_F2;
 
  hipFunction_t mFunctionNppiSum_8u_C1R_F1;

  hipFunction_t mFunctionNppiMean_StdDev_32f_C1R_F1;
  hipFunction_t mFunctionNppiMean_StdDev_32f_C1R_F2;

  hipFunction_t mFunctionNppiMaxIndx_32f_C1R_F1;
  hipFunction_t mFunctionNppiMaxIndx_32f_C1R_F2;

  hipFunction_t mFunctionNppiConvert_16s32f_C1R;
  hipFunction_t mFunctionNppiConvert_16u32f_C1R;
  hipFunction_t mFunctionNppiConvert_32s32f_C1R;
  hipFunction_t mFunctionNppiConvert_32u32f_C1R;

  hipFunction_t mFunctionNppiSet_8u_C1R;
  hipFunction_t mFunctionNppiCompareC_32f_C1R;
  hipFunction_t mFunctionNppiSet_32f_C1MR;
  hipFunction_t mFunctionNppiSqr_32f_C1R;
  hipFunction_t mFunctionNppiCopy_32f_C1R;

 public:	

  NppEmulator(hipModule_t aModule);
  ~NppEmulator();

  void nppiAdd_32f_C1IR( Npp32f *pSrc, int nSrcStep, 
			 Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI );

  void nppiSubC_32f_C1R( Npp32f * pSrc1, int nSrc1Step, const Npp32f nConstant, 
			 Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI             );

  void nppiDivC_32f_C1R( Npp32f *pSrc1, int nSrc1Step,  Npp32f nConstant,
			 Npp32f *pDst,  int nDstStep,  NppiSize oSizeROI               );
  void nppiDivC_32f_C1IR( Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI );


  void nppiMeanGetBufferHostSize_32f_C1R( NppiSize oSizeROI,  int *hpBufferSize );
  void nppiMean_32f_C1R( Npp32f *pSrc, int nSrcStep,  NppiSize oSizeROI,
			 Npp8u *pDeviceBuffer, Npp64f *pMean                    );
 
  void nppiSumGetBufferHostSize_32f_C1R( NppiSize oSizeROI,  int *hpBufferSize );
  void nppiSum_32f_C1R(  Npp32f *pSrc, int nSrcStep,  NppiSize oSizeROI,
			 Npp8u *pDeviceBuffer, Npp64f *pSum                    );
  
  void nppiSum_8u_C1R( Npp8u *pSrc, int nSrcStep,  NppiSize oSizeROI,
		       Npp8u *pDeviceBuffer, Npp64f *pSum                    );


  void nppiMeanStdDevGetBufferHostSize_32f_C1R( NppiSize oSizeROI,  int *hpBufferSize );
  void nppiMean_StdDev_32f_C1R( Npp32f *pSrc,  int nSrcStep,  NppiSize  oSizeROI,
				Npp8u *pDeviceBuffer,  Npp64f *pMean,  Npp64f *pStdDev   );

  void nppiMaxIndxGetBufferHostSize_32f_C1R( NppiSize oSizeROI,  int *hpBufferSize );
  void nppiMaxIndx_32f_C1R(  Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI,
			     Npp8u *pDeviceBuffer,  Npp32f *pMax, int *pIndexX, int *pIndexY );

  void nppiConvert_16s32f_C1R( const Npp16s *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI );
  void nppiConvert_16u32f_C1R( const Npp16u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI );
  void nppiConvert_32s32f_C1R( const Npp32s *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI );
  void nppiConvert_32u32f_C1R( const Npp32u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI );

  void nppiSet_8u_C1R( Npp8u nValue, Npp8u *pDst, int nDstStep, NppiSize oSizeROI );

  void nppiCompareC_32f_C1R( const Npp32f *pSrc, int nSrcStep, Npp32f nConstant,
			     Npp8u *pDst, int nDstStep, NppiSize oSizeROI,  NppCmpOp eComparisonOperation );

  void nppiSet_32f_C1MR( Npp32f nValue, Npp32f *pDst, int nDstStep,  NppiSize oSizeROI, const Npp8u *pMask, int nMaskStep );

  void nppiSqr_32f_C1R( const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,  int nDstStep, NppiSize oSizeROI );

  void nppiCopy_32f_C1R( const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,  int nDstStep, NppiSize oSizeROI );

 private:
 
  void SetWorkSize( size_t nx, size_t ny );
  void SetWorkSize( NppiSize oSizeROI );
  void Launch( hipFunction_t function );

};


inline void NppEmulator::SetWorkSize( size_t nx, size_t ny )
{
  mGridDim.x = (nx + mBlockDim.x - 1) / mBlockDim.x;
  mGridDim.y = (ny + mBlockDim.y - 1) / mBlockDim.y;
  mGridDim.z = 1;
}

inline void NppEmulator::SetWorkSize( NppiSize oSizeROI )
{
  SetWorkSize( oSizeROI.width, oSizeROI.height );
}

#endif
