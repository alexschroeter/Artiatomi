#include "NppEmulator.h"

#include <iostream>
#include "default.h"
#include "hip/HipException.h"

using namespace Hip;
using namespace std;

NppEmulator::NppEmulator(hipModule_t aModule)
{
  cout<<"Initializing NPP emulator.."<<endl;
  mModule = aModule;
  mBlockDim = make_dim3(32, 8, 1);
  mGridDim = make_dim3(1, 1, 1);  

  hipSafeCall( hipModuleGetFunction(&mFunctionNppiAdd_32f_C1IR, mModule, "MYnppiAdd_32f_C1IR") );

  hipSafeCall( hipModuleGetFunction(&mFunctionNppiSubC_32f_C1R, mModule, "MYnppiSubC_32f_C1R") );

  hipSafeCall( hipModuleGetFunction(&mFunctionNppiDivC_32f_C1R,  mModule, "MYnppiDivC_32f_C1R") );
  hipSafeCall( hipModuleGetFunction(&mFunctionNppiDivC_32f_C1IR, mModule, "MYnppiDivC_32f_C1IR") );

  hipSafeCall( hipModuleGetFunction(&mFunctionNppiMean_32f_C1R_F1, mModule, "MYnppiMean_32f_C1R_F1") );
  hipSafeCall( hipModuleGetFunction(&mFunctionNppiMean_32f_C1R_F2, mModule, "MYnppiMean_32f_C1R_F2") );

  hipSafeCall( hipModuleGetFunction(&mFunctionNppiSum_32f_C1R_F1, mModule, "MYnppiSum_32f_C1R_F1") );
  hipSafeCall( hipModuleGetFunction(&mFunctionNppiSum_32f_C1R_F2, mModule, "MYnppiSum_32f_C1R_F2") );

  hipSafeCall( hipModuleGetFunction(&mFunctionNppiSum_8u_C1R_F1, mModule, "MYnppiSum_8u_C1R_F1") );

  hipSafeCall( hipModuleGetFunction(&mFunctionNppiMean_StdDev_32f_C1R_F1, mModule, "MYnppiMean_StdDev_32f_C1R_F1") );
  hipSafeCall( hipModuleGetFunction(&mFunctionNppiMean_StdDev_32f_C1R_F2, mModule, "MYnppiMean_StdDev_32f_C1R_F2") );

  hipSafeCall( hipModuleGetFunction(&mFunctionNppiMaxIndx_32f_C1R_F1, mModule, "MYnppiMaxIndx_32f_C1R_F1") );
  hipSafeCall( hipModuleGetFunction(&mFunctionNppiMaxIndx_32f_C1R_F2, mModule, "MYnppiMaxIndx_32f_C1R_F2") );

  hipSafeCall( hipModuleGetFunction(&mFunctionNppiConvert_16s32f_C1R, mModule, "MYnppiConvert_16s32f_C1R") );
  hipSafeCall( hipModuleGetFunction(&mFunctionNppiConvert_16u32f_C1R, mModule, "MYnppiConvert_16u32f_C1R") );
  hipSafeCall( hipModuleGetFunction(&mFunctionNppiConvert_32s32f_C1R, mModule, "MYnppiConvert_32s32f_C1R") );
  hipSafeCall( hipModuleGetFunction(&mFunctionNppiConvert_32u32f_C1R, mModule, "MYnppiConvert_32u32f_C1R") );

  hipSafeCall( hipModuleGetFunction(&mFunctionNppiSet_8u_C1R, mModule, "MYnppiSet_8u_C1R") );
  hipSafeCall( hipModuleGetFunction(&mFunctionNppiCompareC_32f_C1R, mModule, "MYnppiCompareC_32f_C1R") );
  hipSafeCall( hipModuleGetFunction(&mFunctionNppiSet_32f_C1MR, mModule, "MYnppiSet_32f_C1MR") );

  hipSafeCall( hipModuleGetFunction(&mFunctionNppiSqr_32f_C1R, mModule, "MYnppiSqr_32f_C1R") );

  hipSafeCall( hipModuleGetFunction(&mFunctionNppiCopy_32f_C1R, mModule, "MYnppiCopy_32f_C1R") );
  cout<<"Initializing NPP emulator done."<<endl;
}

NppEmulator::~NppEmulator()
{
}

void NppEmulator::Launch( hipFunction_t function )
{
  
  hipSafeCall(hipDeviceSynchronize());

  hipEvent_t eventStart;
  hipEvent_t eventEnd;
  hipStream_t stream = 0;
  hipSafeCall(hipEventCreateWithFlags(&eventStart, hipEventBlockingSync));
  hipSafeCall(hipEventCreateWithFlags(&eventEnd, hipEventBlockingSync));

  hipSafeCall(hipEventRecord(eventStart, stream));

  // kernel launch with parameters currently is not implemented for AMD - do low-level launch with the input buffer

  //mArglist[0] = &mArgs;
  //hipSafeCall(hipModuleLaunchKernel( function, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, 0, NULL, mArglist, NULL));

  size_t size = sizeof(mArgs);
  void *config[] = {
    HIP_LAUNCH_PARAM_BUFFER_POINTER, (void*) &mArgs,
    HIP_LAUNCH_PARAM_BUFFER_SIZE, &size,
    HIP_LAUNCH_PARAM_END
  };
 
  hipSafeCall(hipModuleLaunchKernel( function, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, 0, 0, NULL, (void**)&config));
  hipSafeCall(hipDeviceSynchronize());

  hipSafeCall(hipCtxSynchronize());
  hipSafeCall(hipEventRecord(eventEnd, stream));
  hipSafeCall(hipEventSynchronize(eventEnd));
  
  float ms;
  hipSafeCall(hipEventElapsedTime(&ms, eventStart, eventEnd));
		
  hipSafeCall(hipEventDestroy(eventStart));
  hipSafeCall(hipEventDestroy(eventEnd));
}


void NppEmulator::nppiAdd_32f_C1IR( Npp32f *pSrc, int nSrcStep, 
				    Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI )
{
  SetWorkSize( oSizeROI );

  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &pSrcDst;
  mArglist[3] = &nSrcDstStep;
  mArglist[4] = &oSizeROI;

  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pSrcDst;
  mArgs.int1 = nSrcStep;
  mArgs.int2 = nSrcDstStep;
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiAdd_32f_C1IR );
}


void NppEmulator::nppiSubC_32f_C1R( Npp32f * pSrc, int nSrcStep,  Npp32f nConstant, 
				    Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI             )
{
  SetWorkSize( oSizeROI );

  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &nConstant;
  mArglist[3] = &pDst;
  mArglist[4] = &nDstStep;
  mArglist[5] = &oSizeROI;

  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDst;
  mArgs.int1 = nSrcStep;
  mArgs.int2 = nDstStep;
  mArgs.float1 = nConstant;
  mArgs.size = oSizeROI;
 
  Launch( mFunctionNppiSubC_32f_C1R ); 
}


void NppEmulator::nppiDivC_32f_C1R( Npp32f *pSrc, int nSrcStep,  Npp32f nConstant,
				    Npp32f *pDst,  int nDstStep,  NppiSize oSizeROI               )
{
  SetWorkSize( oSizeROI );
  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &nConstant;
  mArglist[3] = &pDst;
  mArglist[4] = &nDstStep;
  mArglist[5] = &oSizeROI;
  
  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDst;
  mArgs.int1 = nSrcStep;
  mArgs.int2 = nDstStep;
  mArgs.float1 = nConstant;
  mArgs.size = oSizeROI;
 
  Launch( mFunctionNppiDivC_32f_C1R );
}


void NppEmulator::nppiDivC_32f_C1IR( Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI )
{
  SetWorkSize( oSizeROI );

  mArglist[0] = &nConstant;
  mArglist[1] = &pSrcDst;
  mArglist[2] = &nSrcDstStep;
  mArglist[3] = &oSizeROI;

  mArgs.ptr1 = (hipDeviceptr_t) pSrcDst;
  mArgs.int1 = nSrcDstStep;  
  mArgs.float1 = nConstant;
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiDivC_32f_C1IR ); 
}

void NppEmulator::nppiMeanGetBufferHostSize_32f_C1R( NppiSize oSizeROI,  int *hpBufferSize )
{
  // Buffer size for nppiMean_32f_C1R.
  SetWorkSize( oSizeROI );
  int nBlocks =  mGridDim.x*mGridDim.y;
  *hpBufferSize = nBlocks*2*sizeof(Npp32f);
}

void NppEmulator::nppiMean_32f_C1R( Npp32f *pSrc, int nSrcStep,  NppiSize oSizeROI,
				    Npp8u *pDeviceBuffer, Npp64f *pMean                    )
{
  // One-channel 32-bit floating point image Mean.
  
  // pSrc	Source-Image Pointer.
  // nSrcStep	Source-Image Line Step.
  // oSizeROI	Region-of-Interest (ROI).
  // pMask	Mask-Image Pointer.
  // nMaskStep	Mask-Image Line Step.
  // nCOI	Channel_of_Interest Number.
  // pDeviceBuffer  Pointer to the required device memory allocation, Scratch Buffer and Host Pointer
  //                Use nppiMeanGetBufferHostSize_XX_XXX to determine the minium number of bytes required.
  // pMean	Pointer to the computed mean result.
  
  // C1 means 1-color channel
  // oSizeROI in pixels
  // nSrcStep, nDstStep are in bytes

  SetWorkSize( oSizeROI );
  int nBlocksInBuffer =   mGridDim.x*mGridDim.y;

  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &oSizeROI;
  mArglist[3] = &pDeviceBuffer;
  
  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDeviceBuffer;
  mArgs.int1 = nSrcStep;    
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiMean_32f_C1R_F1 ); 

  SetWorkSize( 1, 1 );  

  mArglist[0] = &pDeviceBuffer;
  mArglist[1] = &nBlocksInBuffer;
  mArglist[2] = &pMean;  
 
  mArgs.ptr1 = (hipDeviceptr_t) pDeviceBuffer;
  mArgs.ptr2 = (hipDeviceptr_t) pMean;
  mArgs.int1 = nBlocksInBuffer;    

  Launch( mFunctionNppiMean_32f_C1R_F2 ); 
}


void NppEmulator::nppiMeanStdDevGetBufferHostSize_32f_C1R( NppiSize oSizeROI,  int *hpBufferSize )
{
  // Buffer size for nppiMean_StdDev_32f_C1R 
  SetWorkSize( oSizeROI );
  int nBlocks =  mGridDim.x*mGridDim.y;
  *hpBufferSize = nBlocks*2*sizeof(Npp32f);
}

void NppEmulator::nppiMean_StdDev_32f_C1R( Npp32f *pSrc,  int nSrcStep,  NppiSize  oSizeROI,
					   Npp8u *pDeviceBuffer,  Npp64f *pMean,  Npp64f *pStdDev   )
{
  // get mean + stdDev
  // pSrc	Source-Image Pointer.
  // nSrcStep	Source-Image Line Step.
  // oSizeROI	Region-of-Interest (ROI).
  // pMask	Mask-Image Pointer.
  // nMaskStep	Mask-Image Line Step.
  // nCOI	Channel_of_Interest Number.
  // pDeviceBuffer  Pointer to the required device memory allocation, Scratch Buffer and Host Pointer
  //                Use nppiMeanGetBufferHostSize_XX_XXX to determine the minium number of bytes required.
  // pMean	Pointer to the computed mean result.

  nppiMean_32f_C1R( pSrc, nSrcStep, oSizeROI, pDeviceBuffer, pMean );

  SetWorkSize( oSizeROI );
  int nBlocksInBuffer =   mGridDim.x*mGridDim.y;

  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &oSizeROI;
  mArglist[3] = &pDeviceBuffer;
  mArglist[4] = &pMean;

  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDeviceBuffer;
  mArgs.ptr3 = (hipDeviceptr_t) pMean;
  mArgs.int1 = nSrcStep;    
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiMean_StdDev_32f_C1R_F1 ); 

  SetWorkSize( 1, 1 ); 

  mArglist[0] = &pDeviceBuffer;
  mArglist[1] = &nBlocksInBuffer;
  mArglist[2] = &pStdDev;     

  mArgs.ptr1 = (hipDeviceptr_t) pDeviceBuffer;
  mArgs.ptr2 = (hipDeviceptr_t) pStdDev;
  mArgs.int1 = nBlocksInBuffer;    

  Launch( mFunctionNppiMean_StdDev_32f_C1R_F2); 
}


void NppEmulator::nppiMaxIndxGetBufferHostSize_32f_C1R( NppiSize oSizeROI,  int *hpBufferSize )
{
  // Buffer size for nppiMean_32f_C1R.
  SetWorkSize( oSizeROI );
  int nBlocks =  mGridDim.x*mGridDim.y;
  *hpBufferSize = nBlocks*3*sizeof(float);
}

void NppEmulator::nppiMaxIndx_32f_C1R(  Npp32f *pSrc, int nSrcStep, NppiSize oSizeROI,
					Npp8u *pDeviceBuffer,  Npp32f *pMax, int *pIndexX, int *pIndexY )
{
  /* One-channel 32-bit floating point image MaxIndx.

    pSrc	Source-Image Pointer.
    nSrcStep	Source-Image Line Step.
    oSizeROI	Region-of-Interest (ROI).
    pDeviceBuffer	Pointer to the required device memory allocation, Scratch Buffer and Host Pointer Use nppiMaxIndxGetBufferHostSize_XX_XXX to determine the minium number of bytes required.
    pMax	Pointer to the computed max result.
    pIndexX	Pointer to the X coordinate of the image max value.
    pIndexY	Ppointer to the Y coordinate of the image max value.
  */
  
  SetWorkSize( oSizeROI );
  int nBlocksInBuffer = mGridDim.x*mGridDim.y;

  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &oSizeROI;
  mArglist[3] = &pDeviceBuffer;
  
  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDeviceBuffer;
  mArgs.int1 = nSrcStep;    
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiMaxIndx_32f_C1R_F1 ); 

  SetWorkSize(1,1);

  mArglist[0] = &pDeviceBuffer;
  mArglist[1] = &nBlocksInBuffer;
  mArglist[2] = &pMax;
  mArglist[3] = &pIndexX;
  mArglist[4] = &pIndexY;

  mArgs.ptr1 = (hipDeviceptr_t) pDeviceBuffer;
  mArgs.ptr2 = (hipDeviceptr_t) pMax;
  mArgs.ptr3 = (hipDeviceptr_t) pIndexX;
  mArgs.ptr4 = (hipDeviceptr_t) pIndexY;
  mArgs.int1 = nBlocksInBuffer;    

  Launch( mFunctionNppiMaxIndx_32f_C1R_F2 ); 
}	


void NppEmulator::nppiSumGetBufferHostSize_32f_C1R( NppiSize oSizeROI,  int *hpBufferSize )
{
  // Buffer size for nppiSum_32f_C1R.
  SetWorkSize( oSizeROI );
  int nBlocks =  mGridDim.x*mGridDim.y;
  *hpBufferSize = nBlocks*sizeof(Npp32f);
}

void NppEmulator::nppiSum_32f_C1R(  Npp32f *pSrc, int nSrcStep,  NppiSize oSizeROI,
				    Npp8u *pDeviceBuffer, Npp64f *pSum                    )
{
  // One-channel 32-bit floating point image sum.
  // pSrc	Source-Image Pointer.
  // nSrcStep	Source-Image Line Step.
  // oSizeROI	Region-of-Interest (ROI).
  // pDeviceBuffer  Pointer to the required device memory allocation. Use nppiSumGetBufferHostSize_XX_XXX to determine the minium number of bytes required.
  // pSum	Pointer to the computed sum.

  SetWorkSize( oSizeROI );
  int nBlocksInBuffer =   mGridDim.x*mGridDim.y;

  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &oSizeROI;
  mArglist[3] = &pDeviceBuffer;

  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDeviceBuffer;
  mArgs.int1 = nSrcStep;    
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiSum_32f_C1R_F1 ); 

  SetWorkSize( 1, 1 );  

  mArglist[0] = &pDeviceBuffer;
  mArglist[1] = &nBlocksInBuffer;
  mArglist[2] = &pSum;  

  mArgs.ptr1 = (hipDeviceptr_t) pDeviceBuffer;
  mArgs.ptr2 = (hipDeviceptr_t) pSum;
  mArgs.int1 = nBlocksInBuffer;    

  Launch( mFunctionNppiSum_32f_C1R_F2 ); 
}


void NppEmulator::nppiSum_8u_C1R( Npp8u *pSrc, int nSrcStep,  NppiSize oSizeROI,
				  Npp8u *pDeviceBuffer, Npp64f *pSum                    )
{
  // One-channel 8-bit unsigned image sum.
  // pSrc	Source-Image Pointer.
  // nSrcStep	Source-Image Line Step.
  // oSizeROI	Region-of-Interest (ROI).
  // pDeviceBuffer  Pointer to the required device memory allocation. Use nppiSumGetBufferHostSize_XX_XXX to determine the minium number of bytes required.
  // pSum	Pointer to the computed sum.

  SetWorkSize( oSizeROI );
  int nBlocksInBuffer =   mGridDim.x*mGridDim.y;

  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &oSizeROI;
  mArglist[3] = &pDeviceBuffer;

  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDeviceBuffer;
  mArgs.int1 = nSrcStep;    
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiSum_8u_C1R_F1 ); 

  SetWorkSize( 1, 1 );  

  mArglist[0] = &pDeviceBuffer;
  mArglist[1] = &nBlocksInBuffer;
  mArglist[2] = &pSum;
 
  mArgs.ptr1 = (hipDeviceptr_t) pDeviceBuffer;
  mArgs.ptr2 = (hipDeviceptr_t) pSum;
  mArgs.int1 = nBlocksInBuffer;    

  Launch( mFunctionNppiSum_32f_C1R_F2 ); // works also for 8u
}








void NppEmulator::nppiConvert_16s32f_C1R( const Npp16s *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI )
{
  // Single channel 16-bit signed to 32-bit floating-point conversion.
  SetWorkSize( oSizeROI );
  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &pDst;
  mArglist[3] = &nDstStep;
  mArglist[4] = &oSizeROI;

  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDst;
  mArgs.int1 = nSrcStep;
  mArgs.int2 = nDstStep;
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiConvert_16s32f_C1R );
}

void NppEmulator::nppiConvert_16u32f_C1R( const Npp16u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI )
{
  // Single channel 16-bit unsigned to 32-bit floating-point conversion.
  SetWorkSize( oSizeROI );
  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &pDst;
  mArglist[3] = &nDstStep;
  mArglist[4] = &oSizeROI;

  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDst;
  mArgs.int1 = nSrcStep;
  mArgs.int2 = nDstStep;
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiConvert_16u32f_C1R );
}

void NppEmulator::nppiConvert_32s32f_C1R( const Npp32s *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI )
{
  // Single channel 32-bit signed to 32-bit floating-point conversion.
  SetWorkSize( oSizeROI );
  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &pDst;
  mArglist[3] = &nDstStep;
  mArglist[4] = &oSizeROI;

  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDst;
  mArgs.int1 = nSrcStep;
  mArgs.int2 = nDstStep;
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiConvert_32s32f_C1R );
}

void NppEmulator::nppiConvert_32u32f_C1R( const Npp32u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI )
{
  // Single channel 32-bit signed to 32-bit floating-point conversion.
  SetWorkSize( oSizeROI );
  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &pDst;
  mArglist[3] = &nDstStep;
  mArglist[4] = &oSizeROI;

  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDst;
  mArgs.int1 = nSrcStep;
  mArgs.int2 = nDstStep;
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiConvert_32u32f_C1R );
}


void NppEmulator::nppiSet_8u_C1R( Npp8u nValue, Npp8u *pDst, int nDstStep, NppiSize oSizeROI )
{
  // 8-bit image set.
  SetWorkSize( oSizeROI );
  mArglist[0] = &nValue;
  mArglist[1] = &pDst;
  mArglist[2] = &nDstStep;
  mArglist[3] = &oSizeROI;

  mArgs.ptr1 = (hipDeviceptr_t) pDst;
  mArgs.int1 = nDstStep;
  mArgs.npp8u1 = nValue;
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiSet_8u_C1R );
}


void NppEmulator::nppiCompareC_32f_C1R(	const Npp32f *pSrc, int nSrcStep, Npp32f nConstant,
					Npp8u *pDst, int nDstStep, NppiSize oSizeROI,  NppCmpOp eComparisonOperation )
{
  // 1 channel 32-bit floating point image compare with constant value.
  // Compare pSrc's pixels with constant value.
  SetWorkSize( oSizeROI ); 
  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &nConstant;
  mArglist[3] = &pDst;
  mArglist[4] = &nDstStep;
  mArglist[5] = &oSizeROI;
  mArglist[6] = &eComparisonOperation;
 
  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDst;
  mArgs.int1 = nSrcStep;
  mArgs.int2 = nDstStep;
  mArgs.float1 = nConstant;
  mArgs.size = oSizeROI;
  mArgs.cmpOp = eComparisonOperation;

  Launch( mFunctionNppiCompareC_32f_C1R );
}


void NppEmulator::nppiSet_32f_C1MR( Npp32f nValue, Npp32f *pDst, int nDstStep,  NppiSize oSizeROI, const Npp8u *pMask, int nMaskStep )
{
  // Masked 32-bit floating point image set. 
  //
  // nValue	The pixel value to be set for single channel functions.
  // aValue	The pixel-value to be set for multi-channel functions.
  // pDst	Pointer Destination-Image Pointer.
  // nDstStep	Destination-Image Line Step.
  // oSizeROI	Region-of-Interest (ROI).
  // pMask	Mask-Image Pointer.
  // nMaskStep	Mask-Image Line Step.
  SetWorkSize( oSizeROI ); 

  mArglist[0] = &nValue;
  mArglist[1] = &pDst;
  mArglist[2] = &nDstStep;
  mArglist[3] = &oSizeROI;
  mArglist[4] = &pMask;
  mArglist[5] = &nMaskStep;

  mArgs.ptr1 = (hipDeviceptr_t) pDst;
  mArgs.ptr2 = (hipDeviceptr_t) pMask;
  mArgs.int1 = nDstStep;
  mArgs.int2 = nMaskStep;
  mArgs.float1 = nValue;
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiSet_32f_C1MR );
}	


void NppEmulator::nppiSqr_32f_C1R( const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,  int nDstStep, NppiSize oSizeROI )
{
  //One 32-bit floating point channel image squared.
  SetWorkSize( oSizeROI );

  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &pDst;
  mArglist[3] = &nDstStep;
  mArglist[4] = &oSizeROI;

  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDst;  
  mArgs.int1 = nSrcStep;
  mArgs.int2 = nDstStep;
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiSqr_32f_C1R );
}	


void NppEmulator::nppiCopy_32f_C1R( const Npp32f *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI )
{		
  // 32-bit floating point image copy.
  SetWorkSize( oSizeROI );

  mArglist[0] = &pSrc;
  mArglist[1] = &nSrcStep;
  mArglist[2] = &pDst;
  mArglist[3] = &nDstStep;
  mArglist[4] = &oSizeROI;

  mArgs.ptr1 = (hipDeviceptr_t) pSrc;
  mArgs.ptr2 = (hipDeviceptr_t) pDst;  
  mArgs.int1 = nSrcStep;
  mArgs.int2 = nDstStep;
  mArgs.size = oSizeROI;

  Launch( mFunctionNppiCopy_32f_C1R );
}
