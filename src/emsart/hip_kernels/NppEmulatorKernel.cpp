#ifndef NPPEMULATORKERNEL_CPP
#define NPPEMULATORKERNEL_CPP

#include "NppEmulatorKernel.h"

#include <hip/hip_runtime.h>
#include <float.h>


extern "C"
//__global__ void MYnppiAdd_32f_C1IR( const Npp32f *pSrc, int nSrcStep, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI )
__global__ void MYnppiAdd_32f_C1IR( DevParamNppEmulator rp )
{
  /*
    One 32-bit floating point channel in place image addition.
    Parameters
    pSrc	        Source-Image Pointer.
    nSrcStep	Source-Image Line Step.
    pSrcDst	In-Place Image Pointer.
    nSrcDstStep	In-Place-Image Line Step.
    oSizeROI	Region-of-Interest (ROI).
  */

  Npp32f* pSrc    = (Npp32f*) rp.ptr1;
  Npp32f* pSrcDst = (Npp32f*) rp.ptr2;
  int &nSrcStep    = rp.int1;
  int &nSrcDstStep = rp.int2;
  NppiSize &oSizeROI = rp.size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= oSizeROI.width || idy >= oSizeROI.height ) return;

  Npp32f *src     = (Npp32f *)((uint8_t*) pSrc     + idy * nSrcStep    );
  Npp32f *srcdst  = (Npp32f *)((uint8_t*) pSrcDst  + idy * nSrcDstStep );
  
  srcdst[idx] += src[idx];
}


extern "C"
//__global__ void MYnppiSubC_32f_C1R( const Npp32f * pSrc, int nSrcStep, const Npp32f nConstant, 
//				    Npp32f * pDst,  int nDstStep,  NppiSize oSizeROI             )
__global__ void MYnppiSubC_32f_C1R(  DevParamNppEmulator rp )
{
  /** 
   * One 32-bit floating point channel image subtract constant.
   * \param pSrc \ref source_image_pointer.
   * \param nSrcStep \ref source_image_line_step.  
   * \param nConstant Constant.
   * \param pDst \ref destination_image_pointer.
   * \param nDstStep \ref destination_image_line_step. 
   * \param oSizeROI \ref roi_specification.
   * \return \ref image_data_error_codes, \ref roi_error_codes
   */
  
  // C1 means 1-color channel
  // oSizeROI in pixels
  // nSrcStep, nDstStep are in bytes

  Npp32f* pSrc = (Npp32f*) rp.ptr1;
  Npp32f* pDst = (Npp32f*) rp.ptr2;
  int &nSrcStep = rp.int1;
  int &nDstStep = rp.int2;
  NppiSize &oSizeROI = rp.size;
  Npp32f &nConstant = rp.float1;

 
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= oSizeROI.width || idy >= oSizeROI.height ) return;
  
  Npp32f *src = (Npp32f *)((uint8_t*) pSrc + idy * nSrcStep);
  Npp32f *dst = (Npp32f *)((uint8_t*) pDst  + idy * nDstStep );
  
  dst[idx] = src[idx] - nConstant;
}


extern "C"
//__global__ void MYnppiDivC_32f_C1R( const Npp32f *pSrc, int nSrcStep,  const Npp32f nConstant,
//				    Npp32f *pDst,  int nDstStep,  NppiSize oSizeROI               )
__global__ void MYnppiDivC_32f_C1R( DevParamNppEmulator rp )
{
  /*
    One 32-bit floating point channel image divided by constant.
    
    Parameters
    pSrc	Source-Image Pointer.
    nSrcStep	Source-Image Line Step.
    nConstant	Constant.
    pDst	Destination-Image Pointer.
    nDstStep	Destination-Image Line Step.
    oSizeROI	Region-of-Interest (ROI).
  */

  // C1 means 1-color channel
  // oSizeROI in pixels
  // nSrcStep, nDstStep are in bytes
 
  Npp32f* pSrc = (Npp32f*) rp.ptr1;
  Npp32f* pDst = (Npp32f*) rp.ptr2;
  int &nSrcStep = rp.int1;
  int &nDstStep = rp.int2;
  NppiSize &oSizeROI = rp.size;
  Npp32f &nConstant = rp.float1;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx >= oSizeROI.width || idy >= oSizeROI.height ) return;
  
  Npp32f *src = (Npp32f *)((uint8_t*) pSrc + idy * nSrcStep);
  Npp32f *dst = (Npp32f *)((uint8_t*) pDst  + idy * nDstStep );
  
  dst[idx] = src[idx] / nConstant;
}




extern "C"
//__global__ void MYnppiDivC_32f_C1IR( const Npp32f nConstant, Npp32f *pSrcDst, int nSrcDstStep, NppiSize oSizeROI )		
__global__ void MYnppiDivC_32f_C1IR( DevParamNppEmulator rp )
{
  /*
    One 32-bit floating point channel in place image divided by constant.

    Parameters
    nConstant	Constant.
    pSrcDst	In-Place Image Pointer.
    nSrcDstStep	In-Place-Image Line Step.
    oSizeROI	Region-of-Interest (ROI).
  */
  Npp32f* pSrcDst = (Npp32f*) rp.ptr1;
  int &nSrcDstStep = rp.int1;
  Npp32f &nConstant = rp.float1;
  NppiSize &oSizeROI = rp.size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= oSizeROI.width || idy >= oSizeROI.height ) return;
  
  Npp32f *srcdst  = (Npp32f *)((uint8_t*) pSrcDst  + idy * nSrcDstStep );  
  srcdst[idx] /= nConstant;
}


extern "C"
//__global__ void MYnppiMean_32f_C1R_F1( const Npp32f *pSrc, int nSrcStep,  NppiSize oSizeROI, Npp8u *pDeviceBuffer )
__global__ void MYnppiMean_32f_C1R_F1( DevParamNppEmulator rp )
{
  // One-channel 32-bit floating point image Mean. First part.

  __shared__ float blockSum;
  __shared__ float blockEntries;

  Npp32f* pSrc = (Npp32f*) rp.ptr1;
  Npp8u* pDeviceBuffer = (Npp8u*) rp.ptr2;
  int &nSrcStep = rp.int1; 
  NppiSize &oSizeROI = rp.size;

  bool firstThread = ( threadIdx.x==0 && threadIdx.y==0 );
  
  if( firstThread ){
    blockSum = 0.;
    blockEntries = 0.;
  }

  //synchronize the local threads writing to the local memory cache
  __syncthreads();
 
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < oSizeROI.width && idy < oSizeROI.height ){  
    Npp32f *src = (Npp32f *)((uint8_t*) pSrc + idy * nSrcStep);  
    atomicAdd( &blockSum, src[idx] );
    atomicAdd( &blockEntries, 1.f );
  }

  //synchronize the local threads writing to the local memory cache
  __syncthreads();
 
  // write to the global memory  
  if( firstThread ){
    int block = blockIdx.y * gridDim.x + blockIdx.x;
    Npp32f *buf = (Npp32f *) pDeviceBuffer;
    buf[block*2+0] = blockSum;
    buf[block*2+1] = blockEntries;
  }
}  


extern "C"
//__global__ void MYnppiMean_32f_C1R_F2( Npp8u *pDeviceBuffer, int nBlocksInBuffer, Npp64f *pMean )
__global__ void MYnppiMean_32f_C1R_F2( DevParamNppEmulator rp )
{
  Npp8u* pDeviceBuffer = (Npp8u*) rp.ptr1;
  Npp64f* pMean = (Npp64f*) rp.ptr2;
  int &nBlocksInBuffer = rp.int1; 

  // One-channel 32-bit floating point image Mean. Second part.
  if( blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0 ){
    Npp32f *buf = (Npp32f *) pDeviceBuffer;
    double sum=0, entr=0;    
    for( int block=0; block<nBlocksInBuffer; block++){
      sum +=buf[block*2 + 0];
      entr+=buf[block*2 + 1];
    }
    double result = 0.;
    if( entr>0 ) result = sum/entr;
    *pMean = result;
  }
}



extern "C"
//__global__ void MYnppiMean_StdDev_32f_C1R_F1( const Npp32f *pSrc,  int nSrcStep,  NppiSize  oSizeROI, Npp8u *pDeviceBuffer,  Npp64f *pMean  )
__global__ void MYnppiMean_StdDev_32f_C1R_F1( DevParamNppEmulator rp )
{
  // One-channel 32-bit floating point image Mean_StdDev. First part.
  // mean is given

  __shared__ float blockSum;
  __shared__ float blockEntries;

  Npp32f* pSrc = (Npp32f*) rp.ptr1;
  Npp8u* pDeviceBuffer = (Npp8u*) rp.ptr2;
  Npp64f* pMean = (Npp64f*) rp.ptr3;
  int &nSrcStep = rp.int1; 
  NppiSize &oSizeROI = rp.size;
 

  bool firstThread = ( threadIdx.x==0 && threadIdx.y==0 );
  
  if( firstThread ){
    blockSum = 0.;
    blockEntries = 0.;
  }

  //synchronize the local threads writing to the local memory cache
  __syncthreads();
 
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < oSizeROI.width && idy < oSizeROI.height ){  
    Npp32f *src = (Npp32f *)((uint8_t*) pSrc + idy * nSrcStep);  
    float d = src[idx] - *pMean;
    atomicAdd( &blockSum, d*d );
    atomicAdd( &blockEntries, 1.f );
  }

  //synchronize the local threads writing to the local memory cache
  __syncthreads();

  // write to the global memory  
  if( firstThread ){
    int block = blockIdx.y * gridDim.x + blockIdx.x;
    Npp32f *buf = (Npp32f *) pDeviceBuffer;
    buf[block*2+0] = blockSum;
    buf[block*2+1] = blockEntries;
  } 
}


extern "C"
//__global__ void MYnppiMean_StdDev_32f_C1R_F2( Npp8u *pDeviceBuffer, int nBlocksInBuffer, Npp64f *pStdDev )
__global__ void MYnppiMean_StdDev_32f_C1R_F2( DevParamNppEmulator rp )
{
  // One-channel 32-bit floating point image Mean_StdDev. Second part.

  Npp8u* pDeviceBuffer = (Npp8u*) rp.ptr1;
  Npp64f* pStdDev = (Npp64f*) rp.ptr2;
  int &nBlocksInBuffer = rp.int1; 

  if( blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0 ){
    Npp32f *buf = (Npp32f *) pDeviceBuffer;
    double sum=0, entr=0;    
    for( int block=0; block<nBlocksInBuffer; block++){
      sum +=buf[block*2 + 0];
      entr+=buf[block*2 + 1];
    }
    double result = 0.;
    if( entr>0 ) result = sqrt(sum/entr);
    *pStdDev = result;
  }
}



extern "C"
//__global__ void MYnppiMaxIndx_32f_C1R_F1( const Npp32f *pSrc, int nSrcStep,  NppiSize oSizeROI, Npp8u *pDeviceBuffer )
__global__ void MYnppiMaxIndx_32f_C1R_F1( DevParamNppEmulator rp )
{
  // One-channel 32-bit floating point image MaxIndx. First part.
  
  __shared__ float blockMax;
  __shared__ int   blockThread;   


  Npp32f* pSrc = (Npp32f*) rp.ptr1;
  Npp8u* pDeviceBuffer = (Npp8u*) rp.ptr2;  
  int &nSrcStep = rp.int1; 
  NppiSize &oSizeROI = rp.size;
 
  // get thread index 
  int iThread = threadIdx.y*blockDim.x + threadIdx.x;
  
  if( iThread==0 ){
    blockMax = -FLT_MAX;
    blockThread = INT_MAX;
  }
  
  __syncthreads();
  
  // image x,y

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
   
  bool inRegion = (idx < oSizeROI.width && idy < oSizeROI.height );

  // get max value withing the thread block

  Npp32f val=0.f;
  int &ival = *(int*)&val; // float as integer variable
  int &iblockMax = *(int*) & blockMax;

  if( inRegion ){
    Npp32f *src = (Npp32f *)((uint8_t*) pSrc + idy * nSrcStep);  
    val = src[idx];    

    // make atomic max for float    

    float oldVal = blockMax;
    int &ioldVal = *(int*) &oldVal;
    int iExpectedOldVal = ioldVal - 1;
    while( (ioldVal!=iExpectedOldVal) && (oldVal<val) ){
      iExpectedOldVal = ioldVal;
      ioldVal = atomicCAS( &iblockMax, iExpectedOldVal, ival);
    }        
  }

  __syncthreads();
  
  // now search for a thread wich has the maximum value and minimal thread number
  
  if( val>=blockMax ) atomicMin( &blockThread, iThread );
  
  __syncthreads();
    
  // write values to the global memory  
  if( iThread == blockThread ){
    int block = blockIdx.y * gridDim.x + blockIdx.x;
    Npp32f *buf = (Npp32f *) pDeviceBuffer;   
    int *ibuf = (int *) pDeviceBuffer;   
    buf [block*3+0] = val;
    ibuf[block*3+1] = idx;
    ibuf[block*3+2] = idy;
  }
}	


extern "C"
//__global__ void MYnppiMaxIndx_32f_C1R_F2( Npp8u *pDeviceBuffer, int nBlocksInBuffer,  Npp32f *pMax,  int *pIndexX, int *pIndexY )
__global__ void MYnppiMaxIndx_32f_C1R_F2( DevParamNppEmulator rp )
{
  // One-channel 32-bit floating point image MaxIndx. Second part.

  Npp8u* pDeviceBuffer = (Npp8u*) rp.ptr1;
  Npp32f* pMax = (Npp32f*) rp.ptr2;
  int* pIndexX = (int*) rp.ptr3;
  int* pIndexY = (int*) rp.ptr4;
  int &nBlocksInBuffer = rp.int1; 

  if( blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0 ){
 
    Npp32f *buf = (Npp32f *) pDeviceBuffer;   
    int *ibuf = (int *) pDeviceBuffer;   
    
    *pMax = -FLT_MAX;
    *pIndexX = 0;
    *pIndexY = 0;
    
    for( int block=0; block<nBlocksInBuffer; block++){
      Npp32f val = buf [block*3+0];
      if( val >= *pMax ){
	*pMax = val;	
	*pIndexX = ibuf[block*3+1];
	*pIndexY = ibuf[block*3+2];
      }
    }
  }
}



extern "C"
//__global__ void MYnppiSum_32f_C1R_F1( const Npp32f *pSrc, int nSrcStep,  NppiSize oSizeROI, Npp8u *pDeviceBuffer )
__global__ void MYnppiSum_32f_C1R_F1( DevParamNppEmulator rp )
{
  // One-channel 32-bit floating point image Sum. First part.

  __shared__ float blockSum;

  Npp32f* pSrc = (Npp32f*) rp.ptr1;
  Npp8u* pDeviceBuffer = (Npp8u*) rp.ptr2;  
  int &nSrcStep = rp.int1; 
  NppiSize &oSizeROI = rp.size;

  bool firstThread = ( threadIdx.x==0 && threadIdx.y==0 );
  
  if( firstThread ){
    blockSum = 0.;
  }

  //synchronize the local threads writing to the local memory cache
  __syncthreads();
 
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < oSizeROI.width && idy < oSizeROI.height ){  
    Npp32f *src = (Npp32f *)((uint8_t*) pSrc + idy * nSrcStep);  
    atomicAdd( &blockSum, src[idx] );
  }

  //synchronize the local threads writing to the local memory cache
  __syncthreads();
 
  // write to the global memory  
  if( firstThread ){
    int block = blockIdx.y * gridDim.x + blockIdx.x;
    Npp32f *buf = (Npp32f *) pDeviceBuffer;
    buf[block] = blockSum;
  }
}


extern "C"
//__global__ void MYnppiSum_32f_C1R_F2( Npp8u *pDeviceBuffer, int nBlocksInBuffer, Npp64f *pSum )
__global__ void MYnppiSum_32f_C1R_F2( DevParamNppEmulator rp )
{
  // One-channel 32-bit floating point image Sum. Second part.

  Npp8u* pDeviceBuffer = (Npp8u*) rp.ptr1;
  Npp64f* pSum = (Npp64f*) rp.ptr2;
  int &nBlocksInBuffer = rp.int1; 

  if( blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0 ){
    Npp32f *buf = (Npp32f *) pDeviceBuffer;
    double sum=0;    
    for( int block=0; block<nBlocksInBuffer; block++){
      sum +=buf[block];
    }
    *pSum = sum;
  }
}


extern "C"
//__global__ void MYnppiSum_8u_C1R_F1( const Npp8u *pSrc, int nSrcStep,  NppiSize oSizeROI, Npp8u *pDeviceBuffer )
__global__ void MYnppiSum_8u_C1R_F1( DevParamNppEmulator rp )
{
  // One-channel 8-bit unsigned image sum.

  __shared__ float blockSum;

  Npp32f* pSrc = (Npp32f*) rp.ptr1;
  Npp8u* pDeviceBuffer = (Npp8u*) rp.ptr2;  
  int &nSrcStep = rp.int1; 
  NppiSize &oSizeROI = rp.size;

  bool firstThread = ( threadIdx.x==0 && threadIdx.y==0 );
  
  if( firstThread ){
    blockSum = 0.;
  }

  //synchronize the local threads writing to the local memory cache
  __syncthreads();
 
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;

  if (idx < oSizeROI.width && idy < oSizeROI.height ){  
    Npp8u *src = (Npp8u *)((uint8_t*) pSrc + idy * nSrcStep);  
    atomicAdd( &blockSum, (float) src[idx] );
  }

  //synchronize the local threads writing to the local memory cache
  __syncthreads();
 
  // write to the global memory  
  if( firstThread ){
    int block = blockIdx.y * gridDim.x + blockIdx.x;
    Npp32f *buf = (Npp32f *) pDeviceBuffer;
    buf[block] = blockSum;
  }
}


extern "C"
//__global__ void MYnppiConvert_16s32f_C1R( const Npp16s *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI )
__global__ void MYnppiConvert_16s32f_C1R( DevParamNppEmulator rp )
{
  Npp16s* pSrc = (Npp16s*) rp.ptr1;
  Npp32f* pDst = (Npp32f*) rp.ptr2;
  int &nSrcStep = rp.int1;
  int &nDstStep = rp.int2;
  NppiSize &oSizeROI = rp.size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= oSizeROI.width || idy >= oSizeROI.height ) return;
  Npp16s *src  = (Npp16s *)((uint8_t*) pSrc + idy*nSrcStep );
  Npp32f *dst  = (Npp32f *)((uint8_t*) pDst + idy*nDstStep ); 
  dst[idx] = (Npp32f) src[idx];
}

extern "C"
//__global__ void MYnppiConvert_16u32f_C1R( const Npp16u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI )
__global__ void MYnppiConvert_16u32f_C1R( DevParamNppEmulator rp )
{
  Npp16u* pSrc = (Npp16u*) rp.ptr1;
  Npp32f* pDst = (Npp32f*) rp.ptr2;
  int &nSrcStep = rp.int1;
  int &nDstStep = rp.int2;
  NppiSize &oSizeROI = rp.size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= oSizeROI.width || idy >= oSizeROI.height ) return;
  Npp16u *src  = (Npp16u *)((uint8_t*) pSrc + idy*nSrcStep );
  Npp32f *dst  = (Npp32f *)((uint8_t*) pDst + idy*nDstStep ); 
  dst[idx] = (Npp32f) src[idx];
}

extern "C"
//__global__ void MYnppiConvert_32s32f_C1R( const Npp32s *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI )
__global__ void MYnppiConvert_32s32f_C1R( DevParamNppEmulator rp )
{
  Npp32s* pSrc = (Npp32s*) rp.ptr1;
  Npp32f* pDst = (Npp32f*) rp.ptr2;
  int &nSrcStep = rp.int1;
  int &nDstStep = rp.int2;
  NppiSize &oSizeROI = rp.size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= oSizeROI.width || idy >= oSizeROI.height ) return;
  Npp32s *src  = (Npp32s *)((uint8_t*) pSrc + idy*nSrcStep );
  Npp32f *dst  = (Npp32f *)((uint8_t*) pDst + idy*nDstStep ); 
  dst[idx] = (Npp32f) src[idx];
}

extern "C"
//__global__ void MYnppiConvert_32u32f_C1R( const Npp32u *pSrc, int nSrcStep, Npp32f *pDst, int nDstStep, NppiSize oSizeROI )
__global__ void MYnppiConvert_32u32f_C1R( DevParamNppEmulator rp )
{
  Npp32u* pSrc = (Npp32u*) rp.ptr1;
  Npp32f* pDst = (Npp32f*) rp.ptr2;
  int &nSrcStep = rp.int1;
  int &nDstStep = rp.int2;
  NppiSize &oSizeROI = rp.size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= oSizeROI.width || idy >= oSizeROI.height ) return;
  Npp32u *src  = (Npp32u *)((uint8_t*) pSrc + idy*nSrcStep );
  Npp32f *dst  = (Npp32f *)((uint8_t*) pDst + idy*nDstStep ); 
  dst[idx] = (Npp32f) src[idx];
}


extern "C"
//__global__ void MYnppiSet_8u_C1R( const Npp8u nValue, Npp8u *pDst, int nDstStep, NppiSize oSizeROI )
__global__ void MYnppiSet_8u_C1R( DevParamNppEmulator rp )
{
  Npp8u* pDst = (Npp8u*) rp.ptr1;
  int &nDstStep = rp.int1;
  Npp8u &nValue = rp.npp8u1;
  NppiSize &oSizeROI = rp.size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= oSizeROI.width || idy >= oSizeROI.height ) return;  
  Npp8u *dst  = (Npp8u *)((uint8_t*) pDst + idy*nDstStep ); 
  dst[idx] = nValue;  
}

extern "C"
//__global__ void MYnppiCompareC_32f_C1R( const Npp32f *pSrc, int nSrcStep, const Npp32f nConstant,
//				      Npp8u *pDst, int nDstStep, NppiSize oSizeROI,  NppCmpOp eComparisonOperation )
__global__ void MYnppiCompareC_32f_C1R( DevParamNppEmulator rp )
{
  // 1 channel 32-bit floating point image compare with constant value.
  // Compare pSrc's pixels with constant value.

  Npp32f* pSrc = (Npp32f*) rp.ptr1;
  Npp8u*  pDst = (Npp8u*)  rp.ptr2;
  int &nSrcStep = rp.int1;
  int &nDstStep = rp.int2;
  Npp32f &nConstant = rp.float1;
  NppiSize &oSizeROI = rp.size;
  NppCmpOp &eComparisonOperation = rp.cmpOp;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= oSizeROI.width || idy >= oSizeROI.height ) return;
  Npp32f *src  = (Npp32f *)((uint8_t*) pSrc + idy*nSrcStep );
  Npp8u  *dst  = (Npp8u *) ((uint8_t*) pDst + idy*nDstStep ); 

  switch( eComparisonOperation ){
  case NPP_CMP_LESS:
    dst[idx] = (src[idx] <  nConstant) ?1 :0;
    break;
  case NPP_CMP_LESS_EQ:
    dst[idx] = (src[idx] <= nConstant) ?1 :0;
    break;
  case NPP_CMP_EQ:
    dst[idx] = (src[idx] == nConstant) ?1 :0;
    break;
  case NPP_CMP_GREATER_EQ:
    dst[idx] = (src[idx] >= nConstant) ?1 :0;
    break;
  case NPP_CMP_GREATER:
    dst[idx] = (src[idx] >  nConstant) ?1 :0;
    break;
  default:
    dst[idx] = 0;
  }
}


extern "C"
//__global__ void MYnppiSet_32f_C1MR( Npp32f nValue, Npp32f *pDst, int nDstStep,  NppiSize oSizeROI, const Npp8u *pMask, int nMaskStep )
__global__ void MYnppiSet_32f_C1MR( DevParamNppEmulator rp )
{
  // Masked 32-bit floating point image set. 

  Npp32f* pDst = (Npp32f*) rp.ptr1;
  Npp8u* pMask = (Npp8u*)  rp.ptr2;
  int &nDstStep  = rp.int1;
  int &nMaskStep = rp.int2;
  Npp32f &nValue = rp.float1;
  NppiSize &oSizeROI = rp.size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= oSizeROI.width || idy >= oSizeROI.height ) return;
  Npp32f *dst  = (Npp32f *) ((uint8_t*) pDst + idy*nDstStep ); 
  const Npp8u *msk  = (const Npp8u *)((const uint8_t*) pMask + idy*nMaskStep );
  if( msk[idx] ) dst[idx] = nValue;
}


extern "C"
//__global__ void MYnppiSqr_32f_C1R( const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,  int nDstStep, NppiSize oSizeROI )
__global__ void MYnppiSqr_32f_C1R( DevParamNppEmulator rp )
{
  //One 32-bit floating point channel image squared.  

  Npp32f* pSrc = (Npp32f*) rp.ptr1;
  Npp32f* pDst = (Npp32f*) rp.ptr2; 
  int &nSrcStep  = rp.int1;
  int &nDstStep  = rp.int2;
  NppiSize &oSizeROI = rp.size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= oSizeROI.width || idy >= oSizeROI.height ) return;
  Npp32f *src  = (Npp32f *)((uint8_t*) pSrc + idy*nSrcStep );
  Npp32f *dst  = (Npp32f *)((uint8_t*) pDst + idy*nDstStep ); 
  Npp32f val = src[idx];
  dst[idx] = val*val;
}

extern "C"
//__global__ void MYnppiCopy_32f_C1R( const Npp32f *pSrc, int nSrcStep, Npp32f *pDst,  int nDstStep, NppiSize oSizeROI )
__global__ void MYnppiCopy_32f_C1R( DevParamNppEmulator rp )
{
  //One 32-bit floating point channel image copy.  
  
  Npp32f* pSrc = (Npp32f*) rp.ptr1;
  Npp32f* pDst = (Npp32f*) rp.ptr2; 
  int &nSrcStep  = rp.int1;
  int &nDstStep  = rp.int2;
  NppiSize &oSizeROI = rp.size;

  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  size_t idy = blockIdx.y * blockDim.y + threadIdx.y;
  if (idx >= oSizeROI.width || idy >= oSizeROI.height ) return;
  Npp32f *src  = (Npp32f *)((uint8_t*) pSrc + idy*nSrcStep );
  Npp32f *dst  = (Npp32f *)((uint8_t*) pDst + idy*nDstStep ); 
  dst[idx] = src[idx];
}

#endif
