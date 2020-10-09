#ifndef HipConvertionUtil_H
#define HipConvertionUtil_H

//!
//!  Utilities needed for CUDA -> HIP  convertion
//!

/*!
  \author Sergey Gorbunov
  \date   Sept 2018
  \version 1.0
*/


#include <hip/hip_runtime_api.h>
#include "HipException.h"

#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)

inline hipChannelFormatDesc myConvertChannelFormat( CUarray_format format, uint numChannels )
{
  hipChannelFormatDesc  d;
  memset(&d, 0, sizeof(d));  

  switch( format )
    {    
    case CU_AD_FORMAT_UNSIGNED_INT8 :
      d.f = cudaChannelFormatKindUnsigned;
      d.x = 8;
      break;
    case  CU_AD_FORMAT_UNSIGNED_INT16:
      d.f = cudaChannelFormatKindUnsigned;
      d.x = 16;
      break;
    case  CU_AD_FORMAT_UNSIGNED_INT32:
      d.f = cudaChannelFormatKindUnsigned;
      d.x = 32;
      break;
    case  CU_AD_FORMAT_SIGNED_INT8:
      d.f = cudaChannelFormatKindSigned;
      d.x = 8;
      break;
    case  CU_AD_FORMAT_SIGNED_INT16:
      d.f = cudaChannelFormatKindSigned;
      d.x = 16;
      break;
    case  CU_AD_FORMAT_SIGNED_INT32:
      d.f = cudaChannelFormatKindSigned;
      d.x = 32;
      break;
    case  CU_AD_FORMAT_HALF:
      d.f = cudaChannelFormatKindFloat;
      d.x = 16;
      break;
    case  CU_AD_FORMAT_FLOAT:
      d.f = cudaChannelFormatKindFloat;
      d.x = 32;
      break;
    default:
      d.f = cudaChannelFormatKindNone;
      d.x = 0;
    }
  
  switch( numChannels )
    {
    case 1:
      d.y = 0;
      d.z = 0;
      d.w = 0;
      break;
    case 2:
      d.y = d.x;
      d.z = 0;
      d.w = 0;
      break;
    case 4:
      d.y = d.x;
      d.z = d.x;
      d.w = d.x;
      break;
    default:
      d.y = 0;
      d.z = 0;
      d.w = 0;      
    }
  return d;
}


inline void myConvertChannelFormatBack( const hipChannelFormatDesc desc, CUarray_format &format, uint &numChannels )
{
  memset(&format, 0, sizeof(format));  

  switch( desc.f )
    {
    case hipChannelFormatKindUnsigned :
      if     ( desc.x ==  8 ) format = CU_AD_FORMAT_UNSIGNED_INT8;
      else if( desc.x == 16 ) format = CU_AD_FORMAT_UNSIGNED_INT16;
      else if( desc.x == 32 ) format = CU_AD_FORMAT_UNSIGNED_INT32;
      break;
    case hipChannelFormatKindSigned :
      if     ( desc.x ==  8 ) format = CU_AD_FORMAT_SIGNED_INT8;
      else if( desc.x == 16 ) format = CU_AD_FORMAT_SIGNED_INT16;
      else if( desc.x == 32 ) format = CU_AD_FORMAT_SIGNED_INT32;
      break;
    case hipChannelFormatKindFloat :
      if     ( desc.x == 16 ) format = CU_AD_FORMAT_HALF;
      else if( desc.x == 32 ) format = CU_AD_FORMAT_FLOAT;
      break;
    case cudaChannelFormatKindNone :      
      format = CU_AD_FORMAT_UNSIGNED_INT8;
      break;
    }
  
  if     ( desc.w>0 ) numChannels = 4;
  else if( desc.y>0 ) numChannels = 2;
  else if( desc.x>0 ) numChannels = 1;
  else numChannels = 0;  
}


inline unsigned int myGetChannelSize(const CUarray_format aFormat)
{
  unsigned int result = 0;
  switch(aFormat)
    {
    case CU_AD_FORMAT_FLOAT:
      result = sizeof(float);
      break;
    case CU_AD_FORMAT_HALF:
      result = sizeof(short);
      break;
    case CU_AD_FORMAT_UNSIGNED_INT8:
      result = sizeof(unsigned char);
      break;
    case CU_AD_FORMAT_UNSIGNED_INT16:
      result = sizeof(unsigned short);
      break;
    case CU_AD_FORMAT_UNSIGNED_INT32:
      result = sizeof(unsigned int);
      break;
    case CU_AD_FORMAT_SIGNED_INT8:
      result = sizeof(char);
      break;
    case CU_AD_FORMAT_SIGNED_INT16:
      result = sizeof(short);
      break;
    case CU_AD_FORMAT_SIGNED_INT32:
      result = sizeof(int);
      break;
    default:
      Hip::HipException ex("Unknown texture format");
      throw ex;
      break;
    }
    
  return result;
}

inline CUDA_ARRAY_DESCRIPTOR myGetCuArrayDescriptor(const hipChannelFormatDesc desc, int widthInElements, int heightInElements )
{   
  CUDA_ARRAY_DESCRIPTOR descriptor;
  memset(&descriptor, 0, sizeof(descriptor));
  descriptor.Width = widthInElements;
  descriptor.Height = heightInElements;
  myConvertChannelFormatBack( desc, descriptor.Format, descriptor.NumChannels );
  return descriptor;
}

inline CUDA_ARRAY3D_DESCRIPTOR myGetCuArrayDescriptor(const hipChannelFormatDesc desc, int widthInElements, int heightInElements, int depthInElements )
{   
  CUDA_ARRAY3D_DESCRIPTOR descriptor;
  memset(&descriptor, 0, sizeof(descriptor));
  descriptor.Width = widthInElements;
  descriptor.Height = heightInElements;
  descriptor.Depth = depthInElements;
  myConvertChannelFormatBack( desc, descriptor.Format, descriptor.NumChannels );
  return descriptor;
}

#endif

#endif
