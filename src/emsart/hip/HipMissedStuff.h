#ifndef HipMissedStuff_H
#define HipMissedStuff_H

//!
//!  Some classes and functions which are necessary for NVCC compilation but currently missing in HIP
//!

/*!
  \author Sergey Gorbunov
  \date   Mary 2018
  \version 1.0
*/


#include <hip/hip_runtime.h>


#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)

typedef cudaExtent hipExtent;

/*  things are already implemented

inline hipError_t hipMalloc3DArray( hipArray ** array, const struct hipChannelFormatDesc* desc,
				    hipExtent extent, unsigned int flags                  )
{
  cudaError_t err = cudaMalloc3DArray( (cudaArray**) array, (const hipChannelFormatDesc*) desc, extent, flags );
  return hipCUDAErrorTohipError( err );
}

typedef struct cudaMemcpy3DParms  hipMemcpy3DParms;

inline hipError_t hipMemcpy3D( const  hipMemcpy3DParms *p )
{
  cudaError_t err = cudaMemcpy3D( p );
  return hipCUDAErrorTohipError( err );
}
*/


#if !defined(__CUDACC__)

// SG: for no reason this HIP functions exist only for nvcc compiler 
// Reproducing them here to let them work also with gcc

inline static hipError_t hipCreateTextureObject(hipTextureObject_t* pTexObject,
					 const hipResourceDesc* pResDesc,
					 const hipTextureDesc* pTexDesc,
					 const hipResourceViewDesc* pResViewDesc) {
  return hipCUDAErrorTohipError(
				cudaCreateTextureObject(pTexObject, pResDesc, pTexDesc, pResViewDesc));
}


inline static hipError_t hipCreateSurfaceObject(hipSurfaceObject_t* pSurfObject,
					 const hipResourceDesc* pResDesc ) {
  return hipCUDAErrorTohipError(
				cudaCreateSurfaceObject(pSurfObject, pResDesc));
}

inline static hipError_t hipDestroyTextureObject(hipTextureObject_t textureObject) {
  return hipCUDAErrorTohipError(cudaDestroyTextureObject(textureObject));
}

inline static hipChannelFormatDesc hipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f ){
  return cudaCreateChannelDesc(x, y, z, w, hipChannelFormatKindToCudaChannelFormatKind(f));
}

#endif


inline static hipChannelFormatDesc myhipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f ){
  // for no reason the original hipCreateChannelDesc exist only for nvcc compiler. 
  // This code makes it work also with gcc
  return cudaCreateChannelDesc(x, y, z, w, hipChannelFormatKindToCudaChannelFormatKind(f));
}


#define hipDeviceScheduleAuto  cudaDeviceScheduleAuto // = 0x0   Automatically select between Spin and Yield
#define hipDeviceScheduleSpin  cudaDeviceScheduleSpin // = 0x01  ///< Dedicate a CPU core to spin-wait.  Provides lowest latency, but burns a CPU core and
                                                            ///< may consume more power.
#define hipDeviceScheduleYield cudaDeviceScheduleYield // = 0x2  ///< Yield the CPU to the operating system when waiting.  May increase latency, but lowers
                                                              ///< power and is friendlier to other threads in the system.
#define hipDeviceScheduleBlockingSync cudaDeviceBlockingSync // = 0x4 

//#define hipDeviceMapHost cudaDeviceMapHost // 0x8 - already defined in HIP
#define hipDeviceLmemResizeToMax cudaDeviceLmemResizeToMax


#endif



#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)

inline static hipChannelFormatDesc myhipCreateChannelDesc(int x, int y, int z, int w, hipChannelFormatKind f ){
  return hipCreateChannelDesc(x, y, z, w, f);
}

inline hipError_t hipCtxDetach(hipCtx_t )
{
  // cuCtx* functions are deprecated for HCC, do nothing
  return hipSuccess;
}

#if defined(__HCC__)
template <class T>
__SURFACE_FUNCTIONS_DECL__  void surf3Dread(T* data, hipSurfaceObject_t surfObj, int x, int y, int z, int boundaryMode = hipBoundaryModeZero)
{
  // x in bytes
    hipArray* arrayPtr = (hipArray*) surfObj;
    size_t width = arrayPtr->width;
    size_t height = arrayPtr->height;
    size_t depth = arrayPtr->depth;    
    int32_t xOffset = x / sizeof(T);
    T* dataPtr = (T*) arrayPtr->data;
    if((xOffset > width) || (xOffset < 0) || (y > height) ||(y < 0) || (z > depth) || (z < 0)) {
        if(boundaryMode == hipBoundaryModeZero) {
            *data = 0;
        }
    } else {
        *data = *(dataPtr + z*width*height + y*width + xOffset);
    }
}

template <class T>
__SURFACE_FUNCTIONS_DECL__  void surf3Dwrite(T data, hipSurfaceObject_t surfObj, int x, int y, int z, int boundaryMode = hipBoundaryModeZero)
{
 // x in bytes
     hipArray* arrayPtr = (hipArray*) surfObj;
    size_t width = arrayPtr->width;
    size_t height = arrayPtr->height;
    size_t depth = arrayPtr->depth;    
    int32_t xOffset = x / sizeof(T);
    T* dataPtr = (T*) arrayPtr->data;
    if(!((xOffset > width) || (xOffset < 0) || (y > height) ||(y < 0) || (z > depth) || (z < 0) )){
        *(dataPtr + z*width*height + y*width + xOffset) = data;
    }
}
#endif

#endif


#endif 
