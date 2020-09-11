#include "HipArrays.h"
#include "HipMissedStuff.h"
#include "HipConvertionUtil.h"
#include <hip/hip_runtime.h>

namespace Hip
{

    
  //========================
  //    HipArray1D   
  //========================

 
  HipArray1D::HipArray1D(hipChannelFormatDesc aFormat, size_t aSizeInElements )
    : mHipArray(0),mFormatDescriptor(aFormat), mWidthInElements(aSizeInElements), mSizeInBytes(0)
  {
    uint height = 0;
    hipSafeCall( hipMallocArray( &mHipArray, &mFormatDescriptor, mWidthInElements, height, hipArrayDefault ) );
    mSizeInBytes = mWidthInElements * ( mFormatDescriptor.x + mFormatDescriptor.y + mFormatDescriptor.z + mFormatDescriptor.w )/8; // bits to bytes
  }

  HipArray1D::~HipArray1D()
  {
    hipSafeCall( hipFreeArray( mHipArray ) );
  }

  void HipArray1D::CopyFromDeviceToArray(HipDeviceVariable& aSource, size_t aOffsetInBytes)
  {
    hipSafeCall( hipMemcpyToArray( mHipArray, aOffsetInBytes, 0, (void*) aSource.GetDevicePtr(), aSource.GetSize(), hipMemcpyDeviceToDevice ) );
  }

  void HipArray1D::CopyFromArrayToDevice(HipDeviceVariable& aDest, size_t aOffsetInBytes)
  {
    // SG!!! note: size of the array is specified by the destination variable
    hipSafeCall( hipMemcpyFromArray( (void*) aDest.GetDevicePtr(),  mHipArray, aOffsetInBytes, 0, aDest.GetSize(), hipMemcpyDeviceToDevice ) );
  }

  void HipArray1D::CopyFromHostToArray(void* aSource, size_t aOffsetInBytes)
  {
    // SG!!! note: size of the array is specified by the destination ( by this array descriptor)   
    hipSafeCall( hipMemcpyToArray( mHipArray, aOffsetInBytes, 0, (void*) aSource, mSizeInBytes, hipMemcpyHostToDevice ) );   
  }
  
  void HipArray1D::CopyFromArrayToHost(void* aDest, size_t aOffsetInBytes)
  {
    hipSafeCall( hipMemcpyFromArray( (void*) aDest,  mHipArray, aOffsetInBytes, 0, mSizeInBytes, hipMemcpyDeviceToHost ) );   
  }

  
 
  

  //========================
  //    HipArray2D   
  //========================

  HipArray2D::HipArray2D(hipChannelFormatDesc aFormat, size_t aWidthInElements, size_t aHeightInElements )
    : mHipArray(0),mFormatDescriptor(), mWidthInElements(0), mHeight(0), mSizeInBytes(0)     
  {
    mFormatDescriptor = aFormat;
    mWidthInElements = aWidthInElements;
    mHeight = aHeightInElements;
    hipSafeCall( hipMallocArray( &mHipArray, &mFormatDescriptor, mWidthInElements, mHeight, hipArrayDefault ) );
    mSizeInBytes = mWidthInElements * mHeight * ( mFormatDescriptor.x + mFormatDescriptor.y + mFormatDescriptor.z + mFormatDescriptor.w )/8;
  } 


  HipArray2D::~HipArray2D()
  {
    hipSafeCall( hipFreeArray( mHipArray ) );
  }

  void HipArray2D::CopyFromDeviceToArray(HipPitchedDeviceVariable& aSource)
  {
    hipSafeCall( hipMemcpy2DToArray( mHipArray, 0, 0,
				      (void*) aSource.GetDevicePtr(), aSource.GetPitch(), aSource.GetWidthInBytes(), aSource.GetHeight(),
				      hipMemcpyDeviceToDevice ) );
  }
  
  void HipArray2D::CopyFromArrayToDevice(HipPitchedDeviceVariable& aDest)
  {
    // SG!!! analog to cudaMemcpy2DFromArray should be called here, but it does not exist in HIP
    /*
      cudaMemcpy2DFromArray( (void*) aDest.GetDevicePtr(), aDest.GetPitch(),
			   mHipArray, 0, 0, aDest.GetWidthInBytes(), aDest.GetHeight(),
			   cudaMemcpyDeviceToDevice );
    */   
    hipSafeCall( hipMemcpyFromArray( (void*) aDest.GetDevicePtr(),
				     mHipArray, 0, 0, mSizeInBytes,
				     hipMemcpyDeviceToDevice )
		 );
  }
  
  void HipArray2D::CopyFromHostToArray(void* aSource)
  {
     size_t srcPitch = mSizeInBytes / mHeight;
    hipSafeCall( hipMemcpy2DToArray( mHipArray, 0, 0,
				      (void*) aSource, srcPitch, srcPitch, mHeight,
				      hipMemcpyHostToDevice ) );   
   
  }

  
  void HipArray2D::CopyFromArrayToHost(void* aDest)
  {
    // SG!!! analog to cudaMemcpy2DFromArray should be called here, but it does not exist in HIP
    /* 
    size_t dstPitch = mSizeInBytes / mHeight;    
    cudaMemcpy2DFromArray( (void*) aDest, dstPitch,
			   mHipArray, 0, 0, dstPitch, mHeight,
			  cudaMemcpyDeviceToHost );
    */
    hipSafeCall( hipMemcpyFromArray( (void*) aDest,
				     mHipArray, 0, 0, mSizeInBytes,
				     hipMemcpyDeviceToHost )
		 );    
  }

  

  //========================
  //    HipArray3D   
  //========================

  HipArray3D::HipArray3D(hipChannelFormatDesc aFormat, size_t aWidthInElements, size_t aHeightInElements, size_t aDepthInElements, uint aFlags)
    : mHipArray(0), mFormatDescriptor(), mWidthInElements(0), mHeightInElements(0), mDepthInElements(0), mPitchInBytes(0)
  {
    //printf("Alloc size: %ld, %ld, %ld\n", aWidthInElements, aHeightInElements, aDepthInElements);
    //fflush(stdout);

    mWidthInElements = aWidthInElements;
    mHeightInElements = aHeightInElements;
    mDepthInElements = aDepthInElements;

    mFormatDescriptor = aFormat;
    mPitchInBytes = mWidthInElements * ( mFormatDescriptor.x + mFormatDescriptor.y + mFormatDescriptor.z + mFormatDescriptor.w )/8; // bits to bytes
    
    hipExtent extent;
    extent.width = mWidthInElements;
    extent.height = mHeightInElements;
    extent.depth = mDepthInElements;
    hipSafeCall( hipMalloc3DArray( &mHipArray, &mFormatDescriptor, extent, aFlags ) );
  }


  HipArray3D::~HipArray3D()
  {
    hipSafeCall( hipFreeArray( mHipArray ) );
  }

  
	void HipArray3D::CopyFromDeviceToArray(HipDeviceVariable& aSource)
	{
/*		CUDA_MEMCPY3D params;
		memset(&params, 0, sizeof(params));
		params.srcDevice = aSource.GetDevicePtr();
		params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        //params.srcHeight = mDescriptor.Height;
		//params.srcPitch = ;
		params.dstArray = mHipArray;
		params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Depth = mDescriptor.Depth;
		params.Height = mDescriptor.Height;
		params.WidthInBytes = mDescriptor.Width * mDescriptor.NumChannels * GetChannelSize(mDescriptor.Format);

		hipSafeCall(cuMemcpy3D_v2(&params));
*/
    hipMemcpy3DParms myParms;

    memset(&myParms, 0, sizeof(myParms));

    myParms.srcArray = 0;
    myParms.srcPtr.pitch = mPitchInBytes;
    myParms.srcPtr.ptr = (void*) aSource.GetDevicePtr();
    myParms.srcPtr.xsize = mWidthInElements;
    myParms.srcPtr.ysize = mHeightInElements;
    myParms.dstArray = mHipArray;
    myParms.extent.width = mWidthInElements;
    myParms.extent.height = mHeightInElements;
    myParms.extent.depth = mDepthInElements;
    //myParms.kind = hipMemcpyKind(3);
#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)
    myParms.kind = cudaMemcpyDeviceToDevice;
#else
    myParms.kind = hipMemcpyDeviceToDevice;
#endif
    hipSafeCall( hipMemcpy3D( &myParms ) );
	}
  /*
	void HipArray3D::CopyFromArrayToDevice(HipDeviceVariable& aDest)
	{
		CUDA_MEMCPY3D params;
		memset(&params, 0, sizeof(params));
		params.dstDevice = aDest.GetDevicePtr();
		params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		params.dstPitch = mDescriptor.Width * mDescriptor.NumChannels * GetChannelSize(mDescriptor.Format);
		params.srcArray = mHipArray;
		params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Depth = mDescriptor.Depth;
		params.Height = mDescriptor.Height;
		params.WidthInBytes = params.srcPitch;

		hipSafeCall(cuMemcpy3D(&params));
	}

	void HipArray3D::CopyFromDeviceToArray(HipPitchedDeviceVariable& aSource)
	{
		CUDA_MEMCPY3D params;
		memset(&params, 0, sizeof(params));
		params.srcDevice = aSource.GetDevicePtr();
		params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
		params.srcPitch = aSource.GetPitch();
		params.dstArray = mHipArray;
		params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Depth = mDescriptor.Depth;
		params.Height = mDescriptor.Height;
		params.WidthInBytes = params.srcPitch;

		hipSafeCall(cuMemcpy3D(&params));
	}
	void HipArray3D::CopyFromArrayToDevice(HipPitchedDeviceVariable& aDest)
	{
		CUDA_MEMCPY3D params;
		memset(&params, 0, sizeof(params));
		params.dstDevice = aDest.GetDevicePtr();
		params.dstMemoryType = CU_MEMORYTYPE_DEVICE;
		params.dstPitch = aDest.GetPitch();
		params.srcArray = mHipArray;
		params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
		params.Depth = mDescriptor.Depth;
		params.Height = mDescriptor.Height;
		params.WidthInBytes = params.srcPitch;

		hipSafeCall(cuMemcpy3D(&params));
	}
  */

  
  void HipArray3D::CopyFromHostToArray(void* aSource)
  {
    /*
    CUDA_MEMCPY3D params;
    memset(&params, 0, sizeof(params));
    params.srcHost = aSource;
    params.srcMemoryType = CU_MEMORYTYPE_HOST;
    params.srcPitch = mDescriptor.Width * mDescriptor.NumChannels * GetChannelSize(mDescriptor.Format);
    params.dstArray = mHipArray;
    params.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    params.Depth = mDescriptor.Depth;
    params.Height = mDescriptor.Height;
    params.WidthInBytes = params.srcPitch;
    
    hipSafeCall(cuMemcpy3D(&params));
    */    
    
    hipMemcpy3DParms myParms;

    memset(&myParms, 0, sizeof(myParms));

    myParms.srcArray = 0;
    myParms.srcPtr.pitch = mPitchInBytes;
    myParms.srcPtr.ptr = aSource;
    myParms.srcPtr.xsize = mWidthInElements;
    myParms.srcPtr.ysize = mHeightInElements;
    myParms.dstArray = mHipArray;
    myParms.extent.width = mWidthInElements;
    myParms.extent.height = mHeightInElements;
    myParms.extent.depth = mDepthInElements;
#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)
    myParms.kind = cudaMemcpyHostToDevice;
#else
    myParms.kind = hipMemcpyHostToDevice;
#endif
    hipSafeCall( hipMemcpy3D( &myParms ) );
  }
  
  void HipArray3D::CopyFromArrayToHost(void* aDest)
  {
    /*
    CUDA_MEMCPY3D params;
    memset(&params, 0, sizeof(params));
    params.dstHost = aDest;
    params.dstMemoryType = CU_MEMORYTYPE_HOST;
    params.dstPitch = mDescriptor.Width * mDescriptor.NumChannels * GetChannelSize(mDescriptor.Format);
    params.srcArray = mHipArray;
    params.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    params.Depth = mDescriptor.Depth;
    params.Height = mDescriptor.Height;
    params.WidthInBytes = params.dstPitch;
    
    hipSafeCall(cuMemcpy3D(&params));
    */

    hipMemcpy3DParms myParms;
    memset(&myParms, 0, sizeof(myParms));
    
    myParms.dstArray = 0;
    myParms.dstPtr.pitch = mPitchInBytes;
    myParms.dstPtr.ptr = aDest;
    myParms.dstPtr.xsize = mWidthInElements;
    myParms.dstPtr.ysize = mHeightInElements;

    // TODO: unify this

#if !defined(__HIP_PLATFORM_NVCC__) && defined(__HIP_PLATFORM_HCC__)
    // copying from array does not work with HCC
    myParms.srcArray = 0;//mHipArray;  
    myParms.srcPtr.ptr = mHipArray->data; 
    myParms.srcPtr.pitch = mPitchInBytes;
    myParms.srcPtr.ysize = mHeightInElements;
    
    myParms.extent.width =  mPitchInBytes;
    myParms.extent.height = mHeightInElements;
    myParms.extent.depth = mDepthInElements;

    myParms.kind = hipMemcpyDeviceToHost;  
 #endif

#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)
    myParms.srcArray = mHipArray; 

    /*
    myParms.dstArray = 0;
    myParms.dstPtr.pitch = mPitchInBytes;
    myParms.dstPtr.ptr = aDest;
    myParms.dstPtr.xsize = mWidthInElements;
    myParms.dstPtr.ysize = mHeightInElements;
    */
    myParms.extent.width =  mWidthInElements;
    myParms.extent.height = mHeightInElements;
    myParms.extent.depth = mDepthInElements;

    myParms.kind = cudaMemcpyDeviceToHost;    
#endif

    hipSafeCall( hipMemcpy3D( &myParms ) );   
  }

} // namespace
