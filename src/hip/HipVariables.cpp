
#include <hip/hip_runtime.h>
#include "HipVariables.h"

namespace Hip
{

  
  //==========                         ===========
  //==========    HipDeviceVariable   ===========
  //==========                         ===========


  
  HipDeviceVariable::HipDeviceVariable(size_t aSizeInBytes)
    :mDevPtr(0), mSizeInBytes(aSizeInBytes)
  {
    hipSafeCall(hipMalloc((void**)&mDevPtr, mSizeInBytes));
  }
  
  HipDeviceVariable::HipDeviceVariable()
    : mDevPtr(0), mSizeInBytes(0)
  {
  }

  void HipDeviceVariable::Alloc(size_t aSizeInBytes)
  {
    if (mDevPtr != 0){
      hipSafeCall(hipFree( (void*) mDevPtr));
      mDevPtr = 0;
    }
    mSizeInBytes = aSizeInBytes;
    hipSafeCall(hipMalloc((void**)&mDevPtr, aSizeInBytes));
  }
  
  HipDeviceVariable::~HipDeviceVariable()
  {
    hipSafeCall( hipFree( (void*) mDevPtr) );    
  }

  void HipDeviceVariable::CopyDeviceToDevice(hipDeviceptr_t aSource)
  {   
    hipSafeCall( hipMemcpy((void*)mDevPtr, (void*)aSource, mSizeInBytes, hipMemcpyDeviceToDevice) );
  }

  void HipDeviceVariable::CopyDeviceToDevice(HipDeviceVariable& aSource)
  {
    hipSafeCall( hipMemcpy((void*)mDevPtr, (void*)aSource.GetDevicePtr(), mSizeInBytes, hipMemcpyDeviceToDevice) );
  }

  void HipDeviceVariable::CopyHostToDevice(void* aSource, size_t aSizeInBytes)
  {
    size_t size = aSizeInBytes;
    if (size == 0)
      size = mSizeInBytes;
    hipSafeCall( hipMemcpy((void*)mDevPtr, aSource, size, hipMemcpyHostToDevice) );
  }

  void HipDeviceVariable::CopyDeviceToHost(void* aDest, size_t aSizeInBytes)
  {
    size_t size = aSizeInBytes;
    if (size == 0)
      size = mSizeInBytes;   
    hipSafeCall( hipMemcpy(aDest, (void*)mDevPtr, size, hipMemcpyDeviceToHost) );
  }

  size_t HipDeviceVariable::GetSize()
  {
    return mSizeInBytes;
  }

  hipDeviceptr_t HipDeviceVariable::GetDevicePtr()
  {
    return mDevPtr;
  }

  void HipDeviceVariable::Memset(uchar aValue)
  {
    hipSafeCall(hipMemsetD8(mDevPtr, aValue, mSizeInBytes));
  }

  void HipDeviceVariable::CopyHostToDeviceAsync(hipStream_t stream, void* aSource, size_t aSizeInBytes)
  {
    size_t size = aSizeInBytes;
    if (size == 0)
      size = mSizeInBytes;
    hipSafeCall(hipMemcpyHtoDAsync(mDevPtr, aSource, size,stream));
  }


  void HipDeviceVariable::CopyDeviceToHostAsync(hipStream_t stream, void* aDest, size_t aSizeInBytes)
  {
    size_t size = aSizeInBytes;
    if (size == 0)
      size = mSizeInBytes;
    hipSafeCall(hipMemcpyDtoHAsync(aDest, mDevPtr, size,stream));
  }


  void HipDeviceVariable::CopyDeviceToDeviceAsync(hipStream_t stream, Hip::HipDeviceVariable& aSource)
  {
    hipSafeCall(hipMemcpyDtoDAsync(mDevPtr, aSource.GetDevicePtr(), mSizeInBytes, stream));
  }   
   
  //==========                                ===========
  //==========    HipPitchedDeviceVariable   ===========
  //==========                                ===========
  

  HipPitchedDeviceVariable::HipPitchedDeviceVariable(size_t aWidthInBytes, size_t aHeight, uint aElementSize)
    : mDevPtr(0), mSizeInBytes(0), mPitch(0), mWidthInBytes(aWidthInBytes), mHeight(aHeight), mElementSize(aElementSize)
  {
    hipSafeCall( hipMallocPitch( (void**)&mDevPtr, &mPitch, mWidthInBytes, mHeight) );
    mSizeInBytes = aHeight * mPitch;
  }

  HipPitchedDeviceVariable::HipPitchedDeviceVariable() 
    : mDevPtr(0), mSizeInBytes(0), mPitch(0),  mWidthInBytes(0), mHeight(0), mElementSize(0)
  {
  }

  void HipPitchedDeviceVariable::Alloc(size_t aWidthInBytes, size_t aHeight, uint aElementSize)
  {
    if (mDevPtr != 0)
      {	
	hipSafeCall( hipFree( (void*)mDevPtr) );
	mDevPtr = 0;
      }
    mPitch = 0;
    mHeight = aHeight;
    mWidthInBytes = aWidthInBytes; 
    mElementSize = aElementSize;
    
    hipSafeCall( hipMallocPitch( (void**)&mDevPtr, &mPitch, mWidthInBytes, mHeight) );
    mSizeInBytes = aHeight * mPitch;
  }

  HipPitchedDeviceVariable::~HipPitchedDeviceVariable()
  {
    // SG!!! change here. The original was w/o hipSafeCall
    hipSafeCall( hipFree( (void*) mDevPtr) );   
  }

  void HipPitchedDeviceVariable::CopyDeviceToDevice(HipPitchedDeviceVariable& aSource)
  {
    hipSafeCall( hipMemcpy2D( (void*) mDevPtr, mPitch,
			      (void*) aSource.GetDevicePtr(), aSource.GetPitch(),
			      mWidthInBytes, mHeight, hipMemcpyDeviceToDevice     )
		 );
  }

  void HipPitchedDeviceVariable::CopyHostToDevice(void* aSource)
  {
    hipSafeCall( hipMemcpy2D( (void*) mDevPtr, mPitch,
			      aSource, mPitch,
			      mWidthInBytes, mHeight, 
			      hipMemcpyHostToDevice      )
		 );    
  }
  
  void HipPitchedDeviceVariable::CopyDeviceToHost(void* aDest)
  {
    hipSafeCall( hipMemcpy2D( aDest, mPitch,
			      (void*) mDevPtr, mPitch,                               
			      mWidthInBytes, mHeight, 
			      hipMemcpyDeviceToHost      )
		 );
    
  }

  size_t HipPitchedDeviceVariable::GetSize()
  {
    return mSizeInBytes;
  }

  hipDeviceptr_t HipPitchedDeviceVariable::GetDevicePtr()
  {
    return mDevPtr;
  }

  size_t HipPitchedDeviceVariable::GetPitch()
  {
    return mPitch;
  }

  uint HipPitchedDeviceVariable::GetElementSize()
  {
    return mElementSize;
  }

  size_t HipPitchedDeviceVariable::GetWidth()
  {
    if (mElementSize == 0)
      return 0;
    
    return mWidthInBytes / mElementSize;
  }

  size_t HipPitchedDeviceVariable::GetWidthInBytes()
  {
    return mWidthInBytes;
  }
  
  size_t HipPitchedDeviceVariable::GetHeight()
  {
    return mHeight;
  }

  void HipPitchedDeviceVariable::Memset(uchar aValue)
  {
    hipSafeCall(hipMemsetD8( mDevPtr, aValue, mSizeInBytes) );
  }



  //==========                                 ===========
  //==========    HipPageLockedHostVariable   ===========
  //==========                                 ===========

  
  HipPageLockedHostVariable::HipPageLockedHostVariable(size_t aSizeInBytes, uint aFlags)
    : mHostPtr(0), mSizeInBytes(aSizeInBytes) 
  {
    hipSafeCall(hipHostMalloc(&mHostPtr, mSizeInBytes, aFlags));
  }

  HipPageLockedHostVariable::~HipPageLockedHostVariable()
  {   
    hipSafeCall( hipHostFree(mHostPtr) );
  }
  
  size_t HipPageLockedHostVariable::GetSize()
  {
    return mSizeInBytes;
  }
  
  void* HipPageLockedHostVariable::GetHostPtr()
  {
    return mHostPtr;
  }
}
