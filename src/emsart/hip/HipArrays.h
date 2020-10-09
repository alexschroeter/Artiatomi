#ifndef HipArrays_H
#define HipArrays_H

#include <hip/hip_runtime.h>
#include "HipDefault.h"
#include "HipException.h"
#include "HipVariables.h"

namespace Hip
{
  //!  A wrapper class for a CUDA Array.
  /*!
    HipArray1D manages all the functionality of a 1D CUDA Array.
    \author Michael Kunz
    \date   January 2010
    \version 1.0
  */
  //A wrapper class for a CUDA Array.
  class HipArray1D
  {
  private:
    hipArray*  mHipArray;
    hipChannelFormatDesc mFormatDescriptor;
    size_t mWidthInElements;
    size_t mSizeInBytes;

  public:
    //! HipArray1D constructor
    /*!
      Creates a new 1D CUDA Array size \p aSizeInElements and \p aNumChannels channels.
      \param aFormat Array format
      \param aSizeInElements Size of the array
      \param aNumChannels Number of array channels. Must be 1, 2 or 4.
    */

    HipArray1D(hipChannelFormatDesc aFormat, size_t aSizeInElements );

    //! HipArray1D destructor
    ~HipArray1D();

    //! Copy data from device memory to this array
    /*!
      Copies the data given by \p aSource in device memory to this array.
      \param aSource Data source in device memory
      \param aOffsetInBytes An Offset in the destination array.
    */
    void CopyFromDeviceToArray(HipDeviceVariable& aSource, size_t aOffsetInBytes = 0);

    //! Copy data from this array to device memory
    /*!
      Copies from this array to device memory given by \p aDest.
      \param aDest Data destination in device memory
      \param aOffsetInBytes An Offset in the source array.
    */
    void CopyFromArrayToDevice(HipDeviceVariable& aDest, size_t aOffsetInBytes = 0);


    //! Copy data from host memory to this array
    /*!
      Copies the data given by \p aSource in host memory to this array.
      \param aSource Data source in host memory
      \param aOffsetInBytes An Offset in the destination array.
    */
    void CopyFromHostToArray(void* aSource, size_t aOffsetInBytes = 0);

    //! Copy data from this array to host memory
    /*!
      Copies from this array to host memory given by \p aDest.
      \param aDest Data destination in host memory
      \param aOffsetInBytes An Offset in the source array.
    */
    void CopyFromArrayToHost(void* aDest, size_t aOffsetInBytes = 0);
  };

  //!  A wrapper class for a CUDA Array.
  /*!
    HipArray2D manages all the functionality of a 2D CUDA Array.
    \author Michael Kunz
    \date   January 2010
    \version 1.0
  */
  class HipArray2D
  {
  private:
    hipArray*  mHipArray;
    hipChannelFormatDesc mFormatDescriptor;
    size_t mWidthInElements;
    size_t mHeight;
    size_t mSizeInBytes;

  public:
    //! HipArray2D constructor
    /*!
      Creates a new 2D CUDA Array of size \p aWidthInElements x \p aHeightInElements and \p aNumChannels channels.
      \param aFormat Array format
      \param aWidthInElements Width of the array
      \param aHeightInElements Height of the array
      \param aNumChannels Number of array channels. Must be 1, 2 or 4.
    */    
    HipArray2D(hipChannelFormatDesc aFormat, size_t aWidthInElements, size_t aHeightInElements );

    //! HipArray2D destructor
    ~HipArray2D();

    //! Copy data from device memory to this array
    /*!
      Copies the data given by \p aSource in device memory to this array.
      \param aSource Data source in device memory
    */
    void CopyFromDeviceToArray(HipPitchedDeviceVariable& aSource);

    //! Copy data from this array to device memory
    /*!
      Copies from this array to device memory given by \p aDest.
      \param aDest Data destination in device memory
    */
    void CopyFromArrayToDevice(HipPitchedDeviceVariable& aDest);


    //! Copy data from host memory to this array
    /*!
      Copies the data given by \p aSource in host memory to this array.
      \param aSource Data source in host memory
    */
    void CopyFromHostToArray(void* aSource);

    //! Copy data from this array to host memory
    /*!
      Copies from this array to host memory given by \p aDest.
      \param aDest Data destination in host memory
    */
    void CopyFromArrayToHost(void* aDest);

  };

  //!  A wrapper class for a CUDA Array.
  /*!
    HipArray3D manages all the functionality of a 3D CUDA Array.
    \author Michael Kunz
    \date   January 2010
    \version 1.0
  */
  class HipArray3D
  {
  private:
    hipArray *mHipArray;
    //CUDA_ARRAY3D_DESCRIPTOR mDescriptor;
    hipChannelFormatDesc mFormatDescriptor;
    size_t mWidthInElements;
    size_t mHeightInElements;
    size_t mDepthInElements;
    size_t mPitchInBytes;

  public:
    //! HipArray3D constructor
    /*!
      Creates a new 3D CUDA Array of size \p aWidthInElements x \p aHeightInElements x \p aDepthInElements and \p aNumChannels channels.
      \param aFormat Array format
      \param aWidthInElements Width of the array
      \param aHeightInElements Height of the array
      \param aDepthInElements Depth of the array
      \param aNumChannels Number of array channels. Must be 1, 2 or 4.
      \param aFlags Array creation flags.
    */   
    HipArray3D(hipChannelFormatDesc aFormat, size_t aWidthInElements, size_t aHeightInElements, size_t aDepthInElements, uint aFlags=hipArrayDefault );

    //! HipArray3D destructor
    ~HipArray3D();

    //====
    // the following methods were not converted to HIP, since they are not in use:
    //		
    // void CopyFromDeviceToArray(HipDeviceVariable& aSource);
    // void CopyFromArrayToDevice(HipDeviceVariable& aDest);
    // void CopyFromDeviceToArray(HipPitchedDeviceVariable& aSource);
    // void CopyFromArrayToDevice(HipPitchedDeviceVariable& aDest);
    //
    //=====
		
    //! Copy data from host memory to this array
    /*!
      Copies the data given by \p aSource in host memory to this array.
      \param aSource Data source in host memory
    */
    void CopyFromHostToArray(void* aSource);

    //! Copy data from this array to host memory
    /*!
      Copies from this array to host memory given by \p aDest.
      \param aDest Data destination in host memory
    */
    void CopyFromArrayToHost(void* aDest);

    hipChannelFormatDesc GetFormatDescriptor(){ return mFormatDescriptor; }

    //! Get the hipArray of this array
    /*!
      Get method for the wrapped CUDA_ARRAY_DESCRIPTOR.
      \return Wrapped hipArray.
    */
    hipArray* GetArray(){ return mHipArray; }
  };
  

} // namespace

#endif 
