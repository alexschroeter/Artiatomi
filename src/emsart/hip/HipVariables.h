#ifndef HIPVARIABLES_H
#define HIPVARIABLES_H


#include <hip/hip_runtime.h>
#include "HipDefault.h"
#include "HipException.h"

namespace Hip
{
  //!  A wrapper class for a HIP Device variable.
  /*!
    \author Michael Kunz
    \date   January 2010
    \version 1.0
  */
  //A wrapper class for a HIP Device variable.
  class HipDeviceVariable
  {
  private:
    hipDeviceptr_t mDevPtr; //Wrapped device pointer
    size_t mSizeInBytes; //Size in bytes of the wrapped device variable

  public:
    //! HipDeviceVariable constructor
    /*!
      Allocates \p aSizeInBytes bytes in device memory
      \param aSizeInBytes Amount of memory to allocate
    */
    //HipDeviceVariable constructor
    HipDeviceVariable(size_t aSizeInBytes);
    //HipDeviceVariable(const hipDeviceptr_t& aDevPtr);

    //! Initializes the object but doesn't allocate GPU memory. Inner ptr is 0;
    HipDeviceVariable();

    //! Reallocates the inner hipDeviceptr_t. If inner ptr isn't 0 it is freed before.
    void Alloc(size_t aSizeInBytes);

    //! HipDeviceVariable destructor
    //HipDeviceVariable destructor
    ~HipDeviceVariable();

    //! Copy data from device memory to this HipDeviceVariable
    /*!
      \param aSource Data source in device memory
    */
    //Copy data from device memory to this HipDeviceVariable
    void CopyDeviceToDevice(hipDeviceptr_t aSource);

    //! Copy data from device memory to this HipDeviceVariable
    /*!
      \param aSource Data source in device memory
    */
    //Copy data from device memory to this HipDeviceVariable
    void CopyDeviceToDevice(HipDeviceVariable& aSource);


    //! Copy data from host memory to this HipDeviceVariable
    /*!
      \param aSource Data source in host memory
      \param aSizeInBytes Number of bytes to copy
    */
    //Copy data from host memory to this HipDeviceVariable
    void CopyHostToDevice(void* aSource, size_t aSizeInBytes = 0);

    //! Copy data from this HipDeviceVariable to host memory
    /*!
      \param aDest Data destination in host memory
      \param aSizeInBytes Number of bytes to copy
    */
    //Copy data from this HipDeviceVariable to host memory
    void CopyDeviceToHost(void* aDest, size_t aSizeInBytes = 0);

    //! Returns the size in bytes of the allocated device memory
    size_t GetSize();

    //! Returns the wrapped hipDeviceptr_t
    hipDeviceptr_t GetDevicePtr();

    //! Sets the allocated memory to \p aValue
    void Memset(uchar aValue);
  };

  //!  A wrapper class for a pitched HIP Device variable.
  /*!
    \author Michael Kunz
    \date   January 2010
    \version 1.0
  */
  //A wrapper class for a pitched HIP Device variable.
  class HipPitchedDeviceVariable
  {
  private:
    hipDeviceptr_t mDevPtr;  //Wrapped hipDeviceptr_t
    size_t mSizeInBytes;     //Total size in bytes allocated in device memory
    size_t mPitch;	     //Memory pitch as returned by hipMallocPitch
    size_t mWidthInBytes;    //Width in bytes of the allocated memory area (<= mPitch)
    size_t mHeight;	     //Height in elements of the allocated memory area
    uint   mElementSize;     //Size of one data element

  public:
    //! HipPitchedDeviceVariable constructor
    /*!
      Allocates at least \p aHeight x \p aWidthInBytes bytes in device memory
      \param aHeight Height in elements
      \param aWidthInBytes Width in bytes (<= mPitch)
      \param aElementSize Size of one data element
    */
    //HipPitchedDeviceVariable constructor
    HipPitchedDeviceVariable(size_t aWidthInBytes, size_t aHeight, uint aElementSize);

    //! Initializes the object but doesn't allocate GPU memory. Inner ptr is 0;
    HipPitchedDeviceVariable();

    //! Reallocates the inner hipDeviceptr_t. If inner ptr isn't 0 it is freed before.
    void Alloc(size_t aWidthInBytes, size_t aHeight, uint aElementSize);

    //! HipPitchedDeviceVariable destructor
    //HipPitchedDeviceVariable destructor
    ~HipPitchedDeviceVariable();

    //! Copy data from device memory to this HipPitchedDeviceVariable
    /*!
      \param aSource Data source in device memory
    */
    //Copy data from device memory to this HipPitchedDeviceVariable
    void CopyDeviceToDevice(hipDeviceptr_t aSource);

    //! Copy data from device memory to this HipPitchedDeviceVariable
    /*!
      \param aSource Data source in device memory
    */
    //Copy data from device memory to this HipPitchedDeviceVariable
    void CopyDeviceToDevice(HipPitchedDeviceVariable& aSource);


    //! Copy data from host memory to this HipPitchedDeviceVariable
    /*!
      \param aSource Data source in host memory
    */
    //Copy data from host memory to this HipPitchedDeviceVariable
    void CopyHostToDevice(void* aSource);

    //! Copy data from this HipPitchedDeviceVariable to host memory
    /*!
      \param aDest Data destination in host memory
    */
    //Copy data from this HipPitchedDeviceVariable to host memory
    void CopyDeviceToHost(void* aDest);

    //! Returns the data size in bytes. NOT the pitched allocated size in device memory
    //Returns the data size in bytes. NOT the pitched allocated size in device memory
    size_t GetSize();

    //! Return the allocation pitch
    //Return the allocation pitch
    size_t GetPitch();

    //! Returns the data element size
    //Returns the data element size
    uint GetElementSize();

    //! Return the width in elements
    //Return the width in elements
    size_t GetWidth();

    //! Return the width in bytes
    //Return the width in bytes
    size_t GetWidthInBytes();

    //! Return the height in elements
    //Return the height in elements
    size_t GetHeight();

    //! Return the wrapped hipDeviceptr_t
    //Return the wrapped hipDeviceptr_t
    hipDeviceptr_t GetDevicePtr();

    //! Sets the allocated memory to \p aValue
    void Memset(uchar aValue);
  };

  //!  A wrapper class for a Page Locked Host variable.
  /*!
    HipPageLockedHostVariable is allocated using hipHostMalloc
    \author Michael Kunz
    \date   January 2010
    \version 1.0
  */
  class HipPageLockedHostVariable
  {
  private:
    void* mHostPtr; //Host pointer allocated by hipHostMalloc
    size_t mSizeInBytes; //Size in bytes

  public:
    //! HipPageLockedHostVariable constructor
    /*!
      Allocates \p aSizeInBytes bytes in page locked host memory using hipHostMalloc
      \param aSizeInBytes Number of bytes to allocate
      \param aFlags Allocation flags.
    */
    //HipPageLockedHostVariable constructor
    HipPageLockedHostVariable(size_t aSizeInBytes, uint aFlags);

    //!HipPageLockedHostVariable destructor
    //HipPageLockedHostVariable destructor
    ~HipPageLockedHostVariable();

    //! Returns the size of the allocated memory area in bytes
    //Returns the size of the allocated memory area in bytes
    size_t GetSize();

    //! Returns the wrapped page locked host memory pointer
    //Returns the wrapped page locked host memory pointer
    void* GetHostPtr();
  };

} // namespace

#endif
