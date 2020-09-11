#ifndef HIPCONTEXT_H
#define HIPCONTEXT_H

#include <hip/hip_runtime.h>
#include "HipKernel.h"
#include "HipDeviceProperties.h"
#include "HipMissedStuff.h"

namespace Hip
{
  //!  A wrapper class for a HIP Context. 
  /*!
    HipContext manages all the functionality of a HIP Context.
    \author Michael Kunz
    \date   January 2010
    \version 1.0
  */
  //A wrapper class for a HIP Context. 
  class HipContext
  {
  private:
    hipCtx_t 	 mHipContext; //The wrapped HIP Context
    hipDevice_t	 mHipDevice;  //The HIP Device linked to this HIP Context
    int		 mDeviceID;   //The ID of the HIP Device
    unsigned int mCtxFlags;   //Context creation flags
				
    //! HipContext constructor
    /*!
      Creates a new HIP Context bound to the HIP Device with the ID \p deviceID
      using the \p ctxFlags context creation flags.
      \param deviceID The ID of the HIP Device to use
      \param ctxFlags Context creation flags
    */
    //HipContext constructor
    HipContext(int deviceID, unsigned int ctxFlags);
	
    //! HipContext destructor
    /*!
      The Wrapped HIP Context will be detached.
    */
    //HipContext destructor
    ~HipContext();
  public:
		
    //! Create a new instance of a HipContext.
    /*!
      Creates a new HIP Context bound to the HIP Device with the ID \p aDeviceID
      using the \p ctxFlags context creation flags.
      \param aDeviceID The ID of the HIP Device to use
      \param ctxFlags Context creation flags
    */
    //Create a new instance of a HipContext.
    static HipContext* CreateInstance(int aDeviceID, unsigned int ctxFlags = hipDeviceScheduleAuto );

    //! Destroys an instance of a HipContext
    /*!
      The Wrapped HIP Context will be detached.
      /param aCtx The HipContext to destroy
    */
    //Destroys an instance of a HipContext
    static void DestroyInstance(HipContext* aCtx);

    //! Destroys an HipContext
    /*!
      The Wrapped HIP Context will be destroyed.
      /param aCtx The HipContext to destroy
    */
    //Destroys an HipContext
    static void DestroyContext(HipContext* aCtx);

    //! Pushes the wrapped HIP Context on the current CPU thread
    /*!
      Pushes the wrapped HIP Context onto the CPU thread's stack of current
      contexts. The specified context becomes the CPU thread's current context, so
      all HIP functions that operate on the current context are affected.
      The wrapped HIP Context must be "floating" before calling HIP::PushContext().
      I.e. not attached to any thread. Contexts are
      made to float by calling HIP::PopContext().
    */
    //Pushes the wrapped HIP Context on the current CPU thread
    void PushContext();		

    //! Pops the wrapped HIP Context from the current CPU thread
    /*!
      Pops the wrapped HIP context from the CPU thread. The HIP context must
      have a usage count of 1. HIP contexts have a usage count of 1 upon
      creation.
    */
    //Pops the wrapped HIP Context from the current CPU thread
    void PopContext();	

    //! Binds the wrapped HIP context to the calling CPU thread
    /*!
      Binds the wrapped HIP context to the calling CPU thread.
      If there exists a HIP context stack on the calling CPU thread, this
      will replace the top of that stack with this context.  
    */
    //Binds the wrapped HIP context to the calling CPU thread
    void SetCurrent();		

    //! Block for a context's tasks to complete
    /*!
      Blocks until the device has completed all preceding requested tasks.
      Synchronize() throws an exception if one of the preceding tasks failed.
      If the context was created with the hipDeviceScheduleBlockingSync flag, the 
      CPU thread will block until the GPU context has finished its work.
    */
    //Block for a context's tasks to complete
    void Synchronize();	

    //! Load a *.ptx Hip Module
    /*!
      Loads the *.ptx file given by \p modulePath and creates a hipModule_t bound to this HIP context.
      \param modulePath Path and filename of the *.ptx file to load.
      \return The created hipModule_t
    */
    //Load a *.ptx Hip Module
    HipKernel* LoadKernel(std::string aModulePath, std::string aKernelName, dim3 aGridDim, dim3 aBlockDim, uint aDynamicSharedMemory = 0);
		
    //! Load a *.ptx Hip Module
    /*!
      Loads the *.ptx file given by \p modulePath and creates a hipModule_t bound to this HIP context.
      \param modulePath Path and filename of the *.ptx file to load.
      \return The created hipModule_t
    */
    //Load a *.ptx Hip Module
    HipKernel* LoadKernel(std::string aModulePath, std::string aKernelName, uint aGridDimX, uint aGridDimY, uint aGridDimZ, uint aBlockDimX, uint aBlockDimY, uint aDynamicSharedMemory = 0);

    //! Load a *.ptx Hip Module
    /*!
      Loads the *.ptx file given by \p modulePath and creates a hipModule_t bound to this HIP context.
      \param modulePath Path and filename of the *.ptx file to load.
      \return The created hipModule_t
    */
    //Load a *.ptx Hip Module
    hipModule_t LoadModule(const char* modulePath);

    //! Load a *.ptx Hip Module
    /*!
      Loads the *.ptx file given by \p aModulePath and creates a hipModule_t bound to this HIP context.
      The PTX file will be compiled using \p aOptionCount compiling options determind in \p aOptions and the option
      values given in \p aOptionValues.
      \param aModulePath Path and filename of the *.ptx file to load.
      \param aOptionCount Number of options
      \param aOptions Options for JIT
      \param aOptionValues Option values for JIT
      \return The created hipModule_t
    */
    //Load a *.ptx Hip Module
    hipModule_t LoadModulePTX(const char* aModulePath, uint aOptionCount, hipJitOption* aOptions, void** aOptionValues);
		
    //! Load a PTX Hip Module from byte array
    /*!
      Loads the *.ptx module given by \p aModuleImage and creates a hipModule_t bound to this HIP context.
      The PTX file will be compiled using \p aOptionCount compiling options determind in \p aOptions and the option
      values given in \p aOptionValues.
      \param aOptionCount Number of options
      \param aOptions Options for JIT
      \param aOptionValues Option values for JIT
      \param aModuleImage Binary image of the *.ptx file to load.
      \return The created hipModule_t
    */
    //Load a PTX Hip Module from byte array
    hipModule_t LoadModulePTX(uint aOptionCount, hipJitOption* aOptions, void** aOptionValues, const void* aModuleImage);
		
    //! Load a PTX Hip Module from byte array
    /*!
      Loads the *.ptx module given by \p aModuleImage and creates a hipModule_t bound to this HIP context.
      The PTX file will be compiled using \p aOptionCount compiling options determind in \p aOptions and the option
      values given in \p aOptionValues.
      \param aOptionCount Number of options
      \param aOptions Options for JIT
      \param aOptionValues Option values for JIT
      \param aModuleImage Binary image of the *.ptx file to load.
      \return The created hipModule_t
    */
    //Load a PTX Hip Module from byte array
    hipModule_t LoadModulePTX(const void* aModuleImage, uint aMaxRegCount, bool showInfoBuffer, bool showErrorBuffer);
		
    //! Unload a Hip Module
    /*!
      Unloads a module \p aModule from the current context.
      \param aModule to unload
    */
    //Unload a Hip Module
    void UnloadModule(hipModule_t& aModule);
				
    //! A memset function for device memory
    /*!
      Sets the memory range of \p aSizeInBytes / sizeof(unsigned int) 32-bit values to the specified value
      \p aValue.
      \param aPtr Destination device pointer
      \param aValue Value to set
      \param aSizeInBytes Size of the memory area to set.
    */
    //A memset function for device memory
    void ClearMemory(hipDeviceptr_t aPtr, unsigned int aValue, size_t aSizeInBytes);
		
    //! Retrieves informations on the current HIP Device
    /*!
      Retrieves informations on the current HIP Device
      \return A HipDeviceProperties object
    */
    //Retrieves informations on the current HIP Device
    HipDeviceProperties* GetDeviceProperties();
		
    size_t GetFreeMemorySize();

    size_t GetMemorySize();
  };

	
} // namespace

#endif
