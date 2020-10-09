
#include "HipContext.h"

namespace Hip
{


  HipContext::HipContext(int deviceID, unsigned int ctxFlags) : 
    mHipContext(0), mHipDevice(0), mDeviceID(deviceID), mCtxFlags(ctxFlags)
  {
    hipSafeCall(hipDeviceGet(&mHipDevice, deviceID));
    hipSafeCall(hipCtxCreate(&mHipContext, mCtxFlags, mHipDevice));
  }

  HipContext::~HipContext()
  {
    if (mHipContext)
      {
	hipSafeCall(hipCtxDetach(mHipContext));
      }
  }
	
  HipContext* HipContext::CreateInstance(int aDeviceID, unsigned int ctxFlags)
  {
    int deviceCount;
    // test whether the driver has already been initialized
    if(hipGetDeviceCount(&deviceCount) == hipErrorNotInitialized)
      {
	hipSafeCall(hipInit(0));
      }

    hipSafeCall(hipGetDeviceCount(&deviceCount));        
    if (deviceCount == 0)
      {
	HipException ex("Hip initialization error: There is no device supporting HIP");
	throw ex;
	return NULL;
      } 
    return new HipContext(aDeviceID, ctxFlags);
  }

  void HipContext::DestroyInstance(HipContext* aCtx)
  {
    delete aCtx;
  }    

  void HipContext::DestroyContext(HipContext* aCtx)
  {
    hipSafeCall(hipCtxDestroy(aCtx->mHipContext));
    aCtx->mHipContext = 0;
  }   

  void HipContext::PushContext()
  {
    hipSafeCall(hipCtxPushCurrent(mHipContext));
  }

  void HipContext::PopContext()
  {
    hipSafeCall(hipCtxPopCurrent(NULL));
  }

  void HipContext::SetCurrent()
  {
    hipSafeCall(hipCtxSetCurrent(mHipContext));
  }

  void HipContext::Synchronize()
  {
    hipSafeCall(hipCtxSynchronize());
  }

  hipModule_t HipContext::LoadModule(const char* aModulePath)
  {   
    hipModule_t hcuModule;
    hipSafeCall(hipModuleLoad(&hcuModule, aModulePath));
    return hcuModule;
  }

  HipKernel* HipContext::LoadKernel(std::string aModulePath, std::string aKernelName, dim3 aGridDim, dim3 aBlockDim, uint aDynamicSharedMemory)
  {   
    hipModule_t hcuModule = LoadModulePTX(aModulePath.c_str(), 0, NULL, NULL);

    HipKernel* kernel = new HipKernel(aKernelName, hcuModule, /*this, */aGridDim, aBlockDim, aDynamicSharedMemory);

    return kernel;
  }

  HipKernel* HipContext::LoadKernel(std::string aModulePath, std::string aKernelName, uint aGridDimX, uint aGridDimY, uint aGridDimZ, uint aBlockDimX, uint aBlockDimY, uint aDynamicSharedMemory)
  {   
    hipModule_t hcuModule = LoadModulePTX(aModulePath.c_str(), 0, NULL, NULL);

    dim3 aGridDim;
    aGridDim.x = aGridDimX;
    aGridDim.y = aGridDimY;
    aGridDim.z = aGridDimZ;
    dim3 aBlockDim;
    aBlockDim.x = aBlockDimX;
    aBlockDim.y = aBlockDimY;

    HipKernel* kernel = new HipKernel(aKernelName, hcuModule, /*this, */aGridDim, aBlockDim, aDynamicSharedMemory);

    return kernel;
  }

  hipModule_t HipContext::LoadModulePTX(const char* aModulePath, uint aOptionCount, hipJitOption* aOptions, void** aOptionValues)
  {	
    std::ifstream file(aModulePath, ios::in|ios::binary|ios::ate);
    if (!file.good())
      {
	std::string filename;
	filename = "File not found: ";
	filename += aModulePath;
	HipException ex(__FILE__, __LINE__, filename, hipErrorFileNotFound );
     }
    ifstream::pos_type size;
    size = file.tellg();
    char* memblock = new char [(size_t)size+1];
    file.seekg (0, ios::beg);
    file.read (memblock, size);
    file.close();
    memblock[size] = 0;
    //cout << endl << endl << "Filesize is: " << size << endl;
	
    hipModule_t hModule = LoadModulePTX(aOptionCount, aOptions, aOptionValues, memblock);
	
    delete[] memblock;
    return hModule;
  }

  hipModule_t HipContext::LoadModulePTX(uint aOptionCount, hipJitOption* aOptions, void** aOptionValues, const void* aModuleImage)
  {   
    hipModule_t hModule;
    hipSafeCall(hipModuleLoadDataEx(&hModule, aModuleImage, aOptionCount, aOptions, aOptionValues));
	
    return hModule;
  }

  hipModule_t HipContext::LoadModulePTX(const void* aModuleImage, uint aMaxRegCount, bool showInfoBuffer, bool showErrorBuffer)
  {
    uint jitOptionCount = 0;
    hipJitOption ptxOptions[6];
    void* jitValues[6];
    int indexSet = 0;
    char infoBuffer[1025];
    char compilerBuffer[1025];
    ulong64 compilerBufferSize = 1024;
    ulong64 infoBufferSize = 1024;
    infoBuffer[0] = '\0';
    compilerBuffer[0] = '\0';
    int indexInfoBufferSize = 0;
    int indexCompilerBufferSize = 0;

    if (showInfoBuffer)
      {
	jitOptionCount += 3;
	ptxOptions[indexSet] = hipJitOptionInfoLogBufferSizeBytes;
	jitValues[indexSet] = (void*)  infoBufferSize;
	indexInfoBufferSize = indexSet;
	indexSet++;
	ptxOptions[indexSet] = hipJitOptionInfoLogBuffer;
	jitValues[indexSet] = infoBuffer;
	indexSet++;
	ptxOptions[indexSet] = hipJitOptionLogVerbose;
	jitValues[indexSet] = (void*) (ulong64) 1; 
	indexSet++;
      }

    if (showErrorBuffer)
      {
	jitOptionCount += 2;
	ptxOptions[indexSet] = hipJitOptionErrorLogBufferSizeBytes;
	jitValues[indexSet] = (void*)compilerBufferSize;
	indexCompilerBufferSize = indexSet;
	indexSet++;
	ptxOptions[indexSet] = hipJitOptionErrorLogBuffer;
	jitValues[indexSet] = compilerBuffer;
	indexSet++;
      }
	
    if (aMaxRegCount > 0)
      {
	jitOptionCount += 1;
	ptxOptions[indexSet] = hipJitOptionMaxRegisters;
	jitValues[indexSet] = (void*)(ulong64)aMaxRegCount;
	indexSet++;
      }
	
    hipModule_t hModule;
    hipSafeCall(hipModuleLoadDataEx(&hModule, aModuleImage, jitOptionCount, ptxOptions, jitValues));
    //hipSafeCall(hipModuleLoadData(&hModule, aModuleImage));

    if (showInfoBuffer)
      {
	if (jitValues[indexInfoBufferSize])
	  printf("Hip JIT Info: \n%s\n", infoBuffer);
      }

    if (showErrorBuffer)
      {
	if (jitValues[indexCompilerBufferSize])
	  printf("Hip JIT Error: \n%s\n", compilerBuffer);
      }
	
    return hModule;
  }

  void HipContext::UnloadModule(hipModule_t& aModule)
  {
    if (aModule)
      {
	hipSafeCall(hipModuleUnload(aModule));
	aModule = 0;
      }
  }		

  //CUarray HipContext::CreateArray1D(unsigned int aNumElements, CUarray_format aFormat, unsigned int aNumChannels)
  //{
  //	CUarray hCuArray;
  //	CUDA_ARRAY_DESCRIPTOR props;
  //	props.Width = aNumElements;
  //	props.Height = 0;
  //	props.Format = aFormat;
  //	props.NumChannels = aNumChannels;
  //	hipSafeCall(cuArrayCreate(&hCuArray, &props));
  //	return hCuArray;
  //}
  //
  //CUarray HipContext::CreateArray2D(unsigned int aWidth, unsigned int aHeight, CUarray_format aFormat, unsigned int aNumChannels)
  //{
  //	CUarray hCuArray;
  //	CUDA_ARRAY_DESCRIPTOR props;
  //	props.Width = aWidth;
  //	props.Height = aHeight;
  //	props.Format = aFormat;
  //	props.NumChannels = aNumChannels;
  //	hipSafeCall(cuArrayCreate(&hCuArray, &props));
  //	return hCuArray;
  //}
  //
  //CUarray HipContext::CreateArray3D(unsigned int aWidth, unsigned int aHeight, unsigned int aDepth, CUarray_format aFormat, unsigned int aNumChannels, int aFlags)
  //{
  //	CUarray hCuArray;
  //	CUDA_ARRAY3D_DESCRIPTOR props;
  //	props.Width = aWidth;
  //	props.Height = aHeight;
  //	props.Depth = aDepth;
  //	props.Format = aFormat;
  //	props.NumChannels = aNumChannels;
  //	props.Flags = aFlags;
  //	hipSafeCall(cuArray3DCreate(&hCuArray, &props));
  //	return hCuArray;
  //}
  //
  //hipDeviceptr_t HipContext::AllocateMemory(size_t aSizeInBytes)
  //{
  //	hipDeviceptr_t dBuffer;
  //	hipSafeCall(cuMemAlloc(&dBuffer, aSizeInBytes));
  //	return dBuffer;
  //}
		
  void HipContext::ClearMemory(hipDeviceptr_t aPtr, unsigned int aValue, size_t aSizeInBytes)
  {
    hipSafeCall(hipMemset((void*)aPtr, aValue, aSizeInBytes / sizeof(unsigned int)));
  }

  /*void HipContext::FreeMemory(hipDeviceptr_t dBuffer)
    {
    hipSafeCall(cuMemFree(dBuffer));
    }	*/	

  //void HipContext::CopyToDevice(hipDeviceptr_t aDest, const void* aSource, unsigned int aSizeInBytes)
  //{
  //	hipSafeCall(cuMemcpyHtoD(aDest, aSource, aSizeInBytes));
  //}
  //
  //void HipContext::CopyToHost(void* aDest, hipDeviceptr_t aSource, unsigned int aSizeInBytes)
  //{
  //	hipSafeCall(cuMemcpyDtoH(aDest, aSource, aSizeInBytes));
  //}

  HipDeviceProperties* HipContext::GetDeviceProperties()
  {
    HipDeviceProperties* props = new HipDeviceProperties(mHipDevice, mDeviceID);
    return props;
  }

  //void HipContext::SetTextureProperties(CUtexref aHcuTexRef, const TextureProperties& aTexProps)
  //{
  //	hipSafeCall(cuTexRefSetAddressMode(aHcuTexRef, 0, aTexProps.addressMode[0]));
  //	hipSafeCall(cuTexRefSetAddressMode(aHcuTexRef, 1, aTexProps.addressMode[1]));
  //	hipSafeCall(cuTexRefSetFilterMode(aHcuTexRef, aTexProps.filterMode));
  //	hipSafeCall(cuTexRefSetFlags(aHcuTexRef, aTexProps.otherFlags));
  //	hipSafeCall(cuTexRefSetFormat(aHcuTexRef, aTexProps.format, aTexProps.numChannels));
  //}      
  //
  //void HipContext::SetTexture3DProperties(CUtexref aHcuTexRef, const Texture3DProperties& aTexProps)
  //{
  //	hipSafeCall(cuTexRefSetAddressMode(aHcuTexRef, 0, aTexProps.addressMode[0]));
  //	hipSafeCall(cuTexRefSetAddressMode(aHcuTexRef, 1, aTexProps.addressMode[1]));
  //	hipSafeCall(cuTexRefSetAddressMode(aHcuTexRef, 2, aTexProps.addressMode[2]));
  //	hipSafeCall(cuTexRefSetFilterMode(aHcuTexRef, aTexProps.filterMode));
  //	hipSafeCall(cuTexRefSetFlags(aHcuTexRef, aTexProps.otherFlags));
  //	hipSafeCall(cuTexRefSetFormat(aHcuTexRef, aTexProps.format, aTexProps.numChannels));
  //}

  size_t HipContext::GetFreeMemorySize()
  {
    size_t sizeTotal, sizeFree;
    hipSafeCall(hipMemGetInfo(&sizeFree, &sizeTotal));
    return sizeFree;
  }

  size_t HipContext::GetMemorySize()
  {
    size_t sizeTotal, sizeFree;
    hipSafeCall(hipMemGetInfo(&sizeFree, &sizeTotal));
    return sizeTotal;
  }


} // namespace
