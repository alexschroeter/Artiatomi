#include "HipKernel.h"

#include "HipException.h"
#include <cstdarg>

namespace Hip
{

  HipKernel::HipKernel(string aKernelName, hipModule_t aModule/*, HipContext* aCtx*/)
    : 
    mModule(aModule), 
    mFunction(),
    mSharedMemSize(0),
    //mParamOffset(0),
    mBlockDim(32,32,1),
    mGridDim(1,1,1),
    mKernelName(aKernelName)
  {
    hipSafeCall(hipModuleGetFunction(&mFunction, mModule, mKernelName.c_str()));
/*    mBlockDim.x = mBlockDim.y = 32;
    mBlockDim.z = 1;
    mGridDim.x = mGridDim.y = mGridDim.z = 1;    
*/
  }

  HipKernel::HipKernel(string aKernelName, hipModule_t aModule, /*HipContext* aCtx, */dim3 aGridDim, dim3 aBlockDim, uint aDynamicSharedMemory)
    :
    mModule(aModule), 
    mFunction(),
    mSharedMemSize( aDynamicSharedMemory ),
    //mParamOffset(0),
    mBlockDim(aBlockDim),
    mGridDim(aGridDim),
    mKernelName(aKernelName)
  {
    hipSafeCall(hipModuleGetFunction(&mFunction, mModule, mKernelName.c_str()));
/*
    mBlockDim.x = aBlockDim.x;
    mBlockDim.y = aBlockDim.y;
    mBlockDim.z = aBlockDim.z;
    mGridDim.x = aGridDim.x;
    mGridDim.y = aGridDim.y;
    mGridDim.z = aGridDim.z;
*/
  }


  HipKernel::~HipKernel()
  {
    //mCtx->UnloadModule(mModule);
  }

  void HipKernel::SetConstantValue(string aName, void* aValue)
  {
    hipDeviceptr_t dVarPtr=NULL;
    size_t varSize=0;    
    hipSafeCall(hipModuleGetGlobal(&dVarPtr, &varSize, mModule, aName.c_str()));
    hipSafeCall(hipMemcpyHtoD(dVarPtr, aValue, varSize));
  }

/*
  void HipKernel::ResetParameterOffset()
  {
    mParamOffset = 0;
  }
*/

  void HipKernel::SetBlockDimensions(uint aX, uint aY, uint aZ)
  {
    mBlockDim.x = aX;
    mBlockDim.y = aY;
    mBlockDim.z = aZ;
  }

  void HipKernel::SetBlockDimensions(dim3 aBlockDim)
  {
    mBlockDim.x = aBlockDim.x;
    mBlockDim.y = aBlockDim.y;
    mBlockDim.z = aBlockDim.z;
  }

  void HipKernel::SetGridDimensions(uint aX, uint aY, uint aZ)
  {
    mGridDim.x = aX;
    mGridDim.y = aY;
    mGridDim.z = aZ;
  }

  void HipKernel::SetGridDimensions(dim3 aGridDim)
  {
    mGridDim.x = aGridDim.x;
    mGridDim.y = aGridDim.y;
    mGridDim.z = aGridDim.z;
  }

  void HipKernel::SetWorkSize(dim3 aWorkSize)
  {
    mGridDim.x = (aWorkSize.x + mBlockDim.x - 1) / mBlockDim.x;
    mGridDim.y = (aWorkSize.y + mBlockDim.y - 1) / mBlockDim.y;
    mGridDim.z = (aWorkSize.z + mBlockDim.z - 1) / mBlockDim.z;
  }

  void HipKernel::SetWorkSize(uint aX, uint aY, uint aZ)
  {
    SetWorkSize(make_dim3(aX, aY, aZ));
  }

  void HipKernel::SetDynamicSharedMemory(uint aSizeInBytes)
  {
    mSharedMemSize = aSizeInBytes;
  }
	
  hipModule_t& HipKernel::GetHipModule()
  { 
    return mModule;
  }

  hipFunction_t& HipKernel::GetHipFunction()
  {
    return mFunction;
  }



  //float HipKernel::operator()()
  //{
  //	float ms = 0;
  //	hipSafeCall(cuParamSetSize(mFunction, mParamOffset));

  //	hipSafeCall(cuFuncSetBlockShape(mFunction, mBlockDim[0], mBlockDim[1], mBlockDim[2]));
  //	hipSafeCall(cuFuncSetSharedSize(mFunction, mSharedMemSize));

  //	hipSafeCall(cuCtxSynchronize());

  //	hipEvent_t eventStart;
  //	hipEvent_t eventEnd;
  //	hipStream_t stream = 0;
  //	hipSafeCall(hipEventCreateWithFlags(&eventStart, CU_EVENT_DEFAULT));
  //	hipSafeCall(hipEventCreateWithFlags(&eventEnd, CU_EVENT_DEFAULT));
  //	
  //	hipSafeCall(cuStreamQuery(stream));
  //	hipSafeCall(cuEventRecord(eventStart, stream));

  //	hipSafeCall(cuLaunchGrid(mFunction, mGridDim[0], mGridDim[1]));

  //	hipSafeCall(cuCtxSynchronize());

  //	hipSafeCall(cuStreamQuery(stream));
  //	hipSafeCall(cuEventRecord(eventEnd, stream));
  //	hipSafeCall(cuEventSynchronize(eventEnd));
  //	hipSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

  //	//reset the parameter stack
  //	mParamOffset = 0;

  //	return ms;
  //}	

  float HipKernel::operator()(int dummy, ...)
  {
    float ms;
    va_list vl;
    va_start(vl,dummy);

    void* argList = (void*) vl;
		
    hipEvent_t eventStart;
    hipEvent_t eventEnd;
    hipStream_t stream = 0;
    hipSafeCall(hipEventCreateWithFlags(&eventStart, hipEventBlockingSync));
    hipSafeCall(hipEventCreateWithFlags(&eventEnd, hipEventBlockingSync));
		
    //hipSafeCall(cuStreamQuery(stream));
    hipSafeCall(hipEventRecord(eventStart, stream));
    hipSafeCall(hipModuleLaunchKernel(mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, (void**)argList, NULL));
		
    hipSafeCall(hipCtxSynchronize());

    //hipSafeCall(cuStreamQuery(stream));
    hipSafeCall(hipEventRecord(eventEnd, stream));
    hipSafeCall(hipEventSynchronize(eventEnd));
    hipSafeCall(hipEventElapsedTime(&ms, eventStart, eventEnd));

    va_end(vl);

    return ms;
  }

  float HipKernel::Launch( void* arglist[] ) const
  {
    float ms;
    hipEvent_t eventStart;
    hipEvent_t eventEnd;
    hipStream_t stream = 0;
    hipSafeCall(hipEventCreateWithFlags(&eventStart, hipEventBlockingSync));
    hipSafeCall(hipEventCreateWithFlags(&eventEnd, hipEventBlockingSync));

    hipSafeCall(hipEventRecord(eventStart, stream));
    hipSafeCall(hipModuleLaunchKernel( mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));

    hipSafeCall(hipCtxSynchronize());
    
    hipSafeCall(hipEventRecord(eventEnd, stream));
    hipSafeCall(hipEventSynchronize(eventEnd));
    hipSafeCall(hipEventElapsedTime(&ms, eventStart, eventEnd));
    
    hipSafeCall(hipEventDestroy(eventStart));
    hipSafeCall(hipEventDestroy(eventEnd));
    return ms;
  }

  float HipKernel::Launch( void* ArgObjectPtr, size_t size ) const
  {
    hipSafeCall(hipDeviceSynchronize());

    float ms;
    hipEvent_t eventStart;
    hipEvent_t eventEnd;
    hipStream_t stream = 0;
    hipSafeCall(hipEventCreateWithFlags(&eventStart, hipEventBlockingSync));
    hipSafeCall(hipEventCreateWithFlags(&eventEnd, hipEventBlockingSync));

    hipSafeCall(hipEventRecord(eventStart, stream));
    
    // kernel launch with parameters currently is not implemented for AMD - do low-level launch with the input buffer

    //void* arglist[1] = { ArgObjectPtr };
    //hipSafeCall(hipModuleLaunchKernel( mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, NULL, arglist, NULL));
    
    size_t vsize = size;
    void *config[] = {
      HIP_LAUNCH_PARAM_BUFFER_POINTER, ArgObjectPtr,
      HIP_LAUNCH_PARAM_BUFFER_SIZE, &vsize,
      HIP_LAUNCH_PARAM_END
    };
 
    hipSafeCall(hipModuleLaunchKernel( mFunction, mGridDim.x, mGridDim.y, mGridDim.z, mBlockDim.x, mBlockDim.y, mBlockDim.z, mSharedMemSize, 0, NULL, (void**)&config));
   
    hipSafeCall(hipDeviceSynchronize());
    hipSafeCall(hipCtxSynchronize());
    
    hipSafeCall(hipEventRecord(eventEnd, stream));
    hipSafeCall(hipEventSynchronize(eventEnd));
    hipSafeCall(hipEventElapsedTime(&ms, eventStart, eventEnd));
    
    hipSafeCall(hipEventDestroy(eventStart));
    hipSafeCall(hipEventDestroy(eventEnd));
    return ms;
  }

	
}//namespace
