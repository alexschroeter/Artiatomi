#include "HipDeviceProperties.h"
#include <hip/hip_runtime.h>

namespace Hip
{

  HipDeviceProperties::HipDeviceProperties(hipDevice_t aDevice, int aDeviceID)
    :
    mDevProp(),
    mDeviceID(aDeviceID)
  {

    hipSafeCall( hipGetDeviceProperties( &mDevProp, aDevice ) );    

    int version=0;
    hipSafeCall(hipDriverGetVersion(&version));

    mDriverVersion = float(version/1000) + ((version%100) / 100.0f);


    // Not supported in HIP:

    // hipSafeCall(cuDeviceGetAttribute(&mMemPitch, CU_DEVICE_ATTRIBUTE_MAX_PITCH, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mTextureAlign, CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT, aDevice));
    // hipSafeCall(cuDeviceGetAttribute(&mKernelExecTimeoutEnabled, CU_DEVICE_ATTRIBUTE_KERNEL_EXEC_TIMEOUT, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mIntegrated, CU_DEVICE_ATTRIBUTE_INTEGRATED, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mMaximumTexture1DWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_WIDTH, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mMaximumTexture2DWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_WIDTH, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mMaximumTexture2DHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_HEIGHT, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mMaximumTexture3DWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_WIDTH, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mMaximumTexture3DHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_HEIGHT, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mMaximumTexture3DDepth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE3D_DEPTH, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mMaximumTexture2DArrayWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_WIDTH, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mMaximumTexture2DArrayHeight, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_HEIGHT, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mMaximumTexture2DArrayNumSlices, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE2D_ARRAY_NUMSLICES, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mSurfaceAllignment, CU_DEVICE_ATTRIBUTE_SURFACE_ALIGNMENT, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mECCEnabled, CU_DEVICE_ATTRIBUTE_ECC_ENABLED, aDevice));
    // hipSafeCall(cuDeviceGetAttribute(&mTCCDriver, CU_DEVICE_ATTRIBUTE_TCC_DRIVER, aDevice));
    // hipSafeCall(cuDeviceGetAttribute(&mAsyncEngineCount, CU_DEVICE_ATTRIBUTE_ASYNC_ENGINE_COUNT, aDevice));    
    //hipSafeCall(cuDeviceGetAttribute(&mUnifiedAddressing, CU_DEVICE_ATTRIBUTE_UNIFIED_ADDRESSING, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mMaximumTexture1DLayeredWidth, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_WIDTH, aDevice));
    //hipSafeCall(cuDeviceGetAttribute(&mMaximumTexture1DLayeredLayers, CU_DEVICE_ATTRIBUTE_MAXIMUM_TEXTURE1D_LAYERED_LAYERS, aDevice));

  }




  void HipDeviceProperties::Print() const
  {
    float capability = (mDevProp.major + mDevProp.minor / 10.0);

    cout << "\nDetailed device info:\n";
    cout << "--------------------------------------------------------------------------------\n\n" ;
    cout << "  Device name:                                      " << GetDeviceName() << endl;
    cout << "  Device ID:                                        " << mDeviceID << endl;

    cout << "  HIP compute capability revision                  " << capability << endl;
    cout << "  HIP driver version:                              " << mDriverVersion << endl;
    cout << "  Total amount of global memory:                    " << mDevProp.totalGlobalMem / 1024 / 1024 << " MByte" << endl;
    cout << "  Clock Rate:                                        " << mDevProp.clockRate / 1000 << " MHz"<< endl;
    cout << "  Clock Instruction Rate:                            " << mDevProp.clockInstructionRate / 1000 << " MHz"<< endl;
    cout << "  Maximum block dimensions:                         " 
	 << mDevProp.maxThreadsDim[0] << "x" << mDevProp.maxThreadsDim[1] << "x" << mDevProp.maxThreadsDim[2] << endl;
    cout << "  Maximum grid dimensions:                          " 
	 << mDevProp.maxGridSize[0] << "x" << mDevProp.maxGridSize[1] << "x" << mDevProp.maxGridSize[2] << endl;
    cout << endl;

    cout << "  Maximum number of threads per block:              " << mDevProp.maxThreadsPerBlock << endl;

    cout << "  Total number of registers available per block:    " << mDevProp.regsPerBlock << endl;

    cout << "  Total amount of shared memory per block:          " << mDevProp.sharedMemPerBlock << " Bytes" << endl;
    cout << "  Total amount of shared memory per Mult.Processor: " << mDevProp.maxSharedMemoryPerMultiProcessor << " Bytes" << endl;

    cout << "  Total amount of constant memory:                  " << mDevProp.totalConstMem << " Bytes"  << endl;

    if (mDevProp.multiProcessorCount > -1){
      cout << "  Number of multiprocessors:                        " << mDevProp.multiProcessorCount << endl;
      cout << "  Number of cores:                                  " 
	   << mDevProp.multiProcessorCount * ( ( capability < 2) ?8 :32) << endl;
    }

    cout << "  Warp size:                                        " << mDevProp.warpSize << endl;

    cout << endl;

    cout << "  Can execute multiple kernels concurrently:        " << (  mDevProp.concurrentKernels ?"True" :"False") << endl;    

    cout << "  Can map host memory:                              " << ( mDevProp.canMapHostMemory ?"True" :"False") << endl;
    
    cout << endl;

    cout << endl;

    cout << "  PCI bus ID:                                       " << mDevProp.pciBusID << endl;
    cout << "  PCI device ID:                                    " << mDevProp.pciDeviceID << endl;
    cout << "  PCI domain ID of the device:                      " << mDevProp.pciDomainID << endl;
    
    cout << "  HIP compute mode:                                ";

    switch( mDevProp.computeMode ){
    case 0:
      cout << "Default compute mode" << endl;
      break;
    case 1:    
      cout << "Compute-exclusive mode" << endl;
      break;
    case 2:
      cout << "Compute-prohibited mode" << endl;
      break;
    default:
      cout << "Unknown compute mode" << endl;
      break;
    }

    cout << "  Peak memory clock frequency:                      " << mDevProp.memoryClockRate / 1000 << " MHz" << endl;
    
    cout << "  Global memory bus width in bits:                  " << mDevProp.memoryBusWidth << endl;

    cout << "  Size of L2 cache in bytes:                        " << mDevProp.l2CacheSize << endl;

    cout << "  Maximum resident threads per multiprocessor:      " << mDevProp.maxThreadsPerMultiProcessor << endl;
    
    cout << endl;
    cout << endl;
  }
}
