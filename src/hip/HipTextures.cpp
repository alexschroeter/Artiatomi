
#include "HipTextures.h"
#include "HipMissedStuff.h"
#include "HipException.h"
#include "HipConvertionUtil.h"

//#include <hip/hip_runtime.h>

namespace Hip
{
  

  void HipTextureLinearPitched2D::Bind(HipKernel* aKernel, string aTexName,
					hipTextureFilterMode aFilterMode, HipPitchedDeviceVariable* aDevVar, uint aNumChannels)
  {

    // SG: Currently, most of the HIP code here has no implementation for NVCC :(
    // Therefore two compilation branches for now.

#if !defined(__HIP_PLATFORM_NVCC__) && defined(__HIP_PLATFORM_HCC__)

    // HIP

    textureReference* texref;
    hipModule_t hipModule = aKernel->GetHipModule();    

    hipSafeCall( hipModuleGetTexRef(&texref, hipModule, aTexName.c_str()) );
    hipSafeCall( hipTexRefSetAddressMode(texref, 0, hipAddressModeClamp) );
    hipSafeCall( hipTexRefSetAddressMode(texref, 1, hipAddressModeClamp) );
    hipSafeCall( hipTexRefSetFilterMode(texref, aFilterMode) );
    hipSafeCall( hipTexRefSetFlags(texref, 0) );
    hipSafeCall( hipTexRefSetFormat(texref, HIP_AD_FORMAT_FLOAT, aNumChannels) );

    HIP_ARRAY_DESCRIPTOR arraydesc;
    memset(&arraydesc, 0, sizeof(arraydesc));
    arraydesc.Format = HIP_AD_FORMAT_FLOAT;
    arraydesc.NumChannels = aNumChannels;
    arraydesc.Width = aDevVar->GetWidth();
    arraydesc.Height = aDevVar->GetHeight();
    //arraydesc.Flags = 0;
    //arraydesc.Depth = 0;
    
    hipSafeCall( hipTexRefSetAddress2D(texref, &arraydesc, aDevVar->GetDevicePtr(), aDevVar->GetPitch()) );
   
    //  SG!!! not implemented in HIP. Seems to be not needed at all??

   // hipSafeCall( cuParamSetTexRef(aKernel->GetHipFunction(), CU_PARAM_TR_DEFAULT, texref) );

#endif

#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)

    // CUDA 

    CUfilter_mode cuFilterMode = CU_TR_FILTER_MODE_LINEAR;
    switch ( aFilterMode )
      {
      case hipFilterModePoint:
	cuFilterMode = CU_TR_FILTER_MODE_POINT;
	break;
      case hipFilterModeLinear:
	cuFilterMode = CU_TR_FILTER_MODE_LINEAR;
	break;
      }
    
    CUtexref texref;
    hipSafeCall( cuModuleGetTexRef(&texref, aKernel->GetHipModule(), aTexName.c_str()) );
    hipSafeCall( cuTexRefSetAddressMode(texref, 0, CU_TR_ADDRESS_MODE_CLAMP) );
    hipSafeCall( cuTexRefSetAddressMode(texref, 1, CU_TR_ADDRESS_MODE_CLAMP) );   
    hipSafeCall( cuTexRefSetFilterMode(texref, cuFilterMode) );
    hipSafeCall( cuTexRefSetFlags(texref, 0) );
    hipSafeCall( cuTexRefSetFormat(texref, CU_AD_FORMAT_FLOAT, aNumChannels) );

    // uint channelSize = Cuda::GetChannelSize(aFormat);
    // size_t sizeInBytes = aDevVar->GetHeight() * aDevVar->GetWidth() * channelSize * aNumChannels;


    CUDA_ARRAY_DESCRIPTOR arraydesc;
    memset(&arraydesc, 0, sizeof(arraydesc));

    arraydesc.Format = CU_AD_FORMAT_FLOAT;
    arraydesc.Height = aDevVar->GetHeight();
    arraydesc.NumChannels = aNumChannels;
    arraydesc.Width = aDevVar->GetWidth();
    
    hipSafeCall(cuTexRefSetAddress2D(texref, &arraydesc, aDevVar->GetDevicePtr(), aDevVar->GetPitch()));
    
    // SG!!! I commented it out, because I can not convert it to HIP. It seems to be be not needed anyhow.
    //hipSafeCall(cuParamSetTexRef(aKernel->GetHipFunction(), CU_PARAM_TR_DEFAULT, texref));

#endif

  }


  HipTextureObject2D::HipTextureObject2D( )
    : mTexObject(0), mVar2D(0)
  {
  }

  HipTextureObject2D::HipTextureObject2D( HipPitchedDeviceVariable* var2D, hipTextureFilterMode aFilterMode )
  : mTexObject(0), mVar2D(0)
  {
    Create( var2D, aFilterMode );
  }


  void HipTextureObject2D::Create( HipPitchedDeviceVariable* var2D, hipTextureFilterMode aFilterMode )
  {
    mVar2D = var2D;

    hipResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));

    resDesc.resType = hipResourceTypePitch2D;
    resDesc.res.pitch2D.devPtr = (void*) mVar2D->GetDevicePtr();   
    resDesc.res.pitch2D.desc = myhipCreateChannelDesc( 32, 0, 0, 0, hipChannelFormatKindFloat);   
    resDesc.res.pitch2D.width = var2D->GetWidth();
    resDesc.res.pitch2D.height = mVar2D->GetHeight();
    resDesc.res.pitch2D.pitchInBytes = var2D->GetPitch();
 
    hipTextureDesc  texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

    texDesc.addressMode[0] = hipAddressModeClamp;
    texDesc.addressMode[1] = hipAddressModeClamp;
    texDesc.filterMode = aFilterMode;
    texDesc.readMode = hipReadModeElementType;
    texDesc.normalizedCoords = 0;

    hipSafeCall(hipCreateTextureObject( &mTexObject, &resDesc, &texDesc, NULL ));    
  }


  HipTextureObject2D::~HipTextureObject2D()
  {
    hipSafeCall(hipDestroyTextureObject(mTexObject));
  }
			
  HipPitchedDeviceVariable* HipTextureObject2D::GetVar2D()
  {
    return mVar2D;
  }

  hipTextureObject_t HipTextureObject2D::GetTexObject()
  {
    return mTexObject;
  }  

 



  HipTextureObject3D::HipTextureObject3D( HipArray3D* aArray)
  {
    mArray = aArray;    

    hipResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));

    resDesc.resType = hipResourceTypeArray;
    resDesc.res.array.array = aArray->GetArray();   

    hipTextureDesc  texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

    texDesc.addressMode[0] = hipAddressModeClamp;
    texDesc.addressMode[1] = hipAddressModeClamp;
    texDesc.addressMode[2] = hipAddressModeClamp;
    texDesc.filterMode = hipFilterModeLinear;
    texDesc.normalizedCoords = false;

    hipCreateTextureObject( &mTexObject, &resDesc, &texDesc, NULL );     
    
  }

   HipTextureObject3D::HipTextureObject3D( HipArray3D* aArray, hipTextureFilterMode filter)
  {
    mArray = aArray;    

    hipResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));

    resDesc.resType = hipResourceTypeArray;
    resDesc.res.array.array = aArray->GetArray();   

    hipTextureDesc  texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

    texDesc.addressMode[0] = hipAddressModeClamp;
    texDesc.addressMode[1] = hipAddressModeClamp;
    texDesc.addressMode[2] = hipAddressModeClamp;
    texDesc.filterMode = filter;
    //texDesc.readMode = hipReadModeElementType;
    texDesc.normalizedCoords = false;

    hipCreateTextureObject( &mTexObject, &resDesc, &texDesc, NULL );     
    
    /* AS testing
    mTexObject->filterMode = hipFilterModePoint;
    mTexObject->normalized = false;
    mTexObject->addressMode[0] = hipAddressModeClamp;
    mTexObject->addressMode[1] = hipAddressModeClamp;
    */
  }


   HipTextureObject3D::HipTextureObject3D( HipArray3D* aArray, hipTextureFilterMode filter, hipTextureAddressMode addressmode, bool normalizedCoords)
  {
    mArray = aArray;    

    hipResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));

    resDesc.resType = hipResourceTypeArray;
    resDesc.res.array.array = aArray->GetArray();   

    hipTextureDesc  texDesc;
    memset(&texDesc, 0, sizeof(texDesc));

    texDesc.addressMode[0] = addressmode;
    texDesc.addressMode[1] = addressmode;
    texDesc.addressMode[2] = addressmode;
    texDesc.filterMode = filter;
    //texDesc.readMode = hipReadModeElementType;
    texDesc.normalizedCoords = normalizedCoords;

    hipCreateTextureObject( &mTexObject, &resDesc, &texDesc, NULL );     
    
    /* AS testing
    mTexObject->filterMode = hipFilterModePoint;
    mTexObject->normalized = false;
    mTexObject->addressMode[0] = hipAddressModeClamp;
    mTexObject->addressMode[1] = hipAddressModeClamp;
    */
  }


  HipTextureObject3D::~HipTextureObject3D()
  {
    hipSafeCall(hipDestroyTextureObject(mTexObject));
  }
			
  HipArray3D* HipTextureObject3D::GetArray()
  {
    return mArray;
  }

  hipTextureObject_t HipTextureObject3D::GetTexObject()
  {
    return mTexObject;
  }  


} // namespace
