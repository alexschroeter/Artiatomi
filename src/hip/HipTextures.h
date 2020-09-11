#ifndef HIPTEXTURES_H
#define HIPTEXTURES_H

#include <hip/hip_runtime.h>
#include "HipKernel.h"
#include "HipVariables.h"
#include "HipArrays.h"

namespace Hip
{
  
//! A wrapper class for a linear 2D HIP Texture. 
/*!
  \author Michael Kunz
  \date   January 2010
  \version 1.0
*/
//A wrapper class for a linear 2D HIP Texture. 
class HipTextureLinearPitched2D
{
 public:

  //! HipTextureLinearPitched2D constructor
  /*!
    Creates a new linear 2D Texture based on pitched \p aDevVar.
    \param aKernel HipKernel using the texture
    \param aTexName Texture name as defined in the *.cu source file
    \param aFilterMode Filter mode (CU_TR_FILTER_MODE_POINT or CU_TR_FILTER_MODE_LINEAR)
    \param aAddressMode0 Texture addess mode in x direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
    \param aAddressMode1 Texture addess mode in y direction (CU_TR_ADDRESS_MODE_WRAP or CU_TR_ADDRESS_MODE_CLAMP or CU_TR_ADDRESS_MODE_MIRROR or CU_TR_ADDRESS_MODE_BORDER)
    \param aTexRefSetFlag TexRefSetFlag
    \param aFormat Array format (CU_AD_FORMAT_FLOAT, CU_AD_FORMAT_SIGNED_INT8, etc)
    \param aDevVar Pitched device variable where the texture data is stored
    \param aNumChannels Number of texture channels  (must be 1, 2 or 4)
  */
  static void Bind(HipKernel* aKernel, string aTexName, 
		   hipTextureFilterMode aFilterMode, HipPitchedDeviceVariable* aDevVar, uint aNumChannels);
  
  
  
};
	
	
class HipTextureObject2D
{
 private:
  hipTextureObject_t mTexObject;
  HipPitchedDeviceVariable* mVar2D; //HIP pitched variable where the texture data is stored   
 public:
  HipTextureObject2D();  
  HipTextureObject2D( HipPitchedDeviceVariable* aVar2D, hipTextureFilterMode aFilterMode );  

  void Create( HipPitchedDeviceVariable* aVar2D, hipTextureFilterMode aFilterMode );  

  ~HipTextureObject2D();  
  HipPitchedDeviceVariable* GetVar2D();  
  hipTextureObject_t GetTexObject();
};


class HipTextureObject3D
{
 private:
  hipTextureObject_t mTexObject;
  HipArray3D* mArray; //HIP Array where the texture data is stored   
 public:
  HipTextureObject3D( HipArray3D* aArray );  
  HipTextureObject3D( HipArray3D* aArray, hipTextureFilterMode filter);  
  HipTextureObject3D( HipArray3D* aArray, hipTextureFilterMode filter, hipTextureAddressMode addressmode, bool normalizedCoords);
  ~HipTextureObject3D();  
  HipArray3D* GetArray();  
  hipTextureObject_t GetTexObject();
};

 
} // namescpace

 
#endif 
