#ifndef HIPDEVICEPROPERTIES_H
#define HIPDEVICEPROPERTIES_H

#include "HipDefault.h"
#include "HipException.h"
#include "HipArrays.h"
#include <hip/hip_runtime.h>

namespace Hip
{
  //!  Retrieves the device properties of a HIP device. 
  /*!
    \author Michael Kunz
    \date   January 2010
    \version 1.0
  */
  //Retrieves the device properties of a HIP device. 

  class HipDeviceProperties
  {
  private:
    hipDeviceProp_t mDevProp;
    int mDeviceID;      
    float mDriverVersion;

  public:	
    //! HipDeviceProperties constructor
    /*!
      While instantiation, HipDeviceProperties retrieves the device properties using the HIP Driver API 
      \param aDevice The HIP Device 
      \param aDeviceID The ID of the HIP Device to use
    */
    //HipDeviceProperties constructor
    HipDeviceProperties(hipDevice_t aDevice, int aDeviceID);
        
    //! HipDeviceProperties destructor
    ~HipDeviceProperties() { }

    const hipDeviceProp_t& operator() () const { return mDevProp; }
	
    //! Name of the device
    const char* GetDeviceName() const { return mDevProp.name; }

    //! HIP Driver API version
    float GetDriverVersion() const { return mDriverVersion; }
                
    //! Prints detailed information on the CUdevice to std::cout
    void Print() const;
  };

} // namespace

#endif 
