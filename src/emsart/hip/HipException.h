#ifndef HIPEXCEPTION_H
#define HIPEXCEPTION_H


#include <hip/hip_runtime.h>
#include "HipDefault.h"
#include "hipfft.h" 

using namespace std;

#define hipSafeCall(err) __hipSafeCall (  err, __FILE__, __LINE__)
#define hipfftSafeCall(err) __hipfftSafeCall (err, __FILE__, __LINE__)


namespace Hip
{
  //!  An exception wrapper class for HIP hipError_t. 
  /*!
    HipException is thrown, if a HIP Driver API call via hipSafeCall does not return HIP_SUCCESS
    \author Michael Kunz
    \date   September 2011
    \version 1.1
  */
  //An exception wrapper class for HIP hipError_t. 
  class HipException: public exception
  {
  protected:
    string mFileName;
    string mMessage;
    int mLine;
    hipError_t mErr;

  public:			
    //! Default constructor
    //Default constructor
    HipException();
			
    ~HipException() throw();
			
    //! HipException constructor
    /*!
      \param aMessage Ecxeption message
    */
    //HipException constructor
    HipException(string aMessage);

    //! HipException constructor
    /*!
      \param aFileName Source code file where the exception was thrown
      \param aLine Code line where the exception was thrown
      \param aMessage Ecxeption message
      \param aErr hipError_t error code
    */
    //HipException constructor
    HipException(string aFileName, int aLine, string aMessage, hipError_t aErr);

#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)
    // for CUDA leftovers     
    HipException(string aFileName, int aLine, string aMessage, CUresult aErr){
      if( aErr == CUDA_SUCCESS )
	HipException(aFileName,aLine,aMessage,hipSuccess);
      else
	HipException(aFileName,aLine,aMessage,hipErrorInvalidValue);		    
    }
#endif
    
		
    //! Returns "HipException"
    //Returns "HipException"
    virtual const char* what() const throw();
		
    //! Returns an error message
    //Returns an error message
    virtual string GetMessage() const;
  };
	
  //! Translates a hipError_t error code into a human readable error description, if \p err is not CUDA_SUCCESS.
  /*!		
    \param file Source code file where the exception was thrown
    \param line Code line where the exception was thrown
    \param err hipError_t error code
  */
  //Translates a hipError_t error code into a human readable error description, if err is not CUDA_SUCCESS.


  inline void __hipSafeCall(hipError_t err, const char *file, const int line)
  {        
    if( hipSuccess != err)
      {
	std::string errMsg = hipGetErrorString(err);
	HipException ex(file, line, errMsg, err);
	throw ex;
      } //if CUDA_SUCCESS
  }


#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)
    // for CUDA leftovers     
  inline void __hipSafeCall(CUresult err, const char *file, const int line)
  {
    // temporary, to be deleted SG!!!
    __hipSafeCall( hipCUResultTohipError(err), file, line);
  }

  inline void __hipSafeCall(cudaError_t  err, const char *file, const int line)
  {
    // temporary, to be deleted SG!!!
    __hipSafeCall( hipCUDAErrorTohipError(err), file, line);
  }
#endif
	
  //!  An exception wrapper class for CUDA hipError_t. 
  /*!
    HipException is thrown, if a CUDA Driver API call via hipSafeCall does not return CUDA_SUCCESS
    \author Michael Kunz
    \date   September 2011
    \version 1.1
  */
  //An exception wrapper class for CUDA hipError_t. 
  class HipfftException: public exception
  {
  protected:
    string mFileName;
    string mMessage;
    int mLine;
    hipfftResult mErr;

  public:			
    //! Default constructor
    //Default constructor
    HipfftException();
			
    ~HipfftException() throw();
			
    //! HipException constructor
    /*!
      \param aMessage Ecxeption message
    */
    //HipException constructor
    HipfftException(string aMessage);

    //! HipException constructor
    /*!
      \param aFileName Source code file where the exception was thrown
      \param aLine Code line where the exception was thrown
      \param aMessage Ecxeption message
      \param aErr hipError_t error code
    */
    //HipException constructor
    HipfftException(string aFileName, int aLine, string aMessage, hipfftResult aErr);
		
    //! Returns "HipException"
    //Returns "HipException"
    virtual const char* what() const throw();
		
    //! Returns an error message
    //Returns an error message
    virtual string GetMessage() const;
  };
	

  //Translates a hipError_t error code into a human readable error description, if err is not CUDA_SUCCESS.

  inline void __hipfftSafeCall(hipfftResult err, const char *file, const int line)
  {        
    if( HIPFFT_SUCCESS != err)
      {
	std::string errMsg;
	switch(err)
	  {         
	  case HIPFFT_INVALID_PLAN:
	    errMsg = "HIPFFT_INVALID_PLAN";
	    break;       
	  case HIPFFT_ALLOC_FAILED:
	    errMsg = "HIPFFT_ALLOC_FAILED";
	    break;    
	  case HIPFFT_INVALID_TYPE:
	    errMsg = "HIPFFT_INVALID_TYPE";
	    break;    
	  case HIPFFT_INVALID_VALUE:
	    errMsg = "HIPFFT_INVALID_VALUE";
	    break;    
	  case HIPFFT_INTERNAL_ERROR:
	    errMsg = "HIPFFT_INTERNAL_ERROR";
	    break;    
	  case HIPFFT_EXEC_FAILED:
	    errMsg = "HIPFFT_EXEC_FAILED";
	    break;    
	  case HIPFFT_SETUP_FAILED:
	    errMsg = "HIPFFT_SETUP_FAILED";
	    break;    
	  case HIPFFT_INVALID_SIZE:
	    errMsg = "HIPFFT_INVALID_SIZE";
	    break;    
	  case HIPFFT_UNALIGNED_DATA:
	    errMsg = "HIPFFT_UNALIGNED_DATA";
	    break;    
	  case HIPFFT_INCOMPLETE_PARAMETER_LIST:
	    errMsg = "HIPFFT_INCOMPLETE_PARAMETER_LIST";
	    break;   
	  case HIPFFT_INVALID_DEVICE:
	    errMsg = "HIPFFT_INVALID_DEVICE";
	    break;   
	  case HIPFFT_PARSE_ERROR:
	    errMsg = "HIPFFT_PARSE_ERROR";
	    break;   
	  case HIPFFT_NO_WORKSPACE:
	    errMsg = "HIPFFT_NO_WORKSPACE";
	    break;
	  default:
	    errMsg="Not handled HIPFFT error";
	  }

	HipfftException ex(file, line, errMsg, err);
	throw ex;
      } //if CUDA_SUCCESS
  }


}

#endif //HIPEXCEPTION_H
