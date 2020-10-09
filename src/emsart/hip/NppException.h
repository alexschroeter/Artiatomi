#ifndef NPPEXCEPTION_H
#define NPPEXCEPTION_H

#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)

#include <hip/hip_runtime.h>
#include "HipDefault.h"
#include "nppdefs.h"

using namespace std;

#define nppSafeCall(err) __nppSafeCall (err, __FILE__, __LINE__, false)

namespace Hip
{	
  //!  An exception wrapper class for HIP CUresult. 
  /*!
    HipException is thrown, if a HIP Driver API call via hipSafeCall does not return HIP_SUCCESS
    \author Michael Kunz
    \date   September 2011
    \version 1.1
  */
  //An exception wrapper class for HIP CUresult. 
  class NppException: public exception
  {
  protected:
    string mFileName;
    string mMessage;
    int mLine;
    NppStatus mErr;

  public:			
    //! Default constructor
    //Default constructor
    NppException();
			
    ~NppException() throw();
			
    //! HipException constructor
    /*!
      \param aMessage Ecxeption message
    */
    //HipException constructor
    NppException(string aMessage);

    //! HipException constructor
    /*!
      \param aFileName Source code file where the exception was thrown
      \param aLine Code line where the exception was thrown
      \param aMessage Ecxeption message
      \param aErr CUresult error code
    */
    //HipException constructor
    NppException(string aFileName, int aLine, string aMessage, NppStatus aErr);
		
    //! Returns "HipException"
    //Returns "HipException"
    virtual const char* what() const throw();
		
    //! Returns an error message
    //Returns an error message
    virtual string GetMessage() const;
  };

	
  //! Translates a CUresult error code into a human readable error description, if \p err is not HIP_SUCCESS.
  /*!		
    \param file Source code file where the exception was thrown
    \param line Code line where the exception was thrown
    \param err CUresult error code
  */
  //Translates a CUresult error code into a human readable error description, if err is not HIP_SUCCESS.
	
  inline void __nppSafeCall(NppStatus err, const char *file, const int line, bool warningAsError)
  {        
    if( NPP_SUCCESS != err)
      {
	std::string errMsg;
			
	switch(err)
	  {       
	    /* negative return-codes indicate errors */
	  case NPP_NOT_SUPPORTED_MODE_ERROR:
	    errMsg = "NPP_NOT_SUPPORTED_MODE_ERROR";
	    break;         
	  case NPP_INVALID_HOST_POINTER_ERROR:
	    errMsg = "NPP_INVALID_HOST_POINTER_ERROR";
	    break;              
	  case NPP_INVALID_DEVICE_POINTER_ERROR:
	    errMsg = "NPP_INVALID_DEVICE_POINTER_ERROR";
	    break;               
	  case NPP_LUT_PALETTE_BITSIZE_ERROR:
	    errMsg = "NPP_LUT_PALETTE_BITSIZE_ERROR";
	    break;                  
	  case NPP_ZC_MODE_NOT_SUPPORTED_ERROR:
	    errMsg = "NPP_ZC_MODE_NOT_SUPPORTED_ERROR";
	    break;                   
	  case NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY:
	    errMsg = "NPP_NOT_SUFFICIENT_COMPUTE_CAPABILITY";
	    break;                   
	  case NPP_TEXTURE_BIND_ERROR:
	    errMsg = "NPP_TEXTURE_BIND_ERROR";
	    break;                   
	  case NPP_WRONG_INTERSECTION_ROI_ERROR:
	    errMsg = "NPP_WRONG_INTERSECTION_ROI_ERROR";
	    break;                   
	  case NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR:
	    errMsg = "NPP_HAAR_CLASSIFIER_PIXEL_MATCH_ERROR";
	    break;                   
	  case NPP_MEMFREE_ERROR:
	    errMsg = "NPP_MEMFREE_ERROR";
	    break;                   
	  case NPP_MEMSET_ERROR:
	    errMsg = "NPP_MEMSET_ERROR";
	    break;                   
	  case NPP_MEMCPY_ERROR:
	    errMsg = "NPP_MEMCPY_ERROR";
	    break;                    
	  case NPP_ALIGNMENT_ERROR:
	    errMsg = "NPP_ALIGNMENT_ERROR";
	    break;                    
	  case NPP_CUDA_KERNEL_EXECUTION_ERROR:
	    errMsg = "NPP_CUDA_KERNEL_EXECUTION_ERROR";
	    break;                    
	  case NPP_ROUND_MODE_NOT_SUPPORTED_ERROR:
	    errMsg = "NPP_ROUND_MODE_NOT_SUPPORTED_ERROR";
	    break;                    
	  case NPP_QUALITY_INDEX_ERROR:
	    errMsg = "Image pixels are constant for quality index";
	    break;                    
	  case NPP_RESIZE_NO_OPERATION_ERROR:
	    errMsg = "One of the output image dimensions is less than 1 pixel";
	    break;                    
	  case NPP_NOT_EVEN_STEP_ERROR:
	    errMsg = "Step value is not pixel multiple";
	    break;                    
	  case NPP_HISTOGRAM_NUMBER_OF_LEVELS_ERROR:
	    errMsg = "Number of levels for histogram is less than 2";
	    break;                      
	  case NPP_LUT_NUMBER_OF_LEVELS_ERROR:
	    errMsg = "Number of levels for LUT is less than 2";
	    break;                      
	  case NPP_CHANNEL_ORDER_ERROR:
	    errMsg = "Wrong order of the destination channels";
	    break;                      
	  case NPP_ZERO_MASK_VALUE_ERROR:
	    errMsg = "All values of the mask are zero";
	    break;                      
	  case NPP_QUADRANGLE_ERROR:
	    errMsg = "The quadrangle is nonconvex or degenerates into triangle, line or point";
	    break;                      
	  case NPP_RECTANGLE_ERROR:
	    errMsg = "Size of the rectangle region is less than or equal to 1";
	    break;                      
	  case NPP_COEFFICIENT_ERROR:
	    errMsg = "Unallowable values of the transformation coefficients";
	    break;                      
	  case NPP_NUMBER_OF_CHANNELS_ERROR:
	    errMsg = "Bad or unsupported number of channels";
	    break;                      
	  case NPP_COI_ERROR:
	    errMsg = "Channel of interest is not 1, 2, or 3";
	    break;                         
	  case NPP_DIVISOR_ERROR:
	    errMsg = "Divisor is equal to zero";
	    break;                         
	  case NPP_CHANNEL_ERROR:
	    errMsg = "Illegal channel index";
	    break;                         
	  case NPP_STRIDE_ERROR:
	    errMsg = "Stride is less than the row length";
	    break;                         
	  case NPP_ANCHOR_ERROR:
	    errMsg = "Anchor point is outside mask";
	    break;                         
	  case NPP_MASK_SIZE_ERROR:
	    errMsg = "Lower bound is larger than upper bound";
	    break;                         
	  case NPP_RESIZE_FACTOR_ERROR:
	    errMsg = "NPP_RESIZE_FACTOR_ERROR";
	    break;                         
	  case NPP_INTERPOLATION_ERROR:
	    errMsg = "NPP_INTERPOLATION_ERROR";
	    break;                         
	  case NPP_MIRROR_FLIP_ERROR:
	    errMsg = "NPP_MIRROR_FLIP_ERROR";
	    break;                         
	  case NPP_MOMENT_00_ZERO_ERROR:
	    errMsg = "NPP_MOMENT_00_ZERO_ERROR";
	    break;                         
	  case NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR:
	    errMsg = "NPP_THRESHOLD_NEGATIVE_LEVEL_ERROR";
	    break;                            
	  case NPP_THRESHOLD_ERROR:
	    errMsg = "NPP_THRESHOLD_ERROR";
	    break;                          
	  case NPP_CONTEXT_MATCH_ERROR:
	    errMsg = "NPP_CONTEXT_MATCH_ERROR";
	    break;                          
	  case NPP_FFT_FLAG_ERROR:
	    errMsg = "NPP_FFT_FLAG_ERROR";
	    break;                          
	  case NPP_FFT_ORDER_ERROR:
	    errMsg = "NPP_FFT_ORDER_ERROR";
	    break;                          
	  case NPP_STEP_ERROR:
	    errMsg = "Step is less or equal zero";
	    break;                          
	  case NPP_SCALE_RANGE_ERROR:
	    errMsg = "NPP_SCALE_RANGE_ERROR";
	    break;                          
	  case NPP_DATA_TYPE_ERROR:
	    errMsg = "NPP_DATA_TYPE_ERROR";
	    break;                              
	  case NPP_OUT_OFF_RANGE_ERROR:
	    errMsg = "NPP_OUT_OFF_RANGE_ERROR";
	    break;                          
	  case NPP_DIVIDE_BY_ZERO_ERROR:
	    errMsg = "NPP_DIVIDE_BY_ZERO_ERROR";
	    break;                          
	  case NPP_MEMORY_ALLOCATION_ERR:
	    errMsg = "NPP_MEMORY_ALLOCATION_ERR";
	    break;                          
	  case NPP_NULL_POINTER_ERROR:
	    errMsg = "NPP_NULL_POINTER_ERROR";
	    break;                          
	  case NPP_RANGE_ERROR:
	    errMsg = "NPP_RANGE_ERROR";
	    break;                          
	  case NPP_SIZE_ERROR:
	    errMsg = "NPP_SIZE_ERROR";
	    break;                          
	  case NPP_BAD_ARGUMENT_ERROR:
	    errMsg = "NPP_BAD_ARGUMENT_ERROR";
	    break;                          
	  case NPP_NO_MEMORY_ERROR:
	    errMsg = "NPP_NO_MEMORY_ERROR";
	    break;                   
	  case NPP_NOT_IMPLEMENTED_ERROR:
	    errMsg = "NPP_NOT_IMPLEMENTED_ERROR";
	    break;                             
	  case NPP_ERROR:
	    errMsg = "NPP_ERROR";
	    break;                             
	  case NPP_ERROR_RESERVED:
	    errMsg = "NPP_ERROR_RESERVED";
	    break;  

	    /* positive return-codes indicate warnings */
				                           
	  case NPP_NO_OPERATION_WARNING:
	    errMsg = "Indicates that no operation was performed";
	    break;                          
	  case NPP_DIVIDE_BY_ZERO_WARNING:
	    errMsg = "Divisor is zero however does not terminate the execution";
	    break;                          
	  case NPP_AFFINE_QUAD_INCORRECT_WARNING:
	    errMsg = "Indicates that the quadrangle passed to one of affine warping functions doesn't have necessary properties. First 3 vertices are used, the fourth vertex discarded.";
	    break;                          
	  case NPP_WRONG_INTERSECTION_ROI_WARNING:
	    errMsg = "The given ROI has no interestion with either the source or destination ROI. Thus no operation was performed.";
	    break;                          
	  case NPP_WRONG_INTERSECTION_QUAD_WARNING:
	    errMsg = "The given quadrangle has no intersection with either the source or destination ROI. Thus no operation was performed.";
	    break;                          
	  case NPP_DOUBLE_SIZE_WARNING:
	    errMsg = "Image size isn't multiple of two. Indicates that in case of 422/411/420 sampling the ROI width/height was modified for proper processing.";
	    break;                          
	  case NPP_MISALIGNED_DST_ROI_WARNING:
	    errMsg = "Speed reduction due to uncoalesced memory accesses warning.";
	    break;
	  default:
	    errMsg = "Not handled NPP error";
	  }

	if (err < 0 || (warningAsError && err > 0))
	  {
	    NppException ex(file, line, errMsg, err);
	    throw ex;
	  }
	else
	  {
	    printf("NPP Warning: %s", errMsg.c_str());
	  }
      } //if NPP_SUCCESS
  }
}

#endif

#endif //NPPEXCEPTION_H
