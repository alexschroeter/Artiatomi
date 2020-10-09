#ifndef NppEmulatorKernel_H
#define NppEmulatorKernel_H

#include <hip/hip_runtime.h>

#ifndef NV_NPPIDEFS_H

typedef float Npp32f;
typedef uint8_t Npp8u;
typedef double Npp64f;
typedef int16_t Npp16s;
typedef uint16_t Npp16u;
typedef int32_t Npp32s;
typedef uint32_t Npp32u;

typedef struct  
{
  Npp32f  re;     
  Npp32f  im;     
} __attribute__ ((aligned (8))) Npp32fc;

typedef struct 
{
  int width;  // in pixels 
  int height; 
} NppiSize;

enum NppCmpOp {
  NPP_CMP_LESS, 	
  NPP_CMP_LESS_EQ, 	
  NPP_CMP_EQ, 	
  NPP_CMP_GREATER_EQ,
  NPP_CMP_GREATER
};

#endif // nppi types



struct DevParamNppEmulator // a struct to pass parameters to Npp Emulator kernels
{
  hipDeviceptr_t ptr1;
  hipDeviceptr_t ptr2;
  hipDeviceptr_t ptr3;
  hipDeviceptr_t ptr4;
  hipDeviceptr_t ptr5;
  NppiSize size;
  double double1;
  float float1;
  int int1;
  int int2;
  NppCmpOp cmpOp;
  Npp8u npp8u1;
};

#endif
