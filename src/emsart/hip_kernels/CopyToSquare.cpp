#ifndef COPYTOSQUARE_CU
#define COPYTOSQUARE_CU


#include <hip/hip_runtime.h>
#include "DeviceReconstructionParameters.h"


extern "C"
__global__ 
//void makeSquare(int proj_x, int proj_y, int maxsize, int stride, float* aIn, float* aOut, int borderSizeX, int borderSizeY, bool mirrorY, bool fillZero)
void makeSquare( DevParamCopyToSquare recParam )
{
  int &proj_x = recParam.proj_x; 
  int &proj_y = recParam.proj_y; 
  int &maxsize = recParam.maxsize; 
  int &stride = recParam.stride; 
  float* &aIn = recParam.aIn; 
  float* &aOut = recParam.aOut; 
  int &borderSizeX = recParam.borderSizeX; 
  int &borderSizeY = recParam.borderSizeY; 
  bool &mirrorY = recParam.mirrorY; 
  bool &fillZero = recParam.fillZero;

  // integer pixel coordinates	
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;	

  if (x >= maxsize || y >= maxsize)
    return;

  if (fillZero)
    {
      float val = 0;

      int xIn = x - borderSizeX;
      int yIn = y - borderSizeY;

      if (xIn >= 0 && xIn < proj_x && yIn >= 0 && yIn < proj_y)
	{
	  if (mirrorY)
	    {
	      yIn = proj_y - yIn - 1;
	    }
	  val = aIn[(yIn * stride) + xIn];
	}
      aOut[y * maxsize + x] = val;
    }
  else //wrap
    {
      int xIn = x - borderSizeX;
      if (xIn < 0) xIn = -xIn - 1;
      if (xIn >= proj_x)
	{
	  xIn = xIn - proj_x;
	  xIn = proj_x - xIn - 1;
	}

      int yIn = y - borderSizeY;
      if (yIn < 0) yIn = -yIn - 1;
      if (yIn >= proj_y)
	{
	  yIn = yIn - proj_y;
	  yIn = proj_y - yIn - 1;
	}
      if (mirrorY)
	{
	  yIn = proj_y - yIn - 1;
	}
	
      aOut[y * maxsize + x] = aIn[(yIn * stride) + xIn];
    }
  /*float a = aIn[(y * stride) + x];
    if (a > 60.0f) a = 60;
    if (a < 0) a = 0;
    aOut[y * maxsize + x] = a;*/
}

#endif
