/**********************************************
*
* CUDA SART FRAMEWORK
* 2009,2010 Michael Kunz, Lukas Marsalek
*
*
* ForwardProjectionAPriori.cu
* DDA forward projection with trilinear
* interpolation
*
**********************************************/
#ifndef FORWARDPROJECTIONRAYMARCHER_TL_CU
#define FORWARDPROJECTIONRAYMARCHER_TL_CU

   
#include <hip/hip_runtime.h>
#include "Constants.h"
#include "DeviceVariables.h"

//texture< float, 3, cudaReadModeElementType > t_dataset;


extern "C"
__global__
//void march(DeviceReconstructionConstantsCommon param, int proj_x, int proj_y, size_t stride, float* projection, float* volume_traversal_length, hipTextureObject_t tex, int2 roiMin, int2 roiMax)
void march( DevParamMarcher recParam )
{
  DeviceReconstructionConstantsCommon &c = recParam.common;
  // int &proj_x = recParam.proj_x; 
  // int &proj_y = recParam.proj_y; 
  size_t &stride = recParam.stride; 
  float* &projection = recParam.projection; 
  float* &volume_traversal_length = recParam.volume_traversal_length; 
  hipTextureObject_t &tex = recParam.tex; 
  int2 &roiMin = recParam.roiMin; 
  int2 &roiMax = recParam.roiMax;

  float t_in;
  float t_out;
  float4 f;											//helper variable
  float3 g;											//helper variable
  float3 source;

  // integer pixel coordinates
  const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

  //if (x >= proj_x || y >= proj_y) return;
  if (x >= roiMax.x || y >= roiMax.y) return;
  if (x < roiMin.x || y < roiMin.y) return;

  source = c.detektor;

  float temp = 0.0f;
  g.z = 0;
  g.x = 0;
  //No oversampling now (to enable OS use osx = osy = 0.25f)
  for (float  osx = 0.25f; osx < 0.9f; osx+=0.5f)
    {
      for (float osy = 0.25f; osy < 0.9f; osy+=0.5f)
	{
	  float xAniso;
	  float yAniso;

	  MatrixVector3Mul(c.magAniso, (float)x + osx, (float)y + osy, xAniso, yAniso);
	  source = c.detektor;
	  source = source + (xAniso) * c.uPitch;
	  source = source + (yAniso) * c.vPitch;

	  float3 tEntry;
	  tEntry.x = (c.bBoxMin.x - source.x) / (c.projNorm.x);
	  tEntry.y = (c.bBoxMin.y - source.y) / (c.projNorm.y);
	  tEntry.z = (c.bBoxMin.z - source.z) / (c.projNorm.z);

	  float3 tExit;
	  tExit.x = (c.bBoxMax.x - source.x) / (c.projNorm.x);
	  tExit.y = (c.bBoxMax.y - source.y) / (c.projNorm.y);
	  tExit.z = (c.bBoxMax.z - source.z) / (c.projNorm.z);


	  float3 tmin = fminf(tEntry, tExit);
	  float3 tmax = fmaxf(tEntry, tExit);

	  t_in  = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	  t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);

	  ////////////////////////////////////////////////////////////////

	  //default grey value
	  g.y = 0.f;

	  // if the ray hits the dataset (partial Volume)
	  if( (t_out - t_in) > 0.0f)
	    {
	      g.x++;
	      g.z += (t_out - t_in);
	      // calculate entry point
	      f.x = source.x;
	      f.y = source.y;
	      f.z = source.z;

	      f.w = t_in;

	      f.x += (f.w * c.projNorm.x);
	      f.y += (f.w * c.projNorm.y);
	      f.z += (f.w * c.projNorm.z);

	      while (t_in <= t_out)
		{
		  f.x = (f.x - c.bBoxMin.x) * c.volumeBBoxRcp.x * c.volumeDim.x;
		  f.y = (f.y - c.bBoxMin.y) * c.volumeBBoxRcp.y * c.volumeDim.y;
		  f.z = (f.z - c.bBoxMin.z) * c.volumeBBoxRcp.z * c.volumeDim.z - c.zShiftForPartialVolume;
			
		  float test = tex3D<float>(tex, f.x, f.y, f.z);
					
		  temp += test * c.voxelSize.x * 0.15f;


		  t_in += c.voxelSize.x * 0.15f;

		  f.x = source.x;
		  f.y = source.y;
		  f.z = source.z;
		  f.w = t_in ;				

		  f.x += f.w * c.projNorm.x;
		  f.y += f.w * c.projNorm.y;
		  f.z += f.w * c.projNorm.z;
		}

	    }
	}
    }

  unsigned int i = (y * stride / sizeof(float)) + x;
  projection[i] += temp * 0.25f; //  With Oversampling use * 0.25f
  volume_traversal_length[i] = fmaxf(0,g.z/g.x);
}

#endif
