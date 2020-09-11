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

#ifndef FORWARDPROJECTIONSLICER_CU
#define FORWARDPROJECTIONSLICER_CU


#include <hip/hip_runtime.h>
#include "DeviceVariables.h"

//texture< ushort, 3, cudaReadModeNormalizedFloat > t_dataset;
//texture< float, 3, cudaReadModeElementType > t_dataset;

typedef unsigned long long int ulli;



extern "C"
__global__
//void slicer(DeviceReconstructionConstantsCommon param, int proj_x, int proj_y, size_t stride, float* projection, float tminDefocus, float tmaxDefocus, hipTextureObject_t tex, int2 roiMin, int2 roiMax)
void slicer( DevParamSlicer recParam )
{
  DeviceReconstructionConstantsCommon &c = recParam.common;
  //int &proj_x = recParam.proj_x; 
  //int &proj_y = recParam.proj_y; 
  size_t &stride = recParam.stride; 
  float* &projection = recParam.projection; 
  float &tminDefocus = recParam.tminDefocus;
  float &tmaxDefocus = recParam.tmaxDefocus;
  hipTextureObject_t &tex = recParam.tex; 
  int2 &roiMin = recParam.roiMin; 
  int2 &roiMax = recParam.roiMax;

  float t_in;
  float t_out;
  float4 f;      //helper variable
  float3 g;											//helper variable
  float3 source;
  //float val = 0;

  // integer pixel coordinates
  const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
  //if (x >= proj_x || y >= proj_y) return;
  if (x >= roiMax.x || y >= roiMax.y) return;
  if (x < roiMin.x || y < roiMin.y) return;


  source = c.detektor;

  //source = source + ((float)x + 0.5f) * c.uPitch;
  //source = source + ((float)y + 0.5f) * c.vPitch;
  float temp = 0.0f;
  g.z = 0;
  g.x = 0;

  //No oversampling now (to enable OS use osx = osy = 0.25f)
  for (float  osx = 0.25f; osx < 0.8f; osx+=0.5f)
    {
      for (float osy = 0.25f; osy < 0.8f; osy+=0.5f)
	{
	  float xAniso;
	  float yAniso;

	  MatrixVector3Mul(c.magAniso, (float)x + osx, (float)y + osy, xAniso, yAniso);
	  source = c.detektor;
	  source = source + (xAniso) * c.uPitch;
	  source = source + (yAniso) * c.vPitch;

	  //////////// BOX INTERSECTION (partial Volume) /////////////////
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
	  //if( t_out > t_in && t_in < t && t < t_out) //
	  if( (t_out - t_in) > 0.0f)
	    {
	      t_in = fmaxf(t_in, tminDefocus);
	      t_out = fminf(t_out, tmaxDefocus);

	      g.x++;
	      g.z += (t_out - t_in);
	      // calculate entry point
	      f.x = source.x;
	      f.y = source.y;
	      f.z = source.z;
				
	      f.w = t_in;
				
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

	      /*f.x += (t * c.projNorm.x);
		f.y += (t * c.projNorm.y);
		f.z += (t * c.projNorm.z);

		f.x = (f.x - c.bBoxMin.x) * c.volumeBBoxRcp.x * c.volumeDim.x;
		f.y = (f.y - c.bBoxMin.y) * c.volumeBBoxRcp.y * c.volumeDim.y;
		f.z = (f.z - c.bBoxMin.z) * c.volumeBBoxRcp.z * c.volumeDim.z - c.zShiftForPartialVolume;
	      *///if (x == 2048 && y == 2048)
	      //val = f.z;
	      //val += tex3D(t_dataset, f.x, f.y, f.z);
	      //float distX = 1.0f;
	      //float distY = 1.0f;
	      //float distZ = 1.0f;
	      ////dim border
	      //float filterWidth = 150.0f;
	      //if (f.y < filterWidth)
	      //{
	      //	float w = f.y / filterWidth;
	      //	if (w<0) w = 0;
	      //	distY = 1.0f - expf(-(w * w * 9.0f));
	      //}
	      //else if (f.y > c.volumeDimComplete.y - filterWidth)
	      //{
	      //	float w = (c.volumeDimComplete.y-f.y-1.0f) / filterWidth;
	      //	if (w<0) w = 0;
	      //	distY = 1.0f - expf(-(w * w * 9.0f));
	      //}

	      //if (f.x < filterWidth)
	      //{
	      //	float w = f.x / filterWidth;
	      //	if (w<0) w = 0;
	      //	distX = 1.0f - expf(-(w * w * 9.0f));
	      //}
	      //else if (f.x > c.volumeDimComplete.x - filterWidth)
	      //{
	      //	float w = (c.volumeDimComplete.x-f.x-1.0f) / filterWidth;
	      //	if (w<0) w = 0;
	      //	distX = 1.0f - expf(-(w * w * 9.0f));
	      //}

	      //if (f.z < 50.0f)
	      //{
	      //	float w = f.z / 50.0f;
	      //	if (w<0) w = 0;
	      //	distZ = 1.0f - expf(-(w * w * 9.0f));
	      //}
	      //else if (f.z > c.volumeDimComplete.z - 50.0f)
	      //{
	      //	float w = (c.volumeDimComplete.z-f.z-1.0f) / 50.0f;
	      //	if (w<0) w = 0;
	      //	distZ = 1.0f - expf(-(w * w * 9.0f));
	      //}
	      //val = val * distX * distY * distZ;
	      //val = val * (expf(-(distX * distX + distY * distY + distZ * distZ)));
				
	    }
	}
    }

  unsigned int i = (y * stride / sizeof(float)) + x;
  projection[i] += temp * 0.25f; // With Oversampling use * 0.25f
}

extern "C"
__global__
//void volTraversalLength(DeviceReconstructionConstantsCommon param, int proj_x, int proj_y, size_t stride, float* volume_traversal_length, int2 roiMin, int2 roiMax)
void volTraversalLength( DevParamVolTravLength recParam )
{

  DeviceReconstructionConstantsCommon &c = recParam.common;
  //int &proj_x = recParam.proj_x; 
  //int &proj_y = recParam.proj_y; 
  size_t &stride = recParam.stride;   
  float* &volume_traversal_length = recParam.volume_traversal_length; 
  int2 &roiMin = recParam.roiMin; 
  int2 &roiMax = recParam.roiMax;


  float t_in;
  float t_out;
  float3 source;
  float val = 0;

  //volume_traversal_length[0] = c.detektor.x;
  //volume_traversal_length[1] = c.detektor.y;
  //volume_traversal_length[2] = c.detektor.z;
  // integer pixel coordinates
  const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;


  //if (x >= proj_x || y >= proj_y) return;
  if (x >= roiMax.x || y >= roiMax.y) return;
  if (x < roiMin.x || y < roiMin.y) return;

  float xAniso;
  float yAniso;

  MatrixVector3Mul(c.magAniso, (float)x + 0.5f, (float)y + 0.5f, xAniso, yAniso);
  source = c.detektor;

  source = source + (xAniso) * c.uPitch;
  source = source + (yAniso) * c.vPitch;


  //No oversampling now (to enable OS use osx = osy = 0.25f)
  for (float  osx = 0.5f; osx < 0.8f; osx+=0.5f)
    {
      for (float osy = 0.5f; osy < 0.8f; osy+=0.5f)
	{
	  //////////// BOX INTERSECTION (partial Volume) /////////////////
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
			
	  // if the ray hits the dataset (partial Volume)
	  if( t_out > t_in)
	    {
	      val = t_out - t_in;				
	    }
	}
    }

  unsigned int i = (y * stride / sizeof(float)) + x;
  volume_traversal_length[i] += val; // With Oversampling use * 0.25f
}
#endif
