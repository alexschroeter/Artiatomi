//  Copyright (c) 2018, Michael Kunz and Frangakis Lab, BMLS,
//  Goethe University, Frankfurt am Main.
//  All rights reserved.
//  http://kunzmi.github.io/Artiatomi
//  
//  This file is part of the Artiatomi package.
//  
//  Artiatomi is free software: you can redistribute it and/or modify
//  it under the terms of the GNU General Public License as published by
//  the Free Software Foundation, either version 3 of the License, or
//  (at your option) any later version.
//  
//  Artiatomi is distributed in the hope that it will be useful,
//  but WITHOUT ANY WARRANTY; without even the implied warranty of
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//  GNU General Public License for more details.
//  
//  You should have received a copy of the GNU General Public License
//  along with Artiatomi. If not, see <http://www.gnu.org/licenses/>.
//  
////////////////////////////////////////////////////////////////////////


#include "Reconstructor.h"
#include "CudaKernelBinarys.h"
#include <typeinfo>
//#include "math.h"
#include <hip/hip_runtime.h>
#include <cuda_fp16.h>
//#include "cuda_kernels/DeviceVariables.cuh"

using namespace std;
using namespace Cuda;

__device__ float sinc(float x)
{
	float res = 1;
	if (x != 0)
	{
		res = sinf(M_PI * x) / (M_PI * x);
	}
	return res;
}

__device__ void MatrixVector3Mul(float3x3& M, float xIn, float yIn, float& xOut, float& yOut)
{
	xOut = M.m[0].x * xIn + M.m[0].y * yIn + M.m[0].z * 1.f;
	yOut = M.m[1].x * xIn + M.m[1].y * yIn + M.m[1].z * 1.f;
	//erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
}

__device__ float distance(float ax, float ay, float bx, float by)
{
	return sqrt((ax - bx) * (ax - bx) + (ay - by) * (ay - by));
}

__device__ float GetDistance(int2 v, int2 w, int x, int y)
{
	float l2 = (v.x - w.x) * (v.x - w.x) + (v.y - w.y) * (v.y - w.y);

	if (l2 == 0)
		return distance(v.x, v.y, x, y);
	
	float x1, y1;
	x1 = x - v.x;
	y1 = y - v.y;
	
	float x2, y2;
	x2 = w.x - v.x;
	y2 = w.y - v.y;

	float dot = x1*x2 + y1 * y2;
	
	float t = dot / l2;

	if (t < 0.0f) return distance(x, y, v.x, v.y);
	else if (t > 1.0f) return distance(x, y, w.x, w.y);
	else 
	{
		float x3, y3;
		x3 = v.x + t * (w.x - v.x);
		y3 = v.y + t * (w.y - v.y);
		return distance(x, y, x3, y3);
	}

}

extern "C" __global__ void cropBorder(int proj_x, int proj_y, size_t stride, float* image, float2 cutLength, float2 dimLength, int2 p1, int2 p2, int2 p3, int2 p4)
{
	// integer pixel coordinates	
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;	

	if (x >= proj_x || y >= proj_y)
		return;
	
    float distX = 0.0f;
    float distY = 0.0f;

	
	if ( (p2.x - p1.x)*(y - p1.y) - (p2.y - p1.y)*(x - p1.x) < 0
	  && (p3.x - p4.x)*(y - p4.y) - (p3.y - p4.y)*(x - p4.x) < 0
	  && (p1.x - p3.x)*(y - p3.y) - (p1.y - p3.y)*(x - p3.x) < 0
	  && (p4.x - p2.x)*(y - p2.y) - (p4.y - p2.y)*(x - p2.x) < 0)
	{
		distX = 1;
		distY = 1;
		
		float minDistX = 3 * proj_x;
		float minDistY = 3 * proj_y;
		minDistX = fminf(minDistX, GetDistance(p1, p2, x, y));
		minDistX = fminf(minDistX, GetDistance(p4, p3, x, y));
		minDistX = fminf(minDistX, x);
		minDistX = fminf(minDistX, proj_x - x - 1);
		
		minDistY = fminf(minDistY, GetDistance(p3, p1, x, y));
		minDistY = fminf(minDistY, GetDistance(p2, p4, x, y));
		minDistY = fminf(minDistY, y);
		minDistY = fminf(minDistY, proj_y - y - 1);


		if (minDistX < cutLength.x + dimLength.x)
		{
			float w = (minDistX - cutLength.x) / dimLength.x;
			if (w < 0) w = 0;
			distX = 1.0f - expf(-(w * w * 9.0f));
		}
		
		if (minDistY < cutLength.y + dimLength.y)
		{
			float w = (minDistY - cutLength.y) / dimLength.y;
			if (w < 0) w = 0;
			distY = 1.0f - expf(-(w * w * 9.0f));
		}
	}


	*(((float*)((char*)image + stride * y)) + x) *= distX * distY;
}

extern "C" __global__ void makeSquare(int proj_x, int proj_y, int maxsize, int stride, float* aIn, float* aOut, int borderSizeX, int borderSizeY, bool mirrorY, bool fillZero)
{
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
			val = *(((float*)((char*)aIn + stride * yIn)) + xIn);
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
	
		aOut[y * maxsize + x] = *(((float*)((char*)aIn + stride * yIn)) + xIn);
	}
}

extern "C" __global__ void march(DeviceReconstructionConstantsCommon param, int proj_x, int proj_y, size_t stride, float* projection, float* volume_traversal_length, CUtexObject tex, int2 roiMin, int2 roiMax)
{
	float t_in;
	float t_out;
	float4 f;											//helper variable
	float3 g;											//helper variable
	float3 c_source;

	// integer pixel coordinates
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= roiMax.x || y >= roiMax.y) return;
	if (x < roiMin.x || y < roiMin.y) return;

	c_source = param.detektor;

	float temp = 0.0f;
	g.z = 0;
	g.x = 0;

	//TODO:changed
	for (float  osx = 0.25f; osx < 0.9f; osx+=0.5f)//(float  osx = 0.0625f; osx < 0.9375f; osx+=0.125f)//(float  osx = 0.5f; osx < 0.9f; osx+=1.f)//(float  osx = 0.25f; osx < 0.9f; osx+=0.5f)//(float  osx = 0.125f; osx < 0.9f; osx+=0.25f)//(float  osx = 0.25f; osx < 0.9f; osx+=0.5f)
	{
		for (float osy = 0.25f; osy < 0.9f; osy+=0.5f)//(float osy = 0.125f; osy < 0.9375f; osy+=0.125f)//(float osy = 0.5f; osy < 0.9f; osy+=1.f)//(float osy = 0.25f; osy < 0.9f; osy+=0.5f)//(float osy = 0.125f; osy < 0.9f; osy+=0.25f)//(float osy = 0.25f; osy < 0.9f; osy+=0.5f)
		{
			float xAniso;
			float yAniso;

			//TODO: jitter
            //curandState stateX;
            //curand_init((unsigned long long)clock() + x, 0, 0, &stateX);
            //float jitx = curand_uniform(&stateX)-0.5f;


            //curandState stateY;
            //curand_init((unsigned long long)clock() + y, 0, 0, &stateY);
            //float jity = curand_uniform(&stateY)-0.5f;

			//MatrixVector3Mul(c_magAniso, (float)x + osx + jitx, (float)y + osy + jity, xAniso, yAniso);
            MatrixVector3Mul(param.magAniso, (float)x + osx, (float)y + osy, xAniso, yAniso);
			c_source = param.detektor;
			c_source = c_source + (xAniso) * param.uPitch;
			c_source = c_source + (yAniso) * param.vPitch;

			float3 tEntry;
			tEntry.x = (param.bBoxMin.x - c_source.x) / (param.projNorm.x);
			tEntry.y = (param.bBoxMin.y - c_source.y) / (param.projNorm.y);
			tEntry.z = (param.bBoxMin.z - c_source.z) / (param.projNorm.z);

			float3 tExit;
			tExit.x = (param.bBoxMax.x - c_source.x) / (param.projNorm.x);
			tExit.y = (param.bBoxMax.y - c_source.y) / (param.projNorm.y);
			tExit.z = (param.bBoxMax.z - c_source.z) / (param.projNorm.z);


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
				f.x = c_source.x;
				f.y = c_source.y;
				f.z = c_source.z;

				f.w = t_in;

				f.x += (f.w * param.projNorm.x);
				f.y += (f.w * param.projNorm.y);
				f.z += (f.w * param.projNorm.z);

				while (t_in <= t_out)
				{
					f.x = (f.x - param.bBoxMin.x) * param.volumeBBoxRcp.x * param.volumeDim.x;
					f.y = (f.y - param.bBoxMin.y) * param.volumeBBoxRcp.y * param.volumeDim.y;
					f.z = (f.z - param.bBoxMin.z) * param.volumeBBoxRcp.z * param.volumeDim.z - param.zShiftForPartialVolume;
			
					float test = tex3D<float>(tex, f.x, f.y, f.z);

					//TODO:add gaussian
					//float d = sqrtf((osx-0.5f)*(osx-0.5f) + (osy-0.5f)*(osy-0.5f));
					temp += test * param.voxelSize.x * 0.15f; //* expf(- (d/0.25f) * (d/0.25f));


					t_in += param.voxelSize.x * 0.15f;

					f.x = c_source.x;
					f.y = c_source.y;
					f.z = c_source.z;
					f.w = t_in ;				

					f.x += f.w * param.projNorm.x;
					f.y += f.w * param.projNorm.y;
					f.z += f.w * param.projNorm.z;
				}
			}
		}
	}
    //TODO:changed
	*(((float*)((char*)projection + stride * y)) + x) += temp / 4.f;//1.f;//0.25f;//0.125f;//0.25f; //  With Oversampling use * 0.25f
	*(((float*)((char*)volume_traversal_length + stride * y)) + x) = fmaxf(0,g.z/g.x);
}

extern "C" __global__ void fourierFilter(float2* img, size_t stride, int pixelcount, float lp, float hp, float lps, float hps)
{
	//compute x,y indices 
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= pixelcount / 2 + 1) return;
	if (y >= pixelcount) return;

	float mx = (float)x;
	float my = (float)y;
	if (my > pixelcount * 0.5f)
		my = (pixelcount - my) * -1.0f;

	float dist = sqrtf(mx * mx + my * my);
	float fil = 0;

	lp = lp - lps;
	hp = hp + hps;
	//Low pass
	if (lp > 0)
	{
		if (dist <= lp) fil = 1;
	}
	else
	{
		if (dist <= pixelcount / 2 - 1) fil = 1;
	}
	//Gauss
	if (lps > 0)
	{
		float fil2;
		if (dist < lp) fil2 = 1;
		else fil2 = 0;

		fil2 = (-fil + 1.0f) * (float)expf(-((dist - lp) * (dist - lp) / (2 * lps * lps)));
		if (fil2 > 0.001f)
			fil = fil2;
	}

	if (lps > 0 && lp == 0 && hp == 0 && hps == 0)
		fil = (float)expf(-((dist - lp) * (dist - lp) / (2 * lps * lps)));

	if (hp > 0)
	{
		float fil2 = 0;
		if (dist >= hp) fil2 = 1;

		fil *= fil2;

		if (hps > 0)
		{
			float fil3 = 0;
			if (dist < hp) fil3 = 1;
			fil3 = (-fil2 + 1) * (float)expf(-((dist - hp) * (dist - hp) / (2 * hps * hps)));
			if (fil3 > 0.001f)
				fil = fil3;
		}
	}

	float2 erg = *(((float2*)((char*)img + stride * y)) + x);
	erg.x *= fil;
	erg.y *= fil;
	*(((float2*)((char*)img + stride * y)) + x) = erg;
}

extern "C" __global__ void doseWeighting(float2* img, size_t stride, int pixelcount, float dose, float pixelsize)
{
	//compute x,y indices 
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= pixelcount / 2 + 1) return;
	if (y >= pixelcount) return;

	float mx = (float)x;
	float my = (float)y;
	if (my > pixelcount * 0.5f)
		my = (pixelcount - my) * -1.0f;

	float dist = sqrtf(mx * mx + my * my);
	float fil = 0;

	dist = dist / (pixelcount / 2 / pixelsize);
	fil = expf(-dose * dist);

	float2 erg = *(((float2*)((char*)img + stride * y)) + x);
	erg.x *= fil;
	erg.y *= fil;
	*(((float2*)((char*)img + stride * y)) + x) = erg;
}

extern "C" __global__ void slicer(DeviceReconstructionConstantsCommon param, int proj_x, int proj_y, size_t stride, float* projection, float tminDefocus, float tmaxDefocus, CUtexObject tex, int2 roiMin, int2 roiMax)
{
	float t_in;
	float t_out;
	float4 f;   //helper variable
	float3 g;	//helper variable
	float3 c_source;

	// integer pixel coordinates
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if (x >= roiMax.x || y >= roiMax.y) return;
	if (x < roiMin.x || y < roiMin.y) return;

	c_source = param.detektor;

	float temp = 0.0f;
	g.z = 0;
	g.x = 0;

	for (float  osx = 0.25f; osx < 0.8f; osx+=0.5f)
	{
		for (float osy = 0.25f; osy < 0.8f; osy+=0.5f)
		{
			float xAniso;
			float yAniso;

			MatrixVector3Mul(param.magAniso, (float)x + osx, (float)y + osy, xAniso, yAniso);
			c_source = param.detektor;
			c_source = c_source + (xAniso) * param.uPitch;
			c_source = c_source + (yAniso) * param.vPitch;

			//////////// BOX INTERSECTION (partial Volume) /////////////////
			float3 tEntry;
			tEntry.x = (param.bBoxMin.x - c_source.x) / (param.projNorm.x);
			tEntry.y = (param.bBoxMin.y - c_source.y) / (param.projNorm.y);
			tEntry.z = (param.bBoxMin.z - c_source.z) / (param.projNorm.z);

			float3 tExit;
			tExit.x = (param.bBoxMax.x - c_source.x) / (param.projNorm.x);
			tExit.y = (param.bBoxMax.y - c_source.y) / (param.projNorm.y);
			tExit.z = (param.bBoxMax.z - c_source.z) / (param.projNorm.z);


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
				f.x = c_source.x;
				f.y = c_source.y;
				f.z = c_source.z;
				
				f.w = t_in;
				
				while (t_in <= t_out)
				{
					f.x = (f.x - param.bBoxMin.x) * param.volumeBBoxRcp.x * param.volumeDim.x;
					f.y = (f.y - param.bBoxMin.y) * param.volumeBBoxRcp.y * param.volumeDim.y;
					f.z = (f.z - param.bBoxMin.z) * param.volumeBBoxRcp.z * param.volumeDim.z - param.zShiftForPartialVolume;
			
					float test = tex3D<float>(tex, f.x, f.y, f.z);
					
					temp += test * param.voxelSize.x * 0.15f;


					t_in += param.voxelSize.x * 0.15f;

					f.x = c_source.x;
					f.y = c_source.y;
					f.z = c_source.z;
					f.w = t_in ;				

					f.x += f.w * param.projNorm.x;
					f.y += f.w * param.projNorm.y;
					f.z += f.w * param.projNorm.z;
				}				
			}
		}
	}

	*(((float*)((char*)projection + stride * y)) + x) += temp * 0.25f; // With Oversampling use * 0.25f	
}

extern "C" __global__ void volTraversalLength(DeviceReconstructionConstantsCommon param, int proj_x, int proj_y, size_t stride, float* volume_traversal_length, int2 roiMin, int2 roiMax)
{
	float t_in;
	float t_out;
	float3 c_source;
	float val = 0;

	// integer pixel coordinates
	const unsigned int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	const unsigned int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x >= roiMax.x || y >= roiMax.y) return;
	if (x < roiMin.x || y < roiMin.y) return;

	float xAniso;
	float yAniso;

	MatrixVector3Mul(param.magAniso, (float)x + 0.5f, (float)y + 0.5f, xAniso, yAniso);
	c_source = param.detektor;

	c_source = c_source + (xAniso) * param.uPitch;
	c_source = c_source + (yAniso) * param.vPitch;


	for (float  osx = 0.5f; osx < 0.8f; osx+=0.5f)
	{
		for (float osy = 0.5f; osy < 0.8f; osy+=0.5f)
		{
			//////////// BOX INTERSECTION (partial Volume) /////////////////
			float3 tEntry;
			tEntry.x = (param.bBoxMin.x - c_source.x) / (param.projNorm.x);
			tEntry.y = (param.bBoxMin.y - c_source.y) / (param.projNorm.y);
			tEntry.z = (param.bBoxMin.z - c_source.z) / (param.projNorm.z);

			float3 tExit;
			tExit.x = (param.bBoxMax.x - c_source.x) / (param.projNorm.x);
			tExit.y = (param.bBoxMax.y - c_source.y) / (param.projNorm.y);
			tExit.z = (param.bBoxMax.z - c_source.z) / (param.projNorm.z);


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

	*(((float*)((char*)volume_traversal_length + stride * y)) + x) += val; // With Oversampling use * 0.25f
	
}


extern "C" __global__ void compare(int proj_x, int proj_y, size_t stride, float* real_raw, float* virtual_raw, float* vol_distance_map, float realLength, float4 cutLength, float4 dimLength, float projValScale)
{
	// integer pixel coordinates	
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	

	if (x >= proj_x || y >= proj_y)
		return;


	// save error difference in virtual projection

	float val = *(((float*)((char*)vol_distance_map + stride * y)) + x);
	float error;

    float distXA = 1.0f;
    float distXB = 1.0f;
    float distYA = 1.0f;
    float distYB = 1.0f;

	if (val >= 1.0f)
	{
		error = ((*(((float*)((char*)real_raw + stride * y)) + x)) - ((*(((float*)((char*)virtual_raw + stride * y)) + x) / projValScale) / val * realLength )) / realLength * projValScale;
	}
	else
	{
		error = 0;
	}

	//dim border
	if (y < cutLength.z + dimLength.z)
	{
		float w = (y - cutLength.z) / dimLength.z;
		if (w<0) w = 0;
		distYB = 1.0f - expf(-(w * w * 9.0f));
	}
	else 
    {
        distYB = 1.0f;
    }
		
	if (y > proj_y - dimLength.w-cutLength.w - 1)
	{

        // incorrect:
        //float w = ((proj_y - y - 1) - cutLength.w - (proj_y - 1)) / dimLength.w;

        //correct:
        float w = (proj_y - 1 - y - cutLength.w) / dimLength.w;

		if (w<0) w = 0.0f;
		distYA = 1.0f - expf(-(w * w * 9.0f));
	}
    else
    {
        distYA = 1.0f;
    }

	if (x < cutLength.y + dimLength.y)
	{
		float w = (x - cutLength.y) / dimLength.y;
		if (w<0) w = 0;
		distXB = 1.0f - expf(-(w * w * 9.0f));
	}
	else
    {
        distXB = 1.0f;
    }

	if (x > proj_x - dimLength.x-cutLength.x - 1)
	{

        // incorrect:
        //float w = ((proj_x - x - 1) - cutLength.x - (proj_x - 1)) / dimLength.x;

        // correct:
        float w = (proj_x - 1 - x - cutLength.x)/dimLength.x;

		if (w<0) w = 0.0f;
		distXA = 1.0f - expf(-(w * w * 9.0f));
	}
    else
    {
        distXA = 1.0f;
    }

	*(((float*)((char*)virtual_raw + stride * y)) + x) = distXA * distXB * distYA * distYB * error;
}

extern "C" __global__ void subtract_error(int proj_x, int proj_y, size_t stride, float* real_raw, const float* error, const float* vol_distance_map)
{
    // integer pixel coordinates
    const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= proj_x || y >= proj_y)
        return;

    unsigned int i = (y * stride / sizeof(float)) + x;

    // Is pixel covered by volume?
    float val = vol_distance_map[i];

    // If yes, subtract error, if no set 0
    if (val >= 1.0f)
    {
        real_raw[i] = real_raw[i] - error[i];
    }
    else
    {
        real_raw[i] = 0.;
    }
}

extern "C" __global__ void wbpWeighting(cuComplex* img, size_t stride, unsigned int pixelcount, float psiAngle, FilterMethod fm, int proj_index, int projectionCount, float thickness, const float* __restrict__ tiltAngles)
{
	//compute x,y,z indiced
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x >= pixelcount/2 + 1) return;
	if (y >= pixelcount) return;

	float xpos = x;
	float ypos = y;
	if (ypos > pixelcount * 0.5f)
		ypos = (pixelcount - ypos) * -1.0f;

	float temp = xpos;
	float sinus =  __sinf(psiAngle);
	float cosin =  __cosf(psiAngle);

	xpos = cosin * xpos - sinus * ypos;
	ypos = sinus * temp + cosin * ypos;

	float length = ypos / (pixelcount / 2.0f);
	float weight = 1;
	switch (fm)
	{
	case FM_RAMP:
		weight = fminf(abs(length), 1.0f);
		break;
	case FM_EXACT:
		{
			//psiAngle += M_PI * 0.5f;
			float x_st = -ypos * cos(tiltAngles[proj_index])*sin(psiAngle) - xpos *cos(psiAngle);
			float y_st =  ypos * cos(tiltAngles[proj_index])*cos(psiAngle) - xpos *sin(psiAngle);
			float z_st =  ypos * sin(tiltAngles[proj_index]);

            //float x_st = -xpos * cos(tiltAngles[proj_index])*sin(psiAngle) - ypos *cos(psiAngle);
            //float y_st =  xpos * cos(tiltAngles[proj_index])*cos(psiAngle) - ypos *sin(psiAngle);
            //float z_st =  xpos * sin(tiltAngles[proj_index]);

			float w = 0;

			for (int tilt = 0; tilt < projectionCount; tilt++)
			{
				if (tilt != proj_index && tiltAngles[tilt] != -999.0f)
				{
					// Berechnung der geometrischen Distanz zu der Ebene
					float d_tmp = x_st*sin(tiltAngles[tilt])*sin(psiAngle) - y_st*sin(tiltAngles[tilt])*cos(psiAngle) + z_st*cos(tiltAngles[tilt]);

					float d2 = abs(sin(tiltAngles[tilt])) * thickness + cos(tiltAngles[tilt]);

					if (abs(d_tmp) > d2)
						d_tmp = d2;

					w += sinc(d_tmp / d2);
				}
			}
			// Normalize, such that the center is 1 / number of projections, and
			// the boundary is one!!
			w += 1.0f;
			w = 1.0f / w;

			//Added normalization(zero frequencies set to zero)
			if (ypos == 0)
			{
				w = 0;
			}
			weight = w;
		}
		break;
	case FM_CONTRAST2:
		{//1.000528623371163   0.006455924123082   0.005311341463650   0.001511856638478 1024
		 //1.000654227857550   0.006008581017124   0.004159659493151   0.000975903396538 1856
			const float p1 = 1.000654227857550f;
			const float p2 = 0.006008581017124f;
			const float p3 = 0.004159659493151f;
			const float p4 = 0.000975903396538f;
			if (length == 0)
			{
				weight = 0;
			}
			else
			{
				float logfl = logf(abs(length));
				weight = p1 + p2 * logfl + p3 * logfl * logfl + p4 * logfl * logfl * logfl;
			}
			weight = fmaxf(0, fminf(weight, 1));
		}
		break;
	case FM_CONTRAST10:		
		{//1.001771328635575   0.019634409648661   0.014871972759515   0.004962873817517 1024
		 //1.003784816598589   0.029016377161629   0.019582940715148   0.004559409669984 1856
			const float p1 = 1.003784816598589f;
			const float p2 = 0.029016377161629f;
			const float p3 = 0.019582940715148f;
			const float p4 = 0.004559409669984f;
			if (length == 0)
			{
				weight = 0;
			}
			else
			{
				float logfl = logf(abs(length));
				weight = p1 + p2 * logfl + p3 * logfl * logfl + p4 * logfl * logfl * logfl;
			}
			weight = fmaxf(0, fminf(weight, 1));
		}
		break;
	case FM_CONTRAST30:		
		{//0.998187224092783   0.019542575617926   0.010359773048706   0.006975890938967 1024
		 //0.999884616010943   0.000307646262566   0.004742915272196   0.004806551368900 1856
			const float p1 = 0.999884616010943f;
			const float p2 = 0.000307646262566f;
			const float p3 = 0.004742915272196f;
			const float p4 = 0.004806551368900f;
			if (length == 0)
			{
				weight = 0;
			}
			else
			{
				float logfl = logf(abs(length));
				weight = p1 + p2 * logfl + p3 * logfl * logfl + p4 * logfl * logfl * logfl;
			}
			weight = fmaxf(0, fminf(weight, 1));
		}
		break;
	}
	
	cuComplex res = *(((cuComplex*)((char*)img + stride * y)) + x);
	res.x *= weight;
	res.y *= weight;
	/*res.x = weight;
	res.y = 0;*/

	*(((cuComplex*)((char*)img + stride * y)) + x) = res;
}


extern "C" __global__ void conjMul(float2* complxA, float2* complxB, size_t stride, int pixelcount)
{
	//compute x,y,z indiced 
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= pixelcount / 2 + 1) return;
	if (y >= pixelcount) return;

	float2 a = *(((float2*)((char*)complxA + stride * y)) + x);
	float2 b = *(((float2*)((char*)complxB + stride * y)) + x);
	float2 erg;
	//conj. complex of a: -a.y
	erg.x = a.x * b.x + a.y * b.y;
	erg.y = a.x * b.y - a.y * b.x;
	*(((float2*)((char*)complxA + stride * y)) + x) = erg;
}

extern "C" __global__ void conjMulPC(float2* complxA, float2* complxB, size_t stride, int pixelcount)
{
    //compute x,y,z indiced
    unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= pixelcount / 2 + 1) return;
    if (y >= pixelcount) return;

    float2 a = *(((float2*)((char*)complxA + stride * y)) + x);
    float2 b = *(((float2*)((char*)complxB + stride * y)) + x);
    float2 erg;

    //conj. complex of a: -a.y
    erg.x = a.x * b.x + a.y * b.y;
    erg.y = a.x * b.y - a.y * b.x;

    float amplitude = sqrtf(erg.x * erg.x + erg.y * erg.y);

    if (amplitude != 0)
    {
        erg.x /= amplitude;
        erg.y /= amplitude;
    }
    else
    {
        erg.x = erg.y = 0;
    }

    *(((float2*)((char*)complxA + stride * y)) + x) = erg;
}


#define M_PI       3.14159265358979323846f
//#define _voltage (300.0f)
#define h ((float)6.63E-34) //Planck's quantum
#define c ((float)3.00E+08) //Light speed
#define CS (param.cs * 0.001f)
#define Cc (param.cs * 0.001f)
				
#define PhaseShift (0)
#define EnergySpread (0.7f) //eV
#define E0 (511) //keV
#define RelativisticCorrectionFactor ((1 + param.voltage / (E0 * 1000))/(1 + ((param.voltage*1000) / (2 * E0 * 1000))))
#define H ((Cc * EnergySpread * RelativisticCorrectionFactor) / (param.voltage * 1000))

#define a1 (1.494f) //Scat.Profile Carbon Amplitude 1
#define a2 (0.937f) //Scat.Profile Carbon Amplitude 2
#define b1 (23.22f * (float)1E-20) //Scat.Profile Carbon Halfwidth 1
#define b2 (3.79f * (float)1E-20)  //Scat.Profile Carbon Halfwidth 2

#define lambda ((h * c) / sqrtf(((2 * E0 * param.voltage * 1000.0f * 1000.0f) + (param.voltage * param.voltage * 1000.0f * 1000.0f)) * 1.602E-19 * 1.602E-19))

// __device__ __constant__ float c_cs;
// __device__ __constant__ float c_voltage;
// __device__ __constant__ float c_openingAngle;
// __device__ __constant__ float c_ampContrast;
// __device__ __constant__ float c_phaseContrast;
// __device__ __constant__ float c_pixelsize;
// __device__ __constant__ float c_pixelcount;
// __device__ __constant__ float c_maxFreq;
// __device__ __constant__ float c_freqStepSize;
// //__device__ __constant__ float c_lambda;
// __device__ __constant__ float c_applyScatteringProfile;
// __device__ __constant__ float c_applyEnvelopeFunction;

extern "C" __global__ void ctf(DeviceReconstructionConstantsCtf param, cuComplex* ctf, size_t stride, float defocusMin, float defocusMax, float angle, bool applyForFP, bool phaseFlipOnly, float WienerFilterNoiseLevel, float4 betaFac)
{
	//compute x,y indiced
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;	
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	
	if (x >= param.pixelcount/2 + 1) return;
	if (y >= param.pixelcount) return;

			
	float xpos = x;
	float ypos = y;
	if (ypos > param.pixelcount * 0.5f)
		ypos = (param.pixelcount - ypos) * -1.0f;
	
	float alpha;
	if (xpos == 0)
		alpha = (M_PI * 0.5f);
	else
		alpha = (atan2(ypos , xpos));
		
	float beta = ((alpha - angle));
	
	float def0 = defocusMin;
	float def1 = defocusMax;

	float defocus = def0 + (1 - cos(2*beta)) * (def1 - def0) * 0.5f;

	float length = sqrtf(xpos * xpos + ypos * ypos);

    length *= param.freqStepSize;

    float o = expf(-14.238829f * (param.openingAngle * param.openingAngle * ((CS * lambda * lambda * length * length * length - defocus * length) * (CS * lambda * lambda * length * length * length - defocus * length))));
    float p = expf(-((0.943359f * lambda * length * length * H) * (0.943359f * lambda * length * length * H)));
    float q = (a1 * expf(-b1 * (length * length)) + a2 * expf(-b2 * (length * length))) / 2.431f;

    float m = -PhaseShift + (M_PI / 2.0f) * (CS * lambda * lambda * lambda * length * length * length * length - 2 * defocus * lambda * length * length);
    float n = param.phaseContrast * sinf(m) + param.ampContrast * cosf(m);
	
	cuComplex res = *(((cuComplex*)((char*)ctf + stride * y)) + x);
	
    if (applyForFP && sqrtf(xpos * xpos + ypos * ypos) > betaFac.x && !phaseFlipOnly)// && length < 317382812)
    {
		length = length / 100000000.0f;
		float coeff1 = betaFac.y;
		float coeff2 = betaFac.z;
		float coeff3 = betaFac.w;
		float expfun = expf((-coeff1 * length - coeff2 * length * length - coeff3 * length * length * length));
		expfun = max(expfun, 0.01f);
		float val = n * expfun;
		if (abs(val) < 0.0001f && val >=0 ) val = 0.0001f;
		if (abs(val) < 0.0001f && val < 0 ) val = -0.0001f;
		
		
		res.x = res.x * -val;
		res.y = res.y * -val;
    }

    if (!applyForFP && sqrtf(xpos * xpos + ypos * ypos) > betaFac.x && !phaseFlipOnly)// && length < 317382812)
    {
		length = length / 100000000.0f;
		float coeff1 = betaFac.y;
		float coeff2 = betaFac.z;
		float coeff3 = betaFac.w;
		float expfun = expf((-coeff1 * length - coeff2 * length * length - coeff3 * length * length * length));
		expfun = max(expfun, WienerFilterNoiseLevel);
		float val = n * expfun;
		
		res.x = res.x * -val / (val * val + WienerFilterNoiseLevel);
		res.y = res.y * -val / (val * val + WienerFilterNoiseLevel);
    }
    
	if (phaseFlipOnly)
	{
		if (n >= 0)
		{
			res.x = -res.x;
			res.y = -res.y;
		}
	}
	*(((cuComplex*)((char*)ctf + stride * y)) + x) = res;
}

extern "C" __global__ void maxShift(float* img, size_t stride, int pixelcount, int maxShift)
{
	//compute x,y,z indiced 
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= pixelcount) return;
	if (y >= pixelcount) return;

	float dist = 0;
	float mx = x;
	float my = y;

	if (mx > pixelcount / 2)
		mx = pixelcount - mx;
	
	if (my > pixelcount / 2)
		my = pixelcount - my;

	dist = sqrtf(mx * mx + my * my);

	if (dist > maxShift)
	{
		*(((float*)((char*)img + stride * y)) + x) = 0;
	}
}

extern "C" __global__ void dimBorders(int proj_x, int proj_y, size_t stride, float* image, float4 cutLength, float4 dimLength)
{
	// integer pixel coordinates
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= proj_x || y >= proj_y)
		return;


	float pixel = *(((float*)((char*)image + stride * y)) + x);
	float distXA = 1.0f;
	float distXB = 1.0f;
	float distYA = 1.0f;
	float distYB = 1.0f;


	//dim border
	if (y < cutLength.z + dimLength.z)
	{
		float w = (y - cutLength.z) / dimLength.z;
		if (w < 0) w = 0;
		distYB = 1.0f - expf(-(w * w * 9.0f));
	}
	else
	{
		distYB = 1.0f;
	}

	if (y > proj_y - dimLength.w - cutLength.w - 1)
	{

        // incorrect:
        //float w = ((proj_y - y - 1) - cutLength.w - (proj_y - 1)) / dimLength.w;

        //correct:
        float w = (proj_y - 1 - y - cutLength.w) / dimLength.w;

		if (w < 0) w = 0.0f;
		distYA = 1.0f - expf(-(w * w * 9.0f));
	}
	else
	{
		distYA = 1.0f;
	}

	if (x < cutLength.y + dimLength.y)
	{
		float w = (x - cutLength.y) / dimLength.y;
		if (w < 0) w = 0;
		distXB = 1.0f - expf(-(w * w * 9.0f));
	}
	else
	{
		distXB = 1.0f;
	}

	if (x > proj_x - dimLength.x - cutLength.x - 1)
	{

        // incorrect:
        //float w = ((proj_x - x - 1) - cutLength.x - (proj_x - 1)) / dimLength.x;

        // correct:
        float w = (proj_x - 1 - x - cutLength.x)/dimLength.x;

		if (w < 0) w = 0.0f;
		distXA = 1.0f - expf(-(w * w * 9.0f));
	}
	else
	{
		distXA = 1.0f;
	}


	*(((float*)((char*)image + stride * y)) + x) = distXA * distXB * distYA * distYB * pixel;
}

// transform vector by matrix
__device__ void MatrixVector3Mul(float4x4 M, float3* v)
{
	float3 erg;
	erg.x = M.m[0].x * v->x + M.m[0].y * v->y + M.m[0].z * v->z + 1.f * M.m[0].w;
	erg.y = M.m[1].x * v->x + M.m[1].y * v->y + M.m[1].z * v->z + 1.f * M.m[1].w;
	erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
	*v = erg;
}

// transform vector by matrix
__device__
void MatrixVector3Mul(float4x4 M, float3& v, float2& erg)
{
	erg.x = M.m[0].x * v.x + M.m[0].y * v.y + M.m[0].z * v.z + 1.f * M.m[0].w;
	erg.y = M.m[1].x * v.x + M.m[1].y * v.y + M.m[1].z * v.z + 1.f * M.m[1].w;
}

extern volatile __shared__ unsigned char sBuffer[];

extern "C" __global__ void backProjection(DeviceReconstructionConstantsCommon param, int proj_x, int proj_y, float my_lambda, int maxOverSample, float maxOverSampleInv, CUtexObject img, CUsurfObject surfref, float distMin, float distMax)
{
	float3 ray;
	float2 pixel;
	float2 borderMin;
	float2 borderMax;
	float3 hitPoint;
	float3 c_source;

	int4 pixelBorders; //--> x = x.min; y = x.max; z = y.min; w = y.max
	
	// index to access shared memory, e.g. thread linear address in a block
	const unsigned int index2 = (threadIdx.z * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= param.volumeDim_x_quarter || y >= param.volumeDim.y || z >= param.volumeDim.z) return;

	//summed up distance per voxel in voxelBlock in shared memory
	volatile float4* distanceD = (float4*)(sBuffer);
	
	//Correction term per voxel in shared memory
	volatile float4* voxelD = distanceD + blockDim.x * blockDim.y * blockDim.z;
	
	
	float4 voxelBlock;

	float3 MC_bBoxMin;
	float3 MC_bBoxMax;
	float t;
	
	float t_in, t_out;
	float3 tEntry;
	float3 tExit;
	float3 tmin, tmax;
	float pixel_y, pixel_x;	

	surf3Dread(&voxelBlock.x, surfref, x * 4 * 4 + 0, y, z);
	surf3Dread(&voxelBlock.y, surfref, x * 4 * 4 + 4, y, z);
	surf3Dread(&voxelBlock.z, surfref, x * 4 * 4 + 8, y, z);
	surf3Dread(&voxelBlock.w, surfref, x * 4 * 4 + 12, y, z);


	//adopt x coordinate to single voxels, not voxelBlocks
	x = x * 4;

	//MacroCell bounding box:
	MC_bBoxMin.x = param.bBoxMin.x + (x) * param.voxelSize.x;
	MC_bBoxMin.y = param.bBoxMin.y + (y) * param.voxelSize.y;
	MC_bBoxMin.z = param.bBoxMin.z + (z) * param.voxelSize.z;
	MC_bBoxMax.x = param.bBoxMin.x + ( x + 4) * param.voxelSize.x;
	MC_bBoxMax.y = param.bBoxMin.y + ( y + 1) * param.voxelSize.y;
	MC_bBoxMax.z = param.bBoxMin.z + ( z + 1) * param.voxelSize.z;

	
	//find maximal projection on detector:
	borderMin = make_float2(FLT_MAX, FLT_MAX);
	borderMax = make_float2(-FLT_MAX, -FLT_MAX);


	//The loop has been manually unrolled: nvcc cannot handle inner loops
	//first corner
	t = (param.projNorm.x * MC_bBoxMin.x + param.projNorm.y * MC_bBoxMin.y + param.projNorm.z * MC_bBoxMin.z);
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);

	if (!(t >= distMin && t < distMax)) return;

	hitPoint.x = t * (-param.projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-param.projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-param.projNorm.z) + MC_bBoxMin.z;

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//second corner
	t = (param.projNorm.x * MC_bBoxMin.x + param.projNorm.y * MC_bBoxMin.y + param.projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-param.projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-param.projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//third corner
	t = (param.projNorm.x * MC_bBoxMin.x + param.projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + param.projNorm.z * MC_bBoxMin.z);
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-param.projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-param.projNorm.z) + MC_bBoxMin.z;

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//fourth corner
	t = (param.projNorm.x * MC_bBoxMin.x + param.projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + param.projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-param.projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-param.projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//fifth corner
	t = (param.projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + param.projNorm.y * MC_bBoxMin.y + param.projNorm.z * MC_bBoxMin.z);
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-param.projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-param.projNorm.z) + MC_bBoxMin.z;

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//sixth corner
	t = (param.projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + param.projNorm.y * MC_bBoxMin.y + param.projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-param.projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-param.projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//seventh corner
	t = (param.projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + param.projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + param.projNorm.z * MC_bBoxMin.z);
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-param.projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-param.projNorm.z) + MC_bBoxMin.z;

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//eighth corner
	t = (param.projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + param.projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + param.projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-param.projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-param.projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);


	//--> pixelBorders.x = x.min; pixelBorders.z = y.min;
	pixelBorders.x = floor(borderMin.x);
	pixelBorders.z = floor(borderMin.y);
	
	//--> pixelBorders.y = x.max; pixelBorders.v = y.max
	hitPoint = fmaxf(MC_bBoxMin, MC_bBoxMax);
	pixelBorders.y = ceil(borderMax.x);
	pixelBorders.w = ceil(borderMax.y);

	//clamp values
	pixelBorders.x = fminf(fmaxf(pixelBorders.x, 0), proj_x);
	pixelBorders.y = fminf(fmaxf(pixelBorders.y, 0), proj_x);
	pixelBorders.z = fminf(fmaxf(pixelBorders.z, 0), proj_y);
	pixelBorders.w = fminf(fmaxf(pixelBorders.w, 0), proj_y);
	
	voxelD[index2].x  = 0;
	voxelD[index2].y  = 0;
	voxelD[index2].z  = 0;
	voxelD[index2].w  = 0;
	distanceD[index2].x  = 0;
	distanceD[index2].y  = 0;
	distanceD[index2].z  = 0;
	distanceD[index2].w  = 0;

	//Loop over detected pixels and shoot rays back	again with manual unrolling
	for( pixel_y = pixelBorders.z + maxOverSampleInv*0.5f ; pixel_y < pixelBorders.w ; pixel_y+=maxOverSampleInv)
	{				
		for ( pixel_x = pixelBorders.x + maxOverSampleInv*0.5f ; pixel_x < pixelBorders.y ; pixel_x+=maxOverSampleInv)	
		{
			float xAniso;
			float yAniso;

			MatrixVector3Mul(param.magAnisoInv, pixel_x, pixel_y, xAniso, yAniso);

			ray.x = param.detektor.x; 
			ray.y = param.detektor.y; 
			ray.z = param.detektor.z; 
			
			ray.x = ray.x + (pixel_x) * param.uPitch.x;
			ray.y = ray.y + (pixel_x) * param.uPitch.y;
			ray.z = ray.z + (pixel_x) * param.uPitch.z;
			
			ray.x = ray.x + (pixel_y) * param.vPitch.x;
			ray.y = ray.y + (pixel_y) * param.vPitch.y;
			ray.z = ray.z + (pixel_y) * param.vPitch.z;
			
			c_source.x = ray.x + 100000.0 * param.projNorm.x;
			c_source.y = ray.y + 100000.0 * param.projNorm.y;
			c_source.z = ray.z + 100000.0 * param.projNorm.z;
			ray.x = ray.x - c_source.x;
			ray.y = ray.y - c_source.y;
			ray.z = ray.z - c_source.z;

			// calculate ray direction
			ray = normalize(ray);
				
			//////////// BOX INTERSECTION (Voxel 1) /////////////////	
			tEntry.x = (param.bBoxMin.x + (x  ) * param.voxelSize.x);
			tEntry.y = (param.bBoxMin.y + (y  ) * param.voxelSize.y);
			tEntry.z = (param.bBoxMin.z + (z  ) * param.voxelSize.z);
			tEntry.x = (tEntry.x - c_source.x) / ray.x;
			tEntry.y = (tEntry.y - c_source.y) / ray.y;
			tEntry.z = (tEntry.z - c_source.z) / ray.z;

			tExit.x = (param.bBoxMin.x + (x+1  ) * param.voxelSize.x);
			tExit.y = (param.bBoxMin.y + (y+1  ) * param.voxelSize.y);
			tExit.z = (param.bBoxMin.z + (z+1  ) * param.voxelSize.z);

			tExit.x = ((tExit.x) - c_source.x) / ray.x;
			tExit.y = ((tExit.y) - c_source.y) / ray.y;
			tExit.z = ((tExit.z) - c_source.z) / ray.z;

			tmin = fminf(tEntry, tExit);
			tmax = fmaxf(tEntry, tExit);
			
			t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
			t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
			////////////////////////////////////////////////////////////////
				
			// if the ray hits the voxel
			if((t_out - t_in) > 0.0f )
			{
				#ifdef CONST_LENGTH_MODE
				voxelD[index2].x += (tex2D<float>(img, xAniso, yAniso)) * (param.voxelSize.x);
				distanceD[index2].x += (param.voxelSize.x);
				#endif
				#ifdef PRECISE_LENGTH_MODE
				voxelD[index2].x += (tex2D<float>(img, xAniso, yAniso)) * (t_out-t_in);
				distanceD[index2].x += (t_out-t_in);
				#endif
			}

			//////////// BOX INTERSECTION (Voxel 2) /////////////////	 
			tEntry.x = (param.bBoxMin.x + (x+1) * param.voxelSize.x);
			tEntry.y = (param.bBoxMin.y + (y  ) * param.voxelSize.y);
			tEntry.z = (param.bBoxMin.z + (z  ) * param.voxelSize.z);
			tEntry.x = (tEntry.x - c_source.x) / ray.x;
			tEntry.y = (tEntry.y - c_source.y) / ray.y;
			tEntry.z = (tEntry.z - c_source.z) / ray.z;

			tExit.x = (param.bBoxMin.x + (x+2) * param.voxelSize.x);
			tExit.y = (param.bBoxMin.y + (y+1  ) * param.voxelSize.y);
			tExit.z = (param.bBoxMin.z + (z+1  ) * param.voxelSize.z);

			tExit.x = ((tExit.x) - c_source.x) / ray.x;
			tExit.y = ((tExit.y) - c_source.y) / ray.y;
			tExit.z = ((tExit.z) - c_source.z) / ray.z;

			tmin = fminf(tEntry, tExit);
			tmax = fmaxf(tEntry, tExit);
			
			t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
			t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
			////////////////////////////////////////////////////////////////
				
			// if the ray hits the voxel
			if((t_out - t_in) > 0.0f )
			{
				#ifdef CONST_LENGTH_MODE
				voxelD[index2].y += (tex2D<float>(img, xAniso, yAniso)) * (param.voxelSize.x);
				distanceD[index2].y += (param.voxelSize.x);
				#endif
				#ifdef PRECISE_LENGTH_MODE
				voxelD[index2].y += (tex2D<float>(img, xAniso, yAniso)) * (t_out-t_in);
				distanceD[index2].y += (t_out-t_in);
				#endif
			}

			//////////// BOX INTERSECTION (Voxel 3) /////////////////	
			tEntry.x = (param.bBoxMin.x + (x+2) * param.voxelSize.x);
			tEntry.y = (param.bBoxMin.y + (y  ) * param.voxelSize.y);
			tEntry.z = (param.bBoxMin.z + (z  ) * param.voxelSize.z);
			tEntry.x = (tEntry.x - c_source.x) / ray.x;
			tEntry.y = (tEntry.y - c_source.y) / ray.y;
			tEntry.z = (tEntry.z - c_source.z) / ray.z;

			tExit.x = (param.bBoxMin.x + (x+3) * param.voxelSize.x);
			tExit.y = (param.bBoxMin.y + (y+1  ) * param.voxelSize.y);
			tExit.z = (param.bBoxMin.z + (z+1  ) * param.voxelSize.z);

			tExit.x = ((tExit.x) - c_source.x) / ray.x;
			tExit.y = ((tExit.y) - c_source.y) / ray.y;
			tExit.z = ((tExit.z) - c_source.z) / ray.z;

			tmin = fminf(tEntry, tExit);
			tmax = fmaxf(tEntry, tExit);
			
			t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
			t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
			////////////////////////////////////////////////////////////////
		
			// if the ray hits the voxel
			if((t_out - t_in) > 0.0f )
			{
				#ifdef CONST_LENGTH_MODE
				voxelD[index2].z += (tex2D<float>(img, xAniso, yAniso)) * (param.voxelSize.x);
				distanceD[index2].z += (param.voxelSize.x);
				#endif
				#ifdef PRECISE_LENGTH_MODE
				voxelD[index2].z += (tex2D<float>(img, xAniso, yAniso)) * (t_out-t_in);
				distanceD[index2].z += (t_out-t_in);
				#endif
			}

			//////////// BOX INTERSECTION (Voxel 4) /////////////////	 
			tEntry.x = (param.bBoxMin.x + (x+3) * param.voxelSize.x);
			tEntry.y = (param.bBoxMin.y + (y  ) * param.voxelSize.y);
			tEntry.z = (param.bBoxMin.z + (z  ) * param.voxelSize.z);
			tEntry.x = (tEntry.x - c_source.x) / ray.x;
			tEntry.y = (tEntry.y - c_source.y) / ray.y;
			tEntry.z = (tEntry.z - c_source.z) / ray.z;

			tExit.x = (param.bBoxMin.x + (x+4) * param.voxelSize.x);
			tExit.y = (param.bBoxMin.y + (y+1  ) * param.voxelSize.y);
			tExit.z = (param.bBoxMin.z + (z+1  ) * param.voxelSize.z);

			tExit.x = ((tExit.x) - c_source.x) / ray.x;
			tExit.y = ((tExit.y) - c_source.y) / ray.y;
			tExit.z = ((tExit.z) - c_source.z) / ray.z;

			tmin = fminf(tEntry, tExit);
			tmax = fmaxf(tEntry, tExit);
			
			t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
			t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
			////////////////////////////////////////////////////////////////
		
			// if the ray hits the voxel
			if((t_out - t_in) > 0.0f )
			{
				#ifdef CONST_LENGTH_MODE
				voxelD[index2].w += (tex2D<float>(img, xAniso, yAniso)) * (param.voxelSize.x);
				distanceD[index2].w += (param.voxelSize.x);
				#endif
				#ifdef PRECISE_LENGTH_MODE
				voxelD[index2].w += (tex2D<float>(img, xAniso, yAniso)) * (t_out-t_in);
				distanceD[index2].w += (t_out-t_in);
				#endif
			}// if hit voxel
		}//for loop y-pixel
	}//for loop x-pixel

	//Only positive distance values are allowed
	distanceD[index2].x = fmaxf (0.f, distanceD[index2].x);
	distanceD[index2].y = fmaxf (0.f, distanceD[index2].y);
	distanceD[index2].z = fmaxf (0.f, distanceD[index2].z);
	distanceD[index2].w = fmaxf (0.f, distanceD[index2].w);	

	//Apply correction term to voxel
	if (distanceD[index2].x != 0.0f) voxelBlock.x += (my_lambda * voxelD[index2].x / (float)distanceD[index2].x);
	if (distanceD[index2].y != 0.0f) voxelBlock.y += (my_lambda * voxelD[index2].y / (float)distanceD[index2].y);
	if (distanceD[index2].z != 0.0f) voxelBlock.z += (my_lambda * voxelD[index2].z / (float)distanceD[index2].z);
	if (distanceD[index2].w != 0.0f) voxelBlock.w += (my_lambda * voxelD[index2].w / (float)distanceD[index2].w);

	surf3Dwrite(voxelBlock.x, surfref, x * 4 + 0, y, z);
	surf3Dwrite(voxelBlock.y, surfref, x * 4 + 4, y, z);
	surf3Dwrite(voxelBlock.z, surfref, x * 4 + 8, y, z);
	surf3Dwrite(voxelBlock.w, surfref, x * 4 + 12, y, z);
}


extern "C" __global__  void backProjectionFP16(DeviceReconstructionConstantsCommon param, int proj_x, int proj_y, float my_lambda, int maxOverSample, float maxOverSampleInv, CUtexObject img, CUsurfObject surfref, float distMin, float distMax)
{
	float3 ray;
	float2 pixel;
	float2 borderMin;
	float2 borderMax;
	float3 hitPoint;
	float3 c_source;

	int4 pixelBorders; //--> x = x.min; y = x.max; z = y.min; w = y.max
	
	// index to access shared memory, e.g. thread linear address in a block
	const unsigned int index2 = (threadIdx.z * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;	

	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= param.volumeDim_x_quarter || y >= param.volumeDim.y || z >= param.volumeDim.z) return;

	//summed up distance per voxel in voxelBlock in shared memory
	volatile float4* distanceD = (float4*)(sBuffer);
	
	//Correction term per voxel in shared memory
	volatile float4* voxelD = distanceD + blockDim.x * blockDim.y * blockDim.z;
	
	float4 voxelBlock;

	float3 MC_bBoxMin;
	float3 MC_bBoxMax;
	float t;
	
	float t_in, t_out;
	float3 tEntry;
	float3 tExit;
	float3 tmin, tmax;
	float pixel_y, pixel_x;	

	unsigned short tempfp16;

	surf3Dread(&tempfp16, surfref, x * 2 * 4 + 0, y, z);
	voxelBlock.x = __half2float(tempfp16);
	surf3Dread(&tempfp16, surfref, x * 2 * 4 + 2, y, z);
	voxelBlock.y = __half2float(tempfp16);
	surf3Dread(&tempfp16, surfref, x * 2 * 4 + 4, y, z);
	voxelBlock.z = __half2float(tempfp16);
	surf3Dread(&tempfp16, surfref, x * 2 * 4 + 6, y, z);
	voxelBlock.w = __half2float(tempfp16);

	//adopt x coordinate to single voxels, not voxelBlocks
	x = x * 4;

	//MacroCell bounding box:
	MC_bBoxMin.x = param.bBoxMin.x + (x) * param.voxelSize.x;
	MC_bBoxMin.y = param.bBoxMin.y + (y) * param.voxelSize.y;
	MC_bBoxMin.z = param.bBoxMin.z + (z) * param.voxelSize.z;
	MC_bBoxMax.x = param.bBoxMin.x + ( x + 4) * param.voxelSize.x;
	MC_bBoxMax.y = param.bBoxMin.y + ( y + 1) * param.voxelSize.y;
	MC_bBoxMax.z = param.bBoxMin.z + ( z + 1) * param.voxelSize.z;

	
	//find maximal projection on detector:
	borderMin = make_float2(FLT_MAX, FLT_MAX);
	borderMax = make_float2(-FLT_MAX, -FLT_MAX);

	//The loop has been manually unrolled: nvcc cannot handle inner loops
	//first corner
	t = (param.projNorm.x * MC_bBoxMin.x + param.projNorm.y * MC_bBoxMin.y + param.projNorm.z * MC_bBoxMin.z);
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);

	if (!(t >= distMin && t < distMax)) return;

	hitPoint.x = t * (-param.projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-param.projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-param.projNorm.z) + MC_bBoxMin.z;

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//second corner
	t = (param.projNorm.x * MC_bBoxMin.x + param.projNorm.y * MC_bBoxMin.y + param.projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-param.projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-param.projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//third corner
	t = (param.projNorm.x * MC_bBoxMin.x + param.projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + param.projNorm.z * MC_bBoxMin.z);
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-param.projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-param.projNorm.z) + MC_bBoxMin.z;

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//fourth corner
	t = (param.projNorm.x * MC_bBoxMin.x + param.projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + param.projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + MC_bBoxMin.x;
	hitPoint.y = t * (-param.projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-param.projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//fifth corner
	t = (param.projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + param.projNorm.y * MC_bBoxMin.y + param.projNorm.z * MC_bBoxMin.z);
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-param.projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-param.projNorm.z) + MC_bBoxMin.z;

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//sixth corner
	t = (param.projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + param.projNorm.y * MC_bBoxMin.y + param.projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-param.projNorm.y) + MC_bBoxMin.y;
	hitPoint.z = t * (-param.projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//seventh corner
	t = (param.projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + param.projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + param.projNorm.z * MC_bBoxMin.z);
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-param.projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-param.projNorm.z) + MC_bBoxMin.z;

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);

	//eighth corner
	t = (param.projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + param.projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + param.projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
	t += (-param.projNorm.x * param.detektor.x - param.projNorm.y * param.detektor.y - param.projNorm.z * param.detektor.z);
	t = abs(t);
	hitPoint.x = t * (-param.projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
	hitPoint.y = t * (-param.projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
	hitPoint.z = t * (-param.projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));

	MatrixVector3Mul(param.DetectorMatrix, hitPoint, pixel);
	borderMin = fminf(pixel, borderMin);
	borderMax = fmaxf(pixel, borderMax);


	//--> pixelBorders.x = x.min; pixelBorders.z = y.min;
	pixelBorders.x = floor(borderMin.x);
	pixelBorders.z = floor(borderMin.y);
	
	//--> pixelBorders.y = x.max; pixelBorders.v = y.max
	hitPoint = fmaxf(MC_bBoxMin, MC_bBoxMax);
	pixelBorders.y = ceil(borderMax.x);
	pixelBorders.w = ceil(borderMax.y);

	//clamp values
	pixelBorders.x = fminf(fmaxf(pixelBorders.x, 0), proj_x);
	pixelBorders.y = fminf(fmaxf(pixelBorders.y, 0), proj_x);
	pixelBorders.z = fminf(fmaxf(pixelBorders.z, 0), proj_y);
	pixelBorders.w = fminf(fmaxf(pixelBorders.w, 0), proj_y);
	
	voxelD[index2].x  = 0;
	voxelD[index2].y  = 0;
	voxelD[index2].z  = 0;
	voxelD[index2].w  = 0;
	distanceD[index2].x  = 0;
	distanceD[index2].y  = 0;
	distanceD[index2].z  = 0;
	distanceD[index2].w  = 0;

	//Loop over detected pixels and shoot rays back	again with manual unrolling
	for( pixel_y = pixelBorders.z + maxOverSampleInv*0.5f ; pixel_y < pixelBorders.w ; pixel_y+=maxOverSampleInv)
	{				
		for ( pixel_x = pixelBorders.x + maxOverSampleInv*0.5f ; pixel_x < pixelBorders.y ; pixel_x+=maxOverSampleInv)	
		{
			float xAniso;
			float yAniso;

			MatrixVector3Mul(param.magAnisoInv, pixel_x, pixel_y, xAniso, yAniso);

			//if (pixel_x < 1) continue;
			ray.x = param.detektor.x; 
			ray.y = param.detektor.y; 
			ray.z = param.detektor.z; 
			
			ray.x = ray.x + (pixel_x) * param.uPitch.x;
			ray.y = ray.y + (pixel_x) * param.uPitch.y;
			ray.z = ray.z + (pixel_x) * param.uPitch.z;
			
			ray.x = ray.x + (pixel_y) * param.vPitch.x;
			ray.y = ray.y + (pixel_y) * param.vPitch.y;
			ray.z = ray.z + (pixel_y) * param.vPitch.z;
			
			c_source.x = ray.x + 100000.0 * param.projNorm.x;
			c_source.y = ray.y + 100000.0 * param.projNorm.y;
			c_source.z = ray.z + 100000.0 * param.projNorm.z;
			ray.x = ray.x - c_source.x;
			ray.y = ray.y - c_source.y;
			ray.z = ray.z - c_source.z;

			// calculate ray direction
			ray = normalize(ray);
				
			//////////// BOX INTERSECTION (Voxel 1) /////////////////	
			tEntry.x = (param.bBoxMin.x + (x  ) * param.voxelSize.x);
			tEntry.y = (param.bBoxMin.y + (y  ) * param.voxelSize.y);
			tEntry.z = (param.bBoxMin.z + (z  ) * param.voxelSize.z);
			tEntry.x = (tEntry.x - c_source.x) / ray.x;
			tEntry.y = (tEntry.y - c_source.y) / ray.y;
			tEntry.z = (tEntry.z - c_source.z) / ray.z;

			tExit.x = (param.bBoxMin.x + (x+1  ) * param.voxelSize.x);
			tExit.y = (param.bBoxMin.y + (y+1  ) * param.voxelSize.y);
			tExit.z = (param.bBoxMin.z + (z+1  ) * param.voxelSize.z);

			tExit.x = ((tExit.x) - c_source.x) / ray.x;
			tExit.y = ((tExit.y) - c_source.y) / ray.y;
			tExit.z = ((tExit.z) - c_source.z) / ray.z;

			tmin = fminf(tEntry, tExit);
			tmax = fmaxf(tEntry, tExit);
			
			t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
			t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
			////////////////////////////////////////////////////////////////
				
			// if the ray hits the voxel
			if((t_out - t_in) > 0.0f )
			{
				#ifdef CONST_LENGTH_MODE
				voxelD[index2].x += (tex2D<float>(img, xAniso, yAniso)) * (param.voxelSize.x);
				distanceD[index2].x += (param.voxelSize.x);
				#endif
				#ifdef PRECISE_LENGTH_MODE
				voxelD[index2].x += (tex2D<float>(img, xAniso, yAniso)) * (t_out-t_in);
				distanceD[index2].x += (t_out-t_in);
				#endif
			}

			//////////// BOX INTERSECTION (Voxel 2) /////////////////	 
			tEntry.x = (param.bBoxMin.x + (x+1) * param.voxelSize.x);
			tEntry.y = (param.bBoxMin.y + (y  ) * param.voxelSize.y);
			tEntry.z = (param.bBoxMin.z + (z  ) * param.voxelSize.z);
			tEntry.x = (tEntry.x - c_source.x) / ray.x;
			tEntry.y = (tEntry.y - c_source.y) / ray.y;
			tEntry.z = (tEntry.z - c_source.z) / ray.z;

			tExit.x = (param.bBoxMin.x + (x+2) * param.voxelSize.x);
			tExit.y = (param.bBoxMin.y + (y+1  ) * param.voxelSize.y);
			tExit.z = (param.bBoxMin.z + (z+1  ) * param.voxelSize.z);

			tExit.x = ((tExit.x) - c_source.x) / ray.x;
			tExit.y = ((tExit.y) - c_source.y) / ray.y;
			tExit.z = ((tExit.z) - c_source.z) / ray.z;

			tmin = fminf(tEntry, tExit);
			tmax = fmaxf(tEntry, tExit);
			
			t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
			t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
			////////////////////////////////////////////////////////////////
				
			// if the ray hits the voxel
			if((t_out - t_in) > 0.0f )
			{
				#ifdef CONST_LENGTH_MODE
				voxelD[index2].y += (tex2D<float>(img, xAniso, yAniso)) * (param.voxelSize.x);
				distanceD[index2].y += (param.voxelSize.x);
				#endif
				#ifdef PRECISE_LENGTH_MODE
				voxelD[index2].y += (tex2D<float>(img, xAniso, yAniso)) * (t_out-t_in);
				distanceD[index2].y += (t_out-t_in);
				#endif
			}

			//////////// BOX INTERSECTION (Voxel 3) /////////////////	
			tEntry.x = (param.bBoxMin.x + (x+2) * param.voxelSize.x);
			tEntry.y = (param.bBoxMin.y + (y  ) * param.voxelSize.y);
			tEntry.z = (param.bBoxMin.z + (z  ) * param.voxelSize.z);
			tEntry.x = (tEntry.x - c_source.x) / ray.x;
			tEntry.y = (tEntry.y - c_source.y) / ray.y;
			tEntry.z = (tEntry.z - c_source.z) / ray.z;

			tExit.x = (param.bBoxMin.x + (x+3) * param.voxelSize.x);
			tExit.y = (param.bBoxMin.y + (y+1  ) * param.voxelSize.y);
			tExit.z = (param.bBoxMin.z + (z+1  ) * param.voxelSize.z);

			tExit.x = ((tExit.x) - c_source.x) / ray.x;
			tExit.y = ((tExit.y) - c_source.y) / ray.y;
			tExit.z = ((tExit.z) - c_source.z) / ray.z;

			tmin = fminf(tEntry, tExit);
			tmax = fmaxf(tEntry, tExit);
			
			t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
			t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
			////////////////////////////////////////////////////////////////
		
			// if the ray hits the voxel
			if((t_out - t_in) > 0.0f )
			{
				#ifdef CONST_LENGTH_MODE
				voxelD[index2].z += (tex2D<float>(img, xAniso, yAniso)) * (param.voxelSize.x);
				distanceD[index2].z += (param.voxelSize.x);
				#endif
				#ifdef PRECISE_LENGTH_MODE
				voxelD[index2].z += (tex2D<float>(img, xAniso, yAniso)) * (t_out-t_in);
				distanceD[index2].z += (t_out-t_in);
				#endif
			}

			//////////// BOX INTERSECTION (Voxel 4) /////////////////	 
			tEntry.x = (param.bBoxMin.x + (x+3) * param.voxelSize.x);
			tEntry.y = (param.bBoxMin.y + (y  ) * param.voxelSize.y);
			tEntry.z = (param.bBoxMin.z + (z  ) * param.voxelSize.z);
			tEntry.x = (tEntry.x - c_source.x) / ray.x;
			tEntry.y = (tEntry.y - c_source.y) / ray.y;
			tEntry.z = (tEntry.z - c_source.z) / ray.z;

			tExit.x = (param.bBoxMin.x + (x+4) * param.voxelSize.x);
			tExit.y = (param.bBoxMin.y + (y+1  ) * param.voxelSize.y);
			tExit.z = (param.bBoxMin.z + (z+1  ) * param.voxelSize.z);

			tExit.x = ((tExit.x) - c_source.x) / ray.x;
			tExit.y = ((tExit.y) - c_source.y) / ray.y;
			tExit.z = ((tExit.z) - c_source.z) / ray.z;

			tmin = fminf(tEntry, tExit);
			tmax = fmaxf(tEntry, tExit);
			
			t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
			t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
			////////////////////////////////////////////////////////////////
		
			// if the ray hits the voxel
			if((t_out - t_in) > 0.0f )
			{
				#ifdef CONST_LENGTH_MODE
				voxelD[index2].w += (tex2D<float>(img, xAniso, yAniso)) * (param.voxelSize.x);
				distanceD[index2].w += (param.voxelSize.x);
				#endif
				#ifdef PRECISE_LENGTH_MODE
				voxelD[index2].w += (tex2D<float>(img, xAniso, yAniso)) * (t_out-t_in);
				distanceD[index2].w += (t_out-t_in);
				#endif
			}// if hit voxel
		}//for loop y-pixel
	}//for loop x-pixel

	//Only positive distance values are allowed
	distanceD[index2].x = fmaxf (0.f, distanceD[index2].x);
	distanceD[index2].y = fmaxf (0.f, distanceD[index2].y);
	distanceD[index2].z = fmaxf (0.f, distanceD[index2].z);
	distanceD[index2].w = fmaxf (0.f, distanceD[index2].w);	

	//Apply correction term to voxel
	if (distanceD[index2].x != 0.0f) voxelBlock.x += (my_lambda * voxelD[index2].x / (float)distanceD[index2].x);
	if (distanceD[index2].y != 0.0f) voxelBlock.y += (my_lambda * voxelD[index2].y / (float)distanceD[index2].y);
	if (distanceD[index2].z != 0.0f) voxelBlock.z += (my_lambda * voxelD[index2].z / (float)distanceD[index2].z);
	if (distanceD[index2].w != 0.0f) voxelBlock.w += (my_lambda * voxelD[index2].w / (float)distanceD[index2].w);

	tempfp16 = __float2half_rn(voxelBlock.x);
	surf3Dwrite(tempfp16, surfref, x * 2 + 0, y, z);
	tempfp16 = __float2half_rn(voxelBlock.y);
	surf3Dwrite(tempfp16, surfref, x * 2 + 2, y, z);
	tempfp16 = __float2half_rn(voxelBlock.z);
	surf3Dwrite(tempfp16, surfref, x * 2 + 4, y, z);
	tempfp16 = __float2half_rn(voxelBlock.w);
	surf3Dwrite(tempfp16, surfref, x * 2 + 6, y, z);
}

extern "C" __global__ void convertVolumeFP16ToFP32(DeviceReconstructionConstantsCommon param, float* volPlane, int stride, CUsurfObject surfref, unsigned int z)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

	if (x >= param.volumeDim_x_quarter || y >= param.volumeDim.y || z >= param.volumeDim.z) return;

	float4 voxelBlock;

	unsigned short tempfp16;

	surf3Dread(&tempfp16, surfref, x * 2 * 4 + 0, y, z);
	voxelBlock.x = __half2float(tempfp16);
	surf3Dread(&tempfp16, surfref, x * 2 * 4 + 2, y, z);
	voxelBlock.y = __half2float(tempfp16);
	surf3Dread(&tempfp16, surfref, x * 2 * 4 + 4, y, z);
	voxelBlock.z = __half2float(tempfp16);
	surf3Dread(&tempfp16, surfref, x * 2 * 4 + 6, y, z);
	voxelBlock.w = __half2float(tempfp16);
	
	//adopt x coordinate to single voxels, not voxelBlocks
	x = x * 4;
	
	*(((float*)((char*)volPlane + stride * y)) + x + 0) = -voxelBlock.x;
	*(((float*)((char*)volPlane + stride * y)) + x + 1) = -voxelBlock.y;
	*(((float*)((char*)volPlane + stride * y)) + x + 2) = -voxelBlock.z;
	*(((float*)((char*)volPlane + stride * y)) + x + 3) = -voxelBlock.w;
}

// extern "C"
// __global__ void rot3d(int size, float3 rotMat0, float3 rotMat1, float3 rotMat2, hipTextureObject_t texVol, float* outVol)
// {
// 	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
// 	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
// 	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

// 	float center = size / 2;

// 	float3 vox = make_float3(x - center, y - center, z - center);
// 	float3 rotVox;
// 	rotVox.x = center + rotMat0.x * vox.x + rotMat1.x * vox.y + rotMat2.x * vox.z;
// 	rotVox.y = center + rotMat0.y * vox.x + rotMat1.y * vox.y + rotMat2.y * vox.z;
// 	rotVox.z = center + rotMat0.z * vox.x + rotMat1.z * vox.y + rotMat2.z * vox.z;

// 	outVol[z * size * size + y * size + x] = tex3D(texVol, rotVox.x + 0.5f, rotVox.y + 0.5f, rotVox.z + 0.5f);
// }

/*
extern "C"
__global__
void convertVolume3DFP16ToFP32(float* volPlane, int stride, CUsurfObject surfref)
{
	unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	if (x >= param.volumeDim_x_quarter || y >= param.volumeDim.y || z >= param.volumeDim.z) return;

	float4 voxelBlock;

	unsigned short tempfp16;

	surf3Dread(&tempfp16, surfref, x * 2 * 4 + 0, y, z);
	voxelBlock.x = __half2float(tempfp16);
	surf3Dread(&tempfp16, surfref, x * 2 * 4 + 2, y, z);
	voxelBlock.y = __half2float(tempfp16);
	surf3Dread(&tempfp16, surfref, x * 2 * 4 + 4, y, z);
	voxelBlock.z = __half2float(tempfp16);
	surf3Dread(&tempfp16, surfref, x * 2 * 4 + 6, y, z);
	voxelBlock.w = __half2float(tempfp16);

	//adopt x coordinate to single voxels, not voxelBlocks
	x = x * 4;

	*(((float*)((char*)volPlane + stride * y)) + x + 0) = -voxelBlock.x;
	*(((float*)((char*)volPlane + stride * y)) + x + 1) = -voxelBlock.y;
	*(((float*)((char*)volPlane + stride * y)) + x + 2) = -voxelBlock.z;
	*(((float*)((char*)volPlane + stride * y)) + x + 3) = -voxelBlock.w;
}
*/

KernelModuls::KernelModuls(Cuda::CudaContext* aCuCtx)
	:compilerOutput(false),
	infoOutput(false)
{
	//modFP = aCuCtx->LoadModulePTX(KernelForwardProjectionRayMarcher_TL, 0, infoOutput, compilerOutput);
	//modSlicer = aCuCtx->LoadModulePTX(KernelForwardProjectionSlicer, 0, infoOutput, compilerOutput);
	//modVolTravLen = modSlicer;
	//modComp = aCuCtx->LoadModulePTX(KernelCompare, 0, infoOutput, compilerOutput);
	//modWBP = aCuCtx->LoadModulePTX(KernelwbpWeighting, 0, infoOutput, compilerOutput);
	//modBP = aCuCtx->LoadModulePTX(KernelBackProjectionSquareOS, 0, infoOutput, compilerOutput);
	//modCTF = aCuCtx->LoadModulePTX(Kernelctf, 0, infoOutput, compilerOutput);
	//modCTS = aCuCtx->LoadModulePTX(KernelCopyToSquare, 0, infoOutput, compilerOutput);
}

void Reconstructor::MatrixVector3Mul(float4x4 M, float3* v)
{
	float3 erg;
	erg.x = M.m[0].x * v->x + M.m[0].y * v->y + M.m[0].z * v->z + 1.f * M.m[0].w;
	erg.y = M.m[1].x * v->x + M.m[1].y * v->y + M.m[1].z * v->z + 1.f * M.m[1].w;
	erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
	*v = erg;
}

void Reconstructor::MatrixVector3Mul(float3x3& M, float xIn, float yIn, float& xOut, float& yOut)
{
    xOut = M.m[0].x * xIn + M.m[0].y * yIn + M.m[0].z * 1.f;
    yOut = M.m[1].x * xIn + M.m[1].y * yIn + M.m[1].z * 1.f;
    //erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
}

template<class TVol>
void Reconstructor::GetDefocusDistances(float & t_in, float & t_out, int index, Volume<TVol>* vol)
{
	//Shoot ray from center of volume:
	float3 c_projNorm = proj.GetNormalVector(index);
	float3 c_detektor = proj.GetPosition(index);
	float3 MC_bBoxMin;
	float3 MC_bBoxMax;
	MC_bBoxMin = vol->GetVolumeBBoxMin();
	MC_bBoxMax = vol->GetVolumeBBoxMax();
	float3 volDim = vol->GetDimension();
	float3 hitPoint;
	float t;
//	printf("PosInVol2: %f, %f, %f\n", (MC_bBoxMin.x + (volDim.x * vol->GetVoxelSize().x * 0.5f + vol->GetVoxelSize().x * 0.5f)),
//		(MC_bBoxMin.y + (volDim.y * vol->GetVoxelSize().y * 0.5f + vol->GetVoxelSize().x * 0.5f)),
//		(MC_bBoxMin.z + (volDim.z * vol->GetVoxelSize().z * 0.5f + vol->GetVoxelSize().x * 0.5f)));

	t = (c_projNorm.x * (MC_bBoxMin.x + (volDim.x * vol->GetVoxelSize().x * 0.5f)) + 
		 c_projNorm.y * (MC_bBoxMin.y + (volDim.y * vol->GetVoxelSize().y * 0.5f)) + 
		 c_projNorm.z * (MC_bBoxMin.z + (volDim.z * vol->GetVoxelSize().z * 0.5f)));
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = abs(t);
	
	hitPoint.x = t * (-c_projNorm.x) + (MC_bBoxMin.x + (volDim.x * vol->GetVoxelSize().x * 0.5f));
	hitPoint.y = t * (-c_projNorm.y) + (MC_bBoxMin.y + (volDim.y * vol->GetVoxelSize().y * 0.5f));
	hitPoint.z = t * (-c_projNorm.z) + (MC_bBoxMin.z + (volDim.z * vol->GetVoxelSize().z * 0.5f));

	float4x4 c_DetectorMatrix;
	
	proj.GetDetectorMatrix(index, (float*)&c_DetectorMatrix, 1);
	MatrixVector3Mul(c_DetectorMatrix, &hitPoint);

	//--> pixelBorders.x = x.min; pixelBorders.z = y.min;
	float hitX = round(hitPoint.x);
	float hitY = round(hitPoint.y);

	//printf("HitX: %d, HitY: %d\n", hitX, hitY);

	//Shoot ray from hit point on projection towards volume to get the distance to entry and exit point
	//float3 pos = proj.GetPosition(index) + hitX * proj.GetPixelUPitch(index) + hitY * proj.GetPixelVPitch(index);
	float3 pos = proj.GetPosition(index) + hitX * proj.GetPixelUPitch(index) + hitY * proj.GetPixelVPitch(index);
	hitX = (float)proj.GetWidth() * 0.5f;
	hitY = (float)proj.GetHeight() * 0.5f;
	float3 pos2 = proj.GetPosition(index) + hitX * proj.GetPixelUPitch(index) + hitX * proj.GetPixelVPitch(index);
	float3 nvec = proj.GetNormalVector(index);

	/*float3 MC_bBoxMin;
	float3 MC_bBoxMax;*/

	

	t_in = 2*-DIST;
	t_out = 2*DIST;

	for (int x = 0; x <= 1; x++)
		for (int y = 0; y <= 1; y++)
			for (int z = 0; z <= 1; z++)
			{
				//float t;

				t = (nvec.x * (MC_bBoxMin.x + x * (MC_bBoxMax.x - MC_bBoxMin.x))
					+ nvec.y * (MC_bBoxMin.y + y * (MC_bBoxMax.y - MC_bBoxMin.y))
					+ nvec.z * (MC_bBoxMin.z + z * (MC_bBoxMax.z - MC_bBoxMin.z)));
				t += (-nvec.x * pos.x - nvec.y * pos.y - nvec.z * pos.z);

				if (t < t_in) t_in = t;
				if (t > t_out) t_out = t;
			}

	//printf("t_in: %f; t_out: %f\n", t_in, t_out);
	//t_in = 2*-DIST;
	//t_out = 2*DIST;

	//for (int x = 0; x <= 1; x++)
	//	for (int y = 0; y <= 1; y++)
	//		for (int z = 0; z <= 1; z++)
	//		{
	//			//float t;

	//			t = (nvec.x * (MC_bBoxMin.x + x * (MC_bBoxMax.x - MC_bBoxMin.x))
	//				+ nvec.y * (MC_bBoxMin.y + y * (MC_bBoxMax.y - MC_bBoxMin.y))
	//				+ nvec.z * (MC_bBoxMin.z + z * (MC_bBoxMax.z - MC_bBoxMin.z)));
	//			t += (-nvec.x * pos2.x - nvec.y * pos2.y - nvec.z * pos2.z);

	//			if (t < t_in) t_in = t;
	//			if (t > t_out) t_out = t;
	//		}
	////printf("t_in: %f; t_out: %f\n", t_in, t_out);



	//{
	//	float xAniso = 2366.25f;
	//	float yAniso = 4527.75f;

	//	float3 c_source = c_detektor;
	//	float3 c_uPitch = proj.GetPixelUPitch(index);
	//	float3 c_vPitch = proj.GetPixelVPitch(index);
	//	c_source = c_source + (xAniso)* c_uPitch;
	//	c_source = c_source + (yAniso)* c_vPitch;

	//	//////////// BOX INTERSECTION (partial Volume) /////////////////
	//	float3 tEntry;
	//	tEntry.x = (MC_bBoxMin.x - c_source.x) / (c_projNorm.x);
	//	tEntry.y = (MC_bBoxMin.y - c_source.y) / (c_projNorm.y);
	//	tEntry.z = (MC_bBoxMin.z - c_source.z) / (c_projNorm.z);

	//	float3 tExit;
	//	tExit.x = (MC_bBoxMax.x - c_source.x) / (c_projNorm.x);
	//	tExit.y = (MC_bBoxMax.y - c_source.y) / (c_projNorm.y);
	//	tExit.z = (MC_bBoxMax.z - c_source.z) / (c_projNorm.z);


	//	float3 tmin = fminf(tEntry, tExit);
	//	float3 tmax = fmaxf(tEntry, tExit);

	//	t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	//	t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
	//	printf("t_in: %f; t_out: %f\n", t_in, t_out);
	//}
}
template void Reconstructor::GetDefocusDistances(float & t_in, float & t_out, int index, Volume<unsigned short>* vol);
template void Reconstructor::GetDefocusDistances(float & t_in, float & t_out, int index, Volume<float>* vol);


void Reconstructor::GetDefocusMinMax(float ray, int index, float & defocusMin, float & defocusMax)
{
	defocusMin = defocus.GetMinDefocus(index);
	defocusMax = defocus.GetMaxDefocus(index);
	float tiltAngle = (markers(MFI_TiltAngle, index, 0) + config.AddTiltAngle) / 180.0f * (float)M_PI;

	float distanceTo0 = ray + DIST; //in pixel
	if (config.IgnoreZShiftForCTF)
	{
		distanceTo0 = (round(distanceTo0 * proj.GetPixelSize() * config.CTFSliceThickness) / config.CTFSliceThickness) + config.CTFSliceThickness / 2.0f;
	}
	else
	{
		distanceTo0 = (round(distanceTo0 * proj.GetPixelSize() * config.CTFSliceThickness) / config.CTFSliceThickness) + config.CTFSliceThickness / 2.0f - (config.VolumeShift.z * proj.GetPixelSize() * cosf(tiltAngle)); //in nm
	}
	if (config.SwitchCTFDirectionForIMOD)
	{
		distanceTo0 *= -1; //IMOD inverses the logic...
	}
	

	defocusMin = defocusMin + distanceTo0;
	defocusMax = defocusMax + distanceTo0;
}

Reconstructor::Reconstructor(Configuration::Config & aConfig,
	Projection & aProj, ProjectionSource* aProjectionSource,
	MarkerFile& aMarkers, CtfFile& aDefocus, KernelModuls& modules, int aMpi_part, int aMpi_size)
	: 
	proj(aProj), 
	projSource(aProjectionSource), 
	//fpKernel(modules.modFP),
	//slicerKernel(modules.modSlicer),
	//volTravLenKernel(modules.modVolTravLen),
	//wbp(modules.modWBP),
	//fourFilterKernel(modules.modWBP),
	//doseWeightingKernel(modules.modWBP),
	//conjKernel(modules.modWBP),
    //pcKernel(modules.modWBP),
	//maxShiftKernel(modules.modWBP),
	//compKernel(modules.modComp),
	//subEKernel(modules.modComp),
	//cropKernel(modules.modComp),
	//bpKernel(modules.modBP, aConfig.FP16Volume),
	//convVolKernel(modules.modBP),
	//convVol3DKernel(modules.modBP),
	//ctf(modules.modCTF),
	//cts(modules.modCTS),
	//dimBordersKernel(modules.modComp),
#ifdef REFINE_MODE
	rotKernel(modules.modWBP, aConfig.SizeSubVol),
	maxShiftWeightedKernel(modules.modWBP),
	findPeakKernel(modules.modWBP),
#endif
	defocus(aDefocus),
	markers(aMarkers),
	config(aConfig),
	mpi_part(aMpi_part),
	mpi_size(aMpi_size),
	skipFilter(aConfig.SkipFilter),
	squareBorderSizeX(0),
	squareBorderSizeY(0),
	squarePointerShift(0),
	magAnisotropy(GetMagAnistropyMatrix(aConfig.MagAnisotropyAmount, aConfig.MagAnisotropyAngleInDeg, (float)proj.GetWidth(), (float)proj.GetHeight())),
	magAnisotropyInv(GetMagAnistropyMatrix(1.0f / aConfig.MagAnisotropyAmount, aConfig.MagAnisotropyAngleInDeg, (float)proj.GetWidth(), (float)proj.GetHeight())),
	mRecParamCtf()
{
	//Set kernel work dimensions for 2D images:
	//fpKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);
	//slicerKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);
	//volTravLenKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);
	//compKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);
	//subEKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);
	//cropKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);

	//wbp.SetComputeSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
	//ctf.SetComputeSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
	//fourFilterKernel.SetComputeSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
	//doseWeightingKernel.SetComputeSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
	//conjKernel.SetComputeSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
    //pcKernel.SetComputeSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
	//cts.SetComputeSize(proj.GetMaxDimension(), proj.GetMaxDimension(), 1);
	//maxShiftKernel.SetComputeSize(proj.GetMaxDimension(), proj.GetMaxDimension(), 1);
	//convVolKernel.SetComputeSize(config.RecDimensions.x, config.RecDimensions.y, 1);
	//dimBordersKernel.SetComputeSize(proj.GetWidth(), proj.GetHeight(), 1);

	//Alloc device variables
	realprojUS_d.Alloc(proj.GetWidth() * sizeof(int), proj.GetHeight(), sizeof(int));
	proj_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));
	realproj_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));
	dist_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));
	filterImage_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));

#ifdef REFINE_MODE
	maxShiftWeightedKernel.SetComputeSize(proj.GetMaxDimension(), proj.GetMaxDimension(), 1);
	findPeakKernel.SetComputeSize(proj.GetMaxDimension(), proj.GetMaxDimension(), 1);


	projSubVols_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));
	ccMap = new float[aConfig.MaxShift * 4 * aConfig.MaxShift * 4];
	ccMapMulti = new float[aConfig.MaxShift * 4 * aConfig.MaxShift * 4];
	ccMap_d.Alloc(4 * aConfig.MaxShift * sizeof(float), 4 * aConfig.MaxShift, sizeof(float));
	ccMap_d.Memset(0);

	roiCC1.x = 0;
	roiCC1.y = 0;
	roiCC1.width = aConfig.MaxShift * 2;
	roiCC1.height = aConfig.MaxShift * 2;

	roiCC2.x = proj.GetMaxDimension() - aConfig.MaxShift * 2;
	roiCC2.y = 0;
	roiCC2.width = aConfig.MaxShift * 2;
	roiCC2.height = aConfig.MaxShift * 2;

	roiCC3.x = 0;
	roiCC3.y = proj.GetMaxDimension() - aConfig.MaxShift * 2;
	roiCC3.width = aConfig.MaxShift * 2;
	roiCC3.height = aConfig.MaxShift * 2;

	roiCC4.x = proj.GetMaxDimension() - aConfig.MaxShift * 2;
	roiCC4.y = proj.GetMaxDimension() - aConfig.MaxShift * 2;
	roiCC4.width = aConfig.MaxShift * 2;
	roiCC4.height = aConfig.MaxShift * 2;

	roiDestCC4.x = 0;
	roiDestCC4.y = 0;
	roiDestCC1.width = aConfig.MaxShift * 2;
	roiDestCC1.height = aConfig.MaxShift * 2;

	roiDestCC3.x = aConfig.MaxShift * 4 - aConfig.MaxShift * 2;
	roiDestCC3.y = 0;
	roiDestCC2.width = aConfig.MaxShift * 2;
	roiDestCC2.height = aConfig.MaxShift * 2;

	roiDestCC2.x = 0;
	roiDestCC2.y = aConfig.MaxShift * 4 - aConfig.MaxShift * 2;
	roiDestCC3.width = aConfig.MaxShift * 2;
	roiDestCC3.height = aConfig.MaxShift * 2;

	roiDestCC1.x = aConfig.MaxShift * 4 - aConfig.MaxShift * 2;
	roiDestCC1.y = aConfig.MaxShift * 4 - aConfig.MaxShift * 2;
	roiDestCC4.width = aConfig.MaxShift * 2;
	roiDestCC4.height = aConfig.MaxShift * 2;
	projSquare2_d.Alloc(proj.GetMaxDimension() * sizeof(float) * proj.GetMaxDimension());
#endif

	//Bind back projection image to texref in BP Kernel
	if (aConfig.CtfMode == Configuration::Config::CTFM_YES)
	{
		texImage.Bind(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &dist_d, CU_AD_FORMAT_FLOAT, 1);
		//CudaTextureLinearPitched2D::Bind(&bpKernel, "tex", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
		//	CU_TR_FILTER_MODE_LINEAR, 0, &dist_d, CU_AD_FORMAT_FLOAT, 1);
	}
	else
	{
		texImage.Bind(CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP, CU_TR_FILTER_MODE_LINEAR, 0, &proj_d, CU_AD_FORMAT_FLOAT, 1);
		//CudaTextureLinearPitched2D::Bind(&bpKernel, "tex", CU_TR_ADDRESS_MODE_CLAMP, CU_TR_ADDRESS_MODE_CLAMP,
		//	CU_TR_FILTER_MODE_POINT, 0, &proj_d, CU_AD_FORMAT_FLOAT, 1);
	}

	ctf_d.Alloc((proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension(), sizeof(cuComplex));
	fft_d.Alloc((proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex) * proj.GetMaxDimension());
	projSquare_d.Alloc(proj.GetMaxDimension() * sizeof(float) * proj.GetMaxDimension());
	badPixelMask_d.Alloc(proj.GetMaxDimension() * sizeof(char), proj.GetMaxDimension(), 4 * sizeof(char));

	int bufferSize = 0;
	size_t squarePointerShiftX = ((proj.GetMaxDimension() - proj.GetWidth()) / 2);
	size_t squarePointerShiftY = ((proj.GetMaxDimension() - proj.GetHeight()) / 2) * proj.GetMaxDimension();
	squarePointerShift = squarePointerShiftX + squarePointerShiftY;
	squareBorderSizeX = (proj.GetMaxDimension() - proj.GetWidth()) / 2;
	squareBorderSizeY = (proj.GetMaxDimension() - proj.GetHeight()) / 2;
	//roiBorderSquare.width = squareBorderSize;
	//roiBorderSquare.height = proj.GetHeight();
	roiSquare.width = proj.GetMaxDimension();
	roiSquare.height = proj.GetMaxDimension();

	roiAll.width = proj.GetWidth();
	roiAll.height = proj.GetHeight();
	roiFFT.width = proj.GetMaxDimension() / 2 + 1;
	roiFFT.height = proj.GetHeight();
	nppiMeanStdDevGetBufferHostSize_32f_C1R(roiAll, &bufferSize);
	int bufferSize2;
	nppiMaxIndxGetBufferHostSize_32f_C1R(roiSquare, &bufferSize2);
	if (bufferSize2 > bufferSize)
		bufferSize = bufferSize2;
	nppiMeanGetBufferHostSize_32f_C1R(roiSquare, &bufferSize2);
	if (bufferSize2 > bufferSize)
		bufferSize = bufferSize2;
	nppiSumGetBufferHostSize_32f_C1R(roiSquare, &bufferSize2);
	if (bufferSize2 > bufferSize)
		bufferSize = bufferSize2;

	if (markers.GetProjectionCount() * sizeof(float) > bufferSize)
	{
		bufferSize = markers.GetProjectionCount() * sizeof(float); //for exact WBP filter
	}

	meanbuffer.Alloc(bufferSize * 10);
	meanval.Alloc(sizeof(double));
	stdval.Alloc(sizeof(double));

	size_t free = 0;
	size_t total = 0;
	cudaMemGetInfo(&free, &total);
    printf("before crash: free: %zu total: %zu", free, total);

	cufftSafeCall(cufftPlan2d(&handleR2C, proj.GetMaxDimension(), proj.GetMaxDimension(), CUFFT_R2C));
	cufftSafeCall(cufftPlan2d(&handleC2R, proj.GetMaxDimension(), proj.GetMaxDimension(), CUFFT_C2R));

	MPIBuffer = new float[proj.GetWidth() * proj.GetHeight()];
	//SetConstantValues(ctf, proj, 0, config.Cs, config.Voltage);
	mRecParamCtf.cs = config.Cs;
	mRecParamCtf.voltage = config.Voltage;
	mRecParamCtf.openingAngle = 0.01f;
	mRecParamCtf.ampContrast = 0.00f;
	mRecParamCtf.phaseContrast = sqrt( 1. - mRecParamCtf.ampContrast*mRecParamCtf.ampContrast );
	mRecParamCtf.pixelsize = proj.GetPixelSize() * pow(10, -9);
	mRecParamCtf.pixelcount = proj.GetMaxDimension();
	mRecParamCtf.maxFreq = 1.0 / (mRecParamCtf.pixelsize * 2.0 );
	mRecParamCtf.freqStepSize = mRecParamCtf.maxFreq / (mRecParamCtf.pixelcount / 2.0f);
	// mRecParamCtf.c_lambda = ;
	mRecParamCtf.applyScatteringProfile = 0.f;
	mRecParamCtf.applyEnvelopeFunction = 0.f;

	ResetProjectionsDevice();
}

Reconstructor::~Reconstructor()
{
	if (MPIBuffer)
	{
		delete[] MPIBuffer;
		MPIBuffer = NULL;
	}

	cufftSafeCall(cufftDestroy(handleR2C));
	cufftSafeCall(cufftDestroy(handleC2R));
}

Matrix<float> Reconstructor::GetMagAnistropyMatrix(float aAmount, float angleInDeg, float dimX, float dimY)
{
	float angle = (float)(angleInDeg / 180.0 * M_PI);

	Matrix<float> shiftCenter(3, 3);
	Matrix<float> shiftBack(3, 3);
	Matrix<float> rotMat1 = Matrix<float>::GetRotationMatrix3DZ(angle);
	Matrix<float> rotMat2 = Matrix<float>::GetRotationMatrix3DZ(-angle);
	Matrix<float> stretch(3, 3);
	shiftCenter(0, 0) = 1;
	shiftCenter(0, 1) = 0;
	shiftCenter(0, 2) = -dimX / 2.0f;
	shiftCenter(1, 0) = 0;
	shiftCenter(1, 1) = 1;
	shiftCenter(1, 2) = -dimY / 2.0f;
	shiftCenter(2, 0) = 0;
	shiftCenter(2, 1) = 0;
	shiftCenter(2, 2) = 1;

	shiftBack(0, 0) = 1;
	shiftBack(0, 1) = 0;
	shiftBack(0, 2) = dimX / 2.0f;
	shiftBack(1, 0) = 0;
	shiftBack(1, 1) = 1;
	shiftBack(1, 2) = dimY / 2.0f;
	shiftBack(2, 0) = 0;
	shiftBack(2, 1) = 0;
	shiftBack(2, 2) = 1;

	stretch(0, 0) = aAmount;
	stretch(0, 1) = 0;
	stretch(0, 2) = 0;
	stretch(1, 0) = 0;
	stretch(1, 1) = 1;
	stretch(1, 2) = 0;
	stretch(2, 0) = 0;
	stretch(2, 1) = 0;
	stretch(2, 2) = 1;

	return shiftBack * rotMat2 * stretch * rotMat1 * shiftCenter;
}

template<class TVol>
void Reconstructor::ForwardProjectionNoCTF(Volume<TVol>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync)
{	
	//proj_d.Memset(0);
	//dist_d.Memset(0);
	float runtime;

  	DeviceReconstructionConstantsCommon param = GetReconstructionParameters( *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

	if (!volumeIsEmpty)
	{
		// AS Replaced by GetReconstructionParameters <- SetConstantValues(fpKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
		//runtime = fpKernel(proj.GetWidth(), proj.GetHeight(), proj_d, dist_d, texVol);
		hipLaunchKernelGGL(march, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
			param, proj_d.GetWidth(), proj_d.GetHeight(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(), (float *)dist_d.GetDevicePtr(),
			texVol.GetTexObject(), make_int2(0, 0), make_int2(proj_d.GetWidth(), proj_d.GetHeight()));

#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)proj_d.GetDevicePtr(), (int)proj_d.GetPitch(), roiAll));
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
#endif

				// To avoid aliasing artifacts low pass filter to Nyquist of tomogram or Projection fourier filter, whichever is lower.
				if ((config.VoxelSize.x > 1) && (config.LimitToNyquist)) // assume cubic voxel sizes
				{
					int2 pA, pB, pC, pD;
					pA.x = 0;
					pA.y = proj.GetHeight() - 1;
					pB.x = proj.GetWidth() - 1;
					pB.y = proj.GetHeight() - 1;
					pC.x = 0;
					pC.y = 0;
					pD.x = proj.GetWidth() - 1;
					pD.y = 0;

					// cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
					hipLaunchKernelGGL(cropBorder, make_dim3(proj_d.GetWidth(), proj_d.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
										proj_d.GetWidth(), proj_d.GetHeight(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(),
                     					config.CutLength, config.DimLength, pA, pB, pC, pD);

					// cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);
					hipLaunchKernelGGL(makeSquare, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
										proj_d.GetWidth(), proj_d.GetHeight(), proj.GetMaxDimension(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(), (float *)projSquare_d.GetDevicePtr(),
                     					squareBorderSizeX, squareBorderSizeY, false, true);

					fft_d.Memset(0);
					cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

					// Set the low pass filter to nyquist of the Tomogram
					float lp = (float)proj.GetMaxDimension() / config.VoxelSize.x - 20.f;
					float lps = 20.f;
					// Use the low pass from the config if it's lower and filter is not skipped
					if (!config.SkipFilter) {
                        if (lp > (float) config.fourFilterLP) {
                            lp = (float) config.fourFilterLP;
                            lps = (float) config.fourFilterLPS;
                        }
                    }

					int size = proj.GetMaxDimension();

					//fourFilterKernel(fft_d, roiFFT.width * sizeof(Npp32fc), size, lp, 0, lps, 0);
					hipLaunchKernelGGL(fourierFilter, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
										(float2 *)fft_d.GetDevicePtr(), roiFFT.width * sizeof(Npp32fc), size, lp, 0, lps, 0);


					cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));

					nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), (float)(proj.GetMaxDimension() * proj.GetMaxDimension()),
						(Npp32f*)proj_d.GetDevicePtr(), (int)proj_d.GetPitch(), roiAll));


					// cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
					hipLaunchKernelGGL(cropBorder, make_dim3(proj_d.GetWidth(), proj_d.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
										proj_d.GetWidth(), proj_d.GetHeight(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(),
										config.CutLength, config.DimLength, pA, pB, pC, pD);

				}
#ifdef USE_MPI
			}
			else
			{
				proj_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif
	}
	else
	{
		// AS Replaced by GetReconstructionParameters <- SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

		//runtime = volTravLenKernel(proj.GetWidth(), proj.GetHeight(), dist_d);
		hipLaunchKernelGGL(volTraversalLength, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
			param, proj.GetWidth(), proj.GetHeight(), dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(), make_int2(0, 0), make_int2(proj.GetWidth(), proj.GetHeight()));
				
#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
			}
			else
			{
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif
	}	
}
template void Reconstructor::ForwardProjectionNoCTF(Volume<unsigned short>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);
template void Reconstructor::ForwardProjectionNoCTF(Volume<float>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);


template<class TVol>
void Reconstructor::ForwardProjectionCTF(Volume<TVol>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync)
{
	//proj_d.Memset(0);
	//dist_d.Memset(0);
	float runtime;
	int x = proj.GetWidth();
	int y = proj.GetHeight();

	//if (mpi_part == 0)
	//	printf("\n");

  	DeviceReconstructionConstantsCommon param = GetReconstructionParameters( *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

	if (!volumeIsEmpty)
	{
		//Forward projection is not done in WBP --> no need to adapt magAnisotropy
		//SetConstantValues(slicerKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
		//SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

		float t_in, t_out;
		GetDefocusDistances(t_in, t_out, index, vol);

		for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize())
		{
			dist_d.Memset(0);

			float defocusAngle = defocus.GetAstigmatismAngle(index) + (float)(proj.GetImageRotationToCompensate((uint)index) / M_PI * 180.0);
			float defocusMin;
			float defocusMax;
			GetDefocusMinMax(ray, index, defocusMin, defocusMax);

			if (mpi_part == 0)
			{
				printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b FP Defocus: %-8d nm", (int)defocusMin);
				fflush(stdout);
			}
			//printf("\n");
			//runtime = slicerKernel(x, y, dist_d, ray, ray + config.CTFSliceThickness / proj.GetPixelSize(), texVol);
			hipLaunchKernelGGL(slicer, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
					param, x, y, dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(), ray, ray + config.CTFSliceThickness / proj.GetPixelSize(), texVol.GetTexObject(), make_int2(0,0), make_int2(x,y));
				
#ifdef USE_MPI
			if (!noSync)
			{
				if (mpi_part == 0)
				{
					for (int mpi = 1; mpi < mpi_size; mpi++)
					{
						MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						realprojUS_d.CopyHostToDevice(MPIBuffer);
						nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
					}
				}
				else
				{
					dist_d.CopyDeviceToHost(MPIBuffer);
					MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
				}
			}
#endif
			//CTF filtering is only done on GPU 0!
			if (mpi_part == 0)
			{
				/*dist_d.CopyDeviceToHost(MPIBuffer);
				writeBMP(string("VorCTF.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/

				int2 pA, pB, pC, pD;
				pA.x = 0;
				pA.y = proj.GetHeight() - 1;
				pB.x = proj.GetWidth() - 1;
				pB.y = proj.GetHeight() - 1;
				pC.x = 0;
				pC.y = 0;
				pD.x = proj.GetWidth() - 1;
				pD.y = 0;

				// cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
				hipLaunchKernelGGL(cropBorder, make_dim3(proj_d.GetWidth(), proj_d.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
					dist_d.GetWidth(), dist_d.GetHeight(), dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(),
					config.CutLength, config.DimLength, pA, pB, pC, pD);

				// cts(dist_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);
				hipLaunchKernelGGL(makeSquare, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
					dist_d.GetWidth(), dist_d.GetHeight(), proj.GetMaxDimension(), dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(), (float *)projSquare_d.GetDevicePtr(),
					squareBorderSizeX, squareBorderSizeY, false, true);

				fft_d.Memset(0);
				cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

				//ctf(fft_d, defocusMin, defocusMax, defocusAngle, true, config.PhaseFlipOnly, config.WienerFilterNoiseLevel, (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), config.CTFBetaFac);
				hipLaunchKernelGGL(ctf, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0,
					mRecParamCtf, (cuComplex *)fft_d.GetDevicePtr(), (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), defocusMin * 0.000000001f, defocusMax * 0.000000001f, defocusAngle / 180.0f * (float)M_PI, true, config.PhaseFlipOnly, config.WienerFilterNoiseLevel, config.CTFBetaFac);

                // To avoid aliasing artifacts low pass filter to Nyquist of Tomogram or Projection fourier filter, whichever is lower.
				if ((config.VoxelSize.x > 1) && (config.LimitToNyquist)) // assume cubic voxel sizes
				{
                    // Set the low pass filter to nyquist of the Tomogram
                    float lp = (float)proj.GetMaxDimension() / config.VoxelSize.x - 20.f;
                    float lps = 20.f;
                    // Use the low pass from the config if it's lower
                    if (!config.SkipFilter) {
                        if (lp > (float) config.fourFilterLP) {
                            lp = (float) config.fourFilterLP;
                            lps = (float) config.fourFilterLPS;
                        }
                    }

					int size = proj.GetMaxDimension();

					//fourFilterKernel(fft_d, roiFFT.width * sizeof(Npp32fc), size, lp, 0, lps, 0);
					hipLaunchKernelGGL(fourierFilter, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
						(float2 *)fft_d.GetDevicePtr(), roiFFT.width * sizeof(Npp32fc), size, lp, 0, lps, 0);

				}

				cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));

				nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), (float)(proj.GetMaxDimension() * proj.GetMaxDimension()),
					(Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));


				// cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
				hipLaunchKernelGGL(cropBorder, make_dim3(proj_d.GetWidth(), proj_d.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
					dist_d.GetWidth(), dist_d.GetHeight(), dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(),
					config.CutLength, config.DimLength, pA, pB, pC, pD);
				nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), (Npp32f*)proj_d.GetDevicePtr(), (int)proj_d.GetPitch(), roiAll));
			}
		}
		/*proj_d.CopyDeviceToHost(MPIBuffer);
		writeBMP(string("Proj.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
		//Get Volume traversal lengths
		dist_d.Memset(0);
		//runtime = volTravLenKernel(x, y, dist_d);
		hipLaunchKernelGGL(volTraversalLength, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
			param, x, y, dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(), make_int2(0, 0), make_int2(x, y));
#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
				/*proj_d.CopyDeviceToHost(MPIBuffer);
				printf("\n");
				writeBMP(string("proj3.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());
				printf("\n");
				dist_d.CopyDeviceToHost(MPIBuffer);
				writeBMP(string("dist.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
			}
			else
			{
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif

	}
	else
	{
		//Forward projection is not done in WBP --> no need to adapt magAnisotropy
		//SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

		//runtime = volTravLenKernel(proj.GetWidth(), proj.GetHeight(), dist_d);
		hipLaunchKernelGGL(volTraversalLength, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
			param, proj.GetWidth(), proj.GetHeight(), dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(), make_int2(0, 0), make_int2(proj.GetWidth(), proj.GetHeight()));
#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
			}
			else
			{
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif
	}
}
template void Reconstructor::ForwardProjectionCTF(Volume<unsigned short>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);
template void Reconstructor::ForwardProjectionCTF(Volume<float>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);

template<class TVol>
void Reconstructor::ForwardProjectionNoCTFROI(Volume<TVol>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync)
{	
	//proj_d.Memset(0);
	//dist_d.Memset(0);
	float runtime;

  	DeviceReconstructionConstantsCommon param = GetReconstructionParameters( *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

	if (!volumeIsEmpty)
	{
		//Forward projection is not done in WBP --> no need to adapt magAnisotropy
		// AS Replaced by DeviceReconstructionConstantsCommon <- SetConstantValues(fpKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
		//runtime = fpKernel(proj.GetWidth(), proj.GetHeight(), proj_d, dist_d, texVol, roiMin, roiMax);
		hipLaunchKernelGGL(march, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
					param, proj_d.GetWidth(), proj_d.GetHeight(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(), (float *)dist_d.GetDevicePtr(),
					texVol.GetTexObject(), roiMin, roiMax);

#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)proj_d.GetDevicePtr(), (int)proj_d.GetPitch(), roiAll));
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}

#endif
                // To avoid aliasing artifacts low pass filter to Nyquist of Tomogram or Projection fourier filter, whichever is lower.
				if ((config.VoxelSize.x > 1) && (config.LimitToNyquist)) // assume cubic voxel sizes
				{
					int2 pA, pB, pC, pD;
					pA.x = 0;
					pA.y = proj.GetHeight() - 1;
					pB.x = proj.GetWidth() - 1;
					pB.y = proj.GetHeight() - 1;
					pC.x = 0;
					pC.y = 0;
					pD.x = proj.GetWidth() - 1;
					pD.y = 0;

					// cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
					hipLaunchKernelGGL(cropBorder, make_dim3(proj_d.GetWidth(), proj_d.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
						proj_d.GetWidth(), proj_d.GetHeight(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(),
						config.CutLength, config.DimLength, pA, pB, pC, pD);

					// cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);
					hipLaunchKernelGGL(makeSquare, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
					proj_d.GetWidth(), proj_d.GetHeight(), proj.GetMaxDimension(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(), (float *)projSquare_d.GetDevicePtr(),
					squareBorderSizeX, squareBorderSizeY, false, true);

					fft_d.Memset(0);
					cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

                    // Set the low pass filter to nyquist of the Tomogram
                    float lp = (float)proj.GetMaxDimension() / config.VoxelSize.x - 20.f;
                    float lps = 20.f;
                    // Use the low pass from the config if it's lower
                    if (!config.SkipFilter) {
                        if (lp > (float) config.fourFilterLP) {
                            lp = (float) config.fourFilterLP;
                            lps = (float) config.fourFilterLPS;
                        }
                    }

					int size = proj.GetMaxDimension();

					//fourFilterKernel(fft_d, roiFFT.width * sizeof(Npp32fc), size, lp, 0, lps, 0);
					hipLaunchKernelGGL(fourierFilter, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
						(float2 *)fft_d.GetDevicePtr(), roiFFT.width * sizeof(Npp32fc), size, lp, 0, lps, 0);


					cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));

					nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), (float)(proj.GetMaxDimension() * proj.GetMaxDimension()),
						(Npp32f*)proj_d.GetDevicePtr(), (int)proj_d.GetPitch(), roiAll));


					// cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
					hipLaunchKernelGGL(cropBorder, make_dim3(proj_d.GetWidth(), proj_d.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
						proj_d.GetWidth(), proj_d.GetHeight(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(),
						config.CutLength, config.DimLength, pA, pB, pC, pD);
				}
#ifdef USE_MPI
			}
			else
			{
				proj_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif
	}
	else
	{
		//Forward projection is not done in WBP --> no need to adapt magAnisotropy
		// AS Replaced by DeviceReconstructionConstantsCommon <- SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

		//runtime = volTravLenKernel(proj.GetWidth(), proj.GetHeight(), dist_d, roiMin, roiMax);
		hipLaunchKernelGGL(volTraversalLength, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
			param, proj.GetWidth(), proj.GetHeight(), dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(), roiMin, roiMax);
#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
			}
			else
			{
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif
	}	
}
template void Reconstructor::ForwardProjectionNoCTFROI(Volume<unsigned short>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
template void Reconstructor::ForwardProjectionNoCTFROI(Volume<float>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);


template<class TVol>
void Reconstructor::ForwardProjectionCTFROI(Volume<TVol>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync)
{
	//proj_d.Memset(0);
	//dist_d.Memset(0);
	float runtime;
	int x = proj.GetWidth();
	int y = proj.GetHeight();

	//if (mpi_part == 0)
	//	printf("\n");

  	DeviceReconstructionConstantsCommon param = GetReconstructionParameters( *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

	if (!volumeIsEmpty)
	{
		//Forward projection is not done in WBP --> no need to adapt magAnisotropy
		//SetConstantValues(slicerKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
		//SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

		float t_in, t_out;
		GetDefocusDistances(t_in, t_out, index, vol);
		/*t_in -= 2*config.CTFSliceThickness / proj.GetPixelSize();
		t_out += 2*config.CTFSliceThickness / proj.GetPixelSize();*/

		for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize())
		{
			dist_d.Memset(0);

			float defocusAngle = defocus.GetAstigmatismAngle(index) + (float)(proj.GetImageRotationToCompensate((uint)index) / M_PI * 180.0);
			float defocusMin;
			float defocusMax;
			GetDefocusMinMax(ray, index, defocusMin, defocusMax);

			if (mpi_part == 0)
			{
				printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b FP Defocus: %-8d nm", (int)defocusMin);
				fflush(stdout);
			}
			//printf("\n");
			//runtime = slicerKernel(x, y, dist_d, ray, ray + config.CTFSliceThickness / proj.GetPixelSize(), texVol, roiMin, roiMax);
			hipLaunchKernelGGL(slicer, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
				param, x, y, dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(), ray, ray + config.CTFSliceThickness / proj.GetPixelSize(), texVol.GetTexObject(), roiMin, roiMax);

#ifdef USE_MPI
			if (!noSync)
			{
				if (mpi_part == 0)
				{
					for (int mpi = 1; mpi < mpi_size; mpi++)
					{
						MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
						realprojUS_d.CopyHostToDevice(MPIBuffer);
						nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
					}
				}
				else
				{
					dist_d.CopyDeviceToHost(MPIBuffer);
					MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
				}
			}
#endif
			//CTF filtering is only done on GPU 0!
			if (mpi_part == 0)
			{
				/*dist_d.CopyDeviceToHost(MPIBuffer);
				writeBMP(string("VorCTF.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/

				int2 pA, pB, pC, pD;
				pA.x = 0;
				pA.y = proj.GetHeight() - 1;
				pB.x = proj.GetWidth() - 1;
				pB.y = proj.GetHeight() - 1;
				pC.x = 0;
				pC.y = 0;
				pD.x = proj.GetWidth() - 1;
				pD.y = 0;

				// cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
				hipLaunchKernelGGL(cropBorder, make_dim3(proj_d.GetWidth(), proj_d.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
					dist_d.GetWidth(), dist_d.GetHeight(), dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(),
					config.CutLength, config.DimLength, pA, pB, pC, pD);

				// cts(dist_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);
				hipLaunchKernelGGL(makeSquare, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
				dist_d.GetWidth(), dist_d.GetHeight(), proj.GetMaxDimension(), dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(), (float *)projSquare_d.GetDevicePtr(),
				squareBorderSizeX, squareBorderSizeY, false, true);

				fft_d.Memset(0);
				cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

				//ctf(fft_d, defocusMin, defocusMax, defocusAngle, true, config.PhaseFlipOnly, config.WienerFilterNoiseLevel, (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), config.CTFBetaFac);
				hipLaunchKernelGGL(ctf, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0,
					mRecParamCtf, (cuComplex *)fft_d.GetDevicePtr(), (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), defocusMin * 0.000000001f, defocusMax * 0.000000001f, defocusAngle / 180.0f * (float)M_PI, true, config.PhaseFlipOnly, config.WienerFilterNoiseLevel, config.CTFBetaFac);

                // To avoid aliasing artifacts low pass filter to Nyquist of Tomogram or Projection fourier filter, whichever is lower.
				if ((config.VoxelSize.x > 1) && (config.LimitToNyquist)) // assume cubic voxel sizes
				{
                    // Set the low pass filter to nyquist of the Tomogram
                    float lp = (float)proj.GetMaxDimension() / config.VoxelSize.x - 20.f;
                    float lps = 20.f;
                    // Use the low pass from the config if it's lower
                    if (!config.SkipFilter) {
                        if (lp > (float) config.fourFilterLP) {
                            lp = (float) config.fourFilterLP;
                            lps = (float) config.fourFilterLPS;
                        }
                    }

					int size = proj.GetMaxDimension();

					//fourFilterKernel(fft_d, roiFFT.width * sizeof(Npp32fc), size, lp, 0, lps, 0);
					hipLaunchKernelGGL(fourierFilter, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
						(float2 *)fft_d.GetDevicePtr(), roiFFT.width * sizeof(Npp32fc), size, lp, 0, lps, 0);

				}

				cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));

				nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), (float)(proj.GetMaxDimension() * proj.GetMaxDimension()),
					(Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));


				// cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
				hipLaunchKernelGGL(cropBorder, make_dim3(proj_d.GetWidth(), proj_d.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
					dist_d.GetWidth(), dist_d.GetHeight(), dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(),
					config.CutLength, config.DimLength, pA, pB, pC, pD);
				nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), (Npp32f*)proj_d.GetDevicePtr(), (int)proj_d.GetPitch(), roiAll));
			}
		}
		/*proj_d.CopyDeviceToHost(MPIBuffer);
		writeBMP(string("Proj.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
		//Get Volume traversal lengths
		dist_d.Memset(0);
		//runtime = volTravLenKernel(x, y, dist_d, roiMin, roiMax);
		hipLaunchKernelGGL(volTraversalLength, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
			param, x, y, dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(), roiMin, roiMax);
#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
				/*proj_d.CopyDeviceToHost(MPIBuffer);
				printf("\n");
				writeBMP(string("proj3.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());
				printf("\n");
				dist_d.CopyDeviceToHost(MPIBuffer);
				writeBMP(string("dist.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
			}
			else
			{
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif

	}
	else
	{
		//Forward projection is not done in WBP --> no need to adapt magAnisotropy
		//SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

		//runtime = volTravLenKernel(proj.GetWidth(), proj.GetHeight(), dist_d, roiMin, roiMax);
		hipLaunchKernelGGL(volTraversalLength, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
			param, proj.GetWidth(), proj.GetHeight(), dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(), roiMin, roiMax);
#ifdef USE_MPI
		if (!noSync)
		{
			if (mpi_part == 0)
			{
				for (int mpi = 1; mpi < mpi_size; mpi++)
				{
					MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					realprojUS_d.CopyHostToDevice(MPIBuffer);
					nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
				}
			}
			else
			{
				dist_d.CopyDeviceToHost(MPIBuffer);
				MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
#endif
	}
}
template void Reconstructor::ForwardProjectionCTFROI(Volume<unsigned short>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
template void Reconstructor::ForwardProjectionCTFROI(Volume<float>* vol, CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);

template<class TVol>
void Reconstructor::BackProjectionNoCTF(Volume<TVol>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount)
{
	float3 volDim = vol->GetSubVolumeDimension(mpi_part);
	// bpKernel.SetComputeSize((int)volDim.x, (int)volDim.y, (int)volDim.z);

	if (config.WBP_NoSART)
	{
		magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
		magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
	}

    // Find area shaded by volume, cut and dim borders
    int2 pA, pB, pC, pD;
    float2 hitA, hitB, hitC, hitD;
    proj.ComputeHitPoints(*vol, proj_index, pA, pB, pC, pD);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pA.x, (float)pA.y, hitA.x, hitA.y);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pB.x, (float)pB.y, hitB.x, hitB.y);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pC.x, (float)pC.y, hitC.x, hitC.y);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pD.x, (float)pD.y, hitD.x, hitD.y);
    pA.x = (int)hitA.x; pA.y = (int)hitA.y;
    pB.x = (int)hitB.x; pB.y = (int)hitB.y;
    pC.x = (int)hitC.x; pC.y = (int)hitC.y;
    pD.x = (int)hitD.x; pD.y = (int)hitD.y;
	// cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
	hipLaunchKernelGGL(cropBorder, make_dim3(proj_d.GetWidth(), proj_d.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
					proj_d.GetWidth(), proj_d.GetHeight(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(),
					config.CutLength, config.DimLength, pA, pB, pC, pD);
	
	// Prepare and execute Backprojection
	// SetConstantValues(bpKernel, *vol, proj, proj_index, mpi_part, magAnisotropy, magAnisotropyInv);
	DeviceReconstructionConstantsCommon param = GetReconstructionParameters( *vol, proj, proj_index, mpi_part, magAnisotropy, magAnisotropyInv);

	//float runtime = bpKernel(proj.GetWidth(), proj.GetHeight(), config.Lambda / SIRTCount, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage, surface, 0, 9999999999999.0f);
	if (config.FP16Volume)
	{
		hipLaunchKernelGGL(backProjectionFP16, make_dim3((int)volDim.x, (int)volDim.y, (int)volDim.z), make_dim3(1, 1, 1), 2 * (int)volDim.x* (int)volDim.y* (int)volDim.z * sizeof(float) * 4, 0, 
			param, proj.GetWidth(), proj.GetHeight(), config.Lambda / SIRTCount, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage.GetTexObject(), surface.GetSurfObject(), 0, 9999999999999.0f);
	}
	else
	{
		hipLaunchKernelGGL(backProjection, make_dim3((int)volDim.x, (int)volDim.y, (int)volDim.z), make_dim3(1, 1, 1), 2 * (int)volDim.x* (int)volDim.y* (int)volDim.z * sizeof(float) * 4, 0, 
			param, proj.GetWidth(), proj.GetHeight(), config.Lambda / SIRTCount, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage.GetTexObject(), surface.GetSurfObject(), 0, 9999999999999.0f);
	}
	

}
template void Reconstructor::BackProjectionNoCTF(Volume<unsigned short>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);
template void Reconstructor::BackProjectionNoCTF(Volume<float>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);


#ifdef SUBVOLREC_MODE
template<class TVol>
void Reconstructor::BackProjectionNoCTF(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index)
{
	size_t batchSize = subVolumes.size();

	if (config.WBP_NoSART)
	{
		magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
		magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
	}

	for (size_t batch = 0; batch < batchSize; batch++)
	{
		//bind surfref to correct array:
		//cudaSafeCall(cuSurfRefSetArray(surfref, vecArrays[batch]->GetCUarray(), 0));
		CudaSurfaceObject3D surface(vecArrays[batch]);

		//set additional shifts:
		proj.SetExtraShift(proj_index, vecExtraShifts[batch]);

		float3 volDim = subVolumes[batch]->GetSubVolumeDimension(0);
		// bpKernel.SetComputeSize((int)volDim.x, (int)volDim.y, (int)volDim.z);
		// SetConstantValues(bpKernel, *(subVolumes[batch]), proj, proj_index, 0, magAnisotropy, magAnisotropyInv);
		DeviceReconstructionConstantsCommon param = GetReconstructionParameters( *(subVolumes[batch]), proj, proj_index, 0, magAnisotropy, magAnisotropyInv);

		//float runtime = bpKernel(proj.GetWidth(), proj.GetHeight(), 1.0f, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage, surface, 0, 9999999999999.0f);
		if (config.FP16Volume)
		{
			hipLaunchKernelGGL(backProjectionFP16, make_dim3((int)volDim.x, (int)volDim.y, (int)volDim.z), make_dim3(1, 1, 1), 2 * (int)volDim.x* (int)volDim.y* (int)volDim.z * sizeof(float) * 4, 0, 
				param, proj.GetWidth(), proj.GetHeight(), 1.0f, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage.GetTexObject(), surface.GetSurfObject(), 0, 9999999999999.0f);
		}
		else
		{
			hipLaunchKernelGGL(backProjection, make_dim3((int)volDim.x, (int)volDim.y, (int)volDim.z), make_dim3(1, 1, 1), 2 * (int)volDim.x* (int)volDim.y* (int)volDim.z * sizeof(float) * 4, 0, 
				param, proj.GetWidth(), proj.GetHeight(), 1.0f, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage.GetTexObject(), surface.GetSurfObject(), 0, 9999999999999.0f);
		}
	}
}
template void Reconstructor::BackProjectionNoCTF(Volume<unsigned short>* vol, vector<Volume<unsigned short>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index);
template void Reconstructor::BackProjectionNoCTF(Volume<float>* vol, vector<Volume<float>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index);
#endif

template<class TVol>
void Reconstructor::BackProjectionCTF(Volume<TVol>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTcount)
{
	float3 volDim = vol->GetSubVolumeDimension(mpi_part);
	// bpKernel.SetComputeSize((int)volDim.x, (int)volDim.y, (int)volDim.z);
	float runtime;
	int x = proj.GetWidth();
	int y = proj.GetHeight();

	if (config.WBP_NoSART)
	{
		magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
		magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
	}

	if (mpi_part == 0)
		printf("\n");

	float t_in, t_out;
	GetDefocusDistances(t_in, t_out, proj_index, vol);

	// Find area shaded by volume, cut and dim borders
    int2 pA, pB, pC, pD;
    float2 hitA, hitB, hitC, hitD;
    proj.ComputeHitPoints(*vol, proj_index, pA, pB, pC, pD);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pA.x, (float)pA.y, hitA.x, hitA.y);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pB.x, (float)pB.y, hitB.x, hitB.y);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pC.x, (float)pC.y, hitC.x, hitC.y);
    MatrixVector3Mul(*(float3x3*)magAnisotropyInv.GetData(), (float)pD.x, (float)pD.y, hitD.x, hitD.y);
    pA.x = (int)hitA.x; pA.y = (int)hitA.y;
    pB.x = (int)hitB.x; pB.y = (int)hitB.y;
    pC.x = (int)hitC.x; pC.y = (int)hitC.y;
    pD.x = (int)hitD.x; pD.y = (int)hitD.y;
    // cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
	hipLaunchKernelGGL(cropBorder, make_dim3(proj_d.GetWidth(), proj_d.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
		proj_d.GetWidth(), proj_d.GetHeight(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(),
		config.CutLength, config.DimLength, pA, pB, pC, pD);


	for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize())
	{
		//SetConstantValues(bpKernel, *vol, proj, proj_index, mpi_part, magAnisotropy, magAnisotropyInv);
		DeviceReconstructionConstantsCommon param = GetReconstructionParameters( *vol, proj, proj_index, mpi_part, magAnisotropy, magAnisotropyInv);

		float defocusAngle = defocus.GetAstigmatismAngle(proj_index) + (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0);
		float defocusMin;
		float defocusMax;
		GetDefocusMinMax(ray, proj_index, defocusMin, defocusMax);
		float tiltAngle = (markers(MFI_TiltAngle, proj_index, 0) + config.AddTiltAngle) / 180.0f * (float)M_PI;
		
		if (mpi_part == 0)
		{
			printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b BP Defocus: %-8d nm", (int)defocusMin);
			fflush(stdout);
		}

		//cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);
		hipLaunchKernelGGL(makeSquare, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
				proj_d.GetWidth(), proj_d.GetHeight(), proj.GetMaxDimension(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(), (float *)projSquare_d.GetDevicePtr(),
				squareBorderSizeX, squareBorderSizeY, false, true);

		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));
		
		//ctf(fft_d, defocusMin, defocusMax, defocusAngle, false, config.PhaseFlipOnly, config.WienerFilterNoiseLevel, (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), config.CTFBetaFac);
		hipLaunchKernelGGL(ctf, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0,
			mRecParamCtf, (cuComplex *)fft_d.GetDevicePtr(), (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), defocusMin * 0.000000001f, defocusMax * 0.000000001f, defocusAngle / 180.0f * (float)M_PI, false, config.PhaseFlipOnly, config.WienerFilterNoiseLevel, config.CTFBetaFac);

		cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));
		nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), (float)(proj.GetMaxDimension() * proj.GetMaxDimension()),
			(Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));

		// cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
		hipLaunchKernelGGL(cropBorder, make_dim3(proj_d.GetWidth(), proj_d.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
					dist_d.GetWidth(), dist_d.GetHeight(), dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(),
					config.CutLength, config.DimLength, pA, pB, pC, pD);
		
		// runtime = bpKernel(x, y, config.Lambda / SIRTcount, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage, surface, ray, ray + config.CTFSliceThickness / proj.GetPixelSize());
		if (config.FP16Volume)
		{
			hipLaunchKernelGGL(backProjectionFP16, make_dim3((int)volDim.x, (int)volDim.y, (int)volDim.z), make_dim3(1, 1, 1), 2 * (int)volDim.x* (int)volDim.y* (int)volDim.z * sizeof(float) * 4, 0, 
				param, x, y, config.Lambda / SIRTcount, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage.GetTexObject(), surface.GetSurfObject(), ray, ray + config.CTFSliceThickness / proj.GetPixelSize());
		}
		else
		{
			hipLaunchKernelGGL(backProjection, make_dim3((int)volDim.x, (int)volDim.y, (int)volDim.z), make_dim3(1, 1, 1), 2 * (int)volDim.x* (int)volDim.y* (int)volDim.z * sizeof(float) * 4, 0, 
				param, x, y, config.Lambda / SIRTcount, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage.GetTexObject(), surface.GetSurfObject(), ray, ray + config.CTFSliceThickness / proj.GetPixelSize());
		}
	}
}
template void Reconstructor::BackProjectionCTF(Volume<unsigned short>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);
template void Reconstructor::BackProjectionCTF(Volume<float>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);

#ifdef SUBVOLREC_MODE
template<class TVol>
void Reconstructor::BackProjectionCTF(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index)
{
	size_t batchSize = subVolumes.size();
	//TODO
	//float3 volDim = vol->GetSubVolumeDimension(mpi_part);
	// bpKernel.SetComputeSize(config.SizeSubVol, config.SizeSubVol, config.SizeSubVol);

	float runtime;
	int x = proj.GetWidth();
	int y = proj.GetHeight();

	if (config.WBP_NoSART)
	{
		magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
		magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
	}

	if (mpi_part == 0)
		printf("\n");

	float t_in, t_out;
	GetDefocusDistances(t_in, t_out, proj_index, vol);

	for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize())
	{
		float defocusAngle = defocus.GetAstigmatismAngle(proj_index) + (float)(proj.GetImageRotationToCompensate((uint)proj_index) / M_PI * 180.0);
		float defocusMin;
		float defocusMax;
		GetDefocusMinMax(ray, proj_index, defocusMin, defocusMax);
		float tiltAngle = (markers(MFI_TiltAngle, proj_index, 0) + config.AddTiltAngle) / 180.0f * (float)M_PI;

		if (mpi_part == 0)
		{
			printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b BP Defocus: %-8d nm", (int)defocusMin);
			fflush(stdout);
		}

		//Do CTF correction:
		// cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);
		hipLaunchKernelGGL(makeSquare, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
			proj_d.GetWidth(), proj_d.GetHeight(), proj.GetMaxDimension(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(), (float *)projSquare_d.GetDevicePtr(),
			squareBorderSizeX, squareBorderSizeY, false, true);
		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

		//ctf(fft_d, defocusMin, defocusMax, defocusAngle, false, config.PhaseFlipOnly, config.WienerFilterNoiseLevel, (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), config.CTFBetaFac);
		hipLaunchKernelGGL(ctf, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0,
			mRecParamCtf, (cuComplex *)fft_d.GetDevicePtr(), (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), defocusMin * 0.000000001f, defocusMax * 0.000000001f, defocusAngle / 180.0f * (float)M_PI, false, config.PhaseFlipOnly, config.WienerFilterNoiseLevel, config.CTFBetaFac);

		cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));
		nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), (float)(proj.GetMaxDimension() * proj.GetMaxDimension()),
			(Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));


		for (size_t batch = 0; batch < batchSize; batch++)
		{
			//bind surfref to correct array:
			//cudaSafeCall(cuSurfRefSetArray(surfref, vecArrays[batch]->GetCUarray(), 0));
			CudaSurfaceObject3D surface(vecArrays[batch]);

			//set additional shifts:
			proj.SetExtraShift(proj_index, vecExtraShifts[batch]);

			float3 volDim = subVolumes[batch]->GetSubVolumeDimension(0);
			//bpKernel.SetComputeSize((int)volDim.x, (int)volDim.y, (int)volDim.z);

			// SetConstantValues(bpKernel, *(subVolumes[batch]), proj, proj_index, 0, magAnisotropy, magAnisotropyInv);
			DeviceReconstructionConstantsCommon param = GetReconstructionParameters( *(subVolumes[batch]), proj, proj_index, 0, magAnisotropy, magAnisotropyInv);

			//Most of the time, no volume should get hit...
			// runtime = bpKernel(x, y, 1.0f, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage, surface, ray, ray + config.CTFSliceThickness / proj.GetPixelSize());
			if (config.FP16Volume)
			{
				hipLaunchKernelGGL(backProjectionFP16, make_dim3((int)volDim.x, (int)volDim.y, (int)volDim.z), make_dim3(1, 1, 1), 2 * (int)volDim.x* (int)volDim.y* (int)volDim.z * sizeof(float) * 4, 0, 
					param, x, y, 1.0f, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage.GetTexObject(), surface.GetSurfObject(), ray, ray + config.CTFSliceThickness / proj.GetPixelSize());
			}
			else
			{
				hipLaunchKernelGGL(backProjection, make_dim3((int)volDim.x, (int)volDim.y, (int)volDim.z), make_dim3(1, 1, 1), 2 * (int)volDim.x* (int)volDim.y* (int)volDim.z * sizeof(float) * 4, 0, 
					param, x, y, 1.0f, config.OverSampling, 1.0f / (float)(config.OverSampling), texImage.GetTexObject(), surface.GetSurfObject(), ray, ray + config.CTFSliceThickness / proj.GetPixelSize());
			}
		}
	}
}
template void Reconstructor::BackProjectionCTF(Volume<unsigned short>* vol, vector<Volume<unsigned short>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index);
template void Reconstructor::BackProjectionCTF(Volume<float>* vol, vector<Volume<float>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index);
#endif

bool Reconstructor::ComputeFourFilter()
{
	//if (skipFilter)
	//{
	//	return false;
	//}

	//if (mpi_part != 0)
	//{
	//	return false;
	//}

	//float lp = config.fourFilterLP, hp = config.fourFilterHP, lps = config.fourFilterLPS, hps = config.fourFilterHPS;
	//int size = proj.GetMaxDimension();
	//float2* filter = new float2[size * size];
	//float2* fourFilter = new float2[(proj.GetMaxDimension() / 2 + 1) * proj.GetMaxDimension()];

	//if ((lp > size || lp < 0 || hp > size || hp < 0 || hps > size || hps < 0) && !skipFilter)
	//{
	//	//Filter parameters are not good!
	//	skipFilter = true;
	//	return false;
	//}

	//lp = lp - lps;
	//hp = hp + hps;


	//for (int y = 0; y < size; y++)
	//{
	//	for (int x = 0; x < size; x++)
	//	{
	//		float _x = -size / 2 + y;
	//		float _y = -size / 2 + x;

	//		float dist = (float)sqrtf(_x * _x + _y * _y);
	//		float fil = 0;
	//		//Low pass
	//		if (lp > 0)
	//		{
	//			if (dist <= lp) fil = 1;
	//		}
	//		else
	//		{
	//			if (dist <= size / 2 - 1) fil = 1;
	//		}

	//		//Gauss
	//		if (lps > 0)
	//		{
	//			float fil2;
	//			if (dist < lp) fil2 = 1;
	//			else fil2 = 0;

	//			fil2 = (-fil + 1.0f) * (float)expf(-((dist - lp) * (dist - lp) / (2 * lps * lps)));
	//			if (fil2 > 0.001f)
	//				fil = fil2;
	//		}

	//		if (lps > 0 && lp == 0 && hp == 0 && hps == 0)
	//			fil = (float)expf(-((dist - lp) * (dist - lp) / (2 * lps * lps)));

	//		if (hp > lp) return -1;

	//		if (hp > 0)
	//		{
	//			float fil2 = 0;
	//			if (dist >= hp) fil2 = 1;

	//			fil *= fil2;

	//			if (hps > 0)
	//			{
	//				float fil3 = 0;
	//				if (dist < hp) fil3 = 1;
	//				fil3 = (-fil2 + 1) * (float)expf(-((dist - hp) * (dist - hp) / (2 * hps * hps)));
	//				if (fil3 > 0.001f)
	//					fil = fil3;

	//			}
	//		}
	//		float2 filcplx;
	//		filcplx.x = fil;
	//		filcplx.y = 0;
	//		filter[y * size + x] = filcplx;
	//	}
	//}
	////writeBMP("test.bmp", test, size, size);

	//cuFloatComplex* filterTemp = new cuFloatComplex[size * (size / 2 + 1)];

	////Do FFT Shift in X direction (delete double coeffs)
	//for (int y = 0; y < size; y++)
	//{
	//	for (int x = size / 2; x <= size; x++)
	//	{
	//		int oldX = x;
	//		if (oldX == size) oldX = 0;
	//		int newX = x - size / 2;
	//		filterTemp[y * (size / 2 + 1) + newX] = filter[y * size + oldX];
	//	}
	//}
	////Do FFT Shift in Y direction
	//for (int y = 0; y < size; y++)
	//{
	//	for (int x = 0; x < size / 2 + 1; x++)
	//	{
	//		int oldY = y + size / 2;
	//		if (oldY >= size) oldY -= size;
	//		fourFilter[y * (size / 2 + 1) + x] = filterTemp[oldY * (size / 2 + 1) + x];
	//	}
	//}
	//
	//ctf_d.CopyHostToDevice(fourFilter);
	//delete[] filterTemp;
	//delete[] filter;
	//delete[] fourFilter;
	return true;
}

void Reconstructor::PrepareProjection(void * img_h, int proj_index, float & meanValue, float & StdValue, int & BadPixels)
{
	if (mpi_part != 0)
	{
		return;
	}

	if (projSource->GetDataType() == DT_SHORT)
	{
		//printf("SIGNED SHORT\n");
		cudaSafeCall(cuMemcpyHtoD(realprojUS_d.GetDevicePtr(), img_h, proj.GetWidth() * proj.GetHeight() * sizeof(short)));
		nppSafeCall(nppiConvert_16s32f_C1R((Npp16s*)realprojUS_d.GetDevicePtr(), proj.GetWidth() * sizeof(short), (Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
	}
	else if (projSource->GetDataType() == DT_USHORT)
	{
		//printf("UNSIGNED SHORT\n");
		cudaSafeCall(cuMemcpyHtoD(realprojUS_d.GetDevicePtr(), img_h, proj.GetWidth() * proj.GetHeight() * sizeof(short)));
		nppSafeCall(nppiConvert_16u32f_C1R((Npp16u*)realprojUS_d.GetDevicePtr(), proj.GetWidth() * sizeof(short), (Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
	}
	else if (projSource->GetDataType() == DT_INT)
	{
		realprojUS_d.CopyHostToDevice(img_h);
		nppSafeCall(nppiConvert_32s32f_C1R((Npp32s*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
	}
	else if (projSource->GetDataType() == DT_UINT)
	{
		realprojUS_d.CopyHostToDevice(img_h);
		nppSafeCall(nppiConvert_32u32f_C1R((Npp32u*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
	}
	else if (projSource->GetDataType() == DT_FLOAT)
	{
		realproj_d.CopyHostToDevice(img_h);
	}
	else
	{
		return;
	}

	projSquare_d.Memset(0);
	if (config.GetFileReadMode() == Configuration::Config::FRM_DM4)
	{
		// float t = cts(realproj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, true, false);
		hipLaunchKernelGGL(makeSquare, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
					realproj_d.GetWidth(), realproj_d.GetHeight(), proj.GetMaxDimension(), realproj_d.GetPitch(), (float *)realproj_d.GetDevicePtr(), (float *)projSquare_d.GetDevicePtr(),
					squareBorderSizeX, squareBorderSizeY, true, false);
	}
	else
	{
		// float t = cts(realproj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
		hipLaunchKernelGGL(makeSquare, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
					realproj_d.GetWidth(), realproj_d.GetHeight(), proj.GetMaxDimension(), realproj_d.GetPitch(), (float *)realproj_d.GetDevicePtr(), (float *)projSquare_d.GetDevicePtr(),
					squareBorderSizeX, squareBorderSizeY, false, false);
	}

	nppSafeCall(nppiMean_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));

	double mean = 0;
	meanval.CopyDeviceToHost(&mean);
	meanValue = (float)mean;

	if (config.CorrectBadPixels)
	{
		nppSafeCall(nppiCompareC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), config.BadPixelValue * meanValue, 
			(Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch(), roiSquare, NPP_CMP_GREATER));
	}
	else
	{
		nppSafeCall(nppiSet_8u_C1R(0, (Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch(), roiSquare));
	}

	nppSafeCall(nppiSum_8u_C1R((Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch(), roiSquare,
		(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));

	meanval.CopyDeviceToHost(&mean);
	BadPixels = (int)(mean / 255.0);

	nppSafeCall(nppiSet_32f_C1MR(meanValue, (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, 
		(Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch()));

	float normVal = 1;

	//When doing WBP we compute mean and std on the RAW image before Fourier filter and WBP weighting
	if (config.WBP_NoSART)
	{
		nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), 1.0f, 
			(Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), roiAll));

		nppSafeCall(nppiMean_StdDev_32f_C1R((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), roiAll,
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr(), (Npp64f*)stdval.GetDevicePtr()));

		mean = 0;
		meanval.CopyDeviceToHost(&mean);
		double std_h = 0;
		stdval.CopyDeviceToHost(&std_h);
		StdValue = (float)std_h;
		float std_hf = StdValue;

		meanValue = (float)(mean);
		float mean_hf = meanValue;

		if (config.ProjectionNormalization == Configuration::Config::PNM_MEAN)
		{
			std_hf = meanValue;
		}
		if (config.ProjectionNormalization == Configuration::Config::PNM_NONE)
		{
			std_hf = 1;
			mean_hf = 0;
		}
		if (config.DownWeightTiltsForWBP)
		{
			//we devide here because we devide later using nppiDivC: add the end we multiply!
			std_hf /= (float)cos(abs(markers(MarkerFileItem_enum::MFI_TiltAngle, proj_index, 0)) / 180.0 * M_PI);
		}

		nppSafeCall(nppiSubC_32f_C1R((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), mean_hf,
			(Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
		nppSafeCall(nppiDivC_32f_C1IR(-std_hf, (Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));


		//rotate image so that tilt axis lies parallel to image axis to avoid smearing if WBP-filter wedge:

		Matrix<double> shiftToCenter(3, 3);
		Matrix<double> rotate(3, 3);
		Matrix<double> shiftBack(3, 3);
		double psiAngle = proj.GetImageRotationToCompensate((uint)proj_index);

		shiftToCenter(0, 0) = 1; shiftToCenter(0, 1) = 0; shiftToCenter(0, 2) = -proj.GetWidth() / 2.0 + 0.5;
		shiftToCenter(1, 0) = 0; shiftToCenter(1, 1) = 1; shiftToCenter(1, 2) = -proj.GetHeight() / 2.0 + 0.5;
		shiftToCenter(2, 0) = 0; shiftToCenter(2, 1) = 0; shiftToCenter(2, 2) = 1;

		rotate(0, 0) = cos(psiAngle); rotate(0, 1) = -sin(psiAngle); rotate(0, 2) = 0;
		rotate(1, 0) = sin(psiAngle); rotate(1, 1) =  cos(psiAngle); rotate(1, 2) = 0;
		rotate(2, 0) = 0;             rotate(2, 1) = 0;              rotate(2, 2) = 1;

		shiftBack(0, 0) = 1; shiftBack(0, 1) = 0; shiftBack(0, 2) = proj.GetWidth() / 2.0 - 0.5;
		shiftBack(1, 0) = 0; shiftBack(1, 1) = 1; shiftBack(1, 2) = proj.GetHeight() / 2.0 - 0.5;
		shiftBack(2, 0) = 0; shiftBack(2, 1) = 0; shiftBack(2, 2) = 1;

		Matrix<double> rotationMatrix = shiftBack * (rotate * shiftToCenter);

		double affineMatrix[2][3];
		affineMatrix[0][0] = rotationMatrix(0, 0); affineMatrix[0][1] = rotationMatrix(0, 1); affineMatrix[0][2] = rotationMatrix(0, 2);
		affineMatrix[1][0] = rotationMatrix(1, 0); affineMatrix[1][1] = rotationMatrix(1, 1); affineMatrix[1][2] = rotationMatrix(1, 2);

		NppiSize imageSize;
		NppiRect roi;
		imageSize.width = proj.GetWidth();
		imageSize.height = proj.GetHeight();
		roi.x = 0;
		roi.y = 0;
		roi.width = proj.GetWidth();
		roi.height = proj.GetHeight();

		//dimBordersKernel(realproj_d, config.Crop, config.CropDim);
		hipLaunchKernelGGL(dimBorders, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
			realproj_d.GetWidth(), realproj_d.GetHeight(), realproj_d.GetPitch(), (float *)realproj_d.GetDevicePtr(), config.Crop, config.CropDim);
		realprojUS_d.Memset(0);

		nppSafeCall(nppiWarpAffine_32f_C1R((Npp32f*)realproj_d.GetDevicePtr(), imageSize, (int)realproj_d.GetPitch(), roi,
			(Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), roi, affineMatrix, NppiInterpolationMode::NPPI_INTER_CUBIC));

		// float t = cts(realprojUS_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
		hipLaunchKernelGGL(makeSquare, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
			realprojUS_d.GetWidth(), realprojUS_d.GetHeight(), proj.GetMaxDimension(), realprojUS_d.GetPitch(), (float *)realprojUS_d.GetDevicePtr(), (float *)projSquare_d.GetDevicePtr(),
			squareBorderSizeX, squareBorderSizeY, false, false);
		

//        float* test = new float[proj.GetMaxDimension()*proj.GetMaxDimension()];
//        projSquare_d.CopyDeviceToHost(test);
//
//        stringstream ss;
//        ss << "test_" << proj_index << ".em";
//        emwrite(ss.str(), (float*)test, proj.GetMaxDimension(), proj.GetMaxDimension());
//
//        delete[] test;
	}

	// good until here

	if (!skipFilter || config.WBP_NoSART)
	{
		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

		if (!skipFilter)
		{
			float lp = (float)config.fourFilterLP, hp = (float)config.fourFilterHP, lps = (float)config.fourFilterLPS, hps = (float)config.fourFilterHPS;
			int size = proj.GetMaxDimension();
			
			//fourFilterKernel(fft_d, roiFFT.width * sizeof(Npp32fc), size, lp, hp, lps, hps);
			hipLaunchKernelGGL(fourierFilter, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
						(float2 *)fft_d.GetDevicePtr(), roiFFT.width * sizeof(Npp32fc), size, lp, hp, lps, hps);

			/*nppSafeCall(nppiMul_32fc_C1IR((Npp32fc*)ctf_d.GetDevicePtr(), ctf_d.GetPitch(),
				(Npp32fc*)fft_d.GetDevicePtr(), roiFFT.width * sizeof(Npp32fc), roiFFT));*/
		}
		if (config.DoseWeighting)
		{
			//cout << "Dose: " << config.AccumulatedDose[proj_index] << "; Pixelsize: " << proj.GetPixelSize() * 10.0f << endl;
			/*Npp32fc* temp = new Npp32fc[roiFFT.width * roiFFT.height];
			float* temp2 = new float[roiFFT.width * roiFFT.height];
			fft_d.CopyDeviceToHost(temp);
			for (size_t i = 0; i < roiFFT.width * roiFFT.height; i++)
			{
				temp2[i] = sqrtf(temp[i].re * temp[i].re + temp[i].im * temp[i].im);
			}
			emwrite("before.em", temp2, roiFFT.width, roiFFT.height);*/
			//doseWeightingKernel(fft_d, roiFFT.width * sizeof(Npp32fc), proj.GetMaxDimension(), config.AccumulatedDose[proj_index], proj.GetPixelSize() * 10.0f);
			hipLaunchKernelGGL(doseWeighting, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
						(float2 *)fft_d.GetDevicePtr(), roiFFT.width * sizeof(Npp32fc), proj.GetMaxDimension(), config.AccumulatedDose[proj_index], proj.GetPixelSize() * 10.0f);
			
			/*fft_d.CopyDeviceToHost(temp);
			for (size_t i = 0; i < roiFFT.width * roiFFT.height; i++)
			{
				temp2[i] = temp[i].re;
			}
			emwrite("after.em", temp2, roiFFT.width, roiFFT.height);*/
		}
		if (config.WBP_NoSART)
		{
			if (config.WBPFilter == FM_EXACT)
			{
				float* tiltAngles = new float[projSource->GetProjectionCount()];
				for (int i = 0; i < projSource->GetProjectionCount(); i++)
				{
					tiltAngles[i] = markers(MFI_TiltAngle, i, 0) * M_PI / 180.0f;
					if (!markers.CheckIfProjIndexIsGood(i))
					{
						tiltAngles[i] = -999.0f;
					}
				}

				meanbuffer.CopyHostToDevice(tiltAngles, projSource->GetProjectionCount() * sizeof(float));
				delete[] tiltAngles;
			}

			float volumeHeight = config.RecDimensions.z;
			float voxelSize = config.VoxelSize.z;
			volumeHeight *= voxelSize;
#ifdef SUBVOLREC_MODE
			/*volumeHeight = config.SizeSubVol;
			voxelSize = config.VoxelSizeSubVol;*/
#endif
			float D = (proj.GetMaxDimension() / 2) / volumeHeight * 2.0f;


			//Do WBP weighting
            double psiAngle = markers(MFI_RotationPsi, (uint)proj_index, 0) / 180.0 * (double)M_PI;
            if (Configuration::Config::GetConfig().UseFixPsiAngle)
                psiAngle = Configuration::Config::GetConfig().PsiAngle / 180.0 * (double)M_PI;

            // Fix for WBP of rectangular images (otherwise stuff is rotated out too far)
            float flipAngle;
            if (abs(abs(psiAngle) - ((double)M_PI/2.)) < ((double)M_PI/4.)){
                flipAngle = 90.;
            } else {
                flipAngle = 0.;
            }

            printf("PSI ANGLE: %f \n", flipAngle);

            //wbp(fft_d, roiFFT.width * sizeof(Npp32fc), proj.GetMaxDimension(), flipAngle, config.WBPFilter, proj_index, projSource->GetProjectionCount(), D, meanbuffer);
			hipLaunchKernelGGL(wbpWeighting, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
				(cuComplex *)fft_d.GetDevicePtr(), roiFFT.width * sizeof(Npp32fc), proj.GetMaxDimension(), -flipAngle / 180.0f * (float)M_PI, config.WBPFilter, proj_index, projSource->GetProjectionCount(), D, (float *)meanbuffer.GetDevicePtr() );

			/*float2* test = new float2[fft_d.GetSize() / 4 / 2];
			float* test2 = new float[fft_d.GetSize() / 4 / 2];
			fft_d.CopyDeviceToHost(test);

			for (size_t i = 0; i < fft_d.GetSize() / 4 / 2; i++)
			{
				test2[i] = test[i].x;
			}

			stringstream ss;
			ss << "projFilter_" << proj_index << ".em";
			emwrite(ss.str(), test2, proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension());
			delete[] test;
			delete[] test2;*/

		}

		cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));

		normVal = (float)(proj.GetMaxDimension() * proj.GetMaxDimension());

//        float* test = new float[proj.GetMaxDimension()*proj.GetMaxDimension()];
//        projSquare_d.CopyDeviceToHost(test);
//
//        stringstream ss;
//        ss << "test_" << proj_index << ".em";
//        emwrite(ss.str(), (float*)test, proj.GetMaxDimension(), proj.GetMaxDimension());
//
//        delete[] test;
	}

	//Normalize from FFT
	nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), 
		normVal, (Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), roiAll));


	//When doing SART we compute mean and std on the filtered image
	if (!config.WBP_NoSART)
	{
		NppiSize roiForMean;
		Npp32f* ptr = (Npp32f*)realprojUS_d.GetDevicePtr();
		//When doing SART the projection must be mean free, so compute mean on center of image only for IMOD aligned stacks...
		if (config.ProjectionNormalization == Configuration::Config::PNM_NONE) //IMOD aligned stack
		{
			roiForMean.height = roiAll.height / 2;
			roiForMean.width = roiAll.width / 2;

			//Move start pointer:
			char* ptrChar = (char*)ptr;
			ptrChar += realprojUS_d.GetPitch() * (roiAll.height / 4); //Y
			ptr = (float*)ptrChar;
			ptr += roiAll.width / 4; //X

		}
		else
		{
			roiForMean.height = roiAll.height;
			roiForMean.width = roiAll.width;
		}

		nppSafeCall(nppiMean_StdDev_32f_C1R(ptr, (int)realprojUS_d.GetPitch(), roiForMean,
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr(), (Npp64f*)stdval.GetDevicePtr()));

		mean = 0;
		meanval.CopyDeviceToHost(&mean);
		double std_h = 0;
		stdval.CopyDeviceToHost(&std_h);
		StdValue = (float)std_h;
		float std_hf = StdValue;

		meanValue = (float)(mean);
		float mean_hf = meanValue;

		if (config.ProjectionNormalization == Configuration::Config::PNM_MEAN)
		{
			std_hf = meanValue;
		}
		if (config.ProjectionNormalization == Configuration::Config::PNM_NONE)
		{
			std_hf = 1;
			//meanValue = 0;
			//printf("I DID NAAAHHT.\n");
		}
        //mean_hf = 0.f;

		nppSafeCall(nppiSubC_32f_C1R((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), mean_hf,
			(Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
		nppSafeCall(nppiDivC_32f_C1IR(-std_hf, (Npp32f*)realproj_d.GetDevicePtr(), (int)realproj_d.GetPitch(), roiAll));
		realproj_d.CopyDeviceToHost(img_h);


	}
	else
	{
		realprojUS_d.CopyDeviceToHost(img_h);

//		stringstream ss;
//		ss << "test_" << proj_index << ".em";
//		emwrite(ss.str(), (float*)img_h, proj.GetWidth(), proj.GetHeight());
	}
}

template<class TVol>
void Reconstructor::Compare(Volume<TVol>* vol, char* originalImage, int index)
{
	if (mpi_part == 0)
	{
		float z_Direction = proj.GetNormalVector(index).z;
		float z_VolMinZ = vol->GetVolumeBBoxMin().z;
		float z_VolMaxZ = vol->GetVolumeBBoxMax().z;
		float volumeTraversalLength = fabs((DIST - z_VolMinZ) / z_Direction - (DIST - z_VolMaxZ) / z_Direction);

        realproj_d.CopyHostToDevice(originalImage);

		//nppiSet_32f_C1R(1.0f, (float*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);

		//float runtime = compKernel(realproj_d, proj_d, dist_d, volumeTraversalLength, config.Crop, config.CropDim, config.ProjectionScaleFactor);
		hipLaunchKernelGGL(compare, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
			(int)realproj_d.GetWidth(), (int)realproj_d.GetHeight(), realproj_d.GetPitch(), (float *)realproj_d.GetDevicePtr(), (float *)proj_d.GetDevicePtr(), (float *)dist_d.GetDevicePtr(), volumeTraversalLength, config.Crop, config.CropDim, config.ProjectionScaleFactor);
		/*proj_d.CopyDeviceToHost(MPIBuffer);
		writeBMP(string("Comp.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
	}
}
template void Reconstructor::Compare(Volume<unsigned short>* vol, char* originalImage, int index);
template void Reconstructor::Compare(Volume<float>* vol, char* originalImage, int index);

void Reconstructor::SubtractError(float* error)
{
    if (mpi_part == 0)
    {
        realproj_d.CopyHostToDevice(error);
        //float runtime = subEKernel(proj_d, realproj_d, dist_d);
		hipLaunchKernelGGL(subtract_error, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
			(int)proj_d.GetWidth(), (int)proj_d.GetHeight(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(), (float *)realproj_d.GetDevicePtr(), (float *)dist_d.GetDevicePtr());
    }
}


template<class TVol>
void Reconstructor::ForwardProjection(Volume<TVol>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync)
{
	if (config.CtfMode == Configuration::Config::CTFM_YES)
	{
		ForwardProjectionCTF(vol, texVol, index, volumeIsEmpty, noSync);
	}
	else
	{
		ForwardProjectionNoCTF(vol, texVol, index, volumeIsEmpty, noSync);
	}
}
template void Reconstructor::ForwardProjection(Volume<unsigned short>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync);
template void Reconstructor::ForwardProjection(Volume<float>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync);


template<class TVol>
void Reconstructor::PrintGeometry(Volume<TVol>* vol, int index)
{
	int x = proj.GetWidth();
	int y = proj.GetHeight();

	printf("\n\nProjection: %d\n", index);

	if (config.WBP_NoSART)
	{
		magAnisotropy = GetMagAnistropyMatrix(config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
		magAnisotropyInv = GetMagAnistropyMatrix(1.0f / config.MagAnisotropyAmount, config.MagAnisotropyAngleInDeg - (float)(proj.GetImageRotationToCompensate((uint)index) / M_PI * 180.0), (float)proj.GetWidth(), (float)proj.GetHeight());
	}

  	DeviceReconstructionConstantsCommon param = GetReconstructionParameters( *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

	//SetConstantValues(slicerKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
	//SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

	float t_in, t_out;
	GetDefocusDistances(t_in, t_out, index, vol);

	//Shoot ray from center of volume:
	float3 c_projNorm = proj.GetNormalVector(index);
	float3 c_detektor = proj.GetPosition(index);
	float3 MC_bBoxMin;
	float3 MC_bBoxMax;
	MC_bBoxMin = vol->GetVolumeBBoxMin();
	MC_bBoxMax = vol->GetVolumeBBoxMax();
	float3 volDim = vol->GetDimension();
	float3 hitPoint;
	float t;

	t = (c_projNorm.x * (MC_bBoxMin.x + (volDim.x * vol->GetVoxelSize().x * 0.5f)) +
		c_projNorm.y * (MC_bBoxMin.y + (volDim.y * vol->GetVoxelSize().y * 0.5f)) +
		c_projNorm.z * (MC_bBoxMin.z + (volDim.z * vol->GetVoxelSize().z * 0.5f)));
	t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
	t = abs(t);

	printf("t: %f\n", t);

	hitPoint.x = t * (-c_projNorm.x) + (MC_bBoxMin.x + (volDim.x * vol->GetVoxelSize().x * 0.5f));
	hitPoint.y = t * (-c_projNorm.y) + (MC_bBoxMin.y + (volDim.y * vol->GetVoxelSize().y * 0.5f));
	hitPoint.z = t * (-c_projNorm.z) + (MC_bBoxMin.z + (volDim.z * vol->GetVoxelSize().z * 0.5f));

	float4x4 c_DetectorMatrix;

	proj.GetDetectorMatrix(index, (float*)&c_DetectorMatrix, 1);
	MatrixVector3Mul(c_DetectorMatrix, &hitPoint);

	//--> pixelBorders.x = x.min; pixelBorders.z = y.min;
	float hitX = round(hitPoint.x);
	float hitY = round(hitPoint.y);

	printf("HitXY: %d %d\n", (int)hitX, (int)hitY);

	//Shoot ray from hit point on projection towards volume to get the distance to entry and exit point
	float3 pos = proj.GetPosition(index) + hitX * proj.GetPixelUPitch(index) + hitY * proj.GetPixelVPitch(index);
	hitX = (float)proj.GetWidth() * 0.5f;
	hitY = (float)proj.GetHeight() * 0.5f;
	float3 pos2 = proj.GetPosition(index) + hitX * proj.GetPixelUPitch(index) + hitX * proj.GetPixelVPitch(index);

	printf("Center: %f %f %f\n", pos2.x, pos2.y, pos2.z);

	float3 nvec = proj.GetNormalVector(index);

	t_in = 2 * -DIST;
	t_out = 2 * DIST;

	for (int x = 0; x <= 1; x++)
		for (int y = 0; y <= 1; y++)
			for (int z = 0; z <= 1; z++)
			{
				t = (nvec.x * (MC_bBoxMin.x + x * (MC_bBoxMax.x - MC_bBoxMin.x))
					+ nvec.y * (MC_bBoxMin.y + y * (MC_bBoxMax.y - MC_bBoxMin.y))
					+ nvec.z * (MC_bBoxMin.z + z * (MC_bBoxMax.z - MC_bBoxMin.z)));
				t += (-nvec.x * pos.x - nvec.y * pos.y - nvec.z * pos.z);

				if (t < t_in) t_in = t;
				if (t > t_out) t_out = t;
			}

	for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize())
	{
		dist_d.Memset(0);

		float defocusAngle = defocus.GetAstigmatismAngle(index) + (float)(proj.GetImageRotationToCompensate((uint)index) / M_PI * 180.0);
		float defocusMin;
		float defocusMax;
		GetDefocusMinMax(ray + config.CTFSliceThickness / proj.GetPixelSize() * 0.5f, index, defocusMin, defocusMax);

		printf("Defocus: %-8d nm\n", (int)defocusMin);
		
	}
}
template void Reconstructor::PrintGeometry(Volume<unsigned short>* vol, int index);
template void Reconstructor::PrintGeometry(Volume<float>* vol, int index);


template<class TVol>
void Reconstructor::ForwardProjectionROI(Volume<TVol>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync)
{
	if (config.CtfMode == Configuration::Config::CTFM_YES)
	{
		ForwardProjectionCTFROI(vol, texVol, index, volumeIsEmpty, roiMin, roiMax, noSync);
	}
	else
	{
		ForwardProjectionNoCTFROI(vol, texVol, index, volumeIsEmpty, roiMin, roiMax, noSync);
	}
}
template void Reconstructor::ForwardProjectionROI(Volume<unsigned short>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
template void Reconstructor::ForwardProjectionROI(Volume<float>* vol, CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);

template<typename TVol>
void Reconstructor::ForwardProjectionDistanceOnly(Volume<TVol>* vol, int index)
{
	DeviceReconstructionConstantsCommon param = GetReconstructionParameters( *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
	//SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

	//float runtime = volTravLenKernel(proj.GetWidth(), proj.GetHeight(), dist_d);
	hipLaunchKernelGGL(volTraversalLength, make_dim3(proj.GetWidth(), proj.GetHeight(), 1), make_dim3(1, 1, 1), 0, 0, 
			param, proj.GetWidth(), proj.GetHeight(), dist_d.GetPitch(), (float *)dist_d.GetDevicePtr(), make_int2(0, 0), make_int2(proj.GetWidth(), proj.GetHeight()));
#ifdef USE_MPI
	if (mpi_part == 0)
	{
		for (int mpi = 1; mpi < mpi_size; mpi++)
		{
			MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			realprojUS_d.CopyHostToDevice(MPIBuffer);
			nppSafeCall(nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), (int)realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), (int)dist_d.GetPitch(), roiAll));
		}
	}
	else
	{
		dist_d.CopyDeviceToHost(MPIBuffer);
		MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	}
#endif
}
template void Reconstructor::ForwardProjectionDistanceOnly(Volume<unsigned short>* vol, int index);
template void Reconstructor::ForwardProjectionDistanceOnly(Volume<float>* vol, int index);



template<class TVol>
void Reconstructor::BackProjection(Volume<TVol>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount)
{
	if (config.CtfMode == Configuration::Config::CTFM_YES)
	{
		BackProjectionCTF(vol, surface, proj_index, SIRTCount);
	}
	else
	{
		BackProjectionNoCTF(vol, surface, proj_index, SIRTCount);
	}
}

template void Reconstructor::BackProjection(Volume<unsigned short>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);
template void Reconstructor::BackProjection(Volume<float>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);


#ifdef SUBVOLREC_MODE
template<class TVol>
void Reconstructor::BackProjection(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index)
{
	if (config.CtfMode == Configuration::Config::CTFM_YES)
	{
		BackProjectionCTF(vol, subVolumes, vecExtraShifts, vecArrays, proj_index);
	}
	else
	{
		BackProjectionNoCTF(vol, subVolumes, vecExtraShifts, vecArrays, proj_index);
	}
}

template void Reconstructor::BackProjection(Volume<unsigned short>* vol, vector<Volume<unsigned short>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index);
template void Reconstructor::BackProjection(Volume<float>* vol, vector<Volume<float>*>& subVolumes, vector<float2>& vecExtraShifts, vector<CudaArray3D*>& vecArrays, int proj_index);
#endif

//template<class TVol>
//void Reconstructor::OneSARTStep(Volume<TVol>* vol, Cuda::CudaTextureObject3D & texVol, Cuda::CudaSurfaceObject3D& surface, int index, bool volumeIsEmpty, char * originalImage, float SIRTCount, float* MPIBuffer)
//{
//	ForwardProjection(vol, texVol, index, volumeIsEmpty);
//	if (mpi_part == 0)
//	{
//		Compare(vol, originalImage, index);
//		CopyProjectionToHost(MPIBuffer);
//	}
//
//	//spread the content to all other nodes:
//	MPIBroadcast(&MPIBuffer, 1);
//	CopyProjectionToDevice(MPIBuffer);
//	BackProjection(vol, surface, index, SIRTCount);
//}
//template void Reconstructor::OneSARTStep(Volume<unsigned short>* vol, Cuda::CudaTextureObject3D & texVol, Cuda::CudaSurfaceObject3D& surface, int index, bool volumeIsEmpty, char * originalImage, float SIRTCount, float* MPIBuffer);
//template void Reconstructor::OneSARTStep(Volume<float>* vol, Cuda::CudaTextureObject3D & texVol, Cuda::CudaSurfaceObject3D& surface, int index, bool volumeIsEmpty, char * originalImage, float SIRTCount, float* MPIBuffer);

//template<class TVol>
//void Reconstructor::BackProjectionWithPriorWBPFilter(Volume<TVol>* vol, int proj_index, char* originalImage, float* MPIBuffer)
//{
//	if (mpi_part == 0)
//	{
//		realproj_d.CopyHostToDevice(originalImage);
//		projSquare_d.Memset(0);
//		cts(realproj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
//
//		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));
//
//		//Do WBP weighting
//		wbp(fft_d, roiFFT.width * sizeof(Npp32fc), proj.GetMaxDimension(), markers(MFI_RotationPsi, proj_index, 0), FilterMethod::FM_RAMP);
//		cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));
//
//		float normVal = proj.GetMaxDimension() * proj.GetMaxDimension();
//
//		//Normalize from FFT
//		nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float),
//			normVal, (Npp32f*)proj_d.GetDevicePtr(), proj_d.GetPitch(), roiAll));
//
//		CopyProjectionToHost(MPIBuffer);
//	}
//	//spread the content to all other nodes:
//	MPIBroadcast(&MPIBuffer, 1);
//	CopyProjectionToDevice(MPIBuffer);
//	BackProjection(vol, proj_index, 1);
//}
//template void Reconstructor::BackProjectionWithPriorWBPFilter(Volume<unsigned short>* vol, int proj_index, char* originalImage, float* MPIBuffer);
//template void Reconstructor::BackProjectionWithPriorWBPFilter(Volume<float>* vol, int proj_index, char* originalImage, float* MPIBuffer);

//template<class TVol>
//void Reconstructor::RemoveProjectionFromVol(Volume<TVol>* vol, int proj_index, char* originalImage, float* MPIBuffer)
//{
//	if (mpi_part == 0)
//	{
//		realproj_d.CopyHostToDevice(originalImage);
//		projSquare_d.Memset(0);
//		cts(realproj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
//
//		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));
//
//		//Do WBP weighting
//		wbp(fft_d, roiFFT.width * sizeof(Npp32fc), proj.GetMaxDimension(), markers(MFI_RotationPsi, proj_index, 0), FilterMethod::FM_RAMP);
//		cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));
//
//		float normVal = proj.GetMaxDimension() * proj.GetMaxDimension();
//		//negate to remove from volume:
//		normVal *= -1;
//
//		//Normalize from FFT
//		nppSafeCall(nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float),
//			normVal, (Npp32f*)proj_d.GetDevicePtr(), proj_d.GetPitch(), roiAll));
//
//		CopyProjectionToHost(MPIBuffer);
//	}
//	//spread the content to all other nodes:
//	MPIBroadcast(&MPIBuffer, 1);
//	CopyProjectionToDevice(MPIBuffer);
//	BackProjection(vol, proj_index, 1);
//}
//template void Reconstructor::RemoveProjectionFromVol(Volume<unsigned short>* vol, int proj_index, char* originalImage, float* MPIBuffer);
//template void Reconstructor::RemoveProjectionFromVol(Volume<float>* vol, int proj_index, char* originalImage, float* MPIBuffer);

void Reconstructor::ResetProjectionsDevice()
{
	proj_d.Memset(0);
	dist_d.Memset(0);
}

void Reconstructor::CopyProjectionToHost(float * buffer)
{
	proj_d.CopyDeviceToHost(buffer);
}

void Reconstructor::CopyDistanceImageToHost(float * buffer)
{
	dist_d.CopyDeviceToHost(buffer);
}

void Reconstructor::CopyRealProjectionToHost(float * buffer)
{
	realproj_d.CopyDeviceToHost(buffer);
}

void Reconstructor::CopyProjectionToDevice(float * buffer)
{
	proj_d.CopyHostToDevice(buffer);
}

void Reconstructor::CopyDistanceImageToDevice(float * buffer)
{
	dist_d.CopyHostToDevice(buffer);
}

void Reconstructor::CopyRealProjectionToDevice(float * buffer)
{
	realproj_d.CopyHostToDevice(buffer);
}

void Reconstructor::MPIBroadcast(float ** buffers, int bufferCount)
{
#ifdef USE_MPI
	for (int i = 0; i < bufferCount; i++)
	{
		MPI_Bcast(buffers[i], proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, MPI_COMM_WORLD);
	}
#endif
}

#ifdef REFINE_MODE
void Reconstructor::CopyProjectionToSubVolumeProjection()
{
	if (mpi_part == 0)
	{
		projSubVols_d.CopyDeviceToDevice(proj_d);
	}
}
#endif

void Reconstructor::ConvertVolumeFP16(Volume<unsigned short>* vol, float * slice, Cuda::CudaSurfaceObject3D& surf, int z)
{
	if (volTemp_d.GetWidth() != config.RecDimensions.x ||
		volTemp_d.GetHeight() != config.RecDimensions.y)
	{
		volTemp_d.Alloc(config.RecDimensions.x * sizeof(float), config.RecDimensions.y, sizeof(float));
	}
	//convVolKernel(volTemp_d, surf, z);
	DeviceReconstructionConstantsCommon param = GetReconstructionParameters( *vol, proj, 0, 0, magAnisotropy, magAnisotropyInv);
	hipLaunchKernelGGL(convertVolumeFP16ToFP32, make_dim3(config.RecDimensions.x, config.RecDimensions.y, 1), make_dim3(1, 1, 1), 0, 0, 
		param, (float *)volTemp_d.GetDevicePtr(), volTemp_d.GetPitch(), surf.GetSurfObject(), z);
	volTemp_d.CopyDeviceToHost(slice);
}
//#define WRITEDEBUG 1
#ifdef REFINE_MODE
float2 Reconstructor::GetDisplacement(bool MultiPeakDetection, float* CCValue)
{
	float2 shift;
	shift.x = 0;
	shift.y = 0;
	
	if (mpi_part == 0)
	{
#ifdef WRITEDEBUG
		float* test = new float[proj.GetMaxDimension() * proj.GetMaxDimension()];
#endif
		/*float* test = new float[proj.GetWidth() * proj.GetHeight()];
		proj_d.CopyDeviceToHost(test);
		
		double summe = 0;
		for (size_t i = 0; i < proj.GetWidth() * proj.GetHeight(); i++)
		{
			summe += test[i];
		}
		emwrite("testCTF2.em", test, proj.GetWidth(), proj.GetHeight());
		delete[] test;*/

		// proj_d contains the original Projection minus the proj(reconstructionWithoutSubVols)
		// make square
		// cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
		hipLaunchKernelGGL(makeSquare, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
			proj_d.GetWidth(), proj_d.GetHeight(), proj.GetMaxDimension(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(), (float *)projSquare_d.GetDevicePtr(),
			squareBorderSizeX, squareBorderSizeY, false, false);

#ifdef WRITEDEBUG
		projSquare_d.CopyDeviceToHost(test);
		emwrite("projection3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif
        // Make mean free
		nppSafeCall(nppiMean_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));
		double MeanA = 0;
		meanval.CopyDeviceToHost(&MeanA, sizeof(double));
		nppSafeCall(nppiSubC_32f_C1IR((float)(MeanA), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

		// Square, and compute the sum of the squared projection
		nppSafeCall(nppiSqr_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), (Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

		nppSafeCall(nppiSum_32f_C1R((Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));

		double SumA = 0;
		meanval.CopyDeviceToHost(&SumA, sizeof(double)); // this now contains the square counter-intuitively

        // Real-to-Complex FFT of background subtracted REAL projection
		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));
		//fourFilterKernel(fft_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension(), config.fourFilterLP, 12, config.fourFilterLPS, 4);

		//missuse ctf_d as second fft variable
		//projSubVols_d contains the projection of the model
		// Make square
		// cts(projSubVols_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
		hipLaunchKernelGGL(makeSquare, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
			projSubVols_d.GetWidth(), projSubVols_d.GetHeight(), proj.GetMaxDimension(), projSubVols_d.GetPitch(), (float *)projSubVols_d.GetDevicePtr(), (float *)projSquare_d.GetDevicePtr(),
			squareBorderSizeX, squareBorderSizeY, false, false);

		// Make mean free
		nppSafeCall(nppiMean_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));
		double MeanB = 0;
		meanval.CopyDeviceToHost(&MeanB, sizeof(double));
		nppSafeCall(nppiSubC_32f_C1IR((float)(MeanB), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

		// Square, and compute the sum of the squared projection
		nppSafeCall(nppiSqr_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), (Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

		nppSafeCall(nppiSum_32f_C1R((Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));

		double SumB = 0;
		meanval.CopyDeviceToHost(&SumB, sizeof(double));

#ifdef WRITEDEBUG
		projSquare_d.CopyDeviceToHost(test);
		emwrite("realprojection3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif
        // Real-to-Complex FFT of FAKE projection
		cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)ctf_d.GetDevicePtr()));
		//fourFilterKernel(ctf_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension(), 150, 2, 20, 1);

		// Cross-correlation
		//conjKernel(fft_d, ctf_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension());
		hipLaunchKernelGGL(conjMul, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
			(cuComplex *)fft_d.GetDevicePtr(), (cuComplex *)ctf_d.GetDevicePtr(), (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension());

		// Get CC map
		cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));
#ifdef WRITEDEBUG
		projSquare_d.CopyDeviceToHost(test);
		emwrite("cc3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif
		
		int maxShift = 10;
#ifdef REFINE_MODE
		maxShift = config.MaxShift;
#endif
		// Normalize cross correlation result
		nppSafeCall(nppiDivC_32f_C1IR((float)(proj.GetMaxDimension() * proj.GetMaxDimension() * sqrt(SumA * SumB)), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

		//printf("Divs: %f %f\n", (float)SumA, (float)SumB);


		NppiSize ccSize;
		ccSize.width = roiCC1.width;
		ccSize.height = roiCC1.height;
		nppSafeCall(nppiCopy_32f_C1R(
			(float*)((char*)projSquare_d.GetDevicePtr() + roiCC1.y * proj.GetMaxDimension() * sizeof(float) + roiCC1.x * sizeof(float)),
			proj.GetMaxDimension() * sizeof(float),
			(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC1.y * ccMap_d.GetPitch() + roiDestCC1.x * sizeof(float)),
			(int)ccMap_d.GetPitch(), ccSize));

		nppSafeCall(nppiCopy_32f_C1R(
			(float*)((char*)projSquare_d.GetDevicePtr() + roiCC2.y * proj.GetMaxDimension() * sizeof(float) + roiCC2.x * sizeof(float)),
			proj.GetMaxDimension() * sizeof(float),
			(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC2.y * ccMap_d.GetPitch() + roiDestCC2.x * sizeof(float)),
			(int)ccMap_d.GetPitch(), ccSize));

		nppSafeCall(nppiCopy_32f_C1R(
			(float*)((char*)projSquare_d.GetDevicePtr() + roiCC3.y * proj.GetMaxDimension() * sizeof(float) + roiCC3.x * sizeof(float)),
			proj.GetMaxDimension() * sizeof(float),
			(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC3.y * ccMap_d.GetPitch() + roiDestCC3.x * sizeof(float)),
			(int)ccMap_d.GetPitch(), ccSize));

		nppSafeCall(nppiCopy_32f_C1R(
			(float*)((char*)projSquare_d.GetDevicePtr() + roiCC4.y * proj.GetMaxDimension() * sizeof(float) + roiCC4.x * sizeof(float)),
			proj.GetMaxDimension() * sizeof(float),
			(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC4.y * ccMap_d.GetPitch() + roiDestCC4.x * sizeof(float)),
			(int)ccMap_d.GetPitch(), ccSize));

		ccMap_d.CopyDeviceToHost(ccMap);

		//maxShiftKernel(projSquare_d, proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension(), maxShift);
		hipLaunchKernelGGL(maxShiftWeighted, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
			(float *)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension(), maxShift);

		nppSafeCall(nppiMaxIndx_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, 
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp32f*)meanval.GetDevicePtr(), (int*)stdval.GetDevicePtr(), 
			(int*)(stdval.GetDevicePtr() + sizeof(int))));


#ifdef WRITEDEBUG
		projSquare_d.CopyDeviceToHost(test);
		emwrite("shiftTest3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif

		int maxPixels[2];
		stdval.CopyDeviceToHost(maxPixels, 2 * sizeof(int));

		float maxVal;
		meanval.CopyDeviceToHost(&maxVal, sizeof(float));
		//printf("\nMaxVal: %f", maxVal);
		if (CCValue != NULL)
		{
			*CCValue = maxVal;
		}

		if (MultiPeakDetection)
		{
			//multiPeak
			nppSafeCall(nppiSet_8u_C1R(255, (Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch(), roiSquare));

			findPeakKernel(projSquare_d, proj.GetMaxDimension() * sizeof(float), badPixelMask_d, proj.GetMaxDimension(), maxVal * 0.9f);

			nppiSet_32f_C1R(1.0f, (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare);
			nppiSet_32f_C1MR(0.0f, (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, (Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch());

			maxShiftWeightedKernel(projSquare_d, proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension(), maxShift);


			nppSafeCall(nppiMaxIndx_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
				(Npp8u*)meanbuffer.GetDevicePtr(), (Npp32f*)meanval.GetDevicePtr(), (int*)stdval.GetDevicePtr(),
				(int*)(stdval.GetDevicePtr() + sizeof(int))));

			stdval.CopyDeviceToHost(maxPixels, 2 * sizeof(int));


			//NppiSize ccSize;
			ccSize.width = roiCC1.width;
			ccSize.height = roiCC1.height;
			nppSafeCall(nppiCopy_32f_C1R(
				(float*)((char*)projSquare_d.GetDevicePtr() + roiCC1.y * proj.GetMaxDimension() * sizeof(float) + roiCC1.x * sizeof(float)),
				proj.GetMaxDimension() * sizeof(float),
				(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC1.y * ccMap_d.GetPitch() + roiDestCC1.x * sizeof(float)),
				(int)ccMap_d.GetPitch(), ccSize));

			nppSafeCall(nppiCopy_32f_C1R(
				(float*)((char*)projSquare_d.GetDevicePtr() + roiCC2.y * proj.GetMaxDimension() * sizeof(float) + roiCC2.x * sizeof(float)),
				proj.GetMaxDimension() * sizeof(float),
				(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC2.y * ccMap_d.GetPitch() + roiDestCC2.x * sizeof(float)),
				(int)ccMap_d.GetPitch(), ccSize));

			nppSafeCall(nppiCopy_32f_C1R(
				(float*)((char*)projSquare_d.GetDevicePtr() + roiCC3.y * proj.GetMaxDimension() * sizeof(float) + roiCC3.x * sizeof(float)),
				proj.GetMaxDimension() * sizeof(float),
				(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC3.y * ccMap_d.GetPitch() + roiDestCC3.x * sizeof(float)),
				(int)ccMap_d.GetPitch(), ccSize));

			nppSafeCall(nppiCopy_32f_C1R(
				(float*)((char*)projSquare_d.GetDevicePtr() + roiCC4.y * proj.GetMaxDimension() * sizeof(float) + roiCC4.x * sizeof(float)),
				proj.GetMaxDimension() * sizeof(float),
				(float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC4.y * ccMap_d.GetPitch() + roiDestCC4.x * sizeof(float)),
				(int)ccMap_d.GetPitch(), ccSize));

			ccMap_d.CopyDeviceToHost(ccMapMulti);
		}

		//Get shift:
		shift.x = (float)maxPixels[0];
		shift.y = (float)maxPixels[1];

		if (shift.x > proj.GetMaxDimension() / 2)
		{
			shift.x -= proj.GetMaxDimension();
		}
		
		if (shift.y > proj.GetMaxDimension() / 2)
		{
			shift.y -= proj.GetMaxDimension();
		}

		if (maxVal <= 0)
		{
			//something went wrong, no shift found
			shift.x = -1000;
			shift.y = -1000;
		}
	}
	return shift;
}

//TODO: The output correlation values are not normalized (not in range 0 < v < 1), but this isn't strictly necessary here, so it would add useless computation. Maybe fix this later
float2 Reconstructor::GetDisplacementPC(bool MultiPeakDetection, float* CCValue)
{
    float2 shift;
    shift.x = 0;
    shift.y = 0;

    if (mpi_part == 0)
    {
#ifdef WRITEDEBUG
        float* test = new float[proj.GetMaxDimension() * proj.GetMaxDimension()];
#endif

        // proj_d contains the original Projection minus the proj(reconstructionWithoutSubVols)
        // make square
        // cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
		hipLaunchKernelGGL(makeSquare, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
			proj_d.GetWidth(), proj_d.GetHeight(), proj.GetMaxDimension(), proj_d.GetPitch(), (float *)proj_d.GetDevicePtr(), (float *)projSquare_d.GetDevicePtr(),
			squareBorderSizeX, squareBorderSizeY, false, false);
#ifdef WRITEDEBUG
        projSquare_d.CopyDeviceToHost(test);
		emwrite("projection3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif
        // Make mean free
        nppSafeCall(nppiMean_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
                                     (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));
        double MeanA = 0;
        meanval.CopyDeviceToHost(&MeanA, sizeof(double));
        nppSafeCall(nppiSubC_32f_C1IR((float)(MeanA), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

        // Square, and compute the sum of the squared projection
        //nppSafeCall(nppiSqr_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), (Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

        //nppSafeCall(nppiSum_32f_C1R((Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));

        //double SumA = 0;
        //meanval.CopyDeviceToHost(&SumA, sizeof(double)); // this now contains the square counter-intuitively

        // Real-to-Complex FFT of background subtracted REAL projection
        cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)fft_d.GetDevicePtr()));

        // missuse ctf_d as second fft variable
        // projSubVols_d contains the projection of the model
        // Make square
        // cts(projSubVols_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
		hipLaunchKernelGGL(makeSquare, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
			projSubVols_d.GetWidth(), projSubVols_d.GetHeight(), proj.GetMaxDimension(), projSubVols_d.GetPitch(), (float *)projSubVols_d.GetDevicePtr(), (float *)projSquare_d.GetDevicePtr(),
			squareBorderSizeX, squareBorderSizeY, false, false);

        // Make mean free
        nppSafeCall(nppiMean_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
                                     (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));
        double MeanB = 0;
        meanval.CopyDeviceToHost(&MeanB, sizeof(double));
        nppSafeCall(nppiSubC_32f_C1IR((float)(MeanB), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

        // Square, and compute the sum of the squared projection
        // nppSafeCall(nppiSqr_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), (Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

        // nppSafeCall(nppiSum_32f_C1R((Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr()));

        //double SumB = 0;
        //meanval.CopyDeviceToHost(&SumB, sizeof(double));

#ifdef WRITEDEBUG
        projSquare_d.CopyDeviceToHost(test);
		emwrite("realprojection3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif
        // Real-to-Complex FFT of FAKE projection
        cufftSafeCall(cufftExecR2C(handleR2C, (cufftReal*)projSquare_d.GetDevicePtr(), (cufftComplex*)ctf_d.GetDevicePtr()));
        //fourFilterKernel(ctf_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension(), 150, 2, 20, 1);

        // Phase-correlation
        //pcKernel(fft_d, ctf_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension());
		hipLaunchKernelGGL(conjMulPC, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
			(cuComplex *)fft_d.GetDevicePtr(), (cuComplex *)ctf_d.GetDevicePtr(), (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension());

        //Cuda::CudaDeviceVariable& img, size_t stride, int pixelcount, float lp, float hp, float lps, float hps
        //fourFilterKernel(fft_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension(), config.PhaseCorrSigma, 0, config.PhaseCorrSigma, 0);
		hipLaunchKernelGGL(fourierFilter, make_dim3(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
			(float2 *)fft_d.GetDevicePtr(), (proj.GetMaxDimension() / 2 + 1) * sizeof(cuComplex), proj.GetMaxDimension(), config.PhaseCorrSigma, 0, config.PhaseCorrSigma, 0);

        // Get CC map (transform back)
        cufftSafeCall(cufftExecC2R(handleC2R, (cufftComplex*)fft_d.GetDevicePtr(), (cufftReal*)projSquare_d.GetDevicePtr()));
#ifdef WRITEDEBUG
        projSquare_d.CopyDeviceToHost(test);
		emwrite("cc3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif

        int maxShift = 10;
#ifdef REFINE_MODE
        maxShift = config.MaxShift;
#endif
        // Normalize cross correlation result
        //nppSafeCall(nppiDivC_32f_C1IR((float)(proj.GetMaxDimension() * proj.GetMaxDimension() * sqrt(SumA * SumB)), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare));

        //printf("Divs: %f %f\n", (float)SumA, (float)SumB);

        // FFT-shift using NPPI
        NppiSize ccSize;
        ccSize.width = roiCC1.width;
        ccSize.height = roiCC1.height;
        nppSafeCall(nppiCopy_32f_C1R(
                (float*)((char*)projSquare_d.GetDevicePtr() + roiCC1.y * proj.GetMaxDimension() * sizeof(float) + roiCC1.x * sizeof(float)),
                proj.GetMaxDimension() * sizeof(float),
                (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC1.y * ccMap_d.GetPitch() + roiDestCC1.x * sizeof(float)),
                (int)ccMap_d.GetPitch(), ccSize));

        nppSafeCall(nppiCopy_32f_C1R(
                (float*)((char*)projSquare_d.GetDevicePtr() + roiCC2.y * proj.GetMaxDimension() * sizeof(float) + roiCC2.x * sizeof(float)),
                proj.GetMaxDimension() * sizeof(float),
                (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC2.y * ccMap_d.GetPitch() + roiDestCC2.x * sizeof(float)),
                (int)ccMap_d.GetPitch(), ccSize));

        nppSafeCall(nppiCopy_32f_C1R(
                (float*)((char*)projSquare_d.GetDevicePtr() + roiCC3.y * proj.GetMaxDimension() * sizeof(float) + roiCC3.x * sizeof(float)),
                proj.GetMaxDimension() * sizeof(float),
                (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC3.y * ccMap_d.GetPitch() + roiDestCC3.x * sizeof(float)),
                (int)ccMap_d.GetPitch(), ccSize));

        nppSafeCall(nppiCopy_32f_C1R(
                (float*)((char*)projSquare_d.GetDevicePtr() + roiCC4.y * proj.GetMaxDimension() * sizeof(float) + roiCC4.x * sizeof(float)),
                proj.GetMaxDimension() * sizeof(float),
                (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC4.y * ccMap_d.GetPitch() + roiDestCC4.x * sizeof(float)),
                (int)ccMap_d.GetPitch(), ccSize));

        ccMap_d.CopyDeviceToHost(ccMap);


        //maxShiftKernel(projSquare_d, proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension(), maxShift);
		hipLaunchKernelGGL(maxShiftWeighted, make_dim3(proj.GetMaxDimension(), proj.GetMaxDimension(), 1), make_dim3(1, 1, 1), 0, 0, 
			(float *)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension(), maxShift);

        nppSafeCall(nppiMaxIndx_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
                                        (Npp8u*)meanbuffer.GetDevicePtr(), (Npp32f*)meanval.GetDevicePtr(), (int*)stdval.GetDevicePtr(),
                                        (int*)(stdval.GetDevicePtr() + sizeof(int))));


#ifdef WRITEDEBUG
        projSquare_d.CopyDeviceToHost(test);
		emwrite("shiftTest3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif

        int maxPixels[2];
        stdval.CopyDeviceToHost(maxPixels, 2 * sizeof(int));

        float maxVal;
        meanval.CopyDeviceToHost(&maxVal, sizeof(float));
        //printf("\nMaxVal: %f", maxVal);
        if (CCValue != NULL)
        {
            *CCValue = maxVal;
        }

        if (MultiPeakDetection)
        {
            //multiPeak
            nppSafeCall(nppiSet_8u_C1R(255, (Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch(), roiSquare));

            findPeakKernel(projSquare_d, proj.GetMaxDimension() * sizeof(float), badPixelMask_d, proj.GetMaxDimension(), maxVal * 0.9f);

            nppiSet_32f_C1R(1.0f, (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare);
            nppiSet_32f_C1MR(0.0f, (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, (Npp8u*)badPixelMask_d.GetDevicePtr(), (int)badPixelMask_d.GetPitch());

            maxShiftWeightedKernel(projSquare_d, proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension(), maxShift);


            nppSafeCall(nppiMaxIndx_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
                                            (Npp8u*)meanbuffer.GetDevicePtr(), (Npp32f*)meanval.GetDevicePtr(), (int*)stdval.GetDevicePtr(),
                                            (int*)(stdval.GetDevicePtr() + sizeof(int))));

            stdval.CopyDeviceToHost(maxPixels, 2 * sizeof(int));


            //NppiSize ccSize;
            ccSize.width = roiCC1.width;
            ccSize.height = roiCC1.height;
            nppSafeCall(nppiCopy_32f_C1R(
                    (float*)((char*)projSquare_d.GetDevicePtr() + roiCC1.y * proj.GetMaxDimension() * sizeof(float) + roiCC1.x * sizeof(float)),
                    proj.GetMaxDimension() * sizeof(float),
                    (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC1.y * ccMap_d.GetPitch() + roiDestCC1.x * sizeof(float)),
                    (int)ccMap_d.GetPitch(), ccSize));

            nppSafeCall(nppiCopy_32f_C1R(
                    (float*)((char*)projSquare_d.GetDevicePtr() + roiCC2.y * proj.GetMaxDimension() * sizeof(float) + roiCC2.x * sizeof(float)),
                    proj.GetMaxDimension() * sizeof(float),
                    (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC2.y * ccMap_d.GetPitch() + roiDestCC2.x * sizeof(float)),
                    (int)ccMap_d.GetPitch(), ccSize));

            nppSafeCall(nppiCopy_32f_C1R(
                    (float*)((char*)projSquare_d.GetDevicePtr() + roiCC3.y * proj.GetMaxDimension() * sizeof(float) + roiCC3.x * sizeof(float)),
                    proj.GetMaxDimension() * sizeof(float),
                    (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC3.y * ccMap_d.GetPitch() + roiDestCC3.x * sizeof(float)),
                    (int)ccMap_d.GetPitch(), ccSize));

            nppSafeCall(nppiCopy_32f_C1R(
                    (float*)((char*)projSquare_d.GetDevicePtr() + roiCC4.y * proj.GetMaxDimension() * sizeof(float) + roiCC4.x * sizeof(float)),
                    proj.GetMaxDimension() * sizeof(float),
                    (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC4.y * ccMap_d.GetPitch() + roiDestCC4.x * sizeof(float)),
                    (int)ccMap_d.GetPitch(), ccSize));

            ccMap_d.CopyDeviceToHost(ccMapMulti);
        }

        //Get shift:
        shift.x = (float)maxPixels[0];
        shift.y = (float)maxPixels[1];

        if (shift.x > proj.GetMaxDimension() / 2)
        {
            shift.x -= proj.GetMaxDimension();
        }

        if (shift.y > proj.GetMaxDimension() / 2)
        {
            shift.y -= proj.GetMaxDimension();
        }

        if (maxVal <= 0)
        {
            //something went wrong, no shift found
            shift.x = -1000;
            shift.y = -1000;
        }
    }
    return shift;
}

void Reconstructor::rotVol(Cuda::CudaDeviceVariable & vol, float phi, float psi, float theta)
{
	rotKernel(vol, phi, psi, theta);
}

void Reconstructor::setRotVolData(float * data)
{
	rotKernel.SetData(data);
}
float * Reconstructor::GetCCMap()
{
	return ccMap;
}
float * Reconstructor::GetCCMapMulti()
{
	return ccMapMulti;
}

void Reconstructor::GetCroppedProjection(float *outImage, int2 roiMin, int2 roiMax) {

    int outW = roiMax.x-roiMin.x + 1;
    int outH = roiMax.y-roiMin.y + 1;
    //printf("outW: %i outH: %i \n", outW, outH);
    memset(outImage, 0, outW*outH*sizeof(float));

    auto buffer = new float[proj.GetHeight()*proj.GetWidth()];
    proj_d.CopyDeviceToHost(buffer);

    //stringstream ss;
    //ss << "projjjjjj.em";
    //emwrite(ss.str(), buffer, proj.GetWidth(), proj.GetHeight());

    for (int x = roiMin.x; x < roiMax.x+1; x++){
        for (int y = roiMin.y; y < roiMax.y+1; y++){
            if(x > proj.GetWidth()-1) continue;
            if(y > proj.GetHeight()-1) continue;

            if(x < 0) continue;
            if(y < 0) continue;

            int xx = x-roiMin.x;
            int yy = y-roiMin.y;

            outImage[xx+outW*yy] = buffer[x+proj.GetWidth()*y];
            //printf("%s", typeid(buffer).name());
            //printf("xx: %i yy: %i x: %i y: %i buffer: %f out: %f\n", xx, yy, x, y, buffer[y+proj.GetHeight()*x], outImage[yy+outH*xx]);
        }
    }

    delete[] buffer;
}

void Reconstructor::GetCroppedProjection(float *outImage, float *inImage, int2 roiMin, int2 roiMax) {

    int outW = roiMax.x-roiMin.x + 1;
    int outH = roiMax.y-roiMin.y + 1;
    //printf("outW: %i outH: %i \n", outW, outH);
    memset(outImage, 0, outW*outH*sizeof(float));

    //auto buffer = new float[proj.GetHeight()*proj.GetWidth()];
    //proj_d.CopyDeviceToHost(buffer);

    //stringstream ss;
    //ss << "projjjjjj.em";
    //emwrite(ss.str(), buffer, proj.GetWidth(), proj.GetHeight());

    for (int x = roiMin.x; x < roiMax.x+1; x++){
        for (int y = roiMin.y; y < roiMax.y+1; y++){
            if(x > proj.GetWidth()-1) continue;
            if(y > proj.GetHeight()-1) continue;

            if(x < 0) continue;
            if(y < 0) continue;

            int xx = x-roiMin.x;
            int yy = y-roiMin.y;

            outImage[xx+outW*yy] = inImage[x+proj.GetWidth()*y];
            //printf("%s", typeid(buffer).name());
            //printf("xx: %i yy: %i x: %i y: %i buffer: %f out: %f\n", xx, yy, x, y, buffer[y+proj.GetHeight()*x], outImage[yy+outH*xx]);
        }
    }
}

#endif

