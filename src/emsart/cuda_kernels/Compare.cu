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


/**********************************************
*
* CUDA SART FRAMEWORK
* 2009,2010 Michael Kunz, Lukas Marsalek
* 
*
* Compare.cu
* Compare kernel. Compares real and virtual
* projection and include volume traversal
* length per pixel.
*
**********************************************/
#ifndef COMPARE_CU
#define COMPARE_CU


#include <cuda.h>
#include "cutil.h"
#include "cutil_math.h"

#include "DeviceVariables.cuh"
#include "float.h"


extern "C"
__global__ 
void compare(int proj_x, int proj_y, size_t stride, float* real_raw, float* virtual_raw, float* vol_distance_map, float realLength, float4 cutLength, float4 dimLength, float projValScale)
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

extern "C"
__global__
void dimBorders(int proj_x, int proj_y, size_t stride, float* image, float4 cutLength, float4 dimLength)
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

extern "C"
__global__
void subtract_error(int proj_x, int proj_y, size_t stride, float* real_raw, const float* error, const float* vol_distance_map)
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

extern "C"
__global__ 
void cropBorder(int proj_x, int proj_y, size_t stride, float* image, float2 cutLength, float2 dimLength, int2 p1, int2 p2, int2 p3, int2 p4)
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

#endif