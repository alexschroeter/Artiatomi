#ifndef DEFAULT_H
#define DEFAULT_H

#ifndef _USE_MATH_DEFINES
#define _USE_MATH_DEFINES
#endif

#ifdef WIN32
typedef long long long64;
typedef unsigned long long ulong64;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
#else
typedef long long long64;
typedef unsigned long long ulong64;
typedef unsigned int uint;
typedef unsigned short ushort;
typedef unsigned char uchar;
#endif


#ifdef WIN32
#include <string>
#include <math.h>
#include <float.h>
#else
#include <string.h>
#include <float.h>
#include <math.h>
#endif

#include <hip/hip_runtime.h>

#include <stdio.h>
#include <vector>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <limits.h>
#include <sstream>

#include "hip_kernels/cutil_math.h"

inline dim3 make_dim3(uint a, uint b, uint c)
{
	dim3 ret;
	ret.x = a;
	ret.y = b;
	ret.z = c;
	return ret;
}

inline dim3 make_dim3(uint3 val)
{
	dim3 ret;
	ret.x = val.x;
	ret.y = val.y;
	ret.z = val.z;
	return ret;
}

//inline float3 make_float3(uint3 val)
//{
//	float3 ret;
//	ret.x = val.x;
//	ret.y = val.y;
//	ret.z = val.z;
//	return ret;
//}
//
//inline uint3 make_uint3(uint a, uint b, uint c)
//{
//	uint3 ret;
//	ret.x = a;
//	ret.y = b;
//	ret.z = c;
//	return ret;
//}
//
//inline uint3 make_uint3(dim3 val)
//{
//	uint3 ret;
//	ret.x = val.x;
//	ret.y = val.y;
//	ret.z = val.z;
//	return ret;
//}
//
//inline float3 make_uint3(dim3 val)
//{
//	uint3 ret;
//	ret.x = val.x;
//	ret.y = val.y;
//	ret.z = val.z;
//	return ret;
//}





#endif
