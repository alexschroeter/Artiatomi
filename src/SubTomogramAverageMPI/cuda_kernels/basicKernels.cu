//Includes for IntelliSense 
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif


#define EPS (0.000001f)

#include <cuda.h>
#include <device_launch_parameters.h>
#include <texture_fetch_functions.h>
#include "float.h"
#include <builtin_types.h>
#include <vector_functions.h>

texture<float, 3, cudaReadModeElementType> texVol;
texture<float, 3, cudaReadModeElementType> texShift;
texture<float2, 3, cudaReadModeElementType> texVolCplx;

extern "C"
__global__ void rot3d(int size, float3 rotMat0, float3 rotMat1, float3 rotMat2, float* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	

	float center = size / 2;

	float3 vox = make_float3(x - center, y - center, z - center);
	float3 rotVox;
	rotVox.x = center + rotMat0.x * vox.x + rotMat1.x * vox.y + rotMat2.x * vox.z;
	rotVox.y = center + rotMat0.y * vox.x + rotMat1.y * vox.y + rotMat2.y * vox.z;
	rotVox.z = center + rotMat0.z * vox.x + rotMat1.z * vox.y + rotMat2.z * vox.z;

	outVol[z * size * size + y * size + x] = tex3D(texVol, rotVox.x + 0.5f, rotVox.y + 0.5f, rotVox.z + 0.5f);

}

extern "C"
__global__ void rot3dCplx(int size, float3 rotMat0, float3 rotMat1, float3 rotMat2, float2* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float center = size / 2;

	float3 vox = make_float3(x - center, y - center, z - center);
	float3 rotVox;
	rotVox.x = center + rotMat0.x * vox.x + rotMat1.x * vox.y + rotMat2.x * vox.z;
	rotVox.y = center + rotMat0.y * vox.x + rotMat1.y * vox.y + rotMat2.y * vox.z;
	rotVox.z = center + rotMat0.z * vox.x + rotMat1.z * vox.y + rotMat2.z * vox.z;

	outVol[z * size * size + y * size + x] = tex3D(texVolCplx, rotVox.x + 0.5f, rotVox.y + 0.5f, rotVox.z + 0.5f);
}


extern "C"
	__global__ void shift(int size, float* outVol, float3 shift)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float sx = float(x - shift.x + 0.5f) / float(size);
	float sy = float(y - shift.y + 0.5f) / float(size);
	float sz = float(z - shift.z + 0.5f) / float(size); 
	
	outVol[z * size * size + y * size + x] = tex3D(texShift, sx, sy, sz);
}


extern "C"
	__global__ void sub(int size, float* inVol, float* outVol, float val)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	outVol[z * size * size + y * size + x] = inVol[z * size * size + y * size + x] - val;
}


extern "C"
	__global__ void add(int size, float* inVol, float* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	outVol[z * size * size + y * size + x] += inVol[z * size * size + y * size + x];
}


extern "C"
__global__ void subCplx(int size, float2* inVol, float2* outVol, float* subval, float divVal)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = inVol[z * size * size + y * size + x];
	temp.x -= subval[0] / divVal;
	outVol[z * size * size + y * size + x] = temp;
}


extern "C"
__global__ void wedgeNorm(int size, float* wedge, float2* part, float* maxVal, int newMethod)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float val = wedge[z * size * size + y * size + x];

	if (newMethod)
	{
		if (val <= 0)
			val = 0;
		else
			val = 1.0f / val;
	}
	else
	{
		if (val < 0.1f * maxVal[0])
			val = 1.0f / (0.1f * maxVal[0]);
		else
			val = 1.0f / val;
	}
	float2 p = part[z * size * size + y * size + x];
	p.x *= val;
	p.y *= val;
	part[z * size * size + y * size + x] = p;
}


extern "C"
__global__ void subCplx2(int size, float2* inVol, float2* outVol, float* subval, float* divVal)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = inVol[z * size * size + y * size + x];
	temp.x -= subval[0] / divVal[0];
	outVol[z * size * size + y * size + x] = temp;
}


extern "C"
__global__ void makeReal(int size, float2* inVol, float* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = inVol[z * size * size + y * size + x];
	outVol[z * size * size + y * size + x] = temp.x;
}


extern "C"
__global__ void makeCplxWithSub(int size, float* inVol, float2* outVol, float val)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = make_float2(inVol[z * size * size + y * size + x] - val, 0);
	outVol[z * size * size + y * size + x] = temp;
}


extern "C"
__global__ void makeCplxWithSquareAndSub(int size, float* inVol, float2* outVol, float val)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = make_float2((inVol[z * size * size + y * size + x] - val) * (inVol[z * size * size + y * size + x] - val), 0);
	outVol[z * size * size + y * size + x] = temp;
}


extern "C"
__global__ void binarize(int size, float* inVol, float* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	outVol[z * size * size + y * size + x] = inVol[z * size * size + y * size + x] > 0.5f ? 1.0f : 0.0f;
}



extern "C"
__global__ void mulVol(int size, float* inVol, float2* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = outVol[z * size * size + y * size + x];
	temp.x *= inVol[z * size * size + y * size + x];
	temp.y *= inVol[z * size * size + y * size + x];
	outVol[z * size * size + y * size + x] = temp;
}



extern "C"
__global__ void mulVolCplx(int size, float2* inVol, float2* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = outVol[z * size * size + y * size + x];
	float2 temp2 = inVol[z * size * size + y * size + x];
	temp.x *= temp2.x; //complex component is meant to be zero
	temp.y *= temp2.x;
	outVol[z * size * size + y * size + x] = temp;
}



extern "C"
__global__ void mul(int size, float in, float2* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = outVol[z * size * size + y * size + x];
	temp.x *= in;
	temp.y *= in;
	outVol[z * size * size + y * size + x] = temp;
}



extern "C"
__global__ void conv(int size, float2* inVol, float2* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 o = outVol[z * size * size + y * size + x];
	float2 i = inVol[z * size * size + y * size + x];
	float2 erg;
	erg.x = (o.x * i.x) - (o.y * i.y);
	erg.y = (o.x * i.y) + (o.y * i.x);
	outVol[z * size * size + y * size + x] = erg;
}



extern "C"
__global__ void correl(int size, float2* inVol, float2* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 o = outVol[z * size * size + y * size + x];
	float2 i = inVol[z * size * size + y * size + x];
	float2 erg;
	erg.x = (o.x * i.x) + (o.y * i.y);
	erg.y = (o.x * i.y) - (o.y * i.x);
	outVol[z * size * size + y * size + x] = erg;
}



extern "C"
__global__ void phaseCorrel(int size, float2 * inVol, float2 * outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

	float2 o = outVol[z * size * size + y * size + x];
	float2 i = inVol[z * size * size + y * size + x];
	float2 erg;
	erg.x = (o.x * i.x) + (o.y * i.y);
	erg.y = (o.x * i.y) - (o.y * i.x);
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
	outVol[z * size * size + y * size + x] = erg;
}


extern "C"
__global__ void bandpass(int size, float2* vol, float rDown, float rUp, float smooth)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = vol[z * size * size + y * size + x];

	//use squared smooth for Gaussian
	smooth = smooth * smooth;

	float center = size / 2;
	float3 vox = make_float3(x - center, y - center, z - center);

	float dist = sqrt(vox.x * vox.x + vox.y * vox.y + vox.z * vox.z);
	float scf = (dist - rUp) * (dist - rUp);
	smooth > 0 ? scf = exp(-scf/smooth) : scf = 0;

	if (dist > rUp)
	{
		temp.x *= scf;
		temp.y *= scf;
	}
	
	scf = (dist - rDown) * (dist - rDown);
	smooth > 0 ? scf = exp(-scf/smooth) : scf = 0;
	
	if (dist < rDown)
	{
		temp.x *= scf;
		temp.y *= scf;
	}
	

	vol[z * size * size + y * size + x] = temp;
}


extern "C"
__global__ void bandpassFFTShift(int size, float2* vol, float rDown, float rUp, float smooth)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;

	float2 temp = vol[z * size * size + y * size + x];

	//use squared smooth for Gaussian
	smooth = smooth * smooth;

	float center = size / 2;
	float3 vox = make_float3(i - center, j - center, k - center);

	float dist = sqrt(vox.x * vox.x + vox.y * vox.y + vox.z * vox.z);
	float scf = (dist - rUp) * (dist - rUp);
	smooth > 0 ? scf = exp(-scf/smooth) : scf = 0;

	if (dist > rUp)
	{
		temp.x *= scf;
		temp.y *= scf;
	}
	
	scf = (dist - rDown) * (dist - rDown);
	smooth > 0 ? scf = exp(-scf/smooth) : scf = 0;
	
	if (dist < rDown)
	{
		temp.x *= scf;
		temp.y *= scf;
	}
	

	vol[z * size * size + y * size + x] = temp;
}


extern "C"
__global__ void fftshift(int size, float2* vol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = vol[z * size * size + y * size + x]; 
	
	int mx = x - size / 2;
	int my = y - size / 2;
	int mz = z - size / 2;

	float a = 1.0f - 2 * (((mx + my + mz) & 1));
	
	temp.x *= a;
	temp.y *= a;

	vol[z * size * size + y * size + x] = temp;
}


extern "C"
__global__ void fftshift2(int size, float2* volIn, float2* volOut)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;


	float2 temp = volIn[k * size * size + j * size + i]; 
	volOut[z * size * size + y * size + x] = temp;
}


extern "C"
__global__ void splitDataset(int size, float2 * dataIn, float2 * dataOutA, float2 * dataOutB)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;
	const int z = blockIdx.z * blockDim.z + threadIdx.z;

	float2 in = dataIn[z * size * size + y * size + x];


	int pattern = x + y + z;
	int isCenter = (!pattern) & 1; //voxel (0,0,0) is center in fft-shifted image
	int patternA = pattern % 2;
	int patternB = 1 - patternA;

	patternA = patternA | isCenter;
	patternB = patternB | isCenter;

	float2 a = in;
	float2 b = in;
	a.x *= patternA;
	a.y *= patternA;
	b.x *= patternB;
	b.y *= patternB;

	dataOutA[z * size * size + y * size + x] = a;
	dataOutB[z * size * size + y * size + x] = b;
}



extern "C"
__global__ void fftshiftReal(int size, float* volIn, float* volOut)
{
	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;


	float temp = volIn[k * size * size + j * size + i]; 
	volOut[z * size * size + y * size + x] = temp;
}



extern "C"
__global__ void energynorm(int size, float2* particle, float2* partSqr, float2* cccMap, float* energyRef, float* nVox)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float part = particle[z * size * size + y * size + x].x; 
	float energyLocal = partSqr[z * size * size + y * size + x].x; 
	
	float2 erg;
	erg.x = 0;
	erg.y = 0;

	energyLocal -= part * part / nVox[0];
	energyLocal = sqrt(energyLocal) * sqrt(energyRef[0]);

	if (energyLocal > EPS)
	{
		erg.x = cccMap[z * size * size + y * size + x].x / energyLocal;
	}

	cccMap[z * size * size + y * size + x] = erg;
}





extern "C"
__global__ void findmax(float* maxVals, float* index, float* val, float rphi, float rpsi, float rthe)
{
	float oldmax = maxVals[0];
	if (val[0] > oldmax)
	{
		maxVals[0] = val[0];
		maxVals[1] = index[0];
		maxVals[2] = rphi;
		maxVals[3] = rpsi;
		maxVals[4] = rthe;
	}
}


extern "C"
__global__ void findmaxWithCertainty(float* maxVals, int* index, float* val, int* indexA, int* indexB, float rphi, float rpsi, float rthe, int volSize, int limit)
{
	int index0 = index[0];
	int zBoth = index0 / volSize / volSize;
	int yBoth = (index0 - zBoth * volSize * volSize) / volSize;
	int xBoth = index0 - zBoth * volSize * volSize - yBoth * volSize;
	xBoth -= volSize / 2;
	yBoth -= volSize / 2;
	zBoth -= volSize / 2;

	int indexA0 = indexA[0];
	int zA = indexA0 / volSize / volSize;
	int yA = (indexA0 - zA * volSize * volSize) / volSize;
	int xA = indexA0 - zA * volSize * volSize - yA * volSize;
	xA -= volSize / 2;
	yA -= volSize / 2;
	zA -= volSize / 2;


	int indexB0 = indexB[0];
	int zB = indexB0 / volSize / volSize;
	int yB = (indexB0 - zB * volSize * volSize) / volSize;
	int xB = indexB0 - zB * volSize * volSize - yB * volSize;
	xB -= volSize / 2;
	yB -= volSize / 2;
	zB -= volSize / 2;

	xA -= xBoth;
	yA -= yBoth;
	zA -= zBoth;
	xB -= xBoth;
	yB -= yBoth;
	zB -= zBoth;

	float dA = xA * xA + yA * yA + zA * zA;
	float dB = xB * xB + yB * yB + zB * zB;

	dA = sqrt(dA);
	dB = sqrt(dB);

	float dist = min(dA, dB);
	float currentCC = val[0];
	if (dist > limit)
		currentCC = abs(currentCC) * -1; //in case that currentCC is already negative...

	float oldmax = maxVals[0];	
	if (currentCC > oldmax)
	{
		float* ll = (float*)&index0;
		maxVals[0] = currentCC;
		maxVals[1] = *ll;
		maxVals[2] = rphi;
		maxVals[3] = rpsi;
		maxVals[4] = rthe;
	}
}


