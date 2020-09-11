//Includes for IntelliSense 
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif


#define EPS (0.000001f)

#include <hip/hip_runtime.h>
//#include <device_launch_parameters.h>
//#include <texture_fetch_functions.h>
#include "float.h"
//#include <builtin_types.h>
//#include <vector_functions.h>
#include "DeviceReconstructionParameters.h"

/* AS removed global variable 
//texture<float, 3, cudaReadModeElementType> texVol;
//texture<float, 3, cudaReadModeElementType> texShift;
//texture<float2, 3, cudaReadModeElementType> texVolCplx;
*/

extern "C"
__global__ void rot3d(DevParaRotate param)
{
	int size = param.size;
	float3 rotMat0 = param.rotmat0;
	float3 rotMat1 = param.rotmat1;
	float3 rotMat2 = param.rotmat2;
	float* outVol = param.in_dptr;
	hipTextureObject_t &tex = param.texture; 

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	

	//if (!(x >= size || y >= size || z >= size)){
		float center = size / 2;
		
		float3 vox = make_float3((float)x - center, (float)y - center, (float)z - center);
		float3 rotVox;
		rotVox.x = center + rotMat0.x * vox.x + rotMat1.x * vox.y + rotMat2.x * vox.z;
		rotVox.y = center + rotMat0.y * vox.x + rotMat1.y * vox.y + rotMat2.y * vox.z;
		rotVox.z = center + rotMat0.z * vox.x + rotMat1.z * vox.y + rotMat2.z * vox.z;

		/* ToDO Check if setting pixels outside the image to 0 will improve the result
		if ((rotVox.x + 0.5 <= 10.5 || rotVox.x + 0.5 >= size-10+0.5) || (rotVox.y + 0.5 <= 10.5 || rotVox.y + 0.5 >= size-10+0.5) || (rotVox.z + 10.5 <= 0.5 || rotVox.z + 0.5 >= size-10+0.5)) {
			outVol[z * size * size + y * size + x] = 0;
		}
		else {
			//outVol[z * size * size + y * size + x] = tex3D<float>(tex, (int)rotVox.x + 0.5f, (int)rotVox.y + 0.5f, (int)rotVox.z + 0.5f);	
			//outVol[z * size * size + y * size + x] = tex3D<float>(tex, rotVox.x + 0.5f, rotVox.y + 0.5f, rotVox.z + 0.5f);	
			//outVol[z * size * size + y * size + x] = tex3D<float>(tex, x + 0.3f, y + 0.4f, z + 0.7f); // different results for amd and nvidia
			//outVol[z * size * size + y * size + x] = tex3D<float>(tex, x , y , z); // same results for amd and nvidia but different from input
			//outVol[z * size * size + y * size + x] = tex3D<float>(tex, x+.5f , y+.5f , z+.5f); // same results for amd and nvidia but different from input
		}
		*/

		outVol[z * size * size + y * size + x] = tex3D<float>(tex, rotVox.x + 0.5f, rotVox.y + 0.5f, rotVox.z + 0.5f);	
		// outVol[z * size * size + y * size + x] = tex3D<float>(tex, vox.x + 0.5f, vox.y + 0.5f, vox.z + 0.5f);
		// outVol[z * size * size + y * size + x] = tex3D<float>(tex, x + 0.3f, y + 0.4f, z + 0.7f); // different results for amd and nvidia
		// outVol[z * size * size + y * size + x] = tex3D<float>(tex, x , y , z); // same results for amd and nvidia
	//}	
}


extern "C" __global__ void rot3d_soft_interpolate(DevParaRotate param){ 
	int size = param.size; 
	float3 rotMat0 = param.rotmat0; 
	float3 rotMat1 = param.rotmat1; 
	float3 rotMat2 = param.rotmat2; 
	float* outVol = param.in_dptr; 
	hipTextureObject_t &tex = param.texture;  
 
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x; 
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	 
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	 
 
	if (!(x >= size || y >= size || z >= size)){ 
		float center = size / 2; 
		 
		float3 vox = make_float3((float)x - center, (float)y - center, (float)z - center); 
		float3 rotVox; 
		rotVox.x = center + rotMat0.x * vox.x + rotMat1.x * vox.y + rotMat2.x * vox.z; 
		rotVox.y = center + rotMat0.y * vox.x + rotMat1.y * vox.y + rotMat2.y * vox.z; 
		rotVox.z = center + rotMat0.z * vox.x + rotMat1.z * vox.y + rotMat2.z * vox.z; 
 
    	float x_low = floor(rotVox.x); float x_high = ceil(rotVox.x); 
		float y_low = floor(rotVox.y); float y_high = ceil(rotVox.y); 
		float z_low = floor(rotVox.z); float z_high = ceil(rotVox.z); 
 
		float xLyLzL = tex3D<float>(tex, x_low + 0.5f, y_low + 0.5f, z_low + 0.5f); 
		float xHyLzL = tex3D<float>(tex, x_high + 0.5f, y_low + 0.5f, z_low + 0.5f); 
		float xLyHzL = tex3D<float>(tex, x_low + 0.5f, y_high + 0.5f, z_low + 0.5f); 
		float xHyHzL = tex3D<float>(tex, x_high + 0.5f, y_high + 0.5f, z_low + 0.5f); 
		float xLyLzH = tex3D<float>(tex, x_low + 0.5f, y_low + 0.5f, z_high + 0.5f); 
		float xHyLzH = tex3D<float>(tex, x_high + 0.5f, y_low + 0.5f, z_high + 0.5f); 
		float xLyHzH = tex3D<float>(tex, x_low + 0.5f, y_high + 0.5f, z_high + 0.5f); 
		float xHyHzH = tex3D<float>(tex, x_high + 0.5f, y_high + 0.5f, z_high + 0.5f); 
 
		float yLzL = fma(rotVox.x-x_low, xHyLzL, fma(-1.f*(rotVox.x-x_low), xLyLzL, xLyLzL)); 
		float yHzL = fma(rotVox.x-x_low, xHyHzL, fma(-1.f*(rotVox.x-x_low), xLyHzL, xLyHzL)); 
		float yLzH = fma(rotVox.x-x_low, xHyLzH, fma(-1.f*(rotVox.x-x_low), xLyLzH, xLyLzH)); 
		float yHzH = fma(rotVox.x-x_low, xHyHzH, fma(-1.f*(rotVox.x-x_low), xLyHzH, xLyHzH)); 
 
		float zL = fma(rotVox.y-y_low, yHzL, fma(-1.f*(rotVox.y-y_low), yLzL, yLzL)); 
		float zH = fma(rotVox.y-y_low, yHzH, fma(-1.f*(rotVox.y-y_low), yLzH, yLzH)); 
 
		outVol[z * size * size + y * size + x]  = fma(rotVox.z-z_low, zH, fma(-1.f*(rotVox.z-z_low), zL, zL)); 
	}	 
} 


extern "C"
__global__ void shiftrot3d(DevParaShiftRot3D param)
{
	int size = param.size;
	float3 rotMat0 = param.rotmat0;
	float3 rotMat1 = param.rotmat1;
	float3 rotMat2 = param.rotmat2;
	float3 shift = param.shift;
	float* outVol = param.in_dptr;
	hipTextureObject_t &tex = param.texture; 

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	

	float sx = float(x - shift.x + 0.5f);
	float sy = float(y - shift.y + 0.5f);
	float sz = float(z - shift.z + 0.5f); 

	float center = size / 2;
	
	float3 vox = make_float3((float)sx - center, (float)sy - center, (float)sz - center);
	float3 rotVox;
	rotVox.x = center + rotMat0.x * vox.x + rotMat1.x * vox.y + rotMat2.x * vox.z;
	rotVox.y = center + rotMat0.y * vox.x + rotMat1.y * vox.y + rotMat2.y * vox.z;
	rotVox.z = center + rotMat0.z * vox.x + rotMat1.z * vox.y + rotMat2.z * vox.z;

	outVol[z * size * size + y * size + x] = tex3D<float>(tex, rotVox.x, rotVox.y, rotVox.z);	
}

extern "C"
__global__ void rot3dCplx(DevParaRotateCplx param)
{
	int size = param.size;
	float3 rotMat0 = param.rotmat0;
	float3 rotMat1 = param.rotmat1;
	float3 rotMat2 = param.rotmat2;
	float2* outVol = param.in_dptr;
	hipTextureObject_t &texCplx = param.texture; 

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float center = size / 2;

	float3 vox = make_float3(x - center, y - center, z - center);
	float3 rotVox;
	rotVox.x = center + rotMat0.x * vox.x + rotMat1.x * vox.y + rotMat2.x * vox.z;
	rotVox.y = center + rotMat0.y * vox.x + rotMat1.y * vox.y + rotMat2.y * vox.z;
	rotVox.z = center + rotMat0.z * vox.x + rotMat1.z * vox.y + rotMat2.z * vox.z;

	outVol[z * size * size + y * size + x] = tex3D<float2>(texCplx, rotVox.x + 0.5f, rotVox.y + 0.5f, rotVox.z + 0.5f);
}


extern "C"
	__global__ void shift(DevParaShift param)
{
	int size = param.size;
	float3 shift = param.shift;
	float* outVol = param.in_dptr;
	hipTextureObject_t &texshift = param.texture; 

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float sx = float(x - shift.x + 0.5f);
	float sy = float(y - shift.y + 0.5f);
	float sz = float(z - shift.z + 0.5f); 
	
	outVol[z * size * size + y * size + x] = tex3D<float>(texshift, sx, sy, sz);
}


extern "C"
	__global__ void substract(DevParaSub param)
{
	int size = param.size;
	float* inVol = param.in_dptr;
	float* outVol = param.out_dptr;
	float val = param.val;

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
		/* TODO AF MK Fragen warum das so gerechnet wird */
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


// extern "C"
// __global__ void subCplx2(int size, float2* inVol, float2* outVol, float* subval, float* divVal)
// {
// 	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
// 	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
// 	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
// 	float2 temp = inVol[z * size * size + y * size + x];
// 	temp.x -= subval[0] / divVal[0];
// 	outVol[z * size * size + y * size + x] = temp;
// }



extern "C"
__global__ void subCplx2(DevParamSubCplx param)
{

	int size = param.size;
	float2* inVol = param.in_dptr;
	float2* outVol = param.out_dptr;
	float* subval = param.val_dptr;
	float* divVal = param.divval_dptr;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = inVol[z * size * size + y * size + x];
	temp.x -= subval[0] / divVal[0];
	outVol[z * size * size + y * size + x] = temp;
}


extern "C"
__global__ void makeReal(DevParamMakeReal param)
{
	int size = param.size;
	float2* inVol = param.in_dptr;
	float* outVol = param.out_dptr;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = inVol[z * size * size + y * size + x];
	outVol[z * size * size + y * size + x] = temp.x;
}


extern "C"
__global__ void makeCplxWithSub(DevParamMakeCplxWithSub param)
{

	int size = param.size;
	float* inVol = param.in_dptr;
	float2* outVol = param.out_dptr;
	float val = param.val;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = make_float2(inVol[z * size * size + y * size + x] - val, 0);
	outVol[z * size * size + y * size + x] = temp;
}


extern "C"
__global__ void makeCplxWithSquareAndSub(DevParamakeCplxWithSqrSub param)
{

	int size = param.size;
	float* inVol = param.in_dptr;
	float2* outVol = param.out_dptr;
	float val = param.val;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = make_float2((inVol[z * size * size + y * size + x] - val) * (inVol[z * size * size + y * size + x] - val), 0);
	outVol[z * size * size + y * size + x] = temp;
}


extern "C"
__global__ void binarize(DevParaBinarize param)
{
	int size = param.size;
	float* inVol = param.in_dptr;
	float* outVol = param.out_dptr;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	outVol[z * size * size + y * size + x] = inVol[z * size * size + y * size + x] > 0.5f ? 1.0f : 0.0f;
}



extern "C"
__global__ void mulVol(DevParamMulVol param)
{

	int size = param.size;
	float* inVol = param.in_dptr;
	float2* outVol = param.out_dptr;

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
__global__ void mul(DevParamMul param)
{
	int size = param.size;
	float in = param.val;
	float2* outVol = param.out_dptr;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = outVol[z * size * size + y * size + x];
	temp.x *= in;
	temp.y *= in;
	outVol[z * size * size + y * size + x] = temp;
}



extern "C"
__global__ void conv(DevParaConv param)
{
	int size = param.size;
	float2* inVol = param.in_dptr;
	float2* outVol = param.out_dptr;

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
__global__ void correl(DevParaCorrel param)
{
	int size = param.size;
	float2* inVol = param.in_dptr;
	float2* outVol = param.out_dptr;

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
__global__ void bandpassFFTShift(DevParamBandpassFFTShift param)
{

	int size = param.size;
	float2* vol = param.in_dptr;
	float rDown = param.rDown;
	float rUp = param.rUp;
	float smooth = param.smooth;

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
__global__ void fftshift2(DevParaFFTShift2 param)
{
	int size = param.size;
	float2* volIn = param.in_dptr;
	float2* volOut = param.out_dptr;
	
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
__global__ void fftshiftReal(DevParamFFTShiftReal param)
{

	int size = param.size;
	float* volIn = param.in_dptr;
	float* volOut = param.out_dptr;

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
__global__ void energynorm(DevParaEnergynorm param)
{
	int size = param.size;
	float2* particle = param.in_dptr;
	float2* partSqr = param.out_dptr;	
	float2* cccMap = param.cccMap_dptr;
	float* energyRef = param.energyRef;
	float* nVox = param.nVox;

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
__global__ void findmax(DevParaMax param)
{
	float* maxVals = param.maxVals;
	float* index = param.index;
	float* val = param.val;
	float rphi = param.rphi;
	float rpsi = param.rpsi;
	float rthe = param.rthe;


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



/***************************************************************************************************************************

    Real to Complex Replacments

/***************************************************************************************************************************/


extern "C"
__global__ void mulVol_RC(DevParamMulVol_RC param)
{
	int size = param.size;
	float* inVol = param.in_dptr;
	float2* outVol = param.out_dptr;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = outVol[z * size * (size / 2 + 1)  + y * (size / 2 + 1) + x];
	temp.x *= inVol[z * size * size + y * size + x];
	temp.y *= inVol[z * size * size + y * size + x];
	outVol[z * size * (size / 2 + 1) + y * (size / 2 + 1) + x] = temp;
}

extern "C"
__global__ void mulVol_RR(DevParamMulVol_RR param)
{
	int size = param.size;
	float* inVol = param.in_dptr;
	float* outVol = param.out_dptr;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float temp = outVol[z * size * size  + y * size + x];
	temp *= inVol[z * size * size + y * size + x];
	outVol[z * size * size  + y * size + x] = temp;
}

extern "C"
__global__ void bandpassFFTShift_RC(DevParamBandpassFFTShift param)
{
	int size = param.size;
	float2* vol = param.in_dptr;
	float rDown = param.rDown;
	float rUp = param.rUp;
	float smooth = param.smooth;

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;	
	int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	int i = (size / 2) - x; // (x + size / 2) % (size /2);
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;

	float2 temp = vol[z * size * (size/2+1) + y * (size/2+1) + x];

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
	

	vol[z * size * (size/2+1) + y * (size/2+1) + x] = temp;
}


extern "C"
__global__ void mul_RC(DevParamMul param)
{
	int size = param.size;
	float in = param.val;
	float2* outVol = param.out_dptr;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 temp = outVol[z * size * (size/2+1) + y * (size/2+1) + x];
	temp.x *= in;
	temp.y *= in;
	outVol[z * size * (size/2+1) + y * (size/2+1) + x] = temp;
}


extern "C"
	__global__ void sqrsub_RC(DevParaSub_RC param)
{
	int size = param.size;
	float* inVol = param.in_dptr;
	float* outVol = param.out_dptr;
	float* h_sum = param.h_sum;
	float val = param.val;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	outVol[z * size * size + y * size + x] = inVol[z * size * size + y * size + x]*inVol[z * size * size + y * size + x] - (h_sum[0]/val);
}


extern "C"
	__global__ void sub_RC(DevParaSub_RC param)
{
	int size = param.size;
	float* inVol = param.in_dptr;
	float* outVol = param.out_dptr;
	float* h_sum = param.h_sum;
	float val = param.val;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	outVol[z * size * size + y * size + x] = inVol[z * size * size + y * size + x] - (h_sum[0]/val);
}

/* AS ToDo Deprecated wrong idea
extern "C"
__global__ void mul_RR(int size, float in, float* outVol)
{
	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float temp = outVol[z * size * (size/2+1) + y * (size/2+1) + x];
	temp *= in;
	outVol[z * size * (size/2+1) + y * (size/2+1) + x] = temp;
}
*/

extern "C"
__global__ void mul_Real(DevParamMul_Real param)
{
	int size = param.size;
	float in = param.val;
	float* outVol = param.out_dptr;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	

	outVol[z * size * size + y * size + x] = outVol[z * size * size + y * size + x] * in;
}


extern "C"
__global__ void correl_RC(DevParaCorrel param)
{
	int size = param.size;
	float2* inVol = param.in_dptr;
	float2* outVol = param.out_dptr;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 o = outVol[z * size * (size/2+1) + y * (size/2+1) + x];
	float2 i = inVol[z * size * (size/2+1) + y * (size/2+1) + x];
	float2 erg;
	erg.x = (o.x * i.x) + (o.y * i.y);
	erg.y = (o.x * i.y) - (o.y * i.x);
	outVol[z * size * (size/2+1) + y * (size/2+1) + x] = erg;
}


extern "C"
__global__ void conv_RC(DevParaConv param)
{
	int size = param.size;
	float2* inVol = param.in_dptr;
	float2* outVol = param.out_dptr;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float2 o = outVol[z * size * (size/2+1) + y * (size/2+1) + x];
	float2 i = inVol[z * size * (size/2+1) + y * (size/2+1) + x];
	float2 erg;
	erg.x = (o.x * i.x) - (o.y * i.y);
	erg.y = (o.x * i.y) + (o.y * i.x);
	outVol[z * size * (size/2+1) + y * (size/2+1) + x] = erg;
}


extern "C" 
__global__ void energynorm_RC(DevParaEnergynorm_RC param)
{
	int size = param.size;
	float* particle = param.in_dptr;
	float* partSqr = param.out_dptr;	
	float* cccMap = param.cccMap_dptr;
	float* energyRef = param.energyRef;
	float* nVox = param.nVox;
	float2* temp = param.temp;
	float* ccMask = param.ccMask;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	float part = particle[z * size * size + y * size + x]; 
	float energyLocal = partSqr[z * size * size + y * size + x]; 
	
	float erg = 0;
	
	energyLocal -= part * part / nVox[0];
	energyLocal = sqrt(energyLocal) * sqrt(energyRef[0]);

	if (energyLocal > EPS)
	{
		erg = cccMap[z * size * size + y * size + x] / energyLocal;
	}

	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;

	cccMap[z * size * size + y * size + x] = erg;
	erg *= ccMask[k * size * size + j * size + i];
	temp[k * size * size + j * size + i].x = erg;
}

/* AS ToDo Deprecated not sure where this is from looks like some early version of energynorm kernel combined with shift 
extern "C" 
__global__ void energynorm(int size, float2* particle, float2* partSqr, float2* cccMap, float* energyRef, float* nVox, float2* temp, float* ccMask)
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
	energyLocal = sqrt(energyLocal * energyRef[0]);

	if (energyLocal > EPS)
	{
		erg.x = cccMap[z * size * size + y * size + x].x / energyLocal;
	}

	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;

	cccMap[z * size * size + y * size + x] = erg;
	erg.x *= ccMask[k * size * size + j * size + i];
	erg.y *= ccMask[k * size * size + j * size + i];
	temp[k * size * size + j * size + i] = erg;
}
*/

/* AS Has been moved inside the Energynorm_RC Kernel to get more calculations per memory access
extern "C"
__global__ void fftshift2_RC(DevParaFFTShift2 param)
{
	int size = param.size;
	float2* volIn = param.in_dptr;
	float2* volOut = param.out_dptr;

	const int x = blockIdx.x * blockDim.x + threadIdx.x;
	const int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const int z = blockIdx.z * blockDim.z + threadIdx.z;	
	
	int i = (x + size / 2) % size;
	int j = (y + size / 2) % size;
	int k = (z + size / 2) % size;


	float2 temp = volIn[k * size * size + j * size + i]; 
	volOut[z * size * size + y * size + x] = temp;
}
*/

extern "C"
__global__ void subCplx_RC(DevParamSubCplx_RC param)
{
	int size = param.size;
	float* inVol = param.in_dptr;
	float* outVol = param.out_dptr;
	float* subval = param.subval_dptr;
	float* divVal = param.divval_dptr;

	const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;	
	const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;	

	outVol[z * size * (size) + y * (size) + x] = inVol[z * size * size + y * (size) + x] - subval[0] / divVal[0];
}
