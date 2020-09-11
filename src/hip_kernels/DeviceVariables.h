#ifndef DEVICEVARIABLES_CU
#define DEVICEVARIABLES_CU


#include <hip/hip_runtime.h>
#include "Constants.h"
#include "DeviceReconstructionParameters.h"


//extern texture<float, 2, hipReadModeElementType> tex;
//extern texture<float, 2, hipReadModeElementType> texDist;
//texture<unsigned short, 3, cudaReadModeNormalizedFloat> texVol;
//texture<float, 3, cudaReadModeNormalizedFloat> texVol;

// transform vector by matrix
__device__
void MatrixVector3Mul(float3x3& M, float xIn, float yIn, float& xOut, float& yOut)
{
	xOut = M.m[0].x * xIn + M.m[0].y * yIn + M.m[0].z * 1.f;
	yOut = M.m[1].x * xIn + M.m[1].y * yIn + M.m[1].z * 1.f;
	//erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
}
#endif
