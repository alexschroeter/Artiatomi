#ifndef KERNELS_H
#define KERNELS_H

#include "default.h"
#include "hip/HipTextures.h"
#include "hip/HipKernel.h"
#include "hip/HipDeviceProperties.h"
#include "Projection.h"
#include "Volume.h"
#include "hip_kernels/DeviceReconstructionParameters.h"


class FPKernel : public Hip::HipKernel
{
public:
	FPKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	FPKernel(hipModule_t aModule);

	float operator()(DeviceReconstructionConstantsCommon &param, int x, int y, Hip::HipPitchedDeviceVariable& projection, Hip::HipPitchedDeviceVariable& distMap, Hip::HipTextureObject3D& texObj);
	float operator()(DeviceReconstructionConstantsCommon &param, int x, int y, Hip::HipPitchedDeviceVariable& projection, Hip::HipPitchedDeviceVariable& distMap, Hip::HipTextureObject3D& texObj, int2 roiMin, int2 roiMax);
};

class SlicerKernel : public Hip::HipKernel
{
public:
	SlicerKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	SlicerKernel(hipModule_t aModule);

	float operator()(DeviceReconstructionConstantsCommon &param, int x, int y, Hip::HipPitchedDeviceVariable& projection, float tmin, float tmax, Hip::HipTextureObject3D& texObj);
	float operator()(DeviceReconstructionConstantsCommon &param, int x, int y, Hip::HipPitchedDeviceVariable& projection, float tmin, float tmax, Hip::HipTextureObject3D& texObj, int2 roiMin, int2 roiMax);
};

class VolTravLengthKernel : public Hip::HipKernel
{
public:
	VolTravLengthKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	VolTravLengthKernel(hipModule_t aModule);

	float operator()(DeviceReconstructionConstantsCommon &param, int x, int y, Hip::HipPitchedDeviceVariable& distMap);
	float operator()(DeviceReconstructionConstantsCommon &param, int x, int y, Hip::HipPitchedDeviceVariable& distMap, int2 roiMin, int2 roiMax);
};

class CompKernel : public Hip::HipKernel
{
public:
	CompKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	CompKernel(hipModule_t aModule);

	float operator()(Hip::HipPitchedDeviceVariable& real_raw, Hip::HipPitchedDeviceVariable& virtual_raw, Hip::HipPitchedDeviceVariable& vol_distance_map, float realLength, float4 crop, float4 cropDim, float projValScale);
};

class CropBorderKernel : public Hip::HipKernel
{
public:
	CropBorderKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	CropBorderKernel(hipModule_t aModule);

	float operator()(Hip::HipPitchedDeviceVariable& image, float2 cutLength, float2 dimLength, int2 p1, int2 p2, int2 p3, int2 p4);
};

class BPKernel : public Hip::HipKernel
{
public:
	BPKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim, bool fp16);
	BPKernel(hipModule_t aModule, bool fp16);

	float operator()(DeviceReconstructionConstantsCommon &param, int proj_x, int proj_y, float lambda, int maxOverSample, float maxOverSampleInv, Hip::HipPitchedDeviceVariable& img, float distMin, float distMax, hipSurfaceObject_t surfObj, Hip::HipTextureObject2D texObj );
};

class CTFKernel : public Hip::HipKernel
{
public:
	CTFKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	CTFKernel(hipModule_t aModule);

	//float operator()(HipPitchedDeviceVariable& ctf, float defocus, bool absolute)
	float operator()( DeviceReconstructionConstantsCtf &param, Hip::HipDeviceVariable& ctf, float defocusMin, float defocusMax, float angle, bool absolute, size_t stride, float4 betaFac);
};

class CopyToSquareKernel : public Hip::HipKernel
{
public:
	CopyToSquareKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	CopyToSquareKernel(hipModule_t aModule);

	float operator()(Hip::HipPitchedDeviceVariable& aIn, int maxsize, Hip::HipDeviceVariable& aOut, int borderSizeX, int borderSizeY, bool mirrorY, bool fillZero);
};

class SamplesToCoefficients2DX : public Hip::HipKernel
{
public:
	SamplesToCoefficients2DX(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);

	float operator()(Hip::HipPitchedDeviceVariable& image);
};

class SamplesToCoefficients2DY : public Hip::HipKernel
{
public:
	SamplesToCoefficients2DY(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);

	float operator()(Hip::HipPitchedDeviceVariable& image);
};



class ConvVolKernel : public Hip::HipKernel
{
public:
	ConvVolKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	ConvVolKernel(hipModule_t aModule);

    float operator()(DeviceReconstructionConstantsCommon &param, Hip::HipPitchedDeviceVariable& img, unsigned int z, hipSurfaceObject_t surfObj);
};

class ConvVol3DKernel : public Hip::HipKernel
{
public:
	ConvVol3DKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	ConvVol3DKernel(hipModule_t aModule);

	float operator()(DeviceReconstructionConstantsCommon &param, Hip::HipPitchedDeviceVariable& img, hipSurfaceObject_t surfObj);
};



class WbpWeightingKernel : public Hip::HipKernel
{
public:
	WbpWeightingKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	WbpWeightingKernel(hipModule_t aModule);

	float operator()(Hip::HipDeviceVariable& img, size_t stride, unsigned int pixelcount, float psiAngle, WbpFilterMethod  fm);
};

class FourFilterKernel : public Hip::HipKernel
{
public:
	FourFilterKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	FourFilterKernel(hipModule_t aModule);

	float operator()(Hip::HipDeviceVariable& img, size_t stride, int pixelcount, float lp, float hp, float lps, float hps);
};

class ConjKernel : public Hip::HipKernel
{
public:
	ConjKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	ConjKernel(hipModule_t aModule);

	float operator()(Hip::HipDeviceVariable& img1, Hip::HipPitchedDeviceVariable& img2, size_t stride, int pixelcount);
};

class MaxShiftKernel : public Hip::HipKernel
{
public:
	MaxShiftKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim);
	MaxShiftKernel(hipModule_t aModule);

	float operator()(Hip::HipDeviceVariable& img1, size_t stride, int pixelcount, int maxShift);
};




namespace kernels
{  

template<class TVol>
DeviceReconstructionConstantsCommon GetReconstructionParameters( Volume<TVol>& vol, Projection& proj, int index, int subVol, Matrix<float>& m, Matrix<float>& mInv)
{
  //Set reconstruction parameters 
  DeviceReconstructionConstantsCommon p;  
  p.volumeBBoxRcp       = vol.GetSubVolumeBBoxRcp(subVol);
  p.volumeDim           = vol.GetSubVolumeDimension(subVol);
  p.volumeDim_x_quarter = (int)vol.GetDimension().x / 4;
  p.volumeDimComplete   = vol.GetDimension();  
  p.voxelSize           = vol.GetVoxelSize();    
  proj.GetDetectorMatrix(index, (float*) &p.DetectorMatrix, 1);
  p.bBoxMin         = vol.GetSubVolumeBBoxMin(subVol);
  p.bBoxMax         = vol.GetSubVolumeBBoxMax(subVol);
  p.bBoxMinComplete = vol.GetVolumeBBoxMin();
  p.bBoxMaxComplete = vol.GetVolumeBBoxMax();
  p.detektor = proj.GetPosition(index);
  p.uPitch   = proj.GetPixelUPitch(index);
  p.vPitch   = proj.GetPixelVPitch(index);
  p.projNorm = proj.GetNormalVector(index);
  p.zShiftForPartialVolume = 0;//vol.GetSubVolumeZShift(subVol);
  //Magnification anisotropy  
  p.magAniso = *(float3x3*) m.GetData();
  p.magAnisoInv = *(float3x3*) mInv.GetData();

  // ray direction == normal to the projection plane,  +-sign is not important
  // t coordinate will be a coordinate along the ray 
  const float3 &ray = p.projNorm;  

  float3 tGradient; // == dt/dx dt/dy dt/dz
  
  if( fabs(ray.x)<1.e-4 ){
    tGradient.x = (ray.x>=0) ?1.e4 : -1.e4;    
  } else tGradient.x = 1.0 / (double) ray.x;
  if( fabs(ray.y)<1.e-4 ){
    tGradient.y = (ray.y>=0) ?1.e4 : -1.e4;
  } else tGradient.y = 1.0 / (double) ray.y;
  if( fabs(ray.z)<1.e-4 ){
    tGradient.z = (ray.z>=0) ?1.e4 : -1.e4;
  } else tGradient.z = 1.0 / (double) ray.z;

  p.tGradient = tGradient;

  return p;
}

} // namespace

#endif //KERNELS_H
