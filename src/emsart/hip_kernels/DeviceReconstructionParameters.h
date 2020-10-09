#ifndef DeviceReconstructionParameters_H
#define DeviceReconstructionParameters_H

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>

typedef struct {
    float4 m[4];
} float4x4;

typedef struct {
	float3 m[3];
} float3x3;


struct DeviceReconstructionConstantsCommon
{  
  float4x4 DetectorMatrix;
  float3x3 magAniso;
  float3x3 magAnisoInv;  
  float3 volumeBBoxRcp;
  float3 volumeDim;
  float3 volumeDimComplete;  
  float3 voxelSize;  
  float3 bBoxMin;
  float3 bBoxMax;
  float3 bBoxMinComplete;
  float3 bBoxMaxComplete;
  float3 detektor;
  float3 uPitch;
  float3 vPitch;
  float3 projNorm;
  float3 tGradient;
  float  zShiftForPartialVolume; 
  int    volumeDim_x_quarter;
 };

struct DeviceReconstructionConstantsCtf
{
  float cs;
  float voltage;
  float openingAngle;
  float ampContrast;
  float phaseContrast;
  float pixelsize;
  float pixelcount;
  float maxFreq;
  float freqStepSize;
  // float lambda;
  float applyScatteringProfile;
  float applyEnvelopeFunction;
};


struct DevParamMarcher
{
  DeviceReconstructionConstantsCommon common;
  size_t stride;
  float* projection;
  float* volume_traversal_length;
  hipTextureObject_t tex;
  int2 roiMin;
  int2 roiMax;
  int proj_x;
  int proj_y;
};

struct DevParamSlicer
{
  DeviceReconstructionConstantsCommon common;
  size_t stride;
  float* projection;  
  hipTextureObject_t tex;
  int2 roiMin;
  int2 roiMax;  
  float tminDefocus;
  float tmaxDefocus;
  int proj_x;
  int proj_y;
};

struct DevParamVolTravLength
{
  DeviceReconstructionConstantsCommon common;
  size_t stride;
  float* volume_traversal_length;
  int2 roiMin;
  int2 roiMax;
  int proj_x;
  int proj_y;
};

struct DevParamCompare
{  
  float4 cutLength; 
  float4 dimLength; 
  size_t stride;
  float* real_raw; 
  float* virtual_raw; 
  float* vol_distance_map; 
  float realLength; 
  float projValScale;
  int proj_x;
  int proj_y;
};

struct DevParamCropBorder
{  
  size_t stride; 
  float* image; 
  float2 cutLength; 
  float2 dimLength; 
  int2 p1; 
  int2 p2; 
  int2 p3; 
  int2 p4;
  int proj_x; 
  int proj_y; 
};

struct DevParamBackProjection
{
  DeviceReconstructionConstantsCommon common; 
  hipSurfaceObject_t surfObj;
  hipTextureObject_t texObj;
  float* img; 
  float lambda; 
  float maxOverSampleInv; 
  float distMin; 
  float distMax; 
  int proj_x; 
  int proj_y; 
  int stride; 
  int maxOverSample; 
};

struct DevParamConvVol
{
  DeviceReconstructionConstantsCommon common;
  hipSurfaceObject_t surfObj;
  float* volPlane;
  int stride;
  unsigned int z;
};

struct DevParamConvVol3D
{
  DeviceReconstructionConstantsCommon common;
  hipSurfaceObject_t surfObj;
  float* volPlane;
  int stride;
};

struct DevParamCtf
{
  DeviceReconstructionConstantsCtf param;
  float4 betaFac;
  hipComplex* ctf;
  size_t stride;
  float defocusMin;
  float defocusMax;
  float angle;
  bool absolut;
};


enum WbpFilterMethod
{
  FM_RAMP,
  FM_EXACT,
  FM_CONTRAST2,
  FM_CONTRAST10,
  FM_CONTRAST30
};


struct DevParamWbpWeighting
{  
  hipComplex* img;
  size_t stride;
  unsigned int pixelcount;
  float psiAngle;
  WbpFilterMethod fm;
};


struct DevParamFourierFilter
{  
  float2* img;
  size_t stride;
  int pixelcount;
  float lp;
  float hp;
  float lps;
  float hps;
};

struct DevParamConjMul
{ 
  float2* complxA;
  float2* complxB;
  size_t stride;
  int pixelcount;
};

struct DevParamMaxShift
{
  float* img;
  size_t stride;
  int pixelcount;
  int maxShift;
};

struct DevParamCopyToSquare
{
  float* aIn; 
  float* aOut; 
  int proj_x; 
  int proj_y; 
  int maxsize; 
  int stride; 
  int borderSizeX; 
  int borderSizeY; 
  bool mirrorY; 
  bool fillZero;
};

struct DevParamSamplesToCoefficients
{
  float* image;		// in-place processing
  uint pitch;			// width in bytes
  uint width;			// width of the image
  uint height;		// height of the image
};

#endif
