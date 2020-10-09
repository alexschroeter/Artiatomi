#include "Kernels.h"

using namespace Hip;



FPKernel::FPKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : Hip::HipKernel("march", aModule, aGridDim, aBlockDim, 0)
{

}

FPKernel::FPKernel(hipModule_t aModule)
  : Hip::HipKernel("march", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float FPKernel::operator()(DeviceReconstructionConstantsCommon &param, int x, int y, Hip::HipPitchedDeviceVariable& projection, Hip::HipPitchedDeviceVariable& distMap, Hip::HipTextureObject3D& texObj)
{
  return (*this)(param, x, y, projection, distMap, texObj, make_int2(0, 0), make_int2(x, y));
}

float FPKernel::operator()(DeviceReconstructionConstantsCommon &param, int x, int y, Hip::HipPitchedDeviceVariable& projection, Hip::HipPitchedDeviceVariable& distMap, Hip::HipTextureObject3D& texObj, int2 roiMin, int2 roiMax)
{
  hipDeviceptr_t proj_dptr = projection.GetDevicePtr();
  //hipDeviceptr_t vol_dptr = 0;//volume.GetDevicePtr();
  hipDeviceptr_t distmap_dptr = distMap.GetDevicePtr();
  size_t stride = projection.GetPitch();
  hipTextureObject_t tex = texObj.GetTexObject();

  void* arglist[10];

  arglist[0] = &param;
  arglist[1] = &x;
  arglist[2] = &y;
  arglist[3] = &stride;
  arglist[4] = &proj_dptr;
  arglist[5] = &distmap_dptr;
  arglist[6] = &tex;
  arglist[7] = &roiMin;
  arglist[8] = &roiMax;

  DevParamMarcher rp;
  rp.common = param;
  rp.proj_x = x;
  rp.proj_y = y;
  rp.stride = stride;
  rp.projection = (float*)proj_dptr;
  rp.volume_traversal_length = (float*)distmap_dptr;
  rp.tex = tex;
  rp.roiMin = roiMin;
  rp.roiMax = roiMax;
  arglist[0] = &rp;
  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  
  return ms;
}



SlicerKernel::SlicerKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("slicer", aModule, aGridDim, aBlockDim, 0)
{

}

SlicerKernel::SlicerKernel(hipModule_t aModule)
  : HipKernel("slicer", aModule, make_dim3(1, 1, 1), make_dim3(32, 8, 1), 0)
{

}

float SlicerKernel::operator()(DeviceReconstructionConstantsCommon &param, int x, int y, HipPitchedDeviceVariable& projection, float tmin, float tmax, Hip::HipTextureObject3D& texObj)
{
  return (*this)(param, x, y, projection, tmin, tmax, texObj, make_int2(0, 0), make_int2(x, y));
}

float SlicerKernel::operator()(DeviceReconstructionConstantsCommon &param, int x, int y, HipPitchedDeviceVariable& projection, float tmin, float tmax, Hip::HipTextureObject3D& texObj, int2 roiMin, int2 roiMax)
{
  hipDeviceptr_t proj_dptr = projection.GetDevicePtr();
  size_t stride = projection.GetPitch();
  hipTextureObject_t tex = texObj.GetTexObject();

  void* arglist[10];

  arglist[0] = &param;
  arglist[1] = &x;
  arglist[2] = &y;
  arglist[3] = &stride;
  arglist[4] = &proj_dptr;
  arglist[5] = &tmin;
  arglist[6] = &tmax;
  arglist[7] = &tex;
  arglist[8] = &roiMin;
  arglist[9] = &roiMax;
 
  DevParamSlicer rp;
  rp.common = param;
  rp.proj_x = x;
  rp.proj_y = y;
  rp.stride = stride;
  rp.projection = (float*)proj_dptr;
  rp.tminDefocus = tmin;
  rp.tmaxDefocus = tmax;
  rp.tex = tex;
  rp.roiMin = roiMin;
  rp.roiMax = roiMax;
  arglist[0] = &rp;
  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}



VolTravLengthKernel::VolTravLengthKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("volTraversalLength", aModule, aGridDim, aBlockDim, 0)
{

}

VolTravLengthKernel::VolTravLengthKernel(hipModule_t aModule)
  : HipKernel("volTraversalLength", aModule, make_dim3(1, 1, 1), make_dim3(32, 8, 1), 0)
{

}

float VolTravLengthKernel::operator()(DeviceReconstructionConstantsCommon &param, int x, int y, HipPitchedDeviceVariable& distMap)
{
  return (*this)(param, x, y, distMap, make_int2(0, 0), make_int2(x, y));
}

float VolTravLengthKernel::operator()(DeviceReconstructionConstantsCommon &param, int x, int y, HipPitchedDeviceVariable& distMap, int2 roiMin, int2 roiMax)
{
  hipDeviceptr_t distMap_dptr = distMap.GetDevicePtr();
  size_t stride = distMap.GetPitch();

  void* arglist[10];

  arglist[0] = &param;
  arglist[1] = &x;
  arglist[2] = &y;
  arglist[3] = &stride;
  arglist[4] = &distMap_dptr;
  arglist[5] = &roiMin;
  arglist[6] = &roiMax;

  DevParamVolTravLength rp;
  rp.common = param;
  rp.proj_x = x;
  rp.proj_y = y;
  rp.stride = stride;
  rp.volume_traversal_length = (float*)distMap_dptr;
  rp.roiMin = roiMin;
  rp.roiMax = roiMax;
  arglist[0] = &rp;
  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}



CompKernel::CompKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("compare", aModule, aGridDim, aBlockDim, 0)
{

}

CompKernel::CompKernel(hipModule_t aModule)
  : HipKernel("compare", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

float CompKernel::operator()(HipPitchedDeviceVariable& real_raw, HipPitchedDeviceVariable& virtual_raw, HipPitchedDeviceVariable& vol_distance_map, float realLength, float4 crop, float4 cropDim, float projValScale)
{
  hipDeviceptr_t real_raw_dptr = real_raw.GetDevicePtr();
  hipDeviceptr_t virtual_raw_dptr = virtual_raw.GetDevicePtr();
  hipDeviceptr_t vol_distance_map_dptr = vol_distance_map.GetDevicePtr();
  int proj_x = (int)real_raw.GetWidth();
  int proj_y = (int)real_raw.GetHeight();
  size_t stride = real_raw.GetPitch();

  void* arglist[10];

  arglist[0] = &proj_x;
  arglist[1] = &proj_y;
  arglist[2] = &stride;
  arglist[3] = &real_raw_dptr;
  arglist[4] = &virtual_raw_dptr;
  arglist[5] = &vol_distance_map_dptr;
  arglist[6] = &realLength;
  arglist[7] = &crop;
  arglist[8] = &cropDim;
  arglist[9] = &projValScale;

  DevParamCompare rp;
  
  rp.proj_x = proj_x;
  rp.proj_y = proj_y;
  rp.stride = stride;
  rp.real_raw = (float*) real_raw_dptr; 
  rp.virtual_raw = (float*) virtual_raw_dptr; 
  rp.vol_distance_map = (float*) vol_distance_map_dptr; 
  rp.realLength = realLength; 
  rp.cutLength = crop; 
  rp.dimLength = cropDim; 
  rp.projValScale = projValScale;
  arglist[0] = &rp;

  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}



CropBorderKernel::CropBorderKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("cropBorder", aModule, aGridDim, aBlockDim, 0)
{

}

CropBorderKernel::CropBorderKernel(hipModule_t aModule)
  : HipKernel("cropBorder", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

float CropBorderKernel::operator()(HipPitchedDeviceVariable& image, float2 cutLength, float2 dimLength, int2 p1, int2 p2, int2 p3, int2 p4)
{
  hipDeviceptr_t image_dptr = image.GetDevicePtr();
  int proj_x = (int)image.GetWidth();
  int proj_y = (int)image.GetHeight();
  size_t stride = image.GetPitch();

  void* arglist[10];

  arglist[0] = &proj_x;
  arglist[1] = &proj_y;
  arglist[2] = &stride;
  arglist[3] = &image_dptr;
  arglist[4] = &cutLength;
  arglist[5] = &dimLength;
  arglist[6] = &p1;
  arglist[7] = &p2;
  arglist[8] = &p3;
  arglist[9] = &p4;

  DevParamCropBorder rp;

  rp.proj_x = proj_x; 
  rp.proj_y = proj_y; 
  rp.stride = stride; 
  rp.image = (float*) image_dptr; 
  rp.cutLength = cutLength; 
  rp.dimLength = dimLength; 
  rp.p1 = p1; 
  rp.p2 = p2; 
  rp.p3 = p3; 
  rp.p4 = p4;
  
  arglist[0] = &rp;
 //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}



BPKernel::BPKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim, bool fp16)
  : HipKernel(fp16 ? "backProjectionFP16" : "backProjection", aModule, aGridDim, aBlockDim, 2 * aBlockDim.x * aBlockDim.y * aBlockDim.z * sizeof(float) * 4)
{
		
}

BPKernel::BPKernel(hipModule_t aModule, bool fp16)
  : HipKernel(fp16 ? "backProjectionFP16" : "backProjection", aModule, make_dim3(1, 1, 1), make_dim3(8, 16, 4), 2 * 8 * 16 * 4 * sizeof(float) * 4)
{

}

float BPKernel::operator()(DeviceReconstructionConstantsCommon &param, int proj_x, int proj_y, float lambda, int maxOverSample, float maxOverSampleInv, HipPitchedDeviceVariable& img, float distMin, float distMax,
			   hipSurfaceObject_t surfObj, HipTextureObject2D texObj)
{
  hipDeviceptr_t img_ptr = img.GetDevicePtr();
  int stride = (int)img.GetPitch();
  hipTextureObject_t tex = texObj.GetTexObject();


  DevParamBackProjection rp;

  rp.common = param; 
  rp.proj_x = proj_x; 
  rp.proj_y = proj_y; 
  rp.lambda = lambda; 
  rp.maxOverSample = maxOverSample; 
  rp.maxOverSampleInv = maxOverSampleInv; 
  rp.img = (float*)img_ptr; 
  rp.stride = stride; 
  rp.distMin = distMin; 
  rp.distMax = distMax; 
  rp.surfObj = surfObj;
  rp.texObj = texObj.GetTexObject();
 
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}


ConvVolKernel::ConvVolKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("convertVolumeFP16ToFP32", aModule, aGridDim, aBlockDim, 0)
{

}

ConvVolKernel::ConvVolKernel(hipModule_t aModule)
  : HipKernel("convertVolumeFP16ToFP32", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

float ConvVolKernel::operator()(DeviceReconstructionConstantsCommon &param, Hip::HipPitchedDeviceVariable& img, unsigned int z, hipSurfaceObject_t surfObj)
{
  hipDeviceptr_t img_ptr = img.GetDevicePtr();
  int stride = (int)img.GetPitch() / img.GetElementSize();

  void* arglist[10];

  arglist[0] = &param;
  arglist[1] = &img_ptr;
  arglist[2] = &stride;
  arglist[3] = &z;
  arglist[4] = &surfObj;

  DevParamConvVol rp;

  rp.common = param;
  rp.volPlane = (float*) img_ptr;
  rp.stride = stride;
  rp.z = z;
  rp.surfObj = surfObj;
  
  arglist[0] = &rp; 
  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}

ConvVol3DKernel::ConvVol3DKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("convertVolume3DFP16ToFP32", aModule, aGridDim, aBlockDim, 0)
{

}

ConvVol3DKernel::ConvVol3DKernel(hipModule_t aModule)
  : HipKernel("convertVolume3DFP16ToFP32", aModule, make_dim3(1, 1, 1), make_dim3(8, 8, 8), 0)
{

}

float ConvVol3DKernel::operator()(DeviceReconstructionConstantsCommon &param, Hip::HipPitchedDeviceVariable& img, hipSurfaceObject_t surfObj)
{
  hipDeviceptr_t img_ptr = img.GetDevicePtr();
  int stride = (int)img.GetPitch() / img.GetElementSize();

  void* arglist[10];

  arglist[0] = &param;
  arglist[1] = &img_ptr;
  arglist[2] = &stride;
  arglist[3] = &surfObj;
 
  DevParamConvVol3D rp;

  rp.common = param;
  rp.volPlane = (float*) img_ptr;
  rp.stride = stride;  
  rp.surfObj = surfObj;
  
  arglist[0] = &rp;
  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}



CTFKernel::CTFKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("ctf", aModule, aGridDim, aBlockDim, 0)
{

}

CTFKernel::CTFKernel(hipModule_t aModule)
  : HipKernel("ctf", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}


float CTFKernel::operator()( DeviceReconstructionConstantsCtf &param, HipDeviceVariable& ctf, float defocusMin, float defocusMax, float angle, bool absolute, size_t stride, float4 betaFac)
{
  hipDeviceptr_t ctf_dptr = ctf.GetDevicePtr();
  //size_t stride = 2049 * sizeof(float2);
  float _defocusMin = defocusMin * 0.000000001f;
  float _defocusMax = defocusMax * 0.000000001f;
  float _angle = angle / 180.0f * (float)M_PI;

  void* arglist[10];

  arglist[0] = &param;
  arglist[1] = &ctf_dptr;
  arglist[2] = &stride;
  arglist[3] = &_defocusMin;
  arglist[4] = &_defocusMax;
  arglist[5] = &_angle;
  arglist[6] = &absolute;
  arglist[7] = &betaFac;

  DevParamCtf rp;

  rp.param = param;
  rp.ctf = (hipComplex*) ctf_dptr;
  rp.stride = stride;
  rp.defocusMin = _defocusMin;
  rp.defocusMax = _defocusMax;
  rp.angle = _angle;
  rp.absolut = absolute;
  rp.betaFac = betaFac;
 
  arglist[0] = &rp;
  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}


WbpWeightingKernel::WbpWeightingKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("wbpWeighting", aModule, aGridDim, aBlockDim, 0)
{

}

WbpWeightingKernel::WbpWeightingKernel(hipModule_t aModule)
  : HipKernel("wbpWeighting", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}


float WbpWeightingKernel::operator()(HipDeviceVariable& img, size_t stride, unsigned int pixelcount, float psiAngle, WbpFilterMethod fm)
{
  hipDeviceptr_t img_dptr = img.GetDevicePtr();
  float _angle = -psiAngle / 180.0f * (float)M_PI;

  void* arglist[5];

  arglist[0] = &img_dptr;
  arglist[1] = &stride;
  arglist[2] = &pixelcount;
  arglist[3] = &_angle;
  arglist[4] = &fm;

  DevParamWbpWeighting rp;

  rp.img = (hipComplex*) img_dptr;
  rp.stride = stride;
  rp.pixelcount = pixelcount;
  rp.psiAngle = _angle;
  rp.fm = fm;
  
  arglist[0] = &rp;
  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}


FourFilterKernel::FourFilterKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("fourierFilter", aModule, aGridDim, aBlockDim, 0)
{

}

FourFilterKernel::FourFilterKernel(hipModule_t aModule)
  : HipKernel("fourierFilter", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}


float FourFilterKernel::operator()(Hip::HipDeviceVariable& img, size_t stride, int pixelcount, float lp, float hp, float lps, float hps)
{
  hipDeviceptr_t img_dptr = img.GetDevicePtr();

  void* arglist[7];

  arglist[0] = &img_dptr;
  arglist[1] = &stride;
  arglist[2] = &pixelcount;
  arglist[3] = &lp;
  arglist[4] = &hp;
  arglist[5] = &lps;
  arglist[6] = &hps;

  DevParamFourierFilter rp;

  rp.img = (float2*) img_dptr;
  rp.stride = stride;
  rp.pixelcount = pixelcount;
  rp.lp  = lp;
  rp.hp  = hp;
  rp.lps = lps;
  rp.hps = hps;
  
  arglist[0] = &rp;
  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}


ConjKernel::ConjKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("conjMul", aModule, aGridDim, aBlockDim, 0)
{

}

ConjKernel::ConjKernel(hipModule_t aModule)
  : HipKernel("conjMul", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}


float ConjKernel::operator()(Hip::HipDeviceVariable& img1, Hip::HipPitchedDeviceVariable& img2, size_t stride, int pixelcount)
{
  hipDeviceptr_t img_dptr1 = img1.GetDevicePtr();
  hipDeviceptr_t img_dptr2 = img2.GetDevicePtr();

  void* arglist[4];

  arglist[0] = &img_dptr1;
  arglist[1] = &img_dptr2;
  arglist[2] = &stride;
  arglist[3] = &pixelcount;

  DevParamConjMul rp;

  rp.complxA = (float2*) img_dptr1;
  rp.complxB = (float2*) img_dptr2;
  rp.stride   = stride;
  rp.pixelcount  = pixelcount;

  arglist[0] = &rp;
  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}

MaxShiftKernel::MaxShiftKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("maxShift", aModule, aGridDim, aBlockDim, 0)
{

}

MaxShiftKernel::MaxShiftKernel(hipModule_t aModule)
  : HipKernel("maxShift", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}


float MaxShiftKernel::operator()(Hip::HipDeviceVariable& img1, size_t stride, int pixelcount, int maxShift)
{
  hipDeviceptr_t img_dptr1 = img1.GetDevicePtr();

  void* arglist[4];

  arglist[0] = &img_dptr1;
  arglist[1] = &stride;
  arglist[2] = &pixelcount;
  arglist[3] = &maxShift;

  DevParamMaxShift rp;

  rp.img = (float*) img_dptr1;
  rp.stride = stride;
  rp.pixelcount = pixelcount;
  rp.maxShift = maxShift;
  
  arglist[0] = &rp;
  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}



CopyToSquareKernel::CopyToSquareKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("makeSquare", aModule, aGridDim, aBlockDim, 0)
{

}

CopyToSquareKernel::CopyToSquareKernel(hipModule_t aModule)
  : HipKernel("makeSquare", aModule, make_dim3(1, 1, 1), make_dim3(16, 16, 1), 0)
{

}

float CopyToSquareKernel::operator()(HipPitchedDeviceVariable& aIn, int maxsize, HipDeviceVariable& aOut, int borderSizeX, int borderSizeY, bool mirrorY, bool fillZero)
{
  hipDeviceptr_t in_dptr = aIn.GetDevicePtr();
  hipDeviceptr_t out_dptr = aOut.GetDevicePtr();
  int _maxsize = maxsize;
  int _borderSizeX = borderSizeX;
  int _borderSizeY = borderSizeY;
  bool _mirrorY = mirrorY;
  bool _fillZero = fillZero;
  int proj_x = aIn.GetWidth();
  int proj_y = aIn.GetHeight();
  int stride = (int)aIn.GetPitch() / sizeof(float);

  //printf("\n\nStride: %ld, %ld\n", stride, stride / sizeof(float));

  void* arglist[10];

  arglist[0] = &proj_x;
  arglist[1] = &proj_y;
  arglist[2] = &_maxsize;
  arglist[3] = &stride;
  arglist[4] = &in_dptr;
  arglist[5] = &out_dptr;
  arglist[6] = &_borderSizeX;
  arglist[7] = &_borderSizeY;
  arglist[8] = &_mirrorY;
  arglist[9] = &_fillZero;

  DevParamCopyToSquare rp;
  
  rp.proj_x = proj_x; 
  rp.proj_y = proj_y; 
  rp.maxsize = _maxsize; 
  rp.stride = stride; 
  rp.aIn = (float*) in_dptr; 
  rp.aOut = (float*) out_dptr; 
  rp.borderSizeX = _borderSizeX; 
  rp.borderSizeY = _borderSizeY; 
  rp.mirrorY = _mirrorY; 
  rp.fillZero = _fillZero;

  arglist[0] = &rp;
  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}



SamplesToCoefficients2DX::SamplesToCoefficients2DX(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("SamplesToCoefficients2DX", aModule, aGridDim, aBlockDim, 0)
{

}

float SamplesToCoefficients2DX::operator()(HipPitchedDeviceVariable& image)
{
  hipDeviceptr_t in_dptr = image.GetDevicePtr();
  uint _pitch = image.GetPitch();
  uint _width = image.GetWidth();
  uint _height = image.GetHeight();

  void* arglist[4];

  arglist[0] = &in_dptr;
  arglist[1] = &_pitch;
  arglist[2] = &_width;
  arglist[3] = &_height;

  DevParamSamplesToCoefficients rp;

  rp.image = (float*) in_dptr;	
  rp.pitch = _pitch;	
  rp.width = _width;	
  rp.height = _height;	
  
  arglist[0] = &rp;
  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}


SamplesToCoefficients2DY::SamplesToCoefficients2DY(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
  : HipKernel("SamplesToCoefficients2DY", aModule, aGridDim, aBlockDim, 0)
{

}

float SamplesToCoefficients2DY::operator()(HipPitchedDeviceVariable& image)
{
  hipDeviceptr_t in_dptr = image.GetDevicePtr();
  uint _pitch = image.GetPitch();
  uint _width = image.GetWidth();
  uint _height = image.GetHeight();

  void* arglist[4];

  arglist[0] = &in_dptr;
  arglist[1] = &_pitch;
  arglist[2] = &_width;
  arglist[3] = &_height;

  DevParamSamplesToCoefficients rp;

  rp.image = (float*) in_dptr;	
  rp.pitch = _pitch;	
  rp.width = _width;	
  rp.height = _height;	
  
  arglist[0] = &rp;
  //float ms = Launch(arglist);
  float ms = Launch( &rp, sizeof(rp) );
  return ms;
}
