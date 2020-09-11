#ifndef DeviceReconstructionParameters_H
#define DeviceReconstructionParameters_H

#include <hip/hip_runtime.h>
#include <hip/hip_complex.h>



struct DevParamSubCplx
{
  int size;
  float2* in_dptr;
  float2* out_dptr;
  float* val_dptr;
  float* divval_dptr;
};

struct DevParamSubCplx_RC
{
  int size;
  float* in_dptr;
  float* out_dptr;
  float* subval_dptr;
  float* divval_dptr;
};

struct DevParamFFTShiftReal
{
  int size;
  float* in_dptr;
  float* out_dptr;
};

struct DevParamMakeCplxWithSub
{
  int size;
  float* in_dptr;
  float2* out_dptr;
  float val;
};

struct DevParamMulVol
{
  int size;
  float* in_dptr;
  float2* out_dptr;
};

struct DevParamMulVol_RR
{
  int size;
  float* in_dptr;
  float* out_dptr;
};

struct DevParamMulVol_RC
{
  int size;
  float* in_dptr;
  float2* out_dptr;
};

struct DevParamBandpassFFTShift
{
  int size;
  float2* in_dptr;
  //float* out_dptr;
  float rDown;
  float rUp;
  float smooth;
};

struct DevParamSum
{
  int size;
  float* in_dptr;
  float2* out_dptr;
};


struct DevParamSumSqr
{
  int size;
  float* in_dptr;
  float* out_dptr;
};


struct DevParamMakeReal
{
  int size;
  float2* in_dptr;
  float* out_dptr;
};

struct DevParamMul
{
  int size;
  float val;
  float2* out_dptr;
};

struct DevParamMul_Real
{
  int size;
  float val;
  float* out_dptr;
};

struct DevParamakeCplxWithSqrSub
{
  int size;
  float val;
  float* in_dptr;
  float2* out_dptr;
};

struct DevParaCorrel
{
  int size;
  float2* in_dptr;
  float2* out_dptr;
};

struct DevParaConv
{
  int size;
  float2* in_dptr;
  float2* out_dptr;
};

struct DevParaEnergynorm
{
  int size;
  float2* in_dptr;
  float2* out_dptr;
  float2* cccMap_dptr;
  float* energyRef;
  float* nVox;
};

struct DevParaEnergynorm_RC
{
  int size;
  float* in_dptr;
  float* out_dptr;
  float* cccMap_dptr;
  float* energyRef;
  float* nVox;
  float2* temp;
  float* ccMask;
};

struct DevParaFFTShift2
{
  int size;
  float2* in_dptr;
  float2* out_dptr;
};

struct DevParaSum
{
  int size;
  float* in_dptr;
  float* out_dptr;
};

struct DevParaSumCplx
{
  int size;
  float2* in_dptr;
  float* out_dptr;
};

struct DevParaSumSqrCplx
{
  int size;
  float2* in_dptr;
  float* out_dptr;
};

struct DevParaSumSqr
{
  int size;
  float* in_dptr;
  float* out_dptr;
};


struct DevParaBinarize
{
  int size;
  float* in_dptr;
  float* out_dptr;
};

struct DevParaMaxIndexCplx
{
  int size;
  float2* in_dptr;
  float* out_dptr;
  int* index;
  bool readIndex;
};

struct DevParaMaxIndex
{
  int size;
  float* in_dptr;
  float* out_dptr;
  int* index;
  bool readIndex;
};

struct DevParaRotate
{
  int size;
  float* in_dptr;
  float3 rotmat0;
  float3 rotmat1;
  float3 rotmat2;
  hipTextureObject_t texture;
};

struct DevParaRotateCplx
{
  int size;
  float2* in_dptr;
  float3 rotmat0;
  float3 rotmat1;
  float3 rotmat2;
  hipTextureObject_t texture;
};

struct DevParaShift
{
  int size;
  float* in_dptr;
  float3 shift;
  hipTextureObject_t texture;
};

struct DevParaShiftRot3D
{
  int size;
  float* in_dptr;
  float3 rotmat0;
  float3 rotmat1;
  float3 rotmat2;
  float3 shift;
  hipTextureObject_t texture;
};

struct DevParaSub
{
  int size;
  float* in_dptr;
  float* out_dptr;
  float val;
};

struct DevParaSub_RC
{
  int size;
  float* in_dptr;
  float* out_dptr;
  float* h_sum;
  float val;
};

struct DevParaMax
{
  float* maxVals;
  float* index;
  float* val;
  float rpsi;
  float rphi;
  float rthe;
};



#endif



