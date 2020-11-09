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


#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H


#ifdef USE_MPI
#include <mpi.h>
#endif
#include "EmSartDefault.h"
#include "Projection.h"
#include "Volume.h"
#include "Kernels.h"
#include "CudaHelpers/CudaArrays.h"
#include "CudaHelpers/CudaContext.h"
#include "CudaHelpers/CudaTextures.h"
#include "CudaHelpers/CudaSurfaces.h"
#include "CudaHelpers/CudaKernel.h"
#include "CudaHelpers/CudaDeviceProperties.h"
#include "utils/Config.h"
#include "utils/CudaConfig.h"
#include "FilterGraph/Matrix.h"
#ifdef USE_MPI
#include "io/MPISource.h"
#endif
#include "FileIO/MarkerFile.h"
#include "io/writeBMP.h"
#include "FileIO/CtfFile.h"
#include <time.h>
#include <cufft.h>
#include <npp.h>
#include <algorithm>

typedef struct {
    float3 m[3];
} float3x3;

typedef struct {
	float4 m[4];
} float4x4;

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

class KernelModuls
{
private:
	bool compilerOutput;
	bool infoOutput;

public:
	KernelModuls(Cuda::CudaContext* aCuCtx);
	//CUmodule modFP;
	//CUmodule modSlicer;
	//CUmodule modVolTravLen;
	//CUmodule modComp;
	//CUmodule modWBP;
	//CUmodule modBP;
	//CUmodule modCTF;
	//CUmodule modCTS;

};


class Reconstructor
{
private:
	//FPKernel fpKernel;
	//SlicerKernel slicerKernel;
	//VolTravLengthKernel volTravLenKernel;
	//CompKernel compKernel;
	//SubEKernel subEKernel;
	//WbpWeightingKernel wbp;
	//CropBorderKernel cropKernel;
	//BPKernel bpKernel;
	//ConvVolKernel convVolKernel;
	//ConvVol3DKernel convVol3DKernel;
	//CTFKernel ctf;
	//CopyToSquareKernel cts;
	//FourFilterKernel fourFilterKernel;
	//DoseWeightingKernel doseWeightingKernel;
	//ConjKernel conjKernel;
	//PCKernel pcKernel;
	//MaxShiftKernel maxShiftKernel;
	//DimBordersKernel dimBordersKernel;
#ifdef REFINE_MODE
	MaxShiftWeightedKernel maxShiftWeightedKernel;
	FindPeakKernel findPeakKernel;
	Cuda::CudaDeviceVariable		projSquare2_d;
	RotKernel rotKernel;
	Cuda::CudaPitchedDeviceVariable projSubVols_d;
	float* ccMap;
	float* ccMapMulti;
	Cuda::CudaPitchedDeviceVariable ccMap_d;
	NppiRect roiCC1, roiCC2, roiCC3, roiCC4;
	NppiRect roiDestCC1, roiDestCC2, roiDestCC3, roiDestCC4;
#endif

	Cuda::CudaPitchedDeviceVariable realprojUS_d;
	Cuda::CudaPitchedDeviceVariable proj_d;
	Cuda::CudaPitchedDeviceVariable realproj_d;
	Cuda::CudaPitchedDeviceVariable dist_d;
	Cuda::CudaPitchedDeviceVariable filterImage_d;
	

	Cuda::CudaPitchedDeviceVariable ctf_d;
	Cuda::CudaDeviceVariable        fft_d;
	Cuda::CudaDeviceVariable		projSquare_d;
	Cuda::CudaPitchedDeviceVariable badPixelMask_d;
	Cuda::CudaPitchedDeviceVariable volTemp_d;

	Cuda::CudaTextureObject2D		texImage;

	cufftHandle handleR2C;
	cufftHandle handleC2R;

	NppiSize roiAll;
	NppiSize roiFFT;
	//NppiSize roiBorderSquare;
	NppiSize roiSquare;

	Cuda::CudaDeviceVariable meanbuffer;
	Cuda::CudaDeviceVariable meanval;
	Cuda::CudaDeviceVariable stdval;

	Projection& proj;
	ProjectionSource* projSource;
	CtfFile& defocus;
	MarkerFile& markers;
	Configuration::Config& config;

	int mpi_part;
	int mpi_size;
	bool skipFilter;
	int squareBorderSizeX;
	int squareBorderSizeY;
	size_t squarePointerShift;
	float* MPIBuffer;

	Matrix<float> magAnisotropy;
	Matrix<float> magAnisotropyInv;

	template<typename TVol>
	void ForwardProjectionCTF(Volume<TVol>* vol, Cuda::CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);
	template<typename TVol>
	void ForwardProjectionNoCTF(Volume<TVol>* vol, Cuda::CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);

	template<typename TVol>
	void ForwardProjectionCTFROI(Volume<TVol>* vol, Cuda::CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
	template<typename TVol>
	void ForwardProjectionNoCTFROI(Volume<TVol>* vol, Cuda::CudaTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
		
	template<typename TVol>
	void BackProjectionNoCTF(Volume<TVol>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);
	template<typename TVol>
	void BackProjectionCTF(Volume<TVol>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);

#ifdef SUBVOLREC_MODE
	template<typename TVol>
	void BackProjectionNoCTF(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<Cuda::CudaArray3D*>& vecArrays, int proj_index);
	template<typename TVol>
	void BackProjectionCTF(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<Cuda::CudaArray3D*>& vecArrays, int proj_index);
#endif

	template<typename TVol>
	void GetDefocusDistances(float& t_in, float& t_out, int index, Volume<TVol>* vol);

	void GetDefocusMinMax(float ray, int index, float& defocusMin, float& defocusMax);

public:
	Reconstructor(Configuration::Config& aConfig, Projection& aProj, ProjectionSource* aProjectionSource,
		 MarkerFile& aMarkers, CtfFile& aDefocus, KernelModuls& modules, int aMpi_part, int aMpi_size);
	~Reconstructor();

	Matrix<float> GetMagAnistropyMatrix(float aAmount, float angleInDeg, float dimX, float dimY);

	//If returns true, ctf_d contains the fourier Filter mask as defined by coefficients given in config file.
	bool ComputeFourFilter();
	//img_h can be of any supported type. After the call, the type is float! Make sure the array is large enough!
	void PrepareProjection(void* img_h, int proj_index, float& meanValue, float& StdValue, int& BadPixels);

	template<typename TVol>
	void PrintGeometry(Volume<TVol>* vol, int index);

	template<typename TVol>
	void ForwardProjection(Volume<TVol>* vol, Cuda::CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync = false);

	template<typename TVol>
	void ForwardProjectionROI(Volume<TVol>* vol, Cuda::CudaTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync = false);

	template<typename TVol>
	void ForwardProjectionDistanceOnly(Volume<TVol>* vol, int index);

	template<typename TVol>
	void Compare(Volume<TVol>* vol, char* originalImage, int index);

	void SubtractError(float* error);

	//Assumes image to back project stored in proj_d. SIRTCount is overridable to config-file!
	template<typename TVol>
	void BackProjection(Volume<TVol>* vol, Cuda::CudaSurfaceObject3D& surface, int proj_index, float SIRTCount);
	
#ifdef SUBVOLREC_MODE
	//Assumes image to back project stored in proj_d.
	template<typename TVol>
	void BackProjection(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<Cuda::CudaArray3D*>& vecArrays, int proj_index);
#endif

	/*template<typename TVol>
	void BackProjectionWithPriorWBPFilter(Volume<TVol>* vol, int proj_index, char* originalImage, float* MPIBuffer);

	template<typename TVol>
	void RemoveProjectionFromVol(Volume<TVol>* vol, int proj_index, char* originalImage, float* MPIBuffer);

	template<typename TVol>
	void OneSARTStep(Volume<TVol>* vol, Cuda::CudaTextureObject3D& texVol, Cuda::CudaSurfaceObject3D& surface, int index, bool volumeIsEmpty, char* originalImage, float SIRTCount, float* MPIBuffer);
	*/

	void ResetProjectionsDevice();
	void CopyProjectionToHost(float* buffer);
	void CopyDistanceImageToHost(float* buffer);//For Debugging...
	void CopyRealProjectionToHost(float* buffer);//For Debugging...
	void CopyProjectionToDevice(float* buffer);
	void CopyDistanceImageToDevice(float* buffer);//For Debugging...
	void CopyRealProjectionToDevice(float* buffer);//For Debugging...
	void MPIBroadcast(float** buffers, int bufferCount);
#ifdef REFINE_MODE
	void GetCroppedProjection(float *outImage, int2 roiMin, int2 roiMax);
    void GetCroppedProjection(float *outImage, float *inImage, int2 roiMin, int2 roiMax);
	void CopyProjectionToSubVolumeProjection();
	float2 GetDisplacement(bool MultiPeakDetection, float* CCValue = NULL);
    float2 GetDisplacementPC(bool MultiPeakDetection, float* CCValue = NULL);
	// AS hipTextureObject_t texVol;
	void rotVol(Cuda::CudaDeviceVariable& vol, float phi, float psi, float theta);
	void setRotVolData(float* data);
	float* GetCCMap();
	float* GetCCMapMulti();
#endif

	void ConvertVolumeFP16(Volume<unsigned short>* vol, float* slice, Cuda::CudaSurfaceObject3D& surf, int z);
	void ConvertVolume3DFP16(float* volume, Cuda::CudaSurfaceObject3D& surf);
	void MatrixVector3Mul(float4x4 M, float3* v);
    void MatrixVector3Mul(float3x3& M, float xIn, float yIn, float& xOut, float& yOut);

	DeviceReconstructionConstantsCtf mRecParamCtf;
};

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


#endif // !RECONSTRUCTOR_H
