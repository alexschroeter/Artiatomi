#ifndef RECONSTRUCTOR_H
#define RECONSTRUCTOR_H

#include <hip/hip_runtime.h>

#ifdef USE_MPI
#include <mpi.h>
#endif
#include "default.h"
#include "Projection.h"
#include "Volume.h"
#include "Kernels.h"
#include "hip/HipArrays.h"
#include "hip/HipTextures.h"
#include "hip/HipContext.h"
#include "hip/HipDeviceProperties.h"
#include "utils/Config.h"
#include "utils/Matrix.h"
#include "io/Dm4FileStack.h"
#include "io/MRCFile.h"
#ifdef USE_MPI
#include "io/MPISource.h"
#endif
#include "io/MarkerFile.h"
#include "io/writeBMP.h"
#include "io/mrcHeader.h"
#include "io/emHeader.h"
#include "io/CtfFile.h"
#include <time.h>
#include <hipfft.h>
//#include <npp.h>
#include "NppEmulator.h"
#include "hip_kernels/DeviceReconstructionParameters.h"


class KernelModuls
{
private:
	bool compilerOutput;
	bool infoOutput;

public:
	KernelModuls(Hip::HipContext* aHipCtx);
	hipModule_t modFP;
	hipModule_t modSlicer;
	hipModule_t modVolTravLen;
	hipModule_t modComp;
	hipModule_t modWBP;
	hipModule_t modBP;
	hipModule_t modCTF;
	hipModule_t modCTS;
	hipModule_t modNppEmulator;	
};

/*
typedef struct {
	float4 m[4];
} float4x4;
*/

class Reconstructor
{
private:
	FPKernel fpKernel;
	SlicerKernel slicerKernel;
	VolTravLengthKernel volTravLenKernel;
	CompKernel compKernel;
	WbpWeightingKernel wbp;
	CropBorderKernel cropKernel;
	BPKernel bpKernel;
	ConvVolKernel convVolKernel;
	ConvVol3DKernel convVol3DKernel;
	CTFKernel ctf;
	CopyToSquareKernel cts;
	FourFilterKernel fourFilterKernel;
	ConjKernel conjKernel;
	MaxShiftKernel maxShiftKernel;
	NppEmulator NppEmu;
#ifdef REFINE_MODE
	Hip::HipDeviceVariable		projSquare2_d;
	RotKernel rotKernel;
	Hip::HipPitchedDeviceVariable projSubVols_d;
	float* ccMap;
	Hip::HipPitchedDeviceVariable ccMap_d;
	NppiRect roiCC1, roiCC2, roiCC3, roiCC4;
	NppiRect roiDestCC1, roiDestCC2, roiDestCC3, roiDestCC4;
#endif

	Hip::HipPitchedDeviceVariable realprojUS_d;
	Hip::HipPitchedDeviceVariable proj_d;
	Hip::HipPitchedDeviceVariable realproj_d;
	Hip::HipPitchedDeviceVariable dist_d;
	Hip::HipPitchedDeviceVariable filterImage_d;
	

	Hip::HipPitchedDeviceVariable ctf_d;
	Hip::HipDeviceVariable        fft_d;
	Hip::HipDeviceVariable		projSquare_d;
	Hip::HipPitchedDeviceVariable badPixelMask_d;
	Hip::HipPitchedDeviceVariable volTemp_d;

	hipfftHandle handleR2C;
	hipfftHandle handleC2R;

	NppiSize roiAll;
	NppiSize roiFFT;
	//NppiSize roiBorderSquare;
	NppiSize roiSquare;

	Hip::HipDeviceVariable meanbuffer;
	Hip::HipDeviceVariable meanval;
	Hip::HipDeviceVariable stdval;

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
	void ForwardProjectionCTF(Volume<TVol>* vol, Hip::HipTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);
	template<typename TVol>
	void ForwardProjectionNoCTF(Volume<TVol>* vol, Hip::HipTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);

	template<typename TVol>
	void ForwardProjectionCTFROI(Volume<TVol>* vol, Hip::HipTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
	template<typename TVol>
	void ForwardProjectionNoCTFROI(Volume<TVol>* vol, Hip::HipTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
		
	template<typename TVol>
	void BackProjectionNoCTF(Volume<TVol>* vol, int proj_index, float SIRTCount);
	template<typename TVol>
	void BackProjectionCTF(Volume<TVol>* vol, int proj_index, float SIRTCount);

#ifdef SUBVOLREC_MODE
	template<typename TVol>
	void BackProjectionNoCTF(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<Hip::HipArray3D*>& vecArrays, CUsurfref surfref, int proj_index);
	template<typename TVol>
	void BackProjectionCTF(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<Hip::HipArray3D*>& vecArrays, CUsurfref surfref, int proj_index);
#endif

	template<typename TVol>
	void GetDefocusDistances(float& t_in, float& t_out, int index, Volume<TVol>* vol);

	void GetDefocusMinMax(float ray, int index, float& defocusMin, float& defocusMax);

public:
	Reconstructor(Configuration::Config& aConfig, Projection& aProj, ProjectionSource* aProjectionSource,
		      MarkerFile& aMarkers, CtfFile& aDefocus, KernelModuls& modules, int aMpi_part, int aMpi_size, hipArray* arr3D );
	~Reconstructor();

	Matrix<float> GetMagAnistropyMatrix(float aAmount, float angleInDeg, float dimX, float dimY);

	//If returns true, ctf_d contains the fourier Filter mask as defined by coefficients given in config file.
	bool ComputeFourFilter();
	//img_h can be of any supported type. After the call, the type is float! Make sure the array is large enough!
	void PrepareProjection(void* img_h, int proj_index, float& meanValue, float& StdValue, int& BadPixels);

	template<typename TVol>
	void ForwardProjection(Volume<TVol>* vol, Hip::HipTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync = false);

	template<typename TVol>
	void ForwardProjectionROI(Volume<TVol>* vol, Hip::HipTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync = false);

	template<typename TVol>
	void ForwardProjectionDistanceOnly(Volume<TVol>* vol, int index);

	template<typename TVol>
	void Compare(Volume<TVol>* vol, char* originalImage, int index);

	//Assumes image to back project stored in proj_d. SIRTCount is overridable to config-file!
	template<typename TVol>
	void BackProjection(Volume<TVol>* vol, int proj_index, float SIRTCount);
	
#ifdef SUBVOLREC_MODE
	//Assumes image to back project stored in proj_d.
	template<typename TVol>
	void BackProjection(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<Hip::HipArray3D*>& vecArrays, CUsurfref surfref, int proj_index);
#endif

	template<typename TVol>
	void BackProjectionWithPriorWBPFilter(Volume<TVol>* vol, int proj_index, char* originalImage, float* MPIBuffer);

	template<typename TVol>
	void RemoveProjectionFromVol(Volume<TVol>* vol, int proj_index, char* originalImage, float* MPIBuffer);

	template<typename TVol>
	void OneSARTStep(Volume<TVol>* vol, Hip::HipTextureObject3D& texVol, int index, bool volumeIsEmpty, char* originalImage, float SIRTCount, float* MPIBuffer);
	

	void ResetProjectionsDevice();
	void CopyProjectionToHost(float* buffer);
	void CopyDistanceImageToHost(float* buffer);//For Debugging...
	void CopyRealProjectionToHost(float* buffer);//For Debugging...
	void CopyProjectionToDevice(float* buffer);
	void CopyDistanceImageToDevice(float* buffer);//For Debugging...
	void CopyRealProjectionToDevice(float* buffer);//For Debugging...
	void MPIBroadcast(float** buffers, int bufferCount);
#ifdef REFINE_MODE
	void CopyProjectionToSubVolumeProjection();
	float2 GetDisplacement();
	void rotVol(Hip::HipDeviceVariable& vol, float phi, float psi, float theta);
	void setRotVolData(float* data);
	float* GetCCMap();
#endif

	void ConvertVolumeFP16(Volume<unsigned short>* vol, float* slice, int z);
	void ConvertVolume3DFP16(Volume<unsigned short>* vol, float* volume);
	void MatrixVector3Mul(float4x4 M, float3* v);

	//SG!!
	hipArray *array3D;
	hipSurfaceObject_t surfObj;	
	Hip::HipTextureObject2D bpTexObj;	
	DeviceReconstructionConstantsCtf mRecParamCtf;
};

#endif // !RECONSTRUCTOR_H

