#include "Reconstructor.h"
#include "HipKernelBinarys.h"

#include "hip/hip_complex.h"
#include <hip/hip_runtime.h>
#include <hip/hip_texture_types.h>

using namespace std;
using namespace Hip;

#if defined(__HCC__)
//texture<float, 2, hipReadModeElementType> tex;
//texture<float, 3, hipReadModeElementType> texVol;
#endif

KernelModuls::KernelModuls(Hip::HipContext* aHipCtx)
  :compilerOutput(false),
   infoOutput(false)
{
  modFP = aHipCtx->LoadModulePTX(KernelForwardProjectionRayMarcher_TL, 0, infoOutput, compilerOutput);
  modSlicer = aHipCtx->LoadModulePTX(KernelForwardProjectionSlicer, 0, infoOutput, compilerOutput);
  modVolTravLen = modSlicer;
  modComp = aHipCtx->LoadModulePTX(KernelCompare, 0, infoOutput, compilerOutput);
  modWBP = aHipCtx->LoadModulePTX(KernelwbpWeighting, 0, infoOutput, compilerOutput);
  modBP = aHipCtx->LoadModulePTX(KernelBackProjectionSquareOS, 0, infoOutput, compilerOutput);
  modCTF = aHipCtx->LoadModulePTX(Kernelctf, 0, infoOutput, compilerOutput);
  modCTS = aHipCtx->LoadModulePTX(KernelCopyToSquare, 0, infoOutput, compilerOutput);
  modNppEmulator = aHipCtx->LoadModulePTX(KernelNppEmulatorKernel, 0, infoOutput, compilerOutput);
}

void Reconstructor::MatrixVector3Mul(float4x4 M, float3* v)
{
  float3 erg;
  erg.x = M.m[0].x * v->x + M.m[0].y * v->y + M.m[0].z * v->z + 1.f * M.m[0].w;
  erg.y = M.m[1].x * v->x + M.m[1].y * v->y + M.m[1].z * v->z + 1.f * M.m[1].w;
  erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
  *v = erg;
}

template<class TVol>
void Reconstructor::GetDefocusDistances(float & t_in, float & t_out, int index, Volume<TVol>* vol)
{
  //Shoot ray from center of volume:
  float3 c_projNorm = proj.GetNormalVector(index);
  float3 c_detektor = proj.GetPosition(index);
  float3 MC_bBoxMin;
  float3 MC_bBoxMax;
  MC_bBoxMin = vol->GetVolumeBBoxMin();
  MC_bBoxMax = vol->GetVolumeBBoxMax();
  float3 volDim = vol->GetDimension();
  float3 hitPoint;
  float t;
  //	printf("PosInVol2: %f, %f, %f\n", (MC_bBoxMin.x + (volDim.x * vol->GetVoxelSize().x * 0.5f + vol->GetVoxelSize().x * 0.5f)),
  //		(MC_bBoxMin.y + (volDim.y * vol->GetVoxelSize().y * 0.5f + vol->GetVoxelSize().x * 0.5f)),
  //		(MC_bBoxMin.z + (volDim.z * vol->GetVoxelSize().z * 0.5f + vol->GetVoxelSize().x * 0.5f)));

  t = (c_projNorm.x * (MC_bBoxMin.x + (volDim.x * vol->GetVoxelSize().x * 0.5f)) + 
       c_projNorm.y * (MC_bBoxMin.y + (volDim.y * vol->GetVoxelSize().y * 0.5f)) + 
       c_projNorm.z * (MC_bBoxMin.z + (volDim.z * vol->GetVoxelSize().z * 0.5f)));
  t += (-c_projNorm.x * c_detektor.x - c_projNorm.y * c_detektor.y - c_projNorm.z * c_detektor.z);
  t = abs(t);
	
  hitPoint.x = t * (-c_projNorm.x) + (MC_bBoxMin.x + (volDim.x * vol->GetVoxelSize().x * 0.5f));
  hitPoint.y = t * (-c_projNorm.y) + (MC_bBoxMin.y + (volDim.y * vol->GetVoxelSize().y * 0.5f));
  hitPoint.z = t * (-c_projNorm.z) + (MC_bBoxMin.z + (volDim.z * vol->GetVoxelSize().z * 0.5f));

  float4x4 c_DetectorMatrix;
	
  proj.GetDetectorMatrix(index, (float*)&c_DetectorMatrix, 1);
  MatrixVector3Mul(c_DetectorMatrix, &hitPoint);

  //--> pixelBorders.x = x.min; pixelBorders.z = y.min;
  int hitX = round(hitPoint.x);
  int hitY = round(hitPoint.y);

  //printf("HitX: %d, HitY: %d\n", hitX, hitY);

  //Shoot ray from hit point on projection towards volume to get the distance to entry and exit point
  //float3 pos = proj.GetPosition(index) + hitX * proj.GetPixelUPitch(index) + hitY * proj.GetPixelVPitch(index);
  float3 pos = proj.GetPosition(index) + hitX * proj.GetPixelUPitch(index) + hitY * proj.GetPixelVPitch(index);
  hitX = proj.GetWidth() * 0.5f;
  hitY = proj.GetHeight() * 0.5f;
  //float3 pos2 = proj.GetPosition(index) + hitX * proj.GetPixelUPitch(index) + hitX * proj.GetPixelVPitch(index);
  float3 nvec = proj.GetNormalVector(index);

  /*float3 MC_bBoxMin;
    float3 MC_bBoxMax;*/

	

  t_in = 2*-DIST;
  t_out = 2*DIST;

  for (int x = 0; x <= 1; x++)
    for (int y = 0; y <= 1; y++)
      for (int z = 0; z <= 1; z++)
	{
	  //float t;

	  t = (nvec.x * (MC_bBoxMin.x + x * (MC_bBoxMax.x - MC_bBoxMin.x))
	       + nvec.y * (MC_bBoxMin.y + y * (MC_bBoxMax.y - MC_bBoxMin.y))
	       + nvec.z * (MC_bBoxMin.z + z * (MC_bBoxMax.z - MC_bBoxMin.z)));
	  t += (-nvec.x * pos.x - nvec.y * pos.y - nvec.z * pos.z);

	  if (t < t_in) t_in = t;
	  if (t > t_out) t_out = t;
	}

  //printf("t_in: %f; t_out: %f\n", t_in, t_out);
  //t_in = 2*-DIST;
  //t_out = 2*DIST;

  //for (int x = 0; x <= 1; x++)
  //	for (int y = 0; y <= 1; y++)
  //		for (int z = 0; z <= 1; z++)
  //		{
  //			//float t;

  //			t = (nvec.x * (MC_bBoxMin.x + x * (MC_bBoxMax.x - MC_bBoxMin.x))
  //				+ nvec.y * (MC_bBoxMin.y + y * (MC_bBoxMax.y - MC_bBoxMin.y))
  //				+ nvec.z * (MC_bBoxMin.z + z * (MC_bBoxMax.z - MC_bBoxMin.z)));
  //			t += (-nvec.x * pos2.x - nvec.y * pos2.y - nvec.z * pos2.z);

  //			if (t < t_in) t_in = t;
  //			if (t > t_out) t_out = t;
  //		}
  ////printf("t_in: %f; t_out: %f\n", t_in, t_out);



  //{
  //	float xAniso = 2366.25f;
  //	float yAniso = 4527.75f;

  //	float3 c_source = c_detektor;
  //	float3 c_uPitch = proj.GetPixelUPitch(index);
  //	float3 c_vPitch = proj.GetPixelVPitch(index);
  //	c_source = c_source + (xAniso)* c_uPitch;
  //	c_source = c_source + (yAniso)* c_vPitch;

  //	//////////// BOX INTERSECTION (partial Volume) /////////////////
  //	float3 tEntry;
  //	tEntry.x = (MC_bBoxMin.x - c_source.x) / (c_projNorm.x);
  //	tEntry.y = (MC_bBoxMin.y - c_source.y) / (c_projNorm.y);
  //	tEntry.z = (MC_bBoxMin.z - c_source.z) / (c_projNorm.z);

  //	float3 tExit;
  //	tExit.x = (MC_bBoxMax.x - c_source.x) / (c_projNorm.x);
  //	tExit.y = (MC_bBoxMax.y - c_source.y) / (c_projNorm.y);
  //	tExit.z = (MC_bBoxMax.z - c_source.z) / (c_projNorm.z);


  //	float3 tmin = fminf(tEntry, tExit);
  //	float3 tmax = fmaxf(tEntry, tExit);

  //	t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
  //	t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
  //	printf("t_in: %f; t_out: %f\n", t_in, t_out);
  //}
}
template void Reconstructor::GetDefocusDistances(float & t_in, float & t_out, int index, Volume<unsigned short>* vol);
template void Reconstructor::GetDefocusDistances(float & t_in, float & t_out, int index, Volume<float>* vol);


void Reconstructor::GetDefocusMinMax(float ray, int index, float & defocusMin, float & defocusMax)
{
  defocusMin = defocus.GetMinDefocus(index);
  defocusMax = defocus.GetMaxDefocus(index);
  float tiltAngle = (markers(MFI_TiltAngle, index, 0) + config.AddTiltAngle) / 180.0f * (float)M_PI;

  float distanceTo0 = ray + DIST; //in pixel
  if (config.IgnoreZShiftForCTF)
    {
      distanceTo0 = (round(distanceTo0 * proj.GetPixelSize(index) * config.CTFSliceThickness) / config.CTFSliceThickness) + config.CTFSliceThickness / 2.0f;
    }
  else
    {
      distanceTo0 = (round(distanceTo0 * proj.GetPixelSize(index) * config.CTFSliceThickness) / config.CTFSliceThickness) + config.CTFSliceThickness / 2.0f - (config.VolumeShift.z * proj.GetPixelSize(index) * cosf(tiltAngle)); //in nm
    }
  if (config.SwitchCTFDirectionForIMOD)
    {
      distanceTo0 *= -1; //IMOD inverses the logic...
    }
	

  defocusMin = defocusMin + distanceTo0;
  defocusMax = defocusMax + distanceTo0;
}



Reconstructor::Reconstructor(Configuration::Config & aConfig,
			     Projection & aProj, ProjectionSource* aProjectionSource,
			     MarkerFile& aMarkers, CtfFile& aDefocus, KernelModuls& modules, int aMpi_part, int aMpi_size, hipArray* arr3D)
  : 
  fpKernel(modules.modFP),
  slicerKernel(modules.modSlicer),
  volTravLenKernel(modules.modVolTravLen),
  compKernel(modules.modComp),
  wbp(modules.modWBP),
  cropKernel(modules.modComp),
  bpKernel(modules.modBP, aConfig.FP16Volume),
  convVolKernel(modules.modBP),
  convVol3DKernel(modules.modBP),
  ctf(modules.modCTF),
  cts(modules.modCTS),
  fourFilterKernel(modules.modWBP),
  conjKernel(modules.modWBP),
  maxShiftKernel(modules.modWBP),
  NppEmu(modules.modNppEmulator),
#ifdef REFINE_MODE
  projSquare2_d(),
  rotKernel(modules.modWBP, aConfig.SizeSubVol),
  projSubVols_d(),
  ccMap(NULL),
  ccMap_d(),
  roiCC1(), roiCC2(), roiCC3(), roiCC4(),
  roiDestCC1(), roiDestCC2(), roiDestCC3(), roiDestCC4(),
#endif
  realprojUS_d(),
  proj_d(),
  realproj_d(),
  dist_d(),
  filterImage_d(),
  ctf_d(),
  fft_d(),
  projSquare_d(),
  badPixelMask_d(),
  volTemp_d(),
  handleR2C(),
  handleC2R(),
  roiAll(),
  roiFFT(),
  roiSquare(),
  meanbuffer(),
  meanval(),
  stdval(),
  proj(aProj), 
  projSource(aProjectionSource), 
  defocus(aDefocus),
  markers(aMarkers),
  config(aConfig),
  mpi_part(aMpi_part),
  mpi_size(aMpi_size),
  skipFilter(aConfig.SkipFilter),
  squareBorderSizeX(0),
  squareBorderSizeY(0),
  squarePointerShift(0),
  MPIBuffer(NULL),
  magAnisotropy(GetMagAnistropyMatrix(aConfig.MagAnisotropyAmount, aConfig.MagAnisotropyAngleInDeg, proj.GetWidth(), proj.GetHeight())),
  magAnisotropyInv(GetMagAnistropyMatrix(1.0f / aConfig.MagAnisotropyAmount, aConfig.MagAnisotropyAngleInDeg, proj.GetWidth(), proj.GetHeight())),
  array3D(arr3D),
  surfObj(),
  bpTexObj(),
  mRecParamCtf()
{  

  //Set kernel work dimensions for 2D images:
  fpKernel.SetWorkSize(proj.GetWidth(), proj.GetHeight(), 1);
  slicerKernel.SetWorkSize(proj.GetWidth(), proj.GetHeight(), 1);
  volTravLenKernel.SetWorkSize(proj.GetWidth(), proj.GetHeight(), 1);
  compKernel.SetWorkSize(proj.GetWidth(), proj.GetHeight(), 1);
  cropKernel.SetWorkSize(proj.GetWidth(), proj.GetHeight(), 1);

  wbp.SetWorkSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
  ctf.SetWorkSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
  fourFilterKernel.SetWorkSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
  conjKernel.SetWorkSize(proj.GetMaxDimension() / 2 + 1, proj.GetMaxDimension(), 1);
  cts.SetWorkSize(proj.GetMaxDimension(), proj.GetMaxDimension(), 1);
  maxShiftKernel.SetWorkSize(proj.GetMaxDimension(), proj.GetMaxDimension(), 1);
  convVolKernel.SetWorkSize(config.RecDimensions.x, config.RecDimensions.y, 1);


  //Alloc device variables

  // SG: allocate non-pitched device variables before pitched ones 
  // due to a bug in AMD implementation of memory manager:
  // sometimes it has problem with releasing memory
  //

#ifdef REFINE_MODE
  projSquare2_d.Alloc(proj.GetMaxDimension() * sizeof(float) * proj.GetMaxDimension());
#endif

  fft_d.Alloc((proj.GetMaxDimension() / 2 + 1) * sizeof(hipComplex) * proj.GetMaxDimension());
  projSquare_d.Alloc(proj.GetMaxDimension() * sizeof(float) * proj.GetMaxDimension());

  size_t squarePointerShiftX = ((proj.GetMaxDimension() - proj.GetWidth()) / 2);
  size_t squarePointerShiftY = ((proj.GetMaxDimension() - proj.GetHeight()) / 2) * proj.GetMaxDimension();
  squarePointerShift = squarePointerShiftX + squarePointerShiftY;
  squareBorderSizeX = (proj.GetMaxDimension() - proj.GetWidth()) / 2;
  squareBorderSizeY = (proj.GetMaxDimension() - proj.GetHeight()) / 2;
  //roiBorderSquare.width = squareBorderSize;
  //roiBorderSquare.height = proj.GetHeight();
  roiSquare.width = proj.GetMaxDimension();
  roiSquare.height = proj.GetMaxDimension();

  roiAll.width = proj.GetWidth();
  roiAll.height = proj.GetHeight();
  roiFFT.width = proj.GetMaxDimension() / 2 + 1;
  roiFFT.height = proj.GetHeight();

  int bufferSize = 0;
  NppEmu.nppiMeanStdDevGetBufferHostSize_32f_C1R(roiAll, &bufferSize);
  int bufferSize2;
  NppEmu.nppiMaxIndxGetBufferHostSize_32f_C1R(roiSquare, &bufferSize2);
  if (bufferSize2 > bufferSize) bufferSize = bufferSize2;
 
  NppEmu.nppiMeanGetBufferHostSize_32f_C1R(roiSquare, &bufferSize2);
  if (bufferSize2 > bufferSize) bufferSize = bufferSize2;

  NppEmu.nppiSumGetBufferHostSize_32f_C1R(roiSquare, &bufferSize2);
  if (bufferSize2 > bufferSize) bufferSize = bufferSize2;

  cout<<"bufferSize = "<<bufferSize<<endl;

  meanbuffer.Alloc(bufferSize * 10);
  meanval.Alloc(sizeof(double));
  stdval.Alloc(sizeof(double));

  // pitched variables

  realprojUS_d.Alloc(proj.GetWidth() * sizeof(int), proj.GetHeight(), sizeof(int));
  proj_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));
  realproj_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));
  dist_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));
  filterImage_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));

#ifdef REFINE_MODE
  projSubVols_d.Alloc(proj.GetWidth() * sizeof(float), proj.GetHeight(), sizeof(float));
  ccMap = new float[aConfig.MaxShift * 4 * aConfig.MaxShift * 4];
  ccMap_d.Alloc(4 * aConfig.MaxShift * sizeof(float), 4 * aConfig.MaxShift, sizeof(float));
  ccMap_d.Memset(0);

  roiCC1.x = 0;
  roiCC1.y = 0;
  roiCC1.width = aConfig.MaxShift * 2;
  roiCC1.height = aConfig.MaxShift * 2;

  roiCC2.x = proj.GetMaxDimension() - aConfig.MaxShift * 2;
  roiCC2.y = 0;
  roiCC2.width = aConfig.MaxShift * 2;
  roiCC2.height = aConfig.MaxShift * 2;

  roiCC3.x = 0;
  roiCC3.y = proj.GetMaxDimension() - aConfig.MaxShift * 2;
  roiCC3.width = aConfig.MaxShift * 2;
  roiCC3.height = aConfig.MaxShift * 2;

  roiCC4.x = proj.GetMaxDimension() - aConfig.MaxShift * 2;
  roiCC4.y = proj.GetMaxDimension() - aConfig.MaxShift * 2;
  roiCC4.width = aConfig.MaxShift * 2;
  roiCC4.height = aConfig.MaxShift * 2;

  roiDestCC4.x = 0;
  roiDestCC4.y = 0;
  roiDestCC1.width = aConfig.MaxShift * 2;
  roiDestCC1.height = aConfig.MaxShift * 2;

  roiDestCC3.x = aConfig.MaxShift * 4 - aConfig.MaxShift * 2;
  roiDestCC3.y = 0;
  roiDestCC2.width = aConfig.MaxShift * 2;
  roiDestCC2.height = aConfig.MaxShift * 2;

  roiDestCC2.x = 0;
  roiDestCC2.y = aConfig.MaxShift * 4 - aConfig.MaxShift * 2;
  roiDestCC3.width = aConfig.MaxShift * 2;
  roiDestCC3.height = aConfig.MaxShift * 2;

  roiDestCC1.x = aConfig.MaxShift * 4 - aConfig.MaxShift * 2;
  roiDestCC1.y = aConfig.MaxShift * 4 - aConfig.MaxShift * 2;
  roiDestCC4.width = aConfig.MaxShift * 2;
  roiDestCC4.height = aConfig.MaxShift * 2;
#endif

  //Bind back projection image to texref in BP Kernel
  if (aConfig.CtfMode == Configuration::Config::CTFM_YES)
    {
      bpTexObj.Create( &dist_d, hipFilterModeLinear );
      //HipTextureLinearPitched2D::Bind(&bpKernel, "tex", hipFilterModeLinear, &dist_d, 1);
    }
  else
    {
      bpTexObj.Create( &proj_d, hipFilterModePoint );
      //HipTextureLinearPitched2D::Bind(&bpKernel, "tex", hipFilterModePoint, &proj_d, 1);      
    }

  ctf_d.Alloc((proj.GetMaxDimension() / 2 + 1) * sizeof(hipComplex), proj.GetMaxDimension(), sizeof(hipComplex));
  badPixelMask_d.Alloc(proj.GetMaxDimension() * sizeof(Npp8u), proj.GetMaxDimension(), sizeof(Npp8u));

  hipfftSafeCall(hipfftPlan2d(&handleR2C, proj.GetMaxDimension(), proj.GetMaxDimension(), HIPFFT_R2C));
  hipfftSafeCall(hipfftPlan2d(&handleC2R, proj.GetMaxDimension(), proj.GetMaxDimension(), HIPFFT_C2R));

#ifdef USE_MPI
  MPIBuffer = new float[proj.GetWidth() * proj.GetHeight()];
#endif
 
  // SG:  set reconstruction parameters for device kernels
  
  mRecParamCtf.cs = config.Cs;
  mRecParamCtf.voltage = config.Voltage;
  mRecParamCtf.openingAngle = 0.01f;
  mRecParamCtf.ampContrast = 0.00f;
  mRecParamCtf.phaseContrast = sqrt( 1. - mRecParamCtf.ampContrast*mRecParamCtf.ampContrast );
  mRecParamCtf.pixelsize = proj.GetPixelSize(0) * pow(10, -9);
  mRecParamCtf.pixelcount = proj.GetMaxDimension();
  mRecParamCtf.maxFreq = 1.0 / (mRecParamCtf.pixelsize * 2.0 );
  mRecParamCtf.freqStepSize = mRecParamCtf.maxFreq / (mRecParamCtf.pixelcount / 2.0f);
  // mRecParamCtf.c_lambda = ;
  mRecParamCtf.applyScatteringProfile = 0.f;
  mRecParamCtf.applyEnvelopeFunction = 0.f;

  //SetConstantValues(ctf, proj, 0, config.Cs, config.Voltage);
   

  ResetProjectionsDevice();

  hipResourceDesc resDesc;
  memset(&resDesc, 0, sizeof(resDesc));
  resDesc.resType = hipResourceTypeArray;
  resDesc.res.array.array = arr3D;
  hipCreateSurfaceObject(&surfObj, &resDesc);
}

Reconstructor::~Reconstructor()
{  
  if (MPIBuffer)
    {
      delete[] MPIBuffer;
      MPIBuffer = NULL;
    }
  hipfftSafeCall(hipfftDestroy(handleR2C));
  hipfftSafeCall(hipfftDestroy(handleC2R));
}

Matrix<float> Reconstructor::GetMagAnistropyMatrix(float aAmount, float angleInDeg, float dimX, float dimY)
{
  float angle = angleInDeg / 180.0f * M_PI;

  Matrix<float> shiftCenter(3, 3);
  Matrix<float> shiftBack(3, 3);
  Matrix<float> rotMat1 = Matrix<float>::GetRotationMatrix3DZ(angle);
  Matrix<float> rotMat2 = Matrix<float>::GetRotationMatrix3DZ(-angle);
  Matrix<float> stretch(3, 3);
  shiftCenter(0, 0) = 1;
  shiftCenter(0, 1) = 0;
  shiftCenter(0, 2) = -dimX / 2.0f;
  shiftCenter(1, 0) = 0;
  shiftCenter(1, 1) = 1;
  shiftCenter(1, 2) = -dimY / 2.0f;
  shiftCenter(2, 0) = 0;
  shiftCenter(2, 1) = 0;
  shiftCenter(2, 2) = 1;

  shiftBack(0, 0) = 1;
  shiftBack(0, 1) = 0;
  shiftBack(0, 2) = dimX / 2.0f;
  shiftBack(1, 0) = 0;
  shiftBack(1, 1) = 1;
  shiftBack(1, 2) = dimY / 2.0f;
  shiftBack(2, 0) = 0;
  shiftBack(2, 1) = 0;
  shiftBack(2, 2) = 1;

  stretch(0, 0) = aAmount;
  stretch(0, 1) = 0;
  stretch(0, 2) = 0;
  stretch(1, 0) = 0;
  stretch(1, 1) = 1;
  stretch(1, 2) = 0;
  stretch(2, 0) = 0;
  stretch(2, 1) = 0;
  stretch(2, 2) = 1;

  return shiftBack * rotMat2 * stretch * rotMat1 * shiftCenter;
}

template<class TVol>
void Reconstructor::ForwardProjectionNoCTF(Volume<TVol>* vol, HipTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync)
{	
  //proj_d.Memset(0);
  //dist_d.Memset(0);
  //float runtime;

  DeviceReconstructionConstantsCommon param = kernels::GetReconstructionParameters( *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

  if (!volumeIsEmpty)
    {
      //SetConstantValues(fpKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
      //runtime = 
      fpKernel(param, proj.GetWidth(), proj.GetHeight(), proj_d, dist_d, texVol);

#ifdef USE_MPI
      if (!noSync)
	{
	  if (mpi_part == 0)
	    {
	      for (int mpi = 1; mpi < mpi_size; mpi++)
		{
		  MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  realprojUS_d.CopyHostToDevice(MPIBuffer);
		  NppEmu.nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)proj_d.GetDevicePtr(), proj_d.GetPitch(), roiAll);
		  MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  realprojUS_d.CopyHostToDevice(MPIBuffer);
		  NppEmu.nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);
		}
	    }
	  else
	    {
	      proj_d.CopyDeviceToHost(MPIBuffer);
	      MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	      dist_d.CopyDeviceToHost(MPIBuffer);
	      MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	    }
	}
#endif
    }
  else
    {
      //SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

      //runtime = 
      volTravLenKernel(param, proj.GetWidth(), proj.GetHeight(), dist_d);
#ifdef USE_MPI
      if (!noSync)
	{
	  if (mpi_part == 0)
	    {
	      for (int mpi = 1; mpi < mpi_size; mpi++)
		{
		  MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  realprojUS_d.CopyHostToDevice(MPIBuffer);
		  NppEmu.nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);
		}
	    }
	  else
	    {
	      dist_d.CopyDeviceToHost(MPIBuffer);
	      MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	    }
	}
#endif
    }	
}
template void Reconstructor::ForwardProjectionNoCTF(Volume<unsigned short>* vol, HipTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);
template void Reconstructor::ForwardProjectionNoCTF(Volume<float>* vol, HipTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);


template<class TVol>
void Reconstructor::ForwardProjectionCTF(Volume<TVol>* vol, HipTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync)
{
  //proj_d.Memset(0);
  //dist_d.Memset(0);
  //float runtime;
  int x = proj.GetWidth();
  int y = proj.GetHeight();

  //if (mpi_part == 0)
  //	printf("\n");
 
  DeviceReconstructionConstantsCommon param = kernels::GetReconstructionParameters( *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

  if (!volumeIsEmpty)
    {
      //SetConstantValues(slicerKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
      //SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

      float t_in, t_out;
      GetDefocusDistances(t_in, t_out, index, vol);

      for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize(index))
	{
	  dist_d.Memset(0);

	  float defocusAngle = defocus.GetAstigmatismAngle(index);
	  float defocusMin;
	  float defocusMax;
	  GetDefocusMinMax(ray, index, defocusMin, defocusMax);

	  if (mpi_part == 0)
	    {
	      printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b FP Defocus: %-8d nm", (int)defocusMin);
	      fflush(stdout);
	    }
	  //printf("\n");
	  //runtime =
	  slicerKernel(param, x, y, dist_d, ray, ray + config.CTFSliceThickness / proj.GetPixelSize(index), texVol);

#ifdef USE_MPI
	  if (!noSync)
	    {
	      if (mpi_part == 0)
		{
		  for (int mpi = 1; mpi < mpi_size; mpi++)
		    {
		      MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		      realprojUS_d.CopyHostToDevice(MPIBuffer);
		      NppEmu.nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);
		    }
		}
	      else
		{
		  dist_d.CopyDeviceToHost(MPIBuffer);
		  MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		}
	    }
#endif
	  //CTF filtering is only done on GPU 0!
	  if (mpi_part == 0)
	    {
	      /*dist_d.CopyDeviceToHost(MPIBuffer);
		writeBMP(string("VorCTF.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/

	      int2 pA, pB, pC, pD;
	      pA.x = 0;
	      pA.y = proj.GetHeight() - 1;
	      pB.x = proj.GetWidth() - 1;
	      pB.y = proj.GetHeight() - 1;
	      pC.x = 0;
	      pC.y = 0;
	      pD.x = proj.GetWidth() - 1;
	      pD.y = 0;

	      cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);

	      cts(dist_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);

	      fft_d.Memset(0);
	      hipfftSafeCall(hipfftExecR2C(handleR2C, (hipfftReal*)projSquare_d.GetDevicePtr(), (hipfftComplex*)fft_d.GetDevicePtr()));

	      ctf(mRecParamCtf, fft_d, defocusMin, defocusMax, defocusAngle, false, (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), config.CTFBetaFac);

	      hipfftSafeCall(hipfftExecC2R(handleC2R, (hipfftComplex*)fft_d.GetDevicePtr(), (hipfftReal*)projSquare_d.GetDevicePtr()));

	      NppEmu.nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension() * proj.GetMaxDimension(),
				      (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);


	      cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
	      NppEmu.nppiAdd_32f_C1IR((Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), (Npp32f*)proj_d.GetDevicePtr(), proj_d.GetPitch(), roiAll);
	    }
	}
      /*proj_d.CopyDeviceToHost(MPIBuffer);
	writeBMP(string("Proj.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
      //Get Volume traversal lengths
      dist_d.Memset(0);
      //runtime = 
      volTravLenKernel(param, x, y, dist_d);

#ifdef USE_MPI
      if (!noSync)
	{
	  if (mpi_part == 0)
	    {
	      for (int mpi = 1; mpi < mpi_size; mpi++)
		{
		  MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  realprojUS_d.CopyHostToDevice(MPIBuffer);
		  NppEmu.nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);
		}
	      /*proj_d.CopyDeviceToHost(MPIBuffer);
		printf("\n");
		writeBMP(string("proj3.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());
		printf("\n");
		dist_d.CopyDeviceToHost(MPIBuffer);
		writeBMP(string("dist.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
	    }
	  else
	    {
	      dist_d.CopyDeviceToHost(MPIBuffer);
	      MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	    }
	}
#endif

    }
  else
    {
      //SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

      //runtime = 
      volTravLenKernel(param, proj.GetWidth(), proj.GetHeight(), dist_d);
#ifdef USE_MPI
      if (!noSync)
	{
	  if (mpi_part == 0)
	    {
	      for (int mpi = 1; mpi < mpi_size; mpi++)
		{
		  MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  realprojUS_d.CopyHostToDevice(MPIBuffer);
		  NppEmu.nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);
		}
	    }
	  else
	    {
	      dist_d.CopyDeviceToHost(MPIBuffer);
	      MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	    }
	}
#endif
    }
}
template void Reconstructor::ForwardProjectionCTF(Volume<unsigned short>* vol, HipTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);
template void Reconstructor::ForwardProjectionCTF(Volume<float>* vol, HipTextureObject3D& tevVol, int index, bool volumeIsEmpty, bool noSync);

template<class TVol>
void Reconstructor::ForwardProjectionNoCTFROI(Volume<TVol>* vol, HipTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync)
{	
  //proj_d.Memset(0);
  //dist_d.Memset(0);
  //float runtime;

  DeviceReconstructionConstantsCommon param = kernels::GetReconstructionParameters( *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

  if (!volumeIsEmpty)
    {
      //SetConstantValues(fpKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
      //runtime = 
      fpKernel(param, proj.GetWidth(), proj.GetHeight(), proj_d, dist_d, texVol, roiMin, roiMax);

#ifdef USE_MPI
      if (!noSync)
	{
	  if (mpi_part == 0)
	    {
	      for (int mpi = 1; mpi < mpi_size; mpi++)
		{
		  MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  realprojUS_d.CopyHostToDevice(MPIBuffer);
		  NppEmu.nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)proj_d.GetDevicePtr(), proj_d.GetPitch(), roiAll);
		  MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  realprojUS_d.CopyHostToDevice(MPIBuffer);
		  NppEmu.nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);
		}
	    }
	  else
	    {
	      proj_d.CopyDeviceToHost(MPIBuffer);
	      MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	      dist_d.CopyDeviceToHost(MPIBuffer);
	      MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	    }
	}
#endif
    }
  else
    {
      //SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

      //runtime = 
      volTravLenKernel(param, proj.GetWidth(), proj.GetHeight(), dist_d, roiMin, roiMax);
#ifdef USE_MPI
      if (!noSync)
	{
	  if (mpi_part == 0)
	    {
	      for (int mpi = 1; mpi < mpi_size; mpi++)
		{
		  MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  realprojUS_d.CopyHostToDevice(MPIBuffer);
		  NppEmu.nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);
		}
	    }
	  else
	    {
	      dist_d.CopyDeviceToHost(MPIBuffer);
	      MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	    }
	}
#endif
    }	
}
template void Reconstructor::ForwardProjectionNoCTFROI(Volume<unsigned short>* vol, HipTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
template void Reconstructor::ForwardProjectionNoCTFROI(Volume<float>* vol, HipTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);


template<class TVol>
void Reconstructor::ForwardProjectionCTFROI(Volume<TVol>* vol, HipTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync)
{
  //proj_d.Memset(0);
  //dist_d.Memset(0);
  //float runtime;
  int x = proj.GetWidth();
  int y = proj.GetHeight();

  //if (mpi_part == 0)
  //	printf("\n");

  DeviceReconstructionConstantsCommon param = kernels::GetReconstructionParameters( *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

  if (!volumeIsEmpty)
    {
      //SetConstantValues(slicerKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
      //SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

      float t_in, t_out;
      GetDefocusDistances(t_in, t_out, index, vol);
      /*t_in -= 2*config.CTFSliceThickness / proj.GetPixelSize(index);
	t_out += 2*config.CTFSliceThickness / proj.GetPixelSize(index);*/

      for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize(index))
	{
	  dist_d.Memset(0);

	  float defocusAngle = defocus.GetAstigmatismAngle(index);
	  float defocusMin;
	  float defocusMax;
	  GetDefocusMinMax(ray, index, defocusMin, defocusMax);

	  if (mpi_part == 0)
	    {
	      printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b FP Defocus: %-8d nm", (int)defocusMin);
	      fflush(stdout);
	    }
	  //printf("\n");
	  //runtime = 
	  slicerKernel(param, x, y, dist_d, ray, ray + config.CTFSliceThickness / proj.GetPixelSize(index), texVol, roiMin, roiMax);

#ifdef USE_MPI
	  if (!noSync)
	    {
	      if (mpi_part == 0)
		{
		  for (int mpi = 1; mpi < mpi_size; mpi++)
		    {
		      MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		      realprojUS_d.CopyHostToDevice(MPIBuffer);
		      NppEmu.nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);
		    }
		}
	      else
		{
		  dist_d.CopyDeviceToHost(MPIBuffer);
		  MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
		}
	    }
#endif
	  //CTF filtering is only done on GPU 0!
	  if (mpi_part == 0)
	    {
	      /*dist_d.CopyDeviceToHost(MPIBuffer);
		writeBMP(string("VorCTF.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/

	      int2 pA, pB, pC, pD;
	      pA.x = 0;
	      pA.y = proj.GetHeight() - 1;
	      pB.x = proj.GetWidth() - 1;
	      pB.y = proj.GetHeight() - 1;
	      pC.x = 0;
	      pC.y = 0;
	      pD.x = proj.GetWidth() - 1;
	      pD.y = 0;

	      cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);

	      cts(dist_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);

	      fft_d.Memset(0);
	      hipfftSafeCall(hipfftExecR2C(handleR2C, (hipfftReal*)projSquare_d.GetDevicePtr(), (hipfftComplex*)fft_d.GetDevicePtr()));

	      ctf(mRecParamCtf, fft_d, defocusMin, defocusMax, defocusAngle, false, (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), config.CTFBetaFac);

	      hipfftSafeCall(hipfftExecC2R(handleC2R, (hipfftComplex*)fft_d.GetDevicePtr(), (hipfftReal*)projSquare_d.GetDevicePtr()));

	      NppEmu.nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension() * proj.GetMaxDimension(),
					   (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);


	      cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);
	      NppEmu.nppiAdd_32f_C1IR((Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), (Npp32f*)proj_d.GetDevicePtr(), proj_d.GetPitch(), roiAll);
	    }
	}
      /*proj_d.CopyDeviceToHost(MPIBuffer);
	writeBMP(string("Proj.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
      //Get Volume traversal lengths
      dist_d.Memset(0);
      //runtime = 
      volTravLenKernel(param, x, y, dist_d, roiMin, roiMax);
      
#ifdef USE_MPI
      if (!noSync)
	{
	  if (mpi_part == 0)
	    {
	      for (int mpi = 1; mpi < mpi_size; mpi++)
		{
		  MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  realprojUS_d.CopyHostToDevice(MPIBuffer);
		  NppEmu.nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);
		}
	      /*proj_d.CopyDeviceToHost(MPIBuffer);
		printf("\n");
		writeBMP(string("proj3.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());
		printf("\n");
		dist_d.CopyDeviceToHost(MPIBuffer);
		writeBMP(string("dist.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
	    }
	  else
	    {
	      dist_d.CopyDeviceToHost(MPIBuffer);
	      MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	    }
	}
#endif

    }
  else
    {
      // SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

      //runtime = 
      volTravLenKernel(param, proj.GetWidth(), proj.GetHeight(), dist_d, roiMin, roiMax);
#ifdef USE_MPI
      if (!noSync)
	{
	  if (mpi_part == 0)
	    {
	      for (int mpi = 1; mpi < mpi_size; mpi++)
		{
		  MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		  realprojUS_d.CopyHostToDevice(MPIBuffer);
		  NppEmu.nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);
		}
	    }
	  else
	    {
	      dist_d.CopyDeviceToHost(MPIBuffer);
	      MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
	    }
	}
#endif
    }
}
template void Reconstructor::ForwardProjectionCTFROI(Volume<unsigned short>* vol, HipTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
template void Reconstructor::ForwardProjectionCTFROI(Volume<float>* vol, HipTextureObject3D& tevVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);

template<class TVol>
void Reconstructor::BackProjectionNoCTF(Volume<TVol>* vol, int proj_index, float SIRTCount)
{
  float3 volDim = vol->GetSubVolumeDimension(mpi_part);
  bpKernel.SetWorkSize((int)volDim.x, (int)volDim.y, (int)volDim.z);

  int2 pA, pB, pC, pD;

  proj.ComputeHitPoints(*vol, proj_index, pA, pB, pC, pD);

  cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);

  DeviceReconstructionConstantsCommon param = kernels::GetReconstructionParameters( *vol, proj, proj_index, mpi_part, magAnisotropy, magAnisotropyInv);	
	
  //SetConstantValues(bpKernel, *vol, proj, proj_index, mpi_part, magAnisotropy, magAnisotropyInv);

  //float runtime = 
  bpKernel(param, proj.GetWidth(), proj.GetHeight(), config.Lambda / SIRTCount, 
	   config.OverSampling, 1.0f / (float)(config.OverSampling), proj_d, 0, 9999999999999.0f,surfObj, bpTexObj);

}
template void Reconstructor::BackProjectionNoCTF(Volume<unsigned short>* vol, int proj_index, float SIRTCount);
template void Reconstructor::BackProjectionNoCTF(Volume<float>* vol, int proj_index, float SIRTCount);


#ifdef SUBVOLREC_MODE
template<class TVol>
void Reconstructor::BackProjectionNoCTF(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<HipArray3D*>& vecArrays, CUsurfref surfref, int proj_index)
{
  size_t batchSize = subVolumes.size();

  for (size_t batch = 0; batch < batchSize; batch++)
    {
      //bind surfref to correct array:
      hipSafeCall(cuSurfRefSetArray(surfref, vecArrays[batch]->GetArray(), 0));

      //set additional shifts:
      proj.SetExtraShift(proj_index, vecExtraShifts[batch]);

      float3 volDim = subVolumes[batch]->GetSubVolumeDimension(0);
      bpKernel.SetWorkSize((int)volDim.x, (int)volDim.y, (int)volDim.z);

      DeviceReconstructionConstantsCommon param = kernels::GetReconstructionParameters( *(subVolumes[batch]), proj, proj_index, 0, magAnisotropy, magAnisotropyInv);	
      //SetConstantValues(bpKernel, *(subVolumes[batch]), proj, proj_index, 0, magAnisotropy, magAnisotropyInv);

      //float runtime = 
      bpKernel(param, proj.GetWidth(), proj.GetHeight(), 1.0f,
	       config.OverSampling, 1.0f / (float)(config.OverSampling), proj_d, 0, 9999999999999.0f,surfObj, bpTexObj);
    }
}
template void Reconstructor::BackProjectionNoCTF(Volume<unsigned short>* vol, vector<Volume<unsigned short>*>& subVolumes, vector<float2>& vecExtraShifts, vector<HipArray3D*>& vecArrays, CUsurfref surfref, int proj_index);
template void Reconstructor::BackProjectionNoCTF(Volume<float>* vol, vector<Volume<float>*>& subVolumes, vector<float2>& vecExtraShifts, vector<HipArray3D*>& vecArrays, CUsurfref surfref, int proj_index);
#endif

template<class TVol>
void Reconstructor::BackProjectionCTF(Volume<TVol>* vol, int proj_index, float SIRTcount)
{
  float3 volDim = vol->GetSubVolumeDimension(mpi_part);
  bpKernel.SetWorkSize((int)volDim.x, (int)volDim.y, (int)volDim.z);
  //float runtime;
  int x = proj.GetWidth();
  int y = proj.GetHeight();

  if (mpi_part == 0)
    printf("\n");

  float t_in, t_out;
  GetDefocusDistances(t_in, t_out, proj_index, vol);

  for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize(proj_index))
    {
      //SetConstantValues(bpKernel, *vol, proj, proj_index, mpi_part, magAnisotropy, magAnisotropyInv);
      DeviceReconstructionConstantsCommon param = kernels::GetReconstructionParameters( *vol, proj, proj_index, mpi_part, magAnisotropy, magAnisotropyInv);	


      float defocusAngle = defocus.GetAstigmatismAngle(proj_index);
      float defocusMin;
      float defocusMax;
      GetDefocusMinMax(ray, proj_index, defocusMin, defocusMax);
      //float tiltAngle = (markers(MFI_TiltAngle, proj_index, 0) + config.AddTiltAngle) / 180.0f * (float)M_PI;
		
      if (mpi_part == 0)
	{
	  printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b BP Defocus: %-8d nm", (int)defocusMin);
	  fflush(stdout);
	}
      int2 pA, pB, pC, pD;

      proj.ComputeHitPoints(*vol, proj_index, pA, pB, pC, pD);

      cropKernel(proj_d, config.CutLength, config.DimLength, pA, pB, pC, pD);

      cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);
      hipfftSafeCall(hipfftExecR2C(handleR2C, (hipfftReal*)projSquare_d.GetDevicePtr(), (hipfftComplex*)fft_d.GetDevicePtr()));

      ctf(mRecParamCtf, fft_d, defocusMin, defocusMax, defocusAngle, true, (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), config.CTFBetaFac);

      hipfftSafeCall(hipfftExecC2R(handleC2R, (hipfftComplex*)fft_d.GetDevicePtr(), (hipfftReal*)projSquare_d.GetDevicePtr()));
      NppEmu.nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension() * proj.GetMaxDimension(),
				   (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);

      cropKernel(dist_d, config.CutLength, config.DimLength, pA, pB, pC, pD);

      //runtime = 
      bpKernel(param,x, y, config.Lambda / SIRTcount, config.OverSampling, 1.0f / (float)(config.OverSampling), filterImage_d, ray, ray + config.CTFSliceThickness / proj.GetPixelSize(proj_index),surfObj, bpTexObj);
    }
}
template void Reconstructor::BackProjectionCTF(Volume<unsigned short>* vol, int proj_index, float SIRTCount);
template void Reconstructor::BackProjectionCTF(Volume<float>* vol, int proj_index, float SIRTCount);

#ifdef SUBVOLREC_MODE
template<class TVol>
void Reconstructor::BackProjectionCTF(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<HipArray3D*>& vecArrays, CUsurfref surfref, int proj_index)
{
  size_t batchSize = subVolumes.size();
  //TODO
  //float3 volDim = vol->GetSubVolumeDimension(mpi_part);
  bpKernel.SetWorkSize(config.SizeSubVol, config.SizeSubVol, config.SizeSubVol);
  //float runtime;
  int x = proj.GetWidth();
  int y = proj.GetHeight();

  if (mpi_part == 0)
    printf("\n");

  float t_in, t_out;
  GetDefocusDistances(t_in, t_out, proj_index, vol);

  for (float ray = t_in; ray < t_out; ray += config.CTFSliceThickness / proj.GetPixelSize(proj_index))
    {
      float defocusAngle = defocus.GetAstigmatismAngle(proj_index);
      float defocusMin;
      float defocusMax;
      GetDefocusMinMax(ray, proj_index, defocusMin, defocusMax);
      float tiltAngle = (markers(MFI_TiltAngle, proj_index, 0) + config.AddTiltAngle) / 180.0f * (float)M_PI;

      if (mpi_part == 0)
	{
	  printf("\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b\b BP Defocus: %-8d nm", (int)defocusMin);
	  fflush(stdout);
	}

      //Do CTF correction:
      cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, true);
      hipfftSafeCall(hipfftExecR2C(handleR2C, (hipfftReal*)projSquare_d.GetDevicePtr(), (hipfftComplex*)fft_d.GetDevicePtr()));

      ctf(mRecParamCtf, fft_d, defocusMin, defocusMax, defocusAngle, true, (proj.GetMaxDimension() / 2 + 1) * sizeof(float2), config.CTFBetaFac);

      hipfftSafeCall(hipfftExecC2R(handleC2R, (hipfftComplex*)fft_d.GetDevicePtr(), (hipfftReal*)projSquare_d.GetDevicePtr()));
      NppEmu.nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension() * proj.GetMaxDimension(),
				   (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);


      for (size_t batch = 0; batch < batchSize; batch++)
	{
	  //bind surfref to correct array:
	  hipSafeCall(cuSurfRefSetArray(surfref, vecArrays[batch]->GetArray(), 0));

	  //set additional shifts:
	  proj.SetExtraShift(proj_index, vecExtraShifts[batch]);

	  float3 volDim = subVolumes[batch]->GetSubVolumeDimension(0);
	  bpKernel.SetWorkSize((int)volDim.x, (int)volDim.y, (int)volDim.z);

	  DeviceReconstructionConstantsCommon param = kernels::GetReconstructionParameters( *(subVolumes[batch]), proj, proj_index, 0, magAnisotropy, magAnisotropyInv);	
	  //SetConstantValues(bpKernel, *(subVolumes[batch]), proj, proj_index, 0, magAnisotropy, magAnisotropyInv);

	  //Most of the time, no volume should get hit...
	  //runtime = 
	  bpKernel(param, x, y, 1.0f, config.OverSampling, 1.0f / (float)(config.OverSampling), filterImage_d, ray, ray + config.CTFSliceThickness / proj.GetPixelSize(proj_index),surfObj, bpTexObj);
	}
    }
}
template void Reconstructor::BackProjectionCTF(Volume<unsigned short>* vol, vector<Volume<unsigned short>*>& subVolumes, vector<float2>& vecExtraShifts, vector<HipArray3D*>& vecArrays, CUsurfref surfref, int proj_index);
template void Reconstructor::BackProjectionCTF(Volume<float>* vol, vector<Volume<float>*>& subVolumes, vector<float2>& vecExtraShifts, vector<HipArray3D*>& vecArrays, CUsurfref surfref, int proj_index);
#endif

bool Reconstructor::ComputeFourFilter()
{
  //if (skipFilter)
  //{
  //	return false;
  //}

  //if (mpi_part != 0)
  //{
  //	return false;
  //}

  //float lp = config.fourFilterLP, hp = config.fourFilterHP, lps = config.fourFilterLPS, hps = config.fourFilterHPS;
  //int size = proj.GetMaxDimension();
  //float2* filter = new float2[size * size];
  //float2* fourFilter = new float2[(proj.GetMaxDimension() / 2 + 1) * proj.GetMaxDimension()];

  //if ((lp > size || lp < 0 || hp > size || hp < 0 || hps > size || hps < 0) && !skipFilter)
  //{
  //	//Filter parameters are not good!
  //	skipFilter = true;
  //	return false;
  //}

  //lp = lp - lps;
  //hp = hp + hps;


  //for (int y = 0; y < size; y++)
  //{
  //	for (int x = 0; x < size; x++)
  //	{
  //		float _x = -size / 2 + y;
  //		float _y = -size / 2 + x;

  //		float dist = (float)sqrtf(_x * _x + _y * _y);
  //		float fil = 0;
  //		//Low pass
  //		if (lp > 0)
  //		{
  //			if (dist <= lp) fil = 1;
  //		}
  //		else
  //		{
  //			if (dist <= size / 2 - 1) fil = 1;
  //		}

  //		//Gauss
  //		if (lps > 0)
  //		{
  //			float fil2;
  //			if (dist < lp) fil2 = 1;
  //			else fil2 = 0;

  //			fil2 = (-fil + 1.0f) * (float)expf(-((dist - lp) * (dist - lp) / (2 * lps * lps)));
  //			if (fil2 > 0.001f)
  //				fil = fil2;
  //		}

  //		if (lps > 0 && lp == 0 && hp == 0 && hps == 0)
  //			fil = (float)expf(-((dist - lp) * (dist - lp) / (2 * lps * lps)));

  //		if (hp > lp) return -1;

  //		if (hp > 0)
  //		{
  //			float fil2 = 0;
  //			if (dist >= hp) fil2 = 1;

  //			fil *= fil2;

  //			if (hps > 0)
  //			{
  //				float fil3 = 0;
  //				if (dist < hp) fil3 = 1;
  //				fil3 = (-fil2 + 1) * (float)expf(-((dist - hp) * (dist - hp) / (2 * hps * hps)));
  //				if (fil3 > 0.001f)
  //					fil = fil3;

  //			}
  //		}
  //		float2 filcplx;
  //		filcplx.x = fil;
  //		filcplx.y = 0;
  //		filter[y * size + x] = filcplx;
  //	}
  //}
  ////writeBMP("test.bmp", test, size, size);

  //cuFloatComplex* filterTemp = new cuFloatComplex[size * (size / 2 + 1)];

  ////Do FFT Shift in X direction (delete double coeffs)
  //for (int y = 0; y < size; y++)
  //{
  //	for (int x = size / 2; x <= size; x++)
  //	{
  //		int oldX = x;
  //		if (oldX == size) oldX = 0;
  //		int newX = x - size / 2;
  //		filterTemp[y * (size / 2 + 1) + newX] = filter[y * size + oldX];
  //	}
  //}
  ////Do FFT Shift in Y direction
  //for (int y = 0; y < size; y++)
  //{
  //	for (int x = 0; x < size / 2 + 1; x++)
  //	{
  //		int oldY = y + size / 2;
  //		if (oldY >= size) oldY -= size;
  //		fourFilter[y * (size / 2 + 1) + x] = filterTemp[oldY * (size / 2 + 1) + x];
  //	}
  //}
  //
  //ctf_d.CopyHostToDevice(fourFilter);
  //delete[] filterTemp;
  //delete[] filter;
  //delete[] fourFilter;
  return true;
}

void Reconstructor::PrepareProjection(void * img_h, int proj_index, float & meanValue, float & StdValue, int & BadPixels)
{
  if (mpi_part != 0)
    {
      return;
    }

  if (projSource->GetDataType() == FDT_SHORT)
    {
      //printf("SIGNED SHORT\n");
      hipSafeCall(hipMemcpyHtoD(realprojUS_d.GetDevicePtr(), img_h, proj.GetWidth() * proj.GetHeight() * sizeof(short)));
      NppEmu.nppiConvert_16s32f_C1R((Npp16s*)realprojUS_d.GetDevicePtr(), proj.GetWidth() * sizeof(short), (Npp32f*)realproj_d.GetDevicePtr(), realproj_d.GetPitch(), roiAll);
    }
  else if (projSource->GetDataType() == FDT_USHORT)
    {
      //printf("UNSIGNED SHORT\n");
      hipSafeCall(hipMemcpyHtoD(realprojUS_d.GetDevicePtr(), img_h, proj.GetWidth() * proj.GetHeight() * sizeof(short)));
      NppEmu.nppiConvert_16u32f_C1R((Npp16u*)realprojUS_d.GetDevicePtr(), proj.GetWidth() * sizeof(short), (Npp32f*)realproj_d.GetDevicePtr(), realproj_d.GetPitch(), roiAll);
    }
  else if (projSource->GetDataType() == FDT_INT)
    {
      realprojUS_d.CopyHostToDevice(img_h);
      NppEmu.nppiConvert_32s32f_C1R((Npp32s*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)realproj_d.GetDevicePtr(), realproj_d.GetPitch(), roiAll);
    }
  else if (projSource->GetDataType() == FDT_UINT)
    {
      realprojUS_d.CopyHostToDevice(img_h);
      NppEmu.nppiConvert_32u32f_C1R((Npp32u*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)realproj_d.GetDevicePtr(), realproj_d.GetPitch(), roiAll);
    }
  else if (projSource->GetDataType() == FDT_FLOAT)
    {
      realproj_d.CopyHostToDevice(img_h);
    }
  else
    {
      return;
    }

  projSquare_d.Memset(0);
  if (config.GetFileReadMode() == Configuration::Config::FRM_DM4)
    {
      cts(realproj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, true, false);
    }
  else
    {
      cts(realproj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
    }

  
  NppEmu.nppiMean_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr());

  double mean = 0;
  meanval.CopyDeviceToHost(&mean);
  meanValue = (float)mean;

  if (config.CorrectBadPixels)
    {
      NppEmu.nppiCompareC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), config.BadPixelValue * meanValue, 
				  (Npp8u*)badPixelMask_d.GetDevicePtr(), badPixelMask_d.GetPitch(), roiSquare, NPP_CMP_GREATER);
    }
  else
    {
      NppEmu.nppiSet_8u_C1R(0, (Npp8u*)badPixelMask_d.GetDevicePtr(), badPixelMask_d.GetPitch(), roiSquare);
    }

  NppEmu.nppiSum_8u_C1R((Npp8u*)badPixelMask_d.GetDevicePtr(), badPixelMask_d.GetPitch(), roiSquare, 
			(Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr());

  meanval.CopyDeviceToHost(&mean);
  BadPixels = (int)(mean / 255.0);

  NppEmu.nppiSet_32f_C1MR(meanValue, (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, 
			  (Npp8u*)badPixelMask_d.GetDevicePtr(), badPixelMask_d.GetPitch());

  float normVal = 1;

  //When doing WBP we compute mean and std on the RAW image before Fourier filter and WBP weighting
  if (config.WBP_NoSART)
    {
      NppEmu.nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), 1.0f, 
			      (Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), roiAll);

      NppEmu.nppiMean_StdDev_32f_C1R((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), roiAll, 
				     (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr(), (Npp64f*)stdval.GetDevicePtr());

      mean = 0;
      meanval.CopyDeviceToHost(&mean);
      double std_h = 0;
      stdval.CopyDeviceToHost(&std_h);
      StdValue = (float)std_h;
      float std_hf = StdValue;

      meanValue = (float)(mean);
      float mean_hf = meanValue;

      if (config.ProjectionNormalization == Configuration::Config::PNM_MEAN)
	{
	  std_hf = meanValue;
	}
      if (config.ProjectionNormalization == Configuration::Config::PNM_NONE)
	{
	  std_hf = 1;
	  mean_hf = 0;
	}
      if (config.DownWeightTiltsForWBP)
	{
	  //we devide here because we devide later using nppiDivC: add the end we multiply!
	  std_hf /= cosf(abs(markers(MarkerFileItem_enum::MFI_TiltAngle, proj_index, 0)) / 180.0f * M_PI);
	}

      NppEmu.nppiSubC_32f_C1R((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), mean_hf,
		       (Npp32f*)realproj_d.GetDevicePtr(), realproj_d.GetPitch(), roiAll);
      //nppSafeCall(nppiSubC_32f_C1R((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), mean_hf,
      //(Npp32f*)realproj_d.GetDevicePtr(), realproj_d.GetPitch(), roiAll));
      NppEmu.nppiDivC_32f_C1IR(-std_hf, (Npp32f*)realproj_d.GetDevicePtr(), realproj_d.GetPitch(), roiAll);
      //projSquare_d.Memset(0);
      cts(realproj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
    }

  if (!skipFilter || config.WBP_NoSART)
    {
      hipfftSafeCall(hipfftExecR2C(handleR2C, (hipfftReal*)projSquare_d.GetDevicePtr(), (hipfftComplex*)fft_d.GetDevicePtr()));

      if (config.WBP_NoSART)
	{
	  //Do WBP weighting
	  wbp(fft_d, roiFFT.width * sizeof(Npp32fc), proj.GetMaxDimension(), markers(MFI_RotationPsi, proj_index, 0), config.WBPFilter);
	}
      if (!skipFilter)
	{
	  float lp = config.fourFilterLP, hp = config.fourFilterHP, lps = config.fourFilterLPS, hps = config.fourFilterHPS;
	  int size = proj.GetMaxDimension();
			
	  fourFilterKernel(fft_d, roiFFT.width * sizeof(Npp32fc), size, lp, hp, lps, hps);
	  /*nppSafeCall(nppiMul_32fc_C1IR((Npp32fc*)ctf_d.GetDevicePtr(), ctf_d.GetPitch(),
	    (Npp32fc*)fft_d.GetDevicePtr(), roiFFT.width * sizeof(Npp32fc), roiFFT));*/
	}
      hipfftSafeCall(hipfftExecC2R(handleC2R, (hipfftComplex*)fft_d.GetDevicePtr(), (hipfftReal*)projSquare_d.GetDevicePtr()));

      normVal = proj.GetMaxDimension() * proj.GetMaxDimension();
    }

  //Normalize from FFT
  NppEmu.nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float), 
			       normVal, (Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), roiAll);


  //When doing SART we compute mean and std on the filtered image
  if (!config.WBP_NoSART)
    {
      NppiSize roiForMean;
      Npp32f* ptr = (Npp32f*)realprojUS_d.GetDevicePtr();
      //When doing SART the projection must be mean free, so compute mean on center of image only for IMOD aligned stacks...
      if (config.ProjectionNormalization == Configuration::Config::PNM_NONE) //IMOD aligned stack
	{
	  roiForMean.height = roiAll.height / 2;
	  roiForMean.width = roiAll.width / 2;

	  //Move start pointer:
	  char* ptrChar = (char*)ptr;
	  ptrChar += realprojUS_d.GetPitch() * (roiAll.height / 4); //Y
	  ptr = (float*)ptrChar;
	  ptr += roiAll.width / 4; //X

	}
      else
	{
	  roiForMean.height = roiAll.height;
	  roiForMean.width = roiAll.width;
	}

      NppEmu.nppiMean_StdDev_32f_C1R(ptr, realprojUS_d.GetPitch(), roiForMean,
				     (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr(), (Npp64f*)stdval.GetDevicePtr());

      mean = 0;
      meanval.CopyDeviceToHost(&mean);
      double std_h = 0;
      stdval.CopyDeviceToHost(&std_h);
      StdValue = (float)std_h;
      float std_hf = StdValue;

      meanValue = (float)(mean);
      float mean_hf = meanValue;

      if (config.ProjectionNormalization == Configuration::Config::PNM_MEAN)
	{
	  std_hf = meanValue;
	}
      if (config.ProjectionNormalization == Configuration::Config::PNM_NONE)
	{
	  std_hf = 1;
	  //meanValue = 0;
	}
		

      NppEmu.nppiSubC_32f_C1R((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), mean_hf,
		       (Npp32f*)realproj_d.GetDevicePtr(), realproj_d.GetPitch(), roiAll);
      NppEmu.nppiDivC_32f_C1IR(-std_hf, (Npp32f*)realproj_d.GetDevicePtr(), realproj_d.GetPitch(), roiAll);
      realproj_d.CopyDeviceToHost(img_h);
    }
  else
    {
      realprojUS_d.CopyDeviceToHost(img_h);
    }
}

template<class TVol>
void Reconstructor::Compare(Volume<TVol>* vol, char* originalImage, int index)
{
  if (mpi_part == 0)
    {
      float z_Direction = proj.GetNormalVector(index).z;
      float z_VolMinZ = vol->GetVolumeBBoxMin().z;
      float z_VolMaxZ = vol->GetVolumeBBoxMax().z;
      float volumeTraversalLength = fabs((DIST - z_VolMinZ) / z_Direction - (DIST - z_VolMaxZ) / z_Direction);

      realproj_d.CopyHostToDevice(originalImage);



      //nppiSet_32f_C1R(1.0f, (float*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);

      //float runtime = 
      compKernel(realproj_d, proj_d, dist_d, volumeTraversalLength, config.Crop, config.CropDim, config.ProjectionScaleFactor);
      /*proj_d.CopyDeviceToHost(MPIBuffer);
	writeBMP(string("Comp.bmp"), MPIBuffer, proj.GetWidth(), proj.GetHeight());*/
    }
}
template void Reconstructor::Compare(Volume<unsigned short>* vol, char* originalImage, int index);
template void Reconstructor::Compare(Volume<float>* vol, char* originalImage, int index);


template<class TVol>
void Reconstructor::ForwardProjection(Volume<TVol>* vol, HipTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync)
{
  if (config.CtfMode == Configuration::Config::CTFM_YES)
    {
      ForwardProjectionCTF(vol, texVol, index, volumeIsEmpty, noSync);
    }
  else
    {
      ForwardProjectionNoCTF(vol, texVol, index, volumeIsEmpty, noSync);
    }
}
template void Reconstructor::ForwardProjection(Volume<unsigned short>* vol, HipTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync);
template void Reconstructor::ForwardProjection(Volume<float>* vol, HipTextureObject3D& texVol, int index, bool volumeIsEmpty, bool noSync);


template<class TVol>
void Reconstructor::ForwardProjectionROI(Volume<TVol>* vol, HipTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync)
{
  if (config.CtfMode == Configuration::Config::CTFM_YES)
    {
      ForwardProjectionCTFROI(vol, texVol, index, volumeIsEmpty, roiMin, roiMax, noSync);
    }
  else
    {
      ForwardProjectionNoCTFROI(vol, texVol, index, volumeIsEmpty, roiMin, roiMax, noSync);
    }
}
template void Reconstructor::ForwardProjectionROI(Volume<unsigned short>* vol, HipTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);
template void Reconstructor::ForwardProjectionROI(Volume<float>* vol, HipTextureObject3D& texVol, int index, bool volumeIsEmpty, int2 roiMin, int2 roiMax, bool noSync);

template<typename TVol>
void Reconstructor::ForwardProjectionDistanceOnly(Volume<TVol>* vol, int index)
{

  DeviceReconstructionConstantsCommon param = kernels::GetReconstructionParameters( *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);
  //SetConstantValues(volTravLenKernel, *vol, proj, index, mpi_part, magAnisotropy, magAnisotropyInv);

  //float runtime = 
  volTravLenKernel(param, proj.GetWidth(), proj.GetHeight(), dist_d);
#ifdef USE_MPI
  if (mpi_part == 0)
    {
      for (int mpi = 1; mpi < mpi_size; mpi++)
	{
	  MPI_Recv(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, mpi, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
	  realprojUS_d.CopyHostToDevice(MPIBuffer);
	  NppEmu.nppiAdd_32f_C1IR((Npp32f*)realprojUS_d.GetDevicePtr(), realprojUS_d.GetPitch(), (Npp32f*)dist_d.GetDevicePtr(), dist_d.GetPitch(), roiAll);
	}
    }
  else
    {
      dist_d.CopyDeviceToHost(MPIBuffer);
      MPI_Send(MPIBuffer, proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
    }
#endif
}
template void Reconstructor::ForwardProjectionDistanceOnly(Volume<unsigned short>* vol, int index);
template void Reconstructor::ForwardProjectionDistanceOnly(Volume<float>* vol, int index);



template<class TVol>
void Reconstructor::BackProjection(Volume<TVol>* vol, int proj_index, float SIRTCount)
{
  if (config.CtfMode == Configuration::Config::CTFM_YES)
    {
      BackProjectionCTF(vol, proj_index, SIRTCount);
    }
  else
    {
      BackProjectionNoCTF(vol, proj_index, SIRTCount);
    }
}

template void Reconstructor::BackProjection(Volume<unsigned short>* vol, int proj_index, float SIRTCount);
template void Reconstructor::BackProjection(Volume<float>* vol, int proj_index, float SIRTCount);


#ifdef SUBVOLREC_MODE
template<class TVol>
void Reconstructor::BackProjection(Volume<TVol>* vol, vector<Volume<TVol>*>& subVolumes, vector<float2>& vecExtraShifts, vector<HipArray3D*>& vecArrays, CUsurfref surfref, int proj_index)
{
  if (config.CtfMode == Configuration::Config::CTFM_YES)
    {
      BackProjectionCTF(vol, subVolumes, vecExtraShifts, vecArrays, surfref, proj_index);
    }
  else
    {
      BackProjectionNoCTF(vol, subVolumes, vecExtraShifts, vecArrays, surfref, proj_index);
    }
}

template void Reconstructor::BackProjection(Volume<unsigned short>* vol, vector<Volume<unsigned short>*>& subVolumes, vector<float2>& vecExtraShifts, vector<HipArray3D*>& vecArrays, CUsurfref surfref, int proj_index);
template void Reconstructor::BackProjection(Volume<float>* vol, vector<Volume<float>*>& subVolumes, vector<float2>& vecExtraShifts, vector<HipArray3D*>& vecArrays, CUsurfref surfref, int proj_index);
#endif

template<class TVol>
void Reconstructor::OneSARTStep(Volume<TVol>* vol, Hip::HipTextureObject3D & texVol, int index, bool volumeIsEmpty, char * originalImage, float SIRTCount, float* MPIBuffer)
{
  ForwardProjection(vol, texVol, index, volumeIsEmpty);
  if (mpi_part == 0)
    {
      Compare(vol, originalImage, index);
      CopyProjectionToHost(MPIBuffer);
    }

  //spread the content to all other nodes:
  MPIBroadcast(&MPIBuffer, 1);
  CopyProjectionToDevice(MPIBuffer);
  BackProjection(vol, index, SIRTCount);
}
template void Reconstructor::OneSARTStep(Volume<unsigned short>* vol, Hip::HipTextureObject3D & texVol, int index, bool volumeIsEmpty, char * originalImage, float SIRTCount, float* MPIBuffer);
template void Reconstructor::OneSARTStep(Volume<float>* vol, Hip::HipTextureObject3D & texVol, int index, bool volumeIsEmpty, char * originalImage, float SIRTCount, float* MPIBuffer);

template<class TVol>
void Reconstructor::BackProjectionWithPriorWBPFilter(Volume<TVol>* vol, int proj_index, char* originalImage, float* MPIBuffer)
{
  if (mpi_part == 0)
    {
      realproj_d.CopyHostToDevice(originalImage);
      projSquare_d.Memset(0);
      cts(realproj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);

      hipfftSafeCall(hipfftExecR2C(handleR2C, (hipfftReal*)projSquare_d.GetDevicePtr(), (hipfftComplex*)fft_d.GetDevicePtr()));

      //Do WBP weighting
      wbp(fft_d, roiFFT.width * sizeof(Npp32fc), proj.GetMaxDimension(), markers(MFI_RotationPsi, proj_index, 0), WbpFilterMethod::FM_RAMP);
      hipfftSafeCall(hipfftExecC2R(handleC2R, (hipfftComplex*)fft_d.GetDevicePtr(), (hipfftReal*)projSquare_d.GetDevicePtr()));

      float normVal = proj.GetMaxDimension() * proj.GetMaxDimension();

      //Normalize from FFT
      NppEmu.nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float),
			      normVal, (Npp32f*)proj_d.GetDevicePtr(), proj_d.GetPitch(), roiAll);

      CopyProjectionToHost(MPIBuffer);
    }
  //spread the content to all other nodes:
  MPIBroadcast(&MPIBuffer, 1);
  CopyProjectionToDevice(MPIBuffer);
  BackProjection(vol, proj_index, 1);
}
template void Reconstructor::BackProjectionWithPriorWBPFilter(Volume<unsigned short>* vol, int proj_index, char* originalImage, float* MPIBuffer);
template void Reconstructor::BackProjectionWithPriorWBPFilter(Volume<float>* vol, int proj_index, char* originalImage, float* MPIBuffer);

template<class TVol>
void Reconstructor::RemoveProjectionFromVol(Volume<TVol>* vol, int proj_index, char* originalImage, float* MPIBuffer)
{
  if (mpi_part == 0)
    {
      realproj_d.CopyHostToDevice(originalImage);
      projSquare_d.Memset(0);
      cts(realproj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);

      hipfftSafeCall(hipfftExecR2C(handleR2C, (hipfftReal*)projSquare_d.GetDevicePtr(), (hipfftComplex*)fft_d.GetDevicePtr()));

      //Do WBP weighting
      wbp(fft_d, roiFFT.width * sizeof(Npp32fc), proj.GetMaxDimension(), markers(MFI_RotationPsi, proj_index, 0), WbpFilterMethod::FM_RAMP);
      hipfftSafeCall(hipfftExecC2R(handleC2R, (hipfftComplex*)fft_d.GetDevicePtr(), (hipfftReal*)projSquare_d.GetDevicePtr()));

      float normVal = proj.GetMaxDimension() * proj.GetMaxDimension();
      //negate to remove from volume:
      normVal *= -1;

      //Normalize from FFT
      NppEmu.nppiDivC_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr() + squarePointerShift, proj.GetMaxDimension() * sizeof(float),
			      normVal, (Npp32f*)proj_d.GetDevicePtr(), proj_d.GetPitch(), roiAll);

      CopyProjectionToHost(MPIBuffer);
    }
  //spread the content to all other nodes:
  MPIBroadcast(&MPIBuffer, 1);
  CopyProjectionToDevice(MPIBuffer);
  BackProjection(vol, proj_index, 1);
}
template void Reconstructor::RemoveProjectionFromVol(Volume<unsigned short>* vol, int proj_index, char* originalImage, float* MPIBuffer);
template void Reconstructor::RemoveProjectionFromVol(Volume<float>* vol, int proj_index, char* originalImage, float* MPIBuffer);

void Reconstructor::ResetProjectionsDevice()
{
  proj_d.Memset(0);
  dist_d.Memset(0);
}

void Reconstructor::CopyProjectionToHost(float * buffer)
{
  proj_d.CopyDeviceToHost(buffer);
}

void Reconstructor::CopyDistanceImageToHost(float * buffer)
{
  dist_d.CopyDeviceToHost(buffer);
}

void Reconstructor::CopyRealProjectionToHost(float * buffer)
{
  realproj_d.CopyDeviceToHost(buffer);
}

void Reconstructor::CopyProjectionToDevice(float * buffer)
{
  proj_d.CopyHostToDevice(buffer);
}

void Reconstructor::CopyDistanceImageToDevice(float * buffer)
{
  dist_d.CopyHostToDevice(buffer);
}

void Reconstructor::CopyRealProjectionToDevice(float * buffer)
{
  realproj_d.CopyHostToDevice(buffer);
}

void Reconstructor::MPIBroadcast(float ** buffers, int bufferCount)
{
#ifdef USE_MPI
  for (int i = 0; i < bufferCount; i++)
    {
      MPI_Bcast(buffers[i], proj.GetWidth() * proj.GetHeight(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    }
#endif
}

#ifdef REFINE_MODE
void Reconstructor::CopyProjectionToSubVolumeProjection()
{
  if (mpi_part == 0)
    {
      projSubVols_d.CopyDeviceToDevice(proj_d);
    }
}
#endif


void Reconstructor::ConvertVolumeFP16(Volume<unsigned short>* vol, float * slice, int z)
{
  if (volTemp_d.GetWidth() != config.RecDimensions.x ||
      volTemp_d.GetHeight() != config.RecDimensions.y)
    {
      volTemp_d.Alloc(config.RecDimensions.x * sizeof(float), config.RecDimensions.y, sizeof(float));
    }
  DeviceReconstructionConstantsCommon param = kernels::GetReconstructionParameters( *vol, proj, 0, 0, magAnisotropy, magAnisotropyInv);
  convVolKernel(param, volTemp_d, z,surfObj);
  volTemp_d.CopyDeviceToHost(slice);
}
//#define WRITEDEBUG 1

#ifdef REFINE_MODE
float2 Reconstructor::GetDisplacement()
{
  float2 shift;
  shift.x = 0;
  shift.y = 0;
	
  if (mpi_part == 0)
    {
#ifdef WRITEDEBUG
      float* test = new float[proj.GetMaxDimension() * proj.GetMaxDimension()];
#endif
      /*float* test = new float[proj.GetWidth() * proj.GetHeight()];
	proj_d.CopyDeviceToHost(test);
		
	double summe = 0;
	for (size_t i = 0; i < proj.GetWidth() * proj.GetHeight(); i++)
	{
	summe += test[i];
	}
	emwrite("testCTF2.em", test, proj.GetWidth(), proj.GetHeight());
	delete[] test;*/

      //proj_d contains the original Projection minus the proj(reconstructionWithoutSubVols)
      cts(proj_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);
#ifdef WRITEDEBUG
      projSquare_d.CopyDeviceToHost(test);
      emwrite("projection3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif

      NppEmu.nppiMean_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
			      (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr());
      double MeanA = 0;
      meanval.CopyDeviceToHost(&MeanA, sizeof(double));
      NppEmu.nppiSubC_32f_C1IR((float)(MeanA), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare);

      NppEmu.nppiSqr_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), (Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare);

      NppEmu.nppiSum_32f_C1R((Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
				  (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr());

      double SumA = 0;
      meanval.CopyDeviceToHost(&SumA, sizeof(double));


      hipfftSafeCall(hipfftExecR2C(handleR2C, (hipfftReal*)projSquare_d.GetDevicePtr(), (hipfftComplex*)fft_d.GetDevicePtr()));
      //fourFilterKernel(fft_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(hipComplex), proj.GetMaxDimension(), config.fourFilterLP, 12, config.fourFilterLPS, 4);

      //missuse ctf_d as second fft variable
      //projSubVols_d contains the projection of the model
      cts(projSubVols_d, proj.GetMaxDimension(), projSquare_d, squareBorderSizeX, squareBorderSizeY, false, false);

      NppEmu.nppiMean_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
			      (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr());
      double MeanB = 0;
      meanval.CopyDeviceToHost(&MeanB, sizeof(double));
      NppEmu.nppiSubC_32f_C1IR((float)(MeanB), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare);
      NppEmu.nppiSqr_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), (Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare);

      NppEmu.nppiSum_32f_C1R((Npp32f*)projSquare2_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare,
			     (Npp8u*)meanbuffer.GetDevicePtr(), (Npp64f*)meanval.GetDevicePtr());

      double SumB = 0;
      meanval.CopyDeviceToHost(&SumB, sizeof(double));

#ifdef WRITEDEBUG
      projSquare_d.CopyDeviceToHost(test);
      emwrite("realprojection3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif
      hipfftSafeCall(hipfftExecR2C(handleR2C, (hipfftReal*)projSquare_d.GetDevicePtr(), (hipfftComplex*)ctf_d.GetDevicePtr()));
      //fourFilterKernel(ctf_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(hipComplex), proj.GetMaxDimension(), 150, 2, 20, 1);

      conjKernel(fft_d, ctf_d, (proj.GetMaxDimension() / 2 + 1) * sizeof(hipComplex), proj.GetMaxDimension());
		
      hipfftSafeCall(hipfftExecC2R(handleC2R, (hipfftComplex*)fft_d.GetDevicePtr(), (hipfftReal*)projSquare_d.GetDevicePtr()));
#ifdef WRITEDEBUG
      projSquare_d.CopyDeviceToHost(test);
      emwrite("cc3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif
		
      int maxShift = 10;
#ifdef REFINE_MODE
      maxShift = config.MaxShift;
#endif
      NppEmu.nppiDivC_32f_C1IR((float)(proj.GetMaxDimension() * proj.GetMaxDimension() * sqrt(SumA * SumB)), (Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare);

      //printf("Divs: %f %f\n", (float)SumA, (float)SumB);

      NppiSize ccSize;
      ccSize.width = roiCC1.width;
      ccSize.height = roiCC1.height;
      NppEmu.nppiCopy_32f_C1R(
			      (float*)((char*)projSquare_d.GetDevicePtr() + roiCC1.y * proj.GetMaxDimension() * sizeof(float) + roiCC1.x * sizeof(float)),
			      proj.GetMaxDimension() * sizeof(float),
			      (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC1.y * ccMap_d.GetPitch() + roiDestCC1.x * sizeof(float)),
			      ccMap_d.GetPitch(), ccSize);

      NppEmu.nppiCopy_32f_C1R(
			      (float*)((char*)projSquare_d.GetDevicePtr() + roiCC2.y * proj.GetMaxDimension() * sizeof(float) + roiCC2.x * sizeof(float)),
			      proj.GetMaxDimension() * sizeof(float),
			      (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC2.y * ccMap_d.GetPitch() + roiDestCC2.x * sizeof(float)),
			      ccMap_d.GetPitch(), ccSize);

      NppEmu.nppiCopy_32f_C1R(
			      (float*)((char*)projSquare_d.GetDevicePtr() + roiCC3.y * proj.GetMaxDimension() * sizeof(float) + roiCC3.x * sizeof(float)),
			      proj.GetMaxDimension() * sizeof(float),
			      (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC3.y * ccMap_d.GetPitch() + roiDestCC3.x * sizeof(float)),
			      ccMap_d.GetPitch(), ccSize);

      NppEmu.nppiCopy_32f_C1R(
			      (float*)((char*)projSquare_d.GetDevicePtr() + roiCC4.y * proj.GetMaxDimension() * sizeof(float) + roiCC4.x * sizeof(float)),
			      proj.GetMaxDimension() * sizeof(float),
			      (float*)((char*)ccMap_d.GetDevicePtr() + roiDestCC4.y * ccMap_d.GetPitch() + roiDestCC4.x * sizeof(float)),
			      ccMap_d.GetPitch(), ccSize);

      ccMap_d.CopyDeviceToHost(ccMap);


      maxShiftKernel(projSquare_d, proj.GetMaxDimension() * sizeof(float), proj.GetMaxDimension(), maxShift);

      NppEmu.nppiMaxIndx_32f_C1R((Npp32f*)projSquare_d.GetDevicePtr(), proj.GetMaxDimension() * sizeof(float), roiSquare, 
				 (Npp8u*)meanbuffer.GetDevicePtr(), (Npp32f*)meanval.GetDevicePtr(), (int*)stdval.GetDevicePtr(), 
				 (int*)(stdval.GetDevicePtr() + sizeof(int)));

#ifdef WRITEDEBUG
      projSquare_d.CopyDeviceToHost(test);
      emwrite("shiftTest3F.em", test, proj.GetMaxDimension(), proj.GetMaxDimension());/**/
#endif

      int maxPixels[2];
      stdval.CopyDeviceToHost(maxPixels, 2 * sizeof(int));

      float maxVal;
      meanval.CopyDeviceToHost(&maxVal, sizeof(float));
      printf("\nMaxVal: %f", maxVal);

      //Get shift:
      shift.x = maxPixels[0];
      shift.y = maxPixels[1];

      if (shift.x > proj.GetMaxDimension() / 2)
	{
	  shift.x -= proj.GetMaxDimension();
	}
		
      if (shift.y > proj.GetMaxDimension() / 2)
	{
	  shift.y -= proj.GetMaxDimension();
	}

      if (maxVal <= 0)
	{
	  //something went wrong, no shift found
	  shift.x = -1000;
	  shift.y = -1000;
	}
    }
  return shift;
}

void Reconstructor::rotVol(Hip::HipDeviceVariable & vol, float phi, float psi, float theta)
{
  rotKernel(vol, phi, psi, theta);
}

void Reconstructor::setRotVolData(float * data)
{
  rotKernel.SetData(data);
}
float * Reconstructor::GetCCMap()
{
  return ccMap;
}
#endif

