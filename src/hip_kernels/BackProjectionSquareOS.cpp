/**********************************************
*
* CUDA SART FRAMEWORK
* 2009,2010 Michael Kunz, Lukas Marsalek
* 
*
* BackProjectionSquareOS.cu
* CVR back projection kernel with squared 
* oversampling pattern
*
**********************************************/
#ifndef BACKPROJECTIONSQUAREOS_CU
#define BACKPROJECTIONSQUAREOS_CU

#define _SIZE_T_DEFINED
#ifndef __HIPCC__
#define __HIPCC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif

#include <hip/hip_runtime.h>

#include <hip/hip_fp16.h>

#include "DeviceVariables.h"
#include "../hip/HipMissedStuff.h"


//#define CONST_LENGTH_MODE
#define PRECISE_LENGTH_MODE


#define SM20 1
#if __CUDA_ARCH__ >= 200
//#warning compiling for SM20
#else
#if __CUDA_ARCH__ >= 130
//#warning compiling for SM13
#endif
#endif


//texture<float, 2, hipReadModeElementType> tex;


//#define SPLINES
#ifdef SPLINES


#define Pole (sqrt(3.0f)-2.0f)  //pole for cubic b-spline
 
//--------------------------------------------------------------------------
// Local GPU device procedures
//--------------------------------------------------------------------------
__host__ __device__ float InitialCausalCoefficient(
	float* c,			// coefficients
	uint DataLength,	// number of coefficients
	int step)			// element interleave in bytes
{
	const uint Horizon = min(12, DataLength);

	// this initialization corresponds to clamping boundaries
	// accelerated loop
	float zn = Pole;
	float Sum = *c;
	for (uint n = 0; n < Horizon; n++) {
		Sum += zn * *c;
		zn *= Pole;
		c = (float*)((uchar*)c + step);
	}
	return(Sum);
}

__host__ __device__ float InitialAntiCausalCoefficient(
	float* c,			// last coefficient
	uint DataLength,	// number of samples or coefficients
	int step)			// element interleave in bytes
{
	// this initialization corresponds to clamping boundaries
	return((Pole / (Pole - 1.0f)) * *c);
}

__host__ __device__ void ConvertToInterpolationCoefficients(
	float* coeffs,		// input samples --> output coefficients
	uint DataLength,	// number of samples or coefficients
	int step)			// element interleave in bytes
{
	// compute the overall gain
	const float Lambda = (1.0f - Pole) * (1.0f - 1.0f / Pole);

	// causal initialization
	float* c = coeffs;
	float previous_c;  //cache the previously calculated c rather than look it up again (faster!)
	*c = previous_c = Lambda * InitialCausalCoefficient(c, DataLength, step);
	// causal recursion
	for (uint n = 1; n < DataLength; n++) {
		c = (float*)((uchar*)c + step);
		*c = previous_c = Lambda * *c + Pole * previous_c;
	}
	// anticausal initialization
	*c = previous_c = InitialAntiCausalCoefficient(c, DataLength, step);
	// anticausal recursion
	for (int n = DataLength - 2; 0 <= n; n--) {
		c = (float*)((uchar*)c - step);
		*c = previous_c = Pole * (previous_c - *c);
	}
}


extern "C"
//__global__ void SamplesToCoefficients2DX(
//	float* image,		// in-place processing
//	uint pitch,			// width in bytes
//	uint width,			// width of the image
//	uint height)		// height of the image
__global__ void SamplesToCoefficients2DX( DevParamSamplesToCoefficients recParam )
{
  float* &image = recParam.image;	
  uint &pitch = recParam.pitch;	
  uint &width = recParam.width;	
  uint &height = recParam.height;	

  // process lines in x-direction
  const uint y = blockIdx.x * blockDim.x + threadIdx.x;
  float* line = (float*)((uchar*)image + y * pitch);  //direct access
  
  ConvertToInterpolationCoefficients(line, width, sizeof(float));
}

extern "C"
//__global__ void SamplesToCoefficients2DY(
//	float* image,		// in-place processing
//	uint pitch,			// width in bytes
//	uint width,			// width of the image
//	uint height)		// height of the image
__global__ void SamplesToCoefficients2DY( DevParamSamplesToCoefficients recParam )
{
  float* &image = recParam.image;	
  uint &pitch = recParam.pitch;	
  uint &width = recParam.width;	
  uint &height = recParam.height;	

  // process lines in x-direction
  const uint x = blockIdx.x * blockDim.x + threadIdx.x;
  float* line = image + x;  //direct access
  
  ConvertToInterpolationCoefficients(line, height, pitch);
}



// Cubic B-spline function
// The 3rd order Maximal Order and Minimum Support function, that it is maximally differentiable.
inline __host__ __device__ float bspline(float t)
{
	t = fabs(t);
	const float a = 2.0f - t;

	if (t < 1.0f) return 2.0f/3.0f - 0.5f*t*t*a;
	else if (t < 2.0f) return a*a*a / 6.0f;
	else return 0.0f;
}


//! Bicubic interpolated texture lookup, using unnormalized coordinates.
//! Straight forward implementation, using 16 nearest neighbour lookups.
//! @param tex  2D texture
//! @param x  unnormalized x texture coordinate
//! @param y  unnormalized y texture coordinate
__device__ float cubicTex2DSimple(texture<float, 2, hipReadModeElementType> _tex, float x, float y)
{
	// transform the coordinate from [0,extent] to [-0.5, extent-0.5]
	const float2 coord_grid = make_float2(x - 0.5f, y - 0.5f);
	float2 index;
	index.x = floor(coord_grid.x);
	index.y = floor(coord_grid.y);
	float2 fraction;
	fraction.x = coord_grid.x - index.x;
	fraction.y = coord_grid.y - index.y;
	index.x += 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]
	index.y += 0.5f;  //move from [-0.5, extent-0.5] to [0, extent]

	float result = 0.0f;
	for (float y=-1; y < 2.5f; y++)
	{
		float bsplineY = bspline(y-fraction.y);
		float v = index.y + y;
		for (float x=-1; x < 2.5f; x++)
		{
			float bsplineXY = bspline(x-fraction.x) * bsplineY;
			float u = index.x + x;
			result += bsplineXY * tex2D(_tex, u, v);
		}
	}
	return result;
}























#endif



// transform vector by matrix
__device__
void MatrixVector3Mul(float4x4 M, float3* v)
{
	float3 erg;
	erg.x = M.m[0].x * v->x + M.m[0].y * v->y + M.m[0].z * v->z + 1.f * M.m[0].w;
	erg.y = M.m[1].x * v->x + M.m[1].y * v->y + M.m[1].z * v->z + 1.f * M.m[1].w;
	erg.z = M.m[2].x * v->x + M.m[2].y * v->y + M.m[2].z * v->z + 1.f * M.m[2].w;
	*v = erg;
}



#ifdef SM20
//surface<void, cudaSurfaceType3D> surfref;
//surfaceReference surfref;
#endif


extern "C"
__global__ 
//void backProjection(DeviceReconstructionConstantsCommon c, int proj_x, int proj_y, float lambda, int maxOverSample, float maxOverSampleInv, float* img, int stride, float distMin, float distMax, /*CUsurfObject*/ hipSurfaceObject_t surfObj)
void backProjection( DevParamBackProjection recParam )
{
  // SG!! TODO: move all initialization code to host

  DeviceReconstructionConstantsCommon &c = recParam.common; 
  int &proj_x = recParam.proj_x; 
  int &proj_y = recParam.proj_y; 
  float &lambda = recParam.lambda; 
  float &maxOverSampleInv = recParam.maxOverSampleInv; 
  float &distMin = recParam.distMin; 
  float &distMax = recParam.distMax; 
  hipSurfaceObject_t &surfObj = recParam.surfObj;
  hipTextureObject_t &texObj = recParam.texObj; 
  float3 &tGradient = c.tGradient; // == dt/dx dt/dy dt/dz

  const float4 float4Zero = make_float4( 0.f, 0.f, 0.f, 0.f );

  // one thread processes a block of 4 voxels, neighbouring in x
  // SG: is it really faster? TODO: check

  const unsigned int xBlock = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;
  const unsigned int x = 4*xBlock;

  if (xBlock >= c.volumeDim_x_quarter || y >= c.volumeDim.y || z >= c.volumeDim.z) return;

  // intensity values for 4 voxels of the 3d image. They need to be updated with the current projection.
  float4 voxelBlock; 

  //read voxel values from global device memory
#ifdef SM20
  const uint xBytes = x*sizeof(float);
  surf3Dread(&voxelBlock.x, surfObj, xBytes                  , y, z);
  surf3Dread(&voxelBlock.y, surfObj, xBytes +   sizeof(float), y, z);
  surf3Dread(&voxelBlock.z, surfObj, xBytes + 2*sizeof(float), y, z);
  surf3Dread(&voxelBlock.w, surfObj, xBytes + 3*sizeof(float), y, z);
#else
  voxelBlock = vol2[voxelIndex];
#endif

  // ray direction == normal to the projection plane,  +-sign is not important
  // t coordinate will be a coordinate along the ray 
 
  const float3 &ray = c.projNorm;

  // bounding box for the voxel block:

  // lower vertex of the box
  float3 boxMin = c.bBoxMin + make_float3(x,y,z)*c.voxelSize;

  // box half-edges.  SG!!! there was a bug here:  box X-size was 5, changed to 4
  float3 boxHalfEdges = make_float3(2.0f,0.5f,0.5f)*c.voxelSize;

  // center of the box
  float3 boxCent = boxMin + boxHalfEdges;      

  // distance between the detector and the boxCent along the ray  
  float t0 = ray.x*(boxCent.x-c.detektor.x) + ray.y*(boxCent.y-c.detektor.y) + ray.z*(boxCent.z-c.detektor.z);
  
  float t0abs = fabs(t0);
  if( t0abs<distMin || t0abs>distMax ) return; // cut on a distance to the detector

 
  // find the size of the voxel box projection on the detector:
  
  // UV are raw "integer" pixel coordinates, not scaled with the pitch

  const float4x4 &M = c.DetectorMatrix; // global XYZ to detector UV

  // box half edges projected to the UV detector plane in UV pixel coordinates
  
  float2 dx = make_float2( M.m[0].x*boxHalfEdges.x, M.m[1].x*boxHalfEdges.x ); 
  float2 dy = make_float2( M.m[0].y*boxHalfEdges.y, M.m[1].y*boxHalfEdges.y ); 
  float2 dz = make_float2( M.m[0].z*boxHalfEdges.z, M.m[1].z*boxHalfEdges.z ); 

  // half-edges of the bounding box at UV detector plane
  // consider only projections of 4 voxel vertices, the other 4 only differ in sign
  float2 max1 = fmaxf( fabs( dx+dy+dz), fabs( dx-dy-dz) );
  float2 max2 = fmaxf( fabs( dx-dy+dz), fabs( dx+dy-dz) );  
  float2 detHalfEdges  = fmaxf( max1, max2 );

  // box center projected to the detector plane in UV pixel coordinates
  float2 detCent; 
  {
    float3 &c = boxCent;
    detCent.x = M.m[0].x*c.x + M.m[0].y*c.y + M.m[0].z*c.z + M.m[0].w;
    detCent.y = M.m[1].x*c.x + M.m[1].y*c.y + M.m[1].z*c.z + M.m[1].w;
  }  

  float uMin = (detCent.x - detHalfEdges.x);
  float vMin = (detCent.y - detHalfEdges.y);
  float uMax = (detCent.x + detHalfEdges.x);
  float vMax = (detCent.y + detHalfEdges.y);

  //clamp values
  uMin = fminf(fmaxf(uMin, 0), proj_x - 1);
  uMax = fminf(fmaxf(uMax, 0), proj_x - 1);
  vMin = fminf(fmaxf(vMin, 0), proj_y - 1);
  vMax = fminf(fmaxf(vMax, 0), proj_y - 1);

  // u,v: raw pixel coordinates, w/o  pitch 
  
  float dU = maxOverSampleInv;
  float dV = dU;
  float u0 = uMin + 0.5f*dU; // SG: why (+0.5f*dU) ? The center of the pixel?
  float v0 = vMin + 0.5f*dV;

  // Anisotropy: some correction applied to pixel u,v to get values from the projection image
  //SG: TODO: can it be applied directly to the detector matrix?

  float3x3& A = c.magAnisoInv;

  float2 a0 = make_float2( A.m[0].x*u0 + A.m[0].y*v0 + A.m[0].z,
			   A.m[1].x*u0 + A.m[1].y*v0 + A.m[1].z  );
  
  float2 daU = make_float2( A.m[0].x*dU, A.m[1].x*dU ); // how the anis. changes with du
  float2 daV = make_float2( A.m[0].y*dV, A.m[1].y*dV );

  // Get t of the ray when it crosses 3 planes: (yz at x==xVox1); (xz at y==yVox1); (xy at z==zVox1)

  // 'source' is some XYZ point on a ray which comes from the first pixel at the detector.
  // the point is geometrically close to the voxels. It defines t==0. 
  // the point will follow the pixel's move in uv.

  float3 source = c.detektor + u0*c.uPitch + v0*c.vPitch + ray*t0;
  
  // get 6 values of t where the ray crosses the sides of the first voxel
  float3 tSideCross1 = (boxMin - source) * tGradient;
  float3 tSideCross2 = (boxMin + c.voxelSize - source) * tGradient;

  // reshuffle t values to define entry and exit t for +-x,+-y,+-z sides of the first voxel.
  // for each coordinate entry and exit sides stays the same  when the uv pixel moves

  float3 tIn0  = fminf( tSideCross1, tSideCross2 );
  float3 tOut0 = fmaxf( tSideCross1, tSideCross2 );

  // how the intersection t changes when U / V pixel coordinate increases by dU / dV
  float3 dtU = -dU*c.uPitch * tGradient;
  float3 dtV = -dV*c.vPitch * tGradient;

  // intersection t for +-x sides for 2,3,4-th voxel with respect to the 1-st voxel
  float dtVoxel2X = c.voxelSize.x * tGradient.x;
  float dtVoxel3X = dtVoxel2X + dtVoxel2X;
  float dtVoxel4X = dtVoxel3X + dtVoxel2X;
  
  float4 voxelD = float4Zero;      //Correction term per voxel in voxelBlock
  float4 distanceD = float4Zero;   //summed up distance per voxel in voxelBlock

  //Loop over detected pixels and shoot rays back
  for( float v = v0; v <= vMax; v+=dV, a0+=daV, tIn0+=dtV, tOut0+=dtV ){
    float2 a1 = a0;
    float3 tIn1 = tIn0;
    float3 tOut1 = tOut0;
    for( float u = u0; u <= uMax; u+=dU, a1+=daU, tIn1+=dtU, tOut1+=dtU ){
      
      float intensity = tex2D<float>(texObj, a1.x, a1.y);     
      
      // entry t of the voxel 1 = maximal t_in  over x,y,z sides
      // exit  t of the voxel 1 = minimal t_out over x,y,z sides
      
      // first partical calculation for y and z: same for all the voxels
      float tyz_in  = fmaxf(tIn1.y,  tIn1.z  ); 
      float tyz_out = fminf(tOut1.y, tOut1.z ); 
      
      // now final with x: different for voxels 1,2,3,4
      // right away calculate (tout - tin) 
      float4 dt = make_float4 
	( fminf( tOut1.x            , tyz_out) - fmaxf( tIn1.x            , tyz_in ), 
	  fminf( tOut1.x + dtVoxel2X, tyz_out) - fmaxf( tIn1.x + dtVoxel2X, tyz_in ), 
	  fminf( tOut1.x + dtVoxel3X, tyz_out) - fmaxf( tIn1.x + dtVoxel3X, tyz_in ), 
	  fminf( tOut1.x + dtVoxel4X, tyz_out) - fmaxf( tIn1.x + dtVoxel4X, tyz_in )  );
      
      dt = fmaxf(dt, float4Zero); // when dt<0, the ray does not hit the voxel
      
#ifdef CONST_LENGTH_MODE
      if( dt.x > 0.0f ) dt.x = 1.0f;
      if( dt.y > 0.0f ) dt.y = 1.0f;
      if( dt.z > 0.0f ) dt.z = 1.0f;
      if( dt.w > 0.0f ) dt.w = 1.0f;
#endif            
      distanceD += dt;
      voxelD += intensity*dt;
    }//for loop v-pixel
  }//for loop u-pixel
 
  //Only positive distance values are allowed  

  //Apply correction term to voxel

  if (distanceD.x > 0.0f) voxelBlock.x += lambda * (voxelD.x / distanceD.x);
  if (distanceD.y > 0.0f) voxelBlock.y += lambda * (voxelD.y / distanceD.y);
  if (distanceD.z > 0.0f) voxelBlock.z += lambda * (voxelD.z / distanceD.z);
  if (distanceD.w > 0.0f) voxelBlock.w += lambda * (voxelD.w / distanceD.w);

  //store values in global memory
  //__syncthreads();
  //vol2[voxelIndex] = temp2;
	
#ifdef SM20
  surf3Dwrite(voxelBlock.x, surfObj, xBytes                  , y, z);
  surf3Dwrite(voxelBlock.y, surfObj, xBytes +   sizeof(float), y, z);
  surf3Dwrite(voxelBlock.z, surfObj, xBytes + 2*sizeof(float), y, z);
  surf3Dwrite(voxelBlock.w, surfObj, xBytes + 3*sizeof(float), y, z);
#else
  vol2[voxelIndex] = voxelBlock;
#endif
}


extern "C"
__global__ 
//void backProjectionFP16(DeviceReconstructionConstantsCommon c, int proj_x, int proj_y, float lambda, int maxOverSample, float maxOverSampleInv, float* img, int stride, float distMin, float distMax, hipSurfaceObject_t surfObj)
void backProjectionFP16( DevParamBackProjection recParam )
{

  HIP_DYNAMIC_SHARED(unsigned char, sBuffer);

  DeviceReconstructionConstantsCommon &c = recParam.common; 
  int &proj_x = recParam.proj_x; 
  int &proj_y = recParam.proj_y; 
  float &lambda = recParam.lambda; 
  //int &maxOverSample = recParam.maxOverSample; 
  float &maxOverSampleInv = recParam.maxOverSampleInv; 
  //float* &img = recParam.img; 
  //int &stride = recParam.stride; 
  float &distMin = recParam.distMin; 
  float &distMax = recParam.distMax; 
  hipSurfaceObject_t &surfObj = recParam.surfObj;
  hipTextureObject_t &texObj = recParam.texObj; 

  float3 ray;
  float3 borderMin;
  float3 borderMax;
  float3 hitPoint;
  float3 source;

  //curandState state;


  int4 pixelBorders; //--> x = x.min; y = x.max; z = y.min; v = y.max   	
	
  // index to access shared memory, e.g. thread linear address in a block
  const unsigned int index2 = (threadIdx.z * blockDim.x * blockDim.y) + (threadIdx.y * blockDim.x) + threadIdx.x;	

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= c.volumeDim_x_quarter || y >= c.volumeDim.y || z >= c.volumeDim.z) return;

  //summed up distance per voxel in voxelBlock in shared memory
  volatile float4* distanceD = (float4*)(sBuffer);
	
  //Correction term per voxel in shared memory
  volatile float4* voxelD = distanceD + blockDim.x * blockDim.y * blockDim.z;
  //volatile float4* voxelD = (float4*)(sBuffer);
	
	
  float4 voxelBlock;

  //float3 detectorNormal;
  float3 MC_bBoxMin;
  float3 MC_bBoxMax;
  float t;
	
  float t_in, t_out;
  float3 tEntry;
  float3 tExit;
  float3 tmin, tmax;
  float pixel_y, pixel_x;	


#ifdef SM20
  /*
    surf3Dread(&tempfp16, surfref, x * 2 * 4 + 0, y, z);
    voxelBlock.x = __half2float(tempfp16);
    surf3Dread(&tempfp16, surfref, x * 2 * 4 + 2, y, z);
    voxelBlock.y = __half2float(tempfp16);
    surf3Dread(&tempfp16, surfref, x * 2 * 4 + 4, y, z);
    voxelBlock.z = __half2float(tempfp16);
    surf3Dread(&tempfp16, surfref, x * 2 * 4 + 6, y, z);
    voxelBlock.w = __half2float(tempfp16);
  */
  {
    unsigned short tempfp16;
    surf3Dread(&tempfp16, surfObj, x * 2 * 4 + 0, y, z);
    voxelBlock.x = __half2float(*(half*)& tempfp16);
    surf3Dread(&tempfp16, surfObj, x * 2 * 4 + 2, y, z);
    voxelBlock.y = __half2float(*(half*)& tempfp16);
    surf3Dread(&tempfp16, surfObj, x * 2 * 4 + 4, y, z);
    voxelBlock.z = __half2float(*(half*)& tempfp16);
    surf3Dread(&tempfp16, surfObj, x * 2 * 4 + 6, y, z);
    voxelBlock.w = __half2float(*(half*)& tempfp16);
  }
#else
  voxelBlock = vol2[voxelIndex];
#endif


  //adopt x coordinate to single voxels, not voxelBlocks
  x = x * 4;

  //MacroCell bounding box:
  MC_bBoxMin.x = c.bBoxMin.x + (x) * c.voxelSize.x;
  MC_bBoxMin.y = c.bBoxMin.y + (y) * c.voxelSize.y;
  MC_bBoxMin.z = c.bBoxMin.z + (z) * c.voxelSize.z;
  MC_bBoxMax.x = c.bBoxMin.x + ( x + 5) * c.voxelSize.x;
  MC_bBoxMax.y = c.bBoxMin.y + ( y + 1) * c.voxelSize.y;
  MC_bBoxMax.z = c.bBoxMin.z + ( z + 1) * c.voxelSize.z;

	
  //find maximal projection on detector:
  borderMin = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
  borderMax = make_float3(-200000.f, -200000.f, -200000.f);


  //The loop has been manually unrolled: nvcc cannot handle inner loops
  //first corner
  //ray.x = -c.projNorm.x;
  //ray.y = -c.projNorm.y;
  //ray.z = -c.projNorm.z;

  //ray = normalize(ray);
  t = (c.projNorm.x * MC_bBoxMin.x + c.projNorm.y * MC_bBoxMin.y + c.projNorm.z * MC_bBoxMin.z);
  t += (-c.projNorm.x * c.detektor.x - c.projNorm.y * c.detektor.y - c.projNorm.z * c.detektor.z);
  t = fabs(t);

  if (!(t >= distMin && t < distMax)) return;

  hitPoint.x = t * (-c.projNorm.x) + MC_bBoxMin.x;
  hitPoint.y = t * (-c.projNorm.y) + MC_bBoxMin.y;
  hitPoint.z = t * (-c.projNorm.z) + MC_bBoxMin.z;

  borderMin = fminf(hitPoint, borderMin);
  borderMax = fmaxf(hitPoint, borderMax);

  //second corner
  t = (c.projNorm.x * MC_bBoxMin.x + c.projNorm.y * MC_bBoxMin.y + c.projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
  t += (-c.projNorm.x * c.detektor.x - c.projNorm.y * c.detektor.y - c.projNorm.z * c.detektor.z);
  t = fabs(t);
  hitPoint.x = t * (-c.projNorm.x) + MC_bBoxMin.x;
  hitPoint.y = t * (-c.projNorm.y) + MC_bBoxMin.y;
  hitPoint.z = t * (-c.projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));

  borderMin = fminf(hitPoint, borderMin);
  borderMax = fmaxf(hitPoint, borderMax);

  //third corner
  t = (c.projNorm.x * MC_bBoxMin.x + c.projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + c.projNorm.z * MC_bBoxMin.z);
  t += (-c.projNorm.x * c.detektor.x - c.projNorm.y * c.detektor.y - c.projNorm.z * c.detektor.z);
  t = fabs(t);
  hitPoint.x = t * (-c.projNorm.x) + MC_bBoxMin.x;
  hitPoint.y = t * (-c.projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
  hitPoint.z = t * (-c.projNorm.z) + MC_bBoxMin.z;

  borderMin = fminf(hitPoint, borderMin);
  borderMax = fmaxf(hitPoint, borderMax);

  //fourth corner
  t = (c.projNorm.x * MC_bBoxMin.x + c.projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + c.projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
  t += (-c.projNorm.x * c.detektor.x - c.projNorm.y * c.detektor.y - c.projNorm.z * c.detektor.z);
  t = fabs(t);
  hitPoint.x = t * (-c.projNorm.x) + MC_bBoxMin.x;
  hitPoint.y = t * (-c.projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
  hitPoint.z = t * (-c.projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));

  borderMin = fminf(hitPoint, borderMin);
  borderMax = fmaxf(hitPoint, borderMax);

  //fifth corner
  t = (c.projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + c.projNorm.y * MC_bBoxMin.y + c.projNorm.z * MC_bBoxMin.z);
  t += (-c.projNorm.x * c.detektor.x - c.projNorm.y * c.detektor.y - c.projNorm.z * c.detektor.z);
  t = fabs(t);
  hitPoint.x = t * (-c.projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
  hitPoint.y = t * (-c.projNorm.y) + MC_bBoxMin.y;
  hitPoint.z = t * (-c.projNorm.z) + MC_bBoxMin.z;

  borderMin = fminf(hitPoint, borderMin);
  borderMax = fmaxf(hitPoint, borderMax);

  //sixth corner
  t = (c.projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + c.projNorm.y * MC_bBoxMin.y + c.projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
  t += (-c.projNorm.x * c.detektor.x - c.projNorm.y * c.detektor.y - c.projNorm.z * c.detektor.z);
  t = fabs(t);
  hitPoint.x = t * (-c.projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
  hitPoint.y = t * (-c.projNorm.y) + MC_bBoxMin.y;
  hitPoint.z = t * (-c.projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));

  borderMin = fminf(hitPoint, borderMin);
  borderMax = fmaxf(hitPoint, borderMax);

  //seventh corner
  t = (c.projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + c.projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + c.projNorm.z * MC_bBoxMin.z);
  t += (-c.projNorm.x * c.detektor.x - c.projNorm.y * c.detektor.y - c.projNorm.z * c.detektor.z);
  t = fabs(t);
  hitPoint.x = t * (-c.projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
  hitPoint.y = t * (-c.projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
  hitPoint.z = t * (-c.projNorm.z) + MC_bBoxMin.z;

  borderMin = fminf(hitPoint, borderMin);
  borderMax = fmaxf(hitPoint, borderMax);

  //eighth corner
  t = (c.projNorm.x * (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x)) + c.projNorm.y * (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y)) + c.projNorm.z * (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z)));
  t += (-c.projNorm.x * c.detektor.x - c.projNorm.y * c.detektor.y - c.projNorm.z * c.detektor.z);
  t = fabs(t);
  hitPoint.x = t * (-c.projNorm.x) + (MC_bBoxMin.x + (MC_bBoxMax.x - MC_bBoxMin.x));
  hitPoint.y = t * (-c.projNorm.y) + (MC_bBoxMin.y + (MC_bBoxMax.y - MC_bBoxMin.y));
  hitPoint.z = t * (-c.projNorm.z) + (MC_bBoxMin.z + (MC_bBoxMax.z - MC_bBoxMin.z));

  borderMin = fminf(hitPoint, borderMin);
  borderMax = fmaxf(hitPoint, borderMax);

  //get largest area
  MC_bBoxMin = fminf(borderMin, borderMax);
  MC_bBoxMax = fmaxf(borderMin, borderMax);

  //swap x values if opposite projection direction
  //	if (c.projNorm.y * c.projNorm.z >= 0)
  {
    //		float ttemp = MC_bBoxMin.x;
    //		MC_bBoxMin.x = MC_bBoxMax.x;
    //		MC_bBoxMax.x = ttemp;
  }
  //	else
  if (c.projNorm.x <= 0)
    {
      float temp = MC_bBoxMin.x;
      MC_bBoxMin.x = MC_bBoxMax.x;
      MC_bBoxMax.x = temp;
      //temp = MC_bBoxMin.y;
      //MC_bBoxMin.y = MC_bBoxMax.y;
      //MC_bBoxMax.y = temp;
    }
  //if (c.projNorm.x * c.projNorm.y >= 0) 
  //{
  //	float temp = MC_bBoxMin.y;
  //	MC_bBoxMin.y = MC_bBoxMax.y;
  //	MC_bBoxMax.y = temp;

  //	
  //}
  //else
  //{
  //	float ttemp = MC_bBoxMin.x;
  //	MC_bBoxMin.x = MC_bBoxMax.x;
  //	MC_bBoxMax.x = ttemp;
  //}

  //Convert global coordinated to projection pixel indices
  MatrixVector3Mul(c.DetectorMatrix, &MC_bBoxMin);    
  MatrixVector3Mul(c.DetectorMatrix, &MC_bBoxMax);

  hitPoint = fminf(MC_bBoxMin, MC_bBoxMax);
  //--> pixelBorders.x = x.min; pixelBorders.z = y.min;
  pixelBorders.x = floor(hitPoint.x); 
  pixelBorders.z = floor(hitPoint.y);
	
  //--> pixelBorders.y = x.max; pixelBorders.v = y.max
  hitPoint = fmaxf(MC_bBoxMin, MC_bBoxMax);
  pixelBorders.y = ceil(hitPoint.x);
  pixelBorders.w = ceil(hitPoint.y);

  //clamp values
  pixelBorders.x = fminf(fmaxf(pixelBorders.x, 0), proj_x - 1);
  pixelBorders.y = fminf(fmaxf(pixelBorders.y, 0), proj_x - 1);
  pixelBorders.z = fminf(fmaxf(pixelBorders.z, 0), proj_y - 1);
  pixelBorders.w = fminf(fmaxf(pixelBorders.w, 0), proj_y - 1);
	


	
  voxelD[index2].x  = 0;
  voxelD[index2].y  = 0;
  voxelD[index2].z  = 0;
  voxelD[index2].w  = 0;
  distanceD[index2].x  = 0;
  distanceD[index2].y  = 0;
  distanceD[index2].z  = 0;
  distanceD[index2].w  = 0;

  //Loop over detected pixels and shoot rays back	again with manual unrolling
  for( pixel_y = pixelBorders.z + maxOverSampleInv*0.5f ; pixel_y < pixelBorders.w ; pixel_y+=maxOverSampleInv)
    {				
      for ( pixel_x = pixelBorders.x + maxOverSampleInv*0.5f ; pixel_x < pixelBorders.y ; pixel_x+=maxOverSampleInv)	
	{
	  /*float pixel_xr = pixel_x + (curand_uniform(&state) - 0.5f) * maxOverSampleInv;
	    float pixel_yr = pixel_y + (curand_uniform(&state) - 0.5f) * maxOverSampleInv;*/

	  float xAniso;
	  float yAniso;

	  MatrixVector3Mul(c.magAnisoInv, pixel_x, pixel_y, xAniso, yAniso);

	  //if (pixel_x < 1) continue;
	  ray.x = c.detektor.x; 
	  ray.y = c.detektor.y; 
	  ray.z = c.detektor.z; 
			
	  ray.x = ray.x + (pixel_x) * c.uPitch.x;
	  ray.y = ray.y + (pixel_x) * c.uPitch.y;
	  ray.z = ray.z + (pixel_x) * c.uPitch.z;
			
	  ray.x = ray.x + (pixel_y) * c.vPitch.x;
	  ray.y = ray.y + (pixel_y) * c.vPitch.y;
	  ray.z = ray.z + (pixel_y) * c.vPitch.z;
			
	  source.x = ray.x + 100000.0 * c.projNorm.x;
	  source.y = ray.y + 100000.0 * c.projNorm.y;
	  source.z = ray.z + 100000.0 * c.projNorm.z;
	  ray.x = ray.x - source.x;
	  ray.y = ray.y - source.y;
	  ray.z = ray.z - source.z;

	  // calculate ray direction
	  ray = normalize(ray);
				
	  //////////// BOX INTERSECTION (Voxel 1) /////////////////	
	  tEntry.x = (c.bBoxMin.x + (x  ) * c.voxelSize.x);
	  tEntry.y = (c.bBoxMin.y + (y  ) * c.voxelSize.y);
	  tEntry.z = (c.bBoxMin.z + (z  ) * c.voxelSize.z);
	  tEntry.x = (tEntry.x - source.x) / ray.x;
	  tEntry.y = (tEntry.y - source.y) / ray.y;
	  tEntry.z = (tEntry.z - source.z) / ray.z;

	  tExit.x = (c.bBoxMin.x + (x+1  ) * c.voxelSize.x);
	  tExit.y = (c.bBoxMin.y + (y+1  ) * c.voxelSize.y);
	  tExit.z = (c.bBoxMin.z + (z+1  ) * c.voxelSize.z);

	  tExit.x = ((tExit.x) - source.x) / ray.x;
	  tExit.y = ((tExit.y) - source.y) / ray.y;
	  tExit.z = ((tExit.z) - source.z) / ray.z;

	  tmin = fminf(tEntry, tExit);
	  tmax = fmaxf(tEntry, tExit);
			
	  t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	  t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
	  ////////////////////////////////////////////////////////////////
				
	  // if the ray hits the voxel
	  if((t_out - t_in) > 0.0f )
	    {
#ifdef CONST_LENGTH_MODE
	      float length = c.voxelSize.x;
#endif
#ifdef PRECISE_LENGTH_MODE
	      float length = (t_out-t_in);
#endif
	      voxelD[index2].x += tex2D<float>(texObj, xAniso, yAniso) * length;
	      distanceD[index2].x += length;
	    }


	  //////////// BOX INTERSECTION (Voxel 2) /////////////////	 
	  tEntry.x = (c.bBoxMin.x + (x+1) * c.voxelSize.x);
	  tEntry.y = (c.bBoxMin.y + (y  ) * c.voxelSize.y);
	  tEntry.z = (c.bBoxMin.z + (z  ) * c.voxelSize.z);
	  tEntry.x = (tEntry.x - source.x) / ray.x;
	  tEntry.y = (tEntry.y - source.y) / ray.y;
	  tEntry.z = (tEntry.z - source.z) / ray.z;

	  tExit.x = (c.bBoxMin.x + (x+2) * c.voxelSize.x);
	  tExit.y = (c.bBoxMin.y + (y+1  ) * c.voxelSize.y);
	  tExit.z = (c.bBoxMin.z + (z+1  ) * c.voxelSize.z);

	  tExit.x = ((tExit.x) - source.x) / ray.x;
	  tExit.y = ((tExit.y) - source.y) / ray.y;
	  tExit.z = ((tExit.z) - source.z) / ray.z;

	  tmin = fminf(tEntry, tExit);
	  tmax = fmaxf(tEntry, tExit);
			
	  t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	  t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
	  ////////////////////////////////////////////////////////////////
				
	  // if the ray hits the voxel
	  if((t_out - t_in) > 0.0f )
	    {
#ifdef CONST_LENGTH_MODE
	      float length = c.voxelSize.x;
#endif
#ifdef PRECISE_LENGTH_MODE
	      float length = (t_out-t_in);
#endif
	      voxelD[index2].y += tex2D<float>(texObj, xAniso, yAniso) * length;
	      distanceD[index2].y += length;
	    }




	  //////////// BOX INTERSECTION (Voxel 3) /////////////////	
	  tEntry.x = (c.bBoxMin.x + (x+2) * c.voxelSize.x);
	  tEntry.y = (c.bBoxMin.y + (y  ) * c.voxelSize.y);
	  tEntry.z = (c.bBoxMin.z + (z  ) * c.voxelSize.z);
	  tEntry.x = (tEntry.x - source.x) / ray.x;
	  tEntry.y = (tEntry.y - source.y) / ray.y;
	  tEntry.z = (tEntry.z - source.z) / ray.z;

	  tExit.x = (c.bBoxMin.x + (x+3) * c.voxelSize.x);
	  tExit.y = (c.bBoxMin.y + (y+1  ) * c.voxelSize.y);
	  tExit.z = (c.bBoxMin.z + (z+1  ) * c.voxelSize.z);

	  tExit.x = ((tExit.x) - source.x) / ray.x;
	  tExit.y = ((tExit.y) - source.y) / ray.y;
	  tExit.z = ((tExit.z) - source.z) / ray.z;

	  tmin = fminf(tEntry, tExit);
	  tmax = fmaxf(tEntry, tExit);
			
	  t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	  t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
	  ////////////////////////////////////////////////////////////////
		
	  // if the ray hits the voxel
	  if((t_out - t_in) > 0.0f )
	    {
#ifdef CONST_LENGTH_MODE
	      float length = c.voxelSize.x;
#endif
#ifdef PRECISE_LENGTH_MODE
	      float length = (t_out-t_in);
#endif
	      voxelD[index2].z += tex2D<float>(texObj, xAniso, yAniso) * length;
	      distanceD[index2].z += length;
	    }


	  //////////// BOX INTERSECTION (Voxel 4) /////////////////	 
	  tEntry.x = (c.bBoxMin.x + (x+3) * c.voxelSize.x);
	  tEntry.y = (c.bBoxMin.y + (y  ) * c.voxelSize.y);
	  tEntry.z = (c.bBoxMin.z + (z  ) * c.voxelSize.z);
	  tEntry.x = (tEntry.x - source.x) / ray.x;
	  tEntry.y = (tEntry.y - source.y) / ray.y;
	  tEntry.z = (tEntry.z - source.z) / ray.z;

	  tExit.x = (c.bBoxMin.x + (x+4) * c.voxelSize.x);
	  tExit.y = (c.bBoxMin.y + (y+1  ) * c.voxelSize.y);
	  tExit.z = (c.bBoxMin.z + (z+1  ) * c.voxelSize.z);

	  tExit.x = ((tExit.x) - source.x) / ray.x;
	  tExit.y = ((tExit.y) - source.y) / ray.y;
	  tExit.z = ((tExit.z) - source.z) / ray.z;

	  tmin = fminf(tEntry, tExit);
	  tmax = fmaxf(tEntry, tExit);
			
	  t_in = fmaxf(fmaxf(tmin.x, tmin.y), tmin.z);
	  t_out = fminf(fminf(tmax.x, tmax.y), tmax.z);
	  ////////////////////////////////////////////////////////////////
		
	  // if the ray hits the voxel
	  if((t_out - t_in) > 0.0f )
	    {
#ifdef CONST_LENGTH_MODE
	      float length = c.voxelSize.x;
#endif
#ifdef PRECISE_LENGTH_MODE
	      float length = (t_out-t_in);
#endif
	      voxelD[index2].w += tex2D<float>(texObj, xAniso, yAniso) * length;
	      distanceD[index2].w += length;
	    }// if hit voxel
	}//for loop y-pixel
    }//for loop x-pixel

  //	float len = 0;
  //	float3 lengths;
  //	
  //	lengths.x = 100;
  //	if (x > c.volumeDim.x * 0.5f)
  //	{
  //		if (c.volumeDim.x - x < 100.0f)
  //			lengths.x = c.volumeDim.x -x;
  //	}
  //	else
  //	{
  //		if (x < 100.0f)
  //			lengths.x = x;
  //	}
  //	
  //	
  //	lengths.y = 100;
  //	if (x > c.volumeDim.y * 0.5f)
  //	{
  //		if (c.volumeDim.y - y < 100.0f)
  //			lengths.y = c.volumeDim.x -y;
  //	}
  //	else
  //	{
  //		if (y < 100.0f)
  //			lengths.y = y;
  //	}
  //	
  //	
  //	lengths.z = 100;
  //	if (x > c.volumeDim.z * 0.5f)
  //	{
  //		if (c.volumeDim.z - z < 100.0f)
  //			lengths.z = c.volumeDim.x -z;
  //	}
  //	else
  //	{
  //		if (z < 100.0f)
  //			lengths.z = z;
  //	}
  //
  //	len = sqrtf(lengths.x * lengths.x + lengths.y * lengths.y + lengths.z * lengths.z);
  //	if (len < 100)
  //		len = 1.0f - expf(-(len * len));
  //	else
  //		len = 1.0f;

  //Only positive distance values are allowed
  distanceD[index2].x = fmaxf (0.f, distanceD[index2].x);
  distanceD[index2].y = fmaxf (0.f, distanceD[index2].y);
  distanceD[index2].z = fmaxf (0.f, distanceD[index2].z);
  distanceD[index2].w = fmaxf (0.f, distanceD[index2].w);	

  //Apply correction term to voxel
  if (distanceD[index2].x != 0.0f) voxelBlock.x += (lambda * voxelD[index2].x / (float)distanceD[index2].x);
  if (distanceD[index2].y != 0.0f) voxelBlock.y += (lambda * voxelD[index2].y / (float)distanceD[index2].y);
  if (distanceD[index2].z != 0.0f) voxelBlock.z += (lambda * voxelD[index2].z / (float)distanceD[index2].z);
  if (distanceD[index2].w != 0.0f) voxelBlock.w += (lambda * voxelD[index2].w / (float)distanceD[index2].w);


  //store values in global memory
  //__syncthreads();
  //vol2[voxelIndex] = temp2;
	
#ifdef SM20
  /*	tempfp16 = __float2half_rn(voxelBlock.x);
	surf3Dwrite(tempfp16, surfref, x * 2 + 0, y, z);
	tempfp16 = __float2half_rn(voxelBlock.y);
	surf3Dwrite(tempfp16, surfref, x * 2 + 2, y, z);
	tempfp16 = __float2half_rn(voxelBlock.z);
	surf3Dwrite(tempfp16, surfref, x * 2 + 4, y, z);
	tempfp16 = __float2half_rn(voxelBlock.w);
	surf3Dwrite(tempfp16, surfref, x * 2 + 6, y, z);
  */
  {
    half tempfp16;
    tempfp16 = __float2half_rn(voxelBlock.x);
    surf3Dwrite(*(ushort*)& tempfp16, surfObj, x * 2 + 0, y, z);
    tempfp16 = __float2half_rn(voxelBlock.y);
    surf3Dwrite(*(ushort*)& tempfp16, surfObj, x * 2 + 2, y, z);
    tempfp16 = __float2half_rn(voxelBlock.z);
    surf3Dwrite(*(ushort*)& tempfp16, surfObj, x * 2 + 4, y, z);
    tempfp16 = __float2half_rn(voxelBlock.w);
    surf3Dwrite(*(ushort*)& tempfp16, surfObj, x * 2 + 6, y, z);
  }
#else
  vol2[voxelIndex] = voxelBlock;
#endif
}


extern "C"
__global__ 
//void convertVolumeFP16ToFP32(DeviceReconstructionConstantsCommon c, float* volPlane, int stride, unsigned int z, hipSurfaceObject_t surfObj)
void convertVolumeFP16ToFP32( DevParamConvVol recParam )
{

  DeviceReconstructionConstantsCommon &c = recParam.common;
  float* &volPlane = recParam.volPlane;
  int &stride = recParam.stride;
  unsigned int &z = recParam.z;
  hipSurfaceObject_t &surfObj = recParam.surfObj;

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x >= c.volumeDim_x_quarter || y >= c.volumeDim.y || z >= c.volumeDim.z) return;

  float4 voxelBlock;

  unsigned short tempfp16;
  /*
    surf3Dread(&tempfp16, surfref, x * 2 * 4 + 0, y, z);
    voxelBlock.x = __half2float(tempfp16);
    surf3Dread(&tempfp16, surfref, x * 2 * 4 + 2, y, z);
    voxelBlock.y = __half2float(tempfp16);
    surf3Dread(&tempfp16, surfref, x * 2 * 4 + 4, y, z);
    voxelBlock.z = __half2float(tempfp16);
    surf3Dread(&tempfp16, surfref, x * 2 * 4 + 6, y, z);
    voxelBlock.w = __half2float(tempfp16);
  */
  surf3Dread(&tempfp16, surfObj, x * 2 * 4 + 0, y, z);
  voxelBlock.x = __half2float(*(half*)& tempfp16);
  surf3Dread(&tempfp16, surfObj, x * 2 * 4 + 2, y, z);
  voxelBlock.y = __half2float(*(half*)& tempfp16);
  surf3Dread(&tempfp16, surfObj, x * 2 * 4 + 4, y, z);
  voxelBlock.z = __half2float(*(half*)& tempfp16);
  surf3Dread(&tempfp16, surfObj, x * 2 * 4 + 6, y, z);
  voxelBlock.w = __half2float(*(half*)& tempfp16);
	
  //adopt x coordinate to single voxels, not voxelBlocks
  x = x * 4;
	
  volPlane[y * stride + x + 0] = -voxelBlock.x;
  volPlane[y * stride + x + 1] = -voxelBlock.y;
  volPlane[y * stride + x + 2] = -voxelBlock.z;
  volPlane[y * stride + x + 3] = -voxelBlock.w;
}


extern "C"
__global__
//void convertVolume3DFP16ToFP32(DeviceReconstructionConstantsCommon c, float* volPlane, int stride, hipSurfaceObject_t surfObj)
void convertVolume3DFP16ToFP32( DevParamConvVol3D recParam )
{
  DeviceReconstructionConstantsCommon &c = recParam.common;
  float* &volPlane = recParam.volPlane;
  int &stride = recParam.stride;
  hipSurfaceObject_t &surfObj = recParam.surfObj;

  unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const unsigned int z = blockIdx.z * blockDim.z + threadIdx.z;

  if (x >= c.volumeDim_x_quarter || y >= c.volumeDim.y || z >= c.volumeDim.z) return;

  float4 voxelBlock;

  unsigned short tempfp16;
  /*
    surf3Dread(&tempfp16, surfref, x * 2 * 4 + 0, y, z);
    voxelBlock.x = __half2float(tempfp16);
    surf3Dread(&tempfp16, surfref, x * 2 * 4 + 2, y, z);
    voxelBlock.y = __half2float(tempfp16);
    surf3Dread(&tempfp16, surfref, x * 2 * 4 + 4, y, z);
    voxelBlock.z = __half2float(tempfp16);
    surf3Dread(&tempfp16, surfref, x * 2 * 4 + 6, y, z);
    voxelBlock.w = __half2float(tempfp16);
  */
  surf3Dread(&tempfp16, surfObj, x * 2 * 4 + 0, y, z);
  voxelBlock.x = __half2float(*(half*)& tempfp16);
  surf3Dread(&tempfp16, surfObj, x * 2 * 4 + 2, y, z);
  voxelBlock.y = __half2float(*(half*)& tempfp16);
  surf3Dread(&tempfp16, surfObj, x * 2 * 4 + 4, y, z);
  voxelBlock.z = __half2float(*(half*)& tempfp16);
  surf3Dread(&tempfp16, surfObj, x * 2 * 4 + 6, y, z);
  voxelBlock.w = __half2float(*(half*)& tempfp16);
	
  //adopt x coordinate to single voxels, not voxelBlocks
  x = x * 4;

  volPlane[y * stride + x + 0] = -voxelBlock.x;
  volPlane[y * stride + x + 1] = -voxelBlock.y;
  volPlane[y * stride + x + 2] = -voxelBlock.z;
  volPlane[y * stride + x + 3] = -voxelBlock.w;
}

#endif //BACKPROJECTIONSQUAREOS_CU
