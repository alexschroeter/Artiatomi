//Includes for IntelliSense 
#define _SIZE_T_DEFINED
#ifndef __CUDACC__
#define __CUDACC__
#endif
#ifndef __cplusplus
#define __cplusplus
#endif


#include <hip/hip_runtime.h>
//#include <device_launch_parameters.h>
//#include <texture_fetch_functions.h>
#include "float.h"
//#include <builtin_types.h>
//#include <vector_functions.h>

#include "DeviceReconstructionParameters.h"

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
	__device__ inline operator       T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};


/*
	This version adds multiple elements per thread sequentially.  This reduces the overall
	cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
	(Brent's Theorem optimization)

	Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
	In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
	If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <unsigned int blockSize>
__global__ void
reduce(DevParaSum param)
{
	float *g_idata = param.in_dptr;
	float *g_odata = param.out_dptr;
	unsigned int n = param.size;

	float *sdata = SharedMemory<float>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		mySum += g_idata[i+blockSize];

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			sdata[tid] = mySum = mySum + sdata[tid +  64];
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float *smem = sdata;

		if (blockSize >=  64)
		{
			smem[tid] = mySum = mySum + smem[tid + 32];
		}

		if (blockSize >=  32)
		{
			smem[tid] = mySum = mySum + smem[tid + 16];
		}

		if (blockSize >=  16)
		{
			smem[tid] = mySum = mySum + smem[tid +  8];
		}

		if (blockSize >=   8)
		{
			smem[tid] = mySum = mySum + smem[tid +  4];
		}

		if (blockSize >=   4)
		{
			smem[tid] = mySum = mySum + smem[tid +  2];
		}

		if (blockSize >=   2)
		{
			smem[tid] = mySum = mySum + smem[tid +  1];
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

 template __global__ void reduce<512>(DevParaSum param);
 template __global__ void reduce<256>(DevParaSum param);
 template __global__ void reduce<128>(DevParaSum param);
 template __global__ void reduce<64>(DevParaSum param);
 template __global__ void reduce<32>(DevParaSum param);
 template __global__ void reduce<16>(DevParaSum param);
 template __global__ void reduce<8>(DevParaSum param);
 template __global__ void reduce<4>(DevParaSum param);
 template __global__ void reduce<2>(DevParaSum param);
 template __global__ void reduce<1>(DevParaSum param); 



template <unsigned int blockSize>
__global__ void
reduceSqr(DevParaSum param)
{
	float *g_idata = param.in_dptr;
	float *g_odata = param.out_dptr;
	unsigned int n = param.size;

	float *sdata = SharedMemory<float>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i] * g_idata[i];

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		mySum += g_idata[i+blockSize] * g_idata[i+blockSize];

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			sdata[tid] = mySum = mySum + sdata[tid +  64];
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float *smem = sdata;

		if (blockSize >=  64)
		{
			smem[tid] = mySum = mySum + smem[tid + 32];
		}

		if (blockSize >=  32)
		{
			smem[tid] = mySum = mySum + smem[tid + 16];
		}

		if (blockSize >=  16)
		{
			smem[tid] = mySum = mySum + smem[tid +  8];
		}

		if (blockSize >=   8)
		{
			smem[tid] = mySum = mySum + smem[tid +  4];
		}

		if (blockSize >=   4)
		{
			smem[tid] = mySum = mySum + smem[tid +  2];
		}

		if (blockSize >=   2)
		{
			smem[tid] = mySum = mySum + smem[tid +  1];
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}

 template __global__ void reduceSqr<512>(DevParaSum param);
 template __global__ void reduceSqr<256>(DevParaSum param);
 template __global__ void reduceSqr<128>(DevParaSum param);
 template __global__ void reduceSqr<64>(DevParaSum param);
 template __global__ void reduceSqr<32>(DevParaSum param);
 template __global__ void reduceSqr<16>(DevParaSum param);
 template __global__ void reduceSqr<8>(DevParaSum param);
 template __global__ void reduceSqr<4>(DevParaSum param);
 template __global__ void reduceSqr<2>(DevParaSum param);
 template __global__ void reduceSqr<1>(DevParaSum param); 





template <unsigned int blockSize>
__global__ void
reduceCplx(DevParaSumCplx param)
{
	float2 *g_idata = param.in_dptr;
	float *g_odata = param.out_dptr;
	unsigned int n = param.size;

	float *sdata = SharedMemory<float>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i].x;

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		mySum += g_idata[i+blockSize].x;

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			sdata[tid] = mySum = mySum + sdata[tid +  64];
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float *smem = sdata;

		if (blockSize >=  64)
		{
			smem[tid] = mySum = mySum + smem[tid + 32];
		}

		if (blockSize >=  32)
		{
			smem[tid] = mySum = mySum + smem[tid + 16];
		}

		if (blockSize >=  16)
		{
			smem[tid] = mySum = mySum + smem[tid +  8];
		}

		if (blockSize >=   8)
		{
			smem[tid] = mySum = mySum + smem[tid +  4];
		}

		if (blockSize >=   4)
		{
			smem[tid] = mySum = mySum + smem[tid +  2];
		}

		if (blockSize >=   2)
		{
			smem[tid] = mySum = mySum + smem[tid +  1];
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}


 template __global__ void reduceCplx<512>(DevParaSumCplx param);
 template __global__ void reduceCplx<256>(DevParaSumCplx param);
 template __global__ void reduceCplx<128>(DevParaSumCplx param);
 template __global__ void reduceCplx<64>(DevParaSumCplx param);
 template __global__ void reduceCplx<32>(DevParaSumCplx param);
 template __global__ void reduceCplx<16>(DevParaSumCplx param);
 template __global__ void reduceCplx<8>(DevParaSumCplx param);
 template __global__ void reduceCplx<4>(DevParaSumCplx param);
 template __global__ void reduceCplx<2>(DevParaSumCplx param);
 template __global__ void reduceCplx<1>(DevParaSumCplx param);




template <unsigned int blockSize>
__global__ void
reduceSqrCplx(DevParaSumSqrCplx param)
{
	float2 *g_idata = param.in_dptr;
	float *g_odata = param.out_dptr;
	unsigned int n = param.size;

	float *sdata = SharedMemory<float>();

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float mySum = 0;

	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		mySum += g_idata[i].x * g_idata[i].x;

		// ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
		mySum += g_idata[i+blockSize].x * g_idata[i+blockSize].x;

		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = mySum;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 256];
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			sdata[tid] = mySum = mySum + sdata[tid + 128];
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			sdata[tid] = mySum = mySum + sdata[tid +  64];
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float *smem = sdata;

		if (blockSize >=  64)
		{
			smem[tid] = mySum = mySum + smem[tid + 32];
		}

		if (blockSize >=  32)
		{
			smem[tid] = mySum = mySum + smem[tid + 16];
		}

		if (blockSize >=  16)
		{
			smem[tid] = mySum = mySum + smem[tid +  8];
		}

		if (blockSize >=   8)
		{
			smem[tid] = mySum = mySum + smem[tid +  4];
		}

		if (blockSize >=   4)
		{
			smem[tid] = mySum = mySum + smem[tid +  2];
		}

		if (blockSize >=   2)
		{
			smem[tid] = mySum = mySum + smem[tid +  1];
		}
	}

	// write result for this block to global mem
	if (tid == 0)
		g_odata[blockIdx.x] = sdata[0];
}


 template __global__ void reduceSqrCplx<512>(DevParaSumSqrCplx param);
 template __global__ void reduceSqrCplx<256>(DevParaSumSqrCplx param);
 template __global__ void reduceSqrCplx<128>(DevParaSumSqrCplx param);
 template __global__ void reduceSqrCplx<64>(DevParaSumSqrCplx param);
 template __global__ void reduceSqrCplx<32>(DevParaSumSqrCplx param);
 template __global__ void reduceSqrCplx<16>(DevParaSumSqrCplx param);
 template __global__ void reduceSqrCplx<8>(DevParaSumSqrCplx param);
 template __global__ void reduceSqrCplx<4>(DevParaSumSqrCplx param);
 template __global__ void reduceSqrCplx<2>(DevParaSumSqrCplx param);
 template __global__ void reduceSqrCplx<1>(DevParaSumSqrCplx param);


/*
	This version adds multiple elements per thread sequentially.  This reduces the overall
	cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
	(Brent's Theorem optimization)

	Note, this kernel needs a minimum of 64*sizeof(T) bytes of shared memory.
	In other words if blockSize <= 32, allocate 64*sizeof(T) bytes.
	If blockSize > 32, allocate blockSize*sizeof(T) bytes.
*/
template <unsigned int blockSize>
__global__ void
maxIndex(DevParaMaxIndex param)
{
	float *g_idata = param.in_dptr ;
	float *g_odata = param.out_dptr ;
	int *index = param.index ;
	unsigned int n = param.size ;
	bool readIndex = param.readIndex ;

	//float *sdata = SharedMemory<float>();
	//int *sindex = (int*)sdata + blockDim.x;
	int *sindex = SharedMemory<int>();
	float *sdata = (float*)sindex + blockDim.x;

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float myMax = -FLT_MAX;
	int myIndex = -1;
	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		if (g_idata[i] > myMax)
		{
			myMax = g_idata[i];
			myIndex = readIndex ? index[i] : i;
			//mySum += g_idata[i];
		}
		
		if (g_idata[i+blockSize] > myMax)
		{
			myMax = g_idata[i+blockSize];
			myIndex = readIndex ? index[i+blockSize] : i+blockSize;
			//mySum += g_idata[i+blockSize];
		}
		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = myMax;
	sindex[tid] = myIndex;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			if (sdata[tid + 256] > myMax)
			{
				sdata[tid] = myMax = sdata[tid + 256];
				sindex[tid] = myIndex = sindex[tid + 256];
				//sdata[tid] = mySum = mySum + sdata[tid + 256];
			}
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			if (sdata[tid + 128] > myMax)
			{
				sdata[tid] = myMax = sdata[tid + 128];
				sindex[tid] = myIndex = sindex[tid + 128];
				//sdata[tid] = mySum = mySum + sdata[tid + 128];
			}
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			if (sdata[tid +  64] > myMax)
			{
				sdata[tid] = myMax = sdata[tid + 64];
				sindex[tid] = myIndex = sindex[tid + 64];
				//sdata[tid] = mySum = mySum + sdata[tid +  64];
			}
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float *smem = sdata;
		volatile int* smemindex = sindex;

		if (blockSize >=  64)
		{
			if (smem[tid + 32] > myMax)
			{
				smem[tid] = myMax = smem[tid + 32];
				smemindex[tid] = myIndex = smemindex[tid + 32];
				//smem[tid] = mySum = mySum + smem[tid + 32];
			}
		}

		if (blockSize >=  32)
		{
			if (smem[tid + 16] > myMax)
			{
				smem[tid] = myMax = smem[tid + 16];
				smemindex[tid] = myIndex = smemindex[tid + 16];
				//smem[tid] = mySum = mySum + smem[tid + 16];
			}
		}

		if (blockSize >=  16)
		{
			if (smem[tid + 8] > myMax)
			{
				smem[tid] = myMax = smem[tid + 8];
				smemindex[tid] = myIndex = smemindex[tid + 8];
				//smem[tid] = mySum = mySum + smem[tid + 8];
			}
		}

		if (blockSize >=   8)
		{
			if (smem[tid + 4] > myMax)
			{
				smem[tid] = myMax = smem[tid + 4];
				smemindex[tid] = myIndex = smemindex[tid + 4];
				//smem[tid] = mySum = mySum + smem[tid + 4];
			}
		}

		if (blockSize >=   4)
		{
			if (smem[tid + 2] > myMax)
			{
				smem[tid] = myMax = smem[tid + 2];
				smemindex[tid] = myIndex = smemindex[tid + 2];
				//smem[tid] = mySum = mySum + smem[tid + 2];
			}
		}

		if (blockSize >=   2)
		{
			if (smem[tid + 1] > myMax)
			{
				smem[tid] = myMax = smem[tid + 1];
				smemindex[tid] = myIndex = smemindex[tid + 1];
				//smem[tid] = mySum = mySum + smem[tid + 1];
			}
		}
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0];
		index[blockIdx.x] = sindex[0];
	}
}

template __global__ void maxIndex<512>(DevParaMaxIndex param);
template __global__ void maxIndex<256>(DevParaMaxIndex param);
template __global__ void maxIndex<128>(DevParaMaxIndex param);
template __global__ void maxIndex<64>(DevParaMaxIndex param);
template __global__ void maxIndex<32>(DevParaMaxIndex param);
template __global__ void maxIndex<16>(DevParaMaxIndex param);
template __global__ void maxIndex<8>(DevParaMaxIndex param);
template __global__ void maxIndex<4>(DevParaMaxIndex param);
template __global__ void maxIndex<2>(DevParaMaxIndex param);
template __global__ void maxIndex<1>(DevParaMaxIndex param);



template <unsigned int blockSize>
__global__ void
maxIndexCplx(DevParaMaxIndexCplx param)
{
	float2 *g_idata = param.in_dptr ;
	float *g_odata = param.out_dptr ;
	int* index = param.index ;
	unsigned int n = param.size ;
	bool readIndex = param.readIndex ;

	//float *sdata = SharedMemory<float>();
	//int *sindex = (int*)sdata + blockDim.x;
	int *sindex = SharedMemory<int>();
	float *sdata = (float*)sindex + blockDim.x;

	// perform first level of reduction,
	// reading from global memory, writing to shared memory
	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x*blockSize*2 + threadIdx.x;
	unsigned int gridSize = blockSize*2*gridDim.x;

	float myMax = -FLT_MAX;
	int myIndex = -1;
	// we reduce multiple elements per thread.  The number is determined by the
	// number of active thread blocks (via gridDim).  More blocks will result
	// in a larger gridSize and therefore fewer elements per thread
	while (i < n)
	{
		if (g_idata[i].x > myMax)
		{
			myMax = g_idata[i].x;
			myIndex = readIndex ? index[i] : i;
			//mySum += g_idata[i];
		}
		
		if (g_idata[i+blockSize].x > myMax)
		{
			myMax = g_idata[i+blockSize].x;
			myIndex = readIndex ? index[i+blockSize] : i+blockSize;
			//mySum += g_idata[i+blockSize];
		}
		i += gridSize;
	}

	// each thread puts its local sum into shared memory
	sdata[tid] = myMax;
	sindex[tid] = myIndex;
	__syncthreads();


	// do reduction in shared mem
	if (blockSize >= 512)
	{
		if (tid < 256)
		{
			if (sdata[tid + 256] > myMax)
			{
				sdata[tid] = myMax = sdata[tid + 256];
				sindex[tid] = myIndex = sindex[tid + 256];
				//sdata[tid] = mySum = mySum + sdata[tid + 256];
			}
		}

		__syncthreads();
	}

	if (blockSize >= 256)
	{
		if (tid < 128)
		{
			if (sdata[tid + 128] > myMax)
			{
				sdata[tid] = myMax = sdata[tid + 128];
				sindex[tid] = myIndex = sindex[tid + 128];
				//sdata[tid] = mySum = mySum + sdata[tid + 128];
			}
		}

		__syncthreads();
	}

	if (blockSize >= 128)
	{
		if (tid <  64)
		{
			if (sdata[tid +  64] > myMax)
			{
				sdata[tid] = myMax = sdata[tid + 64];
				sindex[tid] = myIndex = sindex[tid + 64];
				//sdata[tid] = mySum = mySum + sdata[tid +  64];
			}
		}

		__syncthreads();
	}

	if (tid < 32)
	{
		// now that we are using warp-synchronous programming (below)
		// we need to declare our shared memory volatile so that the compiler
		// doesn't reorder stores to it and induce incorrect behavior.
		volatile float *smem = sdata;
		volatile int* smemindex = sindex;

		if (blockSize >=  64)
		{
			if (smem[tid + 32] > myMax)
			{
				smem[tid] = myMax = smem[tid + 32];
				smemindex[tid] = myIndex = smemindex[tid + 32];
				//smem[tid] = mySum = mySum + smem[tid + 32];
			}
		}

		if (blockSize >=  32)
		{
			if (smem[tid + 16] > myMax)
			{
				smem[tid] = myMax = smem[tid + 16];
				smemindex[tid] = myIndex = smemindex[tid + 16];
				//smem[tid] = mySum = mySum + smem[tid + 16];
			}
		}

		if (blockSize >=  16)
		{
			if (smem[tid + 8] > myMax)
			{
				smem[tid] = myMax = smem[tid + 8];
				smemindex[tid] = myIndex = smemindex[tid + 8];
				//smem[tid] = mySum = mySum + smem[tid + 8];
			}
		}

		if (blockSize >=   8)
		{
			if (smem[tid + 4] > myMax)
			{
				smem[tid] = myMax = smem[tid + 4];
				smemindex[tid] = myIndex = smemindex[tid + 4];
				//smem[tid] = mySum = mySum + smem[tid + 4];
			}
		}

		if (blockSize >=   4)
		{
			if (smem[tid + 2] > myMax)
			{
				smem[tid] = myMax = smem[tid + 2];
				smemindex[tid] = myIndex = smemindex[tid + 2];
				//smem[tid] = mySum = mySum + smem[tid + 2];
			}
		}

		if (blockSize >=   2)
		{
			if (smem[tid + 1] > myMax)
			{
				smem[tid] = myMax = smem[tid + 1];
				smemindex[tid] = myIndex = smemindex[tid + 1];
				//smem[tid] = mySum = mySum + smem[tid + 1];
			}
		}
	}

	// write result for this block to global mem
	if (tid == 0)
	{
		g_odata[blockIdx.x] = sdata[0];
		index[blockIdx.x] = sindex[0];
	}
}


template __global__ void maxIndexCplx<512>(DevParaMaxIndexCplx param);
template __global__ void maxIndexCplx<256>(DevParaMaxIndexCplx param);
template __global__ void maxIndexCplx<128>(DevParaMaxIndexCplx param);
template __global__ void maxIndexCplx<64>(DevParaMaxIndexCplx param);
template __global__ void maxIndexCplx<32>(DevParaMaxIndexCplx param);
template __global__ void maxIndexCplx<16>(DevParaMaxIndexCplx param);
template __global__ void maxIndexCplx<8>(DevParaMaxIndexCplx param);
template __global__ void maxIndexCplx<4>(DevParaMaxIndexCplx param);
template __global__ void maxIndexCplx<2>(DevParaMaxIndexCplx param);
template __global__ void maxIndexCplx<1>(DevParaMaxIndexCplx param);

