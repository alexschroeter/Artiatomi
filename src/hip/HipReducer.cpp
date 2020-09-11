#include "HipReducer.h"

HipReducer::HipReducer(int aVoxelCount, hipStream_t aStream, Hip::HipContext* context)
	: voxelCount(aVoxelCount), stream(aStream), ctx(context)
{
	hipModule_t kernelModule = ctx->LoadModule("kernel.ptx");
		
	sum512 = new HipKernel("_Z6reduceILj512EEvPfS0_j", kernelModule);
	sum256 = new HipKernel("_Z6reduceILj256EEvPfS0_j", kernelModule);
	sum128 = new HipKernel("_Z6reduceILj128EEvPfS0_j", kernelModule);
	sum64  = new HipKernel("_Z6reduceILj64EEvPfS0_j",  kernelModule);
	sum32  = new HipKernel("_Z6reduceILj32EEvPfS0_j",  kernelModule);
	sum16  = new HipKernel("_Z6reduceILj16EEvPfS0_j",  kernelModule);
	sum8   = new HipKernel("_Z6reduceILj8EEvPfS0_j",   kernelModule);
	sum4   = new HipKernel("_Z6reduceILj4EEvPfS0_j",   kernelModule);
	sum2   = new HipKernel("_Z6reduceILj2EEvPfS0_j",   kernelModule);
	sum1   = new HipKernel("_Z6reduceILj1EEvPfS0_j",   kernelModule);

		
	sumCplx512 = new HipKernel("_Z10reduceCplxILj512EEvP6float2Pfj", kernelModule);
	sumCplx256 = new HipKernel("_Z10reduceCplxILj256EEvP6float2Pfj", kernelModule);
	sumCplx128 = new HipKernel("_Z10reduceCplxILj128EEvP6float2Pfj", kernelModule);
	sumCplx64  = new HipKernel("_Z10reduceCplxILj64EEvP6float2Pfj",  kernelModule);
	sumCplx32  = new HipKernel("_Z10reduceCplxILj32EEvP6float2Pfj",  kernelModule);
	sumCplx16  = new HipKernel("_Z10reduceCplxILj16EEvP6float2Pfj",  kernelModule);
	sumCplx8   = new HipKernel("_Z10reduceCplxILj8EEvP6float2Pfj",   kernelModule);
	sumCplx4   = new HipKernel("_Z10reduceCplxILj4EEvP6float2Pfj",   kernelModule);
	sumCplx2   = new HipKernel("_Z10reduceCplxILj2EEvP6float2Pfj",   kernelModule);
	sumCplx1   = new HipKernel("_Z10reduceCplxILj1EEvP6float2Pfj",   kernelModule);

		
	sumSqrCplx512 = new HipKernel("_Z13reduceSqrCplxILj512EEvP6float2Pfj", kernelModule);
	sumSqrCplx256 = new HipKernel("_Z13reduceSqrCplxILj256EEvP6float2Pfj", kernelModule);
	sumSqrCplx128 = new HipKernel("_Z13reduceSqrCplxILj128EEvP6float2Pfj", kernelModule);
	sumSqrCplx64  = new HipKernel("_Z13reduceSqrCplxILj64EEvP6float2Pfj",  kernelModule);
	sumSqrCplx32  = new HipKernel("_Z13reduceSqrCplxILj32EEvP6float2Pfj",  kernelModule);
	sumSqrCplx16  = new HipKernel("_Z13reduceSqrCplxILj16EEvP6float2Pfj",  kernelModule);
	sumSqrCplx8   = new HipKernel("_Z13reduceSqrCplxILj8EEvP6float2Pfj",   kernelModule);
	sumSqrCplx4   = new HipKernel("_Z13reduceSqrCplxILj4EEvP6float2Pfj",   kernelModule);
	sumSqrCplx2   = new HipKernel("_Z13reduceSqrCplxILj2EEvP6float2Pfj",   kernelModule);
	sumSqrCplx1   = new HipKernel("_Z13reduceSqrCplxILj1EEvP6float2Pfj",   kernelModule);

	maxIndex512 = new HipKernel("_Z8maxIndexILj512EEvPfS0_Pijb", kernelModule);
	maxIndex256 = new HipKernel("_Z8maxIndexILj256EEvPfS0_Pijb", kernelModule);
	maxIndex128 = new HipKernel("_Z8maxIndexILj128EEvPfS0_Pijb", kernelModule);
	maxIndex64  = new HipKernel("_Z8maxIndexILj64EEvPfS0_Pijb",  kernelModule);
	maxIndex32  = new HipKernel("_Z8maxIndexILj32EEvPfS0_Pijb",  kernelModule);
	maxIndex16  = new HipKernel("_Z8maxIndexILj16EEvPfS0_Pijb",  kernelModule);
	maxIndex8   = new HipKernel("_Z8maxIndexILj8EEvPfS0_Pijb",   kernelModule);
	maxIndex4   = new HipKernel("_Z8maxIndexILj4EEvPfS0_Pijb",   kernelModule);
	maxIndex2   = new HipKernel("_Z8maxIndexILj2EEvPfS0_Pijb",   kernelModule);
	maxIndex1   = new HipKernel("_Z8maxIndexILj1EEvPfS0_Pijb",   kernelModule);

	maxIndexCplx512 = new HipKernel("_Z12maxIndexCplxILj512EEvP6float2PfPijb", kernelModule);
	maxIndexCplx256 = new HipKernel("_Z12maxIndexCplxILj256EEvP6float2PfPijb", kernelModule);
	maxIndexCplx128 = new HipKernel("_Z12maxIndexCplxILj128EEvP6float2PfPijb", kernelModule);
	maxIndexCplx64  = new HipKernel("_Z12maxIndexCplxILj64EEvP6float2PfPijb",  kernelModule);
	maxIndexCplx32  = new HipKernel("_Z12maxIndexCplxILj32EEvP6float2PfPijb",  kernelModule);
	maxIndexCplx16  = new HipKernel("_Z12maxIndexCplxILj16EEvP6float2PfPijb",  kernelModule);
	maxIndexCplx8   = new HipKernel("_Z12maxIndexCplxILj8EEvP6float2PfPijb",   kernelModule);
	maxIndexCplx4   = new HipKernel("_Z12maxIndexCplxILj4EEvP6float2PfPijb",   kernelModule);
	maxIndexCplx2   = new HipKernel("_Z12maxIndexCplxILj2EEvP6float2PfPijb",   kernelModule);
	maxIndexCplx1   = new HipKernel("_Z12maxIndexCplxILj1EEvP6float2PfPijb",   kernelModule);

    /*_AS
    This Kernel has been added 16.08.2019

    The kernel has been written in kernel.cu and after creating the ptx with "nvcc -ptx kernel.cu"
    the internal name of the functions can be found in the kernel.ptx files

    Z9reduceSqrILj512EEvPfS0_j
    */
    sumSqr512 = new HipKernel("_Z9reduceSqrILj512EEvPfS0_j", kernelModule);
    sumSqr256 = new HipKernel("_Z9reduceSqrILj256EEvPfS0_j", kernelModule);
    sumSqr128 = new HipKernel("_Z9reduceSqrILj128EEvPfS0_j", kernelModule);
    sumSqr64  = new HipKernel("_Z9reduceSqrILj64EEvPfS0_j",  kernelModule);
    sumSqr32  = new HipKernel("_Z9reduceSqrILj32EEvPfS0_j",  kernelModule);
    sumSqr16  = new HipKernel("_Z9reduceSqrILj16EEvPfS0_j",  kernelModule);
    sumSqr8   = new HipKernel("_Z9reduceSqrILj8EEvPfS0_j",   kernelModule);
    sumSqr4   = new HipKernel("_Z9reduceSqrILj4EEvPfS0_j",   kernelModule);
    sumSqr2   = new HipKernel("_Z9reduceSqrILj2EEvPfS0_j",   kernelModule);
    sumSqr1   = new HipKernel("_Z9reduceSqrILj1EEvPfS0_j",   kernelModule);
}

void HipReducer::MaxIndex(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& d_index)
{
	int blocks;
	int threads;
	float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;

	
    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runMaxIndexKernel(voxelCount, blocks, threads, d_idata, d_odata, d_index, false);

    // sum partial block sums on GPU
    int s=blocks;

    while (s > 1)
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, blocks, threads);

        runMaxIndexKernel(s, blocks, threads, d_odata, d_odata, d_index, true);
		s = (s + (threads*2-1)) / (threads*2);
    }

    if (s > 1)
    {
		printf("Oops, not a power of 2?\n");
		//      // copy result from device to host
			//d_odata.CopyDeviceToHost(h_odata, s * sizeof(float));

		//      for (int i=0; i < s; i++)
		//      {
		//          gpu_result += h_odata[i];
		//      }

		//      needReadBack = false;
    }
    


    if (needReadBack)
    {
        // copy final sum from device to host
		//d_odata.CopyDeviceToHost(&gpu_result, sizeof(float));
    }

}

void HipReducer::MaxIndexCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& d_index)
{
	int blocks;
	int threads;
	float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;

	
    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runMaxIndexCplxKernel(voxelCount, blocks, threads, d_idata, d_odata, d_index, false);

    // sum partial block sums on GPU
    int s=blocks;

    while (s > 1)
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, blocks, threads);

        runMaxIndexKernel(s, blocks, threads, d_odata, d_odata, d_index, true);
		s = (s + (threads*2-1)) / (threads*2);
    }

    if (s > 1)
    {
		printf("Oops, not a power of 2?\n");
		//      // copy result from device to host
			//d_odata.CopyDeviceToHost(h_odata, s * sizeof(float));

		//      for (int i=0; i < s; i++)
		//      {
		//          gpu_result += h_odata[i];
		//      }

		//      needReadBack = false;
    }
    


    if (needReadBack)
    {
        // copy final sum from device to host
		//d_odata.CopyDeviceToHost(&gpu_result, sizeof(float));
    }

}

void HipReducer::Sum(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	int blocks;
	int threads;
	float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;

	
    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runSumKernel(voxelCount, blocks, threads, d_idata, d_odata);

    // sum partial block sums on GPU
    int s=blocks;

    while (s > 1)
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, blocks, threads);

        runSumKernel(s, blocks, threads, d_odata, d_odata);
		s = (s + (threads*2-1)) / (threads*2);
    }

    if (s > 1)
    {
		printf("Oops, not a power of 2?\n");
		//      // copy result from device to host
			//d_odata.CopyDeviceToHost(h_odata, s * sizeof(float));

		//      for (int i=0; i < s; i++)
		//      {
		//          gpu_result += h_odata[i];
		//      }

		//      needReadBack = false;
    }
    


    if (needReadBack)
    {
        // copy final sum from device to host
		//d_odata.CopyDeviceToHost(&gpu_result, sizeof(float));
    }

}

void HipReducer::SumSqrCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	int blocks;
	int threads;
	float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;

	
    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runSumSqrCplxKernel(voxelCount, blocks, threads, d_idata, d_odata);

    // sum partial block sums on GPU
    int s=blocks;

    while (s > 1)
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, blocks, threads);

        /* AS ToDo: Shouldn't this be runSumSqrCplxKernel ? */
        runSumKernel(s, blocks, threads, d_odata, d_odata);
		s = (s + (threads*2-1)) / (threads*2);
    }

    if (s > 1)
    {
		printf("Oops, not a power of 2?\n");
		//      // copy result from device to host
			//d_odata.CopyDeviceToHost(h_odata, s * sizeof(float));

		//      for (int i=0; i < s; i++)
		//      {
		//          gpu_result += h_odata[i];
		//      }

		//      needReadBack = false;
    }
    


    if (needReadBack)
    {
        // copy final sum from device to host
		//d_odata.CopyDeviceToHost(&gpu_result, sizeof(float));
    }

}


void HipReducer::SumSqr(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
/*
 *  AS: Added on 16.08.2019
 *
 */
{
    int blocks;
    int threads;
    float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;

    
    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runSumSqrKernel(voxelCount, blocks, threads, d_idata, d_odata);

    // sum partial block sums on GPU
    int s=blocks;

    while (s > 1)
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, blocks, threads);

        runSumKernel(s, blocks, threads, d_odata, d_odata);
        s = (s + (threads*2-1)) / (threads*2);
    }

    if (s > 1)
    {
        printf("Oops, not a power of 2?\n");
        //      // copy result from device to host
            //d_odata.CopyDeviceToHost(h_odata, s * sizeof(float));

        //      for (int i=0; i < s; i++)
        //      {
        //          gpu_result += h_odata[i];
        //      }

        //      needReadBack = false;
    }
    


    if (needReadBack)
    {
        // copy final sum from device to host
        //d_odata.CopyDeviceToHost(&gpu_result, sizeof(float));
    }

}


void HipReducer::SumCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	int blocks;
	int threads;
	float gpu_result = 0;
    bool needReadBack = true;

    gpu_result = 0;

	
    getNumBlocksAndThreads(voxelCount, blocks, threads);
    // execute the kernel
    runSumCplxKernel(voxelCount, blocks, threads, d_idata, d_odata);

    // sum partial block sums on GPU
    int s=blocks;

    while (s > 1)
    {
        int threads = 0, blocks = 0;
        getNumBlocksAndThreads(s, blocks, threads);

        runSumKernel(s, blocks, threads, d_odata, d_odata);
		s = (s + (threads*2-1)) / (threads*2);
    }

    if (s > 1)
    {
		printf("Oops, not a power of 2?\n");
		//      // copy result from device to host
			//d_odata.CopyDeviceToHost(h_odata, s * sizeof(float));

		//      for (int i=0; i < s; i++)
		//      {
		//          gpu_result += h_odata[i];
		//      }

		//      needReadBack = false;
    }
    


    if (needReadBack)
    {
        // copy final sum from device to host
		//d_odata.CopyDeviceToHost(&gpu_result, sizeof(float));
    }

}



void HipReducer::runMaxIndexKernel(int size, int blocks, int threads, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& d_index, bool readIndex)
{
	HipKernel* kernel;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 4 * threads * sizeof(float) : 2 * threads * sizeof(float);

	switch (threads)
    {
        case 512:
            kernel = maxIndex512; break;
        case 256:
            kernel = maxIndex256; break;
        case 128:
            kernel = maxIndex128; break;
        case 64:
            kernel = maxIndex64; break;
        case 32:
            kernel = maxIndex32; break;
        case 16:
            kernel = maxIndex16; break;
        case  8:
            kernel = maxIndex8; break;
        case  4:
            kernel = maxIndex4; break;
        case  2:
            kernel = maxIndex2; break;
        case  1:
            kernel = maxIndex1; break;
    }
	
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();
    hipDeviceptr_t index_dptr = d_index.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[5];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &index_dptr;
    arglist[3] = &n;
	arglist[4] = &readIndex;

    hipSafeCall(hipModuleLaunchKernel(kernel->GetHipFunction(), dimGrid.x, dimGrid.y,
		dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, smemSize, stream, arglist,NULL));

    delete[] arglist;
}

void HipReducer::runMaxIndexCplxKernel(int size, int blocks, int threads, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& d_index, bool readIndex)
{
	HipKernel* kernel;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 4 * threads * sizeof(float) : 2 * threads * sizeof(float);

	switch (threads)
    {
        case 512:
            kernel = maxIndexCplx512; break;
        case 256:
            kernel = maxIndexCplx256; break;
        case 128:
            kernel = maxIndexCplx128; break;
        case 64:
            kernel = maxIndexCplx64; break;
        case 32:
            kernel = maxIndexCplx32; break;
        case 16:
            kernel = maxIndexCplx16; break;
        case  8:
            kernel = maxIndexCplx8; break;
        case  4:
            kernel = maxIndexCplx4; break;
        case  2:
            kernel = maxIndexCplx2; break;
        case  1:
            kernel = maxIndexCplx1; break;
    }
	
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();
    hipDeviceptr_t index_dptr = d_index.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[5];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &index_dptr;
    arglist[3] = &n;
	arglist[4] = &readIndex;

    hipSafeCall(hipModuleLaunchKernel(kernel->GetHipFunction(), dimGrid.x, dimGrid.y,
		dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, smemSize, stream, arglist,NULL));

    delete[] arglist;
}

void HipReducer::runSumKernel(int size, int blocks, int threads, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	HipKernel* kernel;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads)
    {
        case 512:
            kernel = sum512; break;
        case 256:
            kernel = sum256; break;
        case 128:
            kernel = sum128; break;
        case 64:
            kernel = sum64; break;
        case 32:
            kernel = sum32; break;
        case 16:
            kernel = sum16; break;
        case  8:
            kernel = sum8; break;
        case  4:
            kernel = sum4; break;
        case  2:
            kernel = sum2; break;
        case  1:
            kernel = sum1; break;
    }
	
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[3];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &n;

    //float ms;

    //CUevent eventStart;
    //CUevent eventEnd;
    //hipStream_t stream = 0;
    //hipSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    //hipSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    //hipSafeCall(hipStream_tQuery(stream));
    //hipSafeCall(cuEventRecord(eventStart, stream));
    hipSafeCall(hipModuleLaunchKernel(kernel->GetHipFunction(), dimGrid.x, dimGrid.y,
		dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, smemSize, stream, arglist,NULL));

    //hipSafeCall(cuCtxSynchronize());

    //hipSafeCall(hipStream_tQuery(stream));
    //hipSafeCall(cuEventRecord(eventEnd, stream));
    //hipSafeCall(cuEventSynchronize(eventEnd));
    //hipSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    delete[] arglist;
}


void HipReducer::runSumSqrCplxKernel(int size, int blocks, int threads, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	HipKernel* kernel;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads)
    {
        case 512:
            kernel = sumSqrCplx512; break;
        case 256:
            kernel = sumSqrCplx256; break;
        case 128:
            kernel = sumSqrCplx128; break;
        case 64:
            kernel = sumSqrCplx64; break;
        case 32:
            kernel = sumSqrCplx32; break;
        case 16:
            kernel = sumSqrCplx16; break;
        case  8:
            kernel = sumSqrCplx8; break;
        case  4:
            kernel = sumSqrCplx4; break;
        case  2:
            kernel = sumSqrCplx2; break;
        case  1:
            kernel = sumSqrCplx1; break;
    }
	
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[3];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &n;

    //float ms;

    //CUevent eventStart;
    //CUevent eventEnd;
    //hipStream_t stream = 0;
    //hipSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    //hipSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    //hipSafeCall(hipStream_tQuery(stream));
    //hipSafeCall(cuEventRecord(eventStart, stream));
    hipSafeCall(hipModuleLaunchKernel(kernel->GetHipFunction(), dimGrid.x, dimGrid.y,
		dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, smemSize, stream, arglist,NULL));

    //hipSafeCall(cuCtxSynchronize());

    //hipSafeCall(hipStream_tQuery(stream));
    //hipSafeCall(cuEventRecord(eventEnd, stream));
    //hipSafeCall(cuEventSynchronize(eventEnd));
    //hipSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    delete[] arglist;
}


void HipReducer::runSumSqrKernel(int size, int blocks, int threads, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    HipKernel* kernel;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

    switch (threads)
    {
        case 512:
            kernel = sumSqr512; break;
        case 256:
            kernel = sumSqr256; break;
        case 128:
            kernel = sumSqr128; break;
        case 64:
            kernel = sumSqr64; break;
        case 32:
            kernel = sumSqr32; break;
        case 16:
            kernel = sumSqr16; break;
        case  8:
            kernel = sumSqr8; break;
        case  4:
            kernel = sumSqr4; break;
        case  2:
            kernel = sumSqr2; break;
        case  1:
            kernel = sumSqr1; break;
    }
    
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[3];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &n;

    //float ms;

    //CUevent eventStart;
    //CUevent eventEnd;
    //hipStream_t stream = 0;
    //hipSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    //hipSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    //hipSafeCall(hipStream_tQuery(stream));
    //hipSafeCall(cuEventRecord(eventStart, stream));
    hipSafeCall(hipModuleLaunchKernel(kernel->GetHipFunction(), dimGrid.x, dimGrid.y,
        dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, smemSize, stream, arglist,NULL));

    //hipSafeCall(cuCtxSynchronize());

    //hipSafeCall(hipStream_tQuery(stream));
    //hipSafeCall(cuEventRecord(eventEnd, stream));
    //hipSafeCall(cuEventSynchronize(eventEnd));
    //hipSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    delete[] arglist;
}


void HipReducer::runSumCplxKernel(int size, int blocks, int threads, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	HipKernel* kernel;
    dim3 dimBlock(threads, 1, 1);
    dim3 dimGrid(blocks, 1, 1);

    // when there is only one warp per block, we need to allocate two warps
    // worth of shared memory so that we don't index shared memory out of bounds
    int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

	switch (threads)
    {
        case 512:
            kernel = sumCplx512; break;
        case 256:
            kernel = sumCplx256; break;
        case 128:
            kernel = sumCplx128; break;
        case 64:
            kernel = sumCplx64; break;
        case 32:
            kernel = sumCplx32; break;
        case 16:
            kernel = sumCplx16; break;
        case  8:
            kernel = sumCplx8; break;
        case  4:
            kernel = sumCplx4; break;
        case  2:
            kernel = sumCplx2; break;
        case  1:
            kernel = sumCplx1; break;
    }
	
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();
    int n = size;

    void** arglist = (void**)new void*[3];

    arglist[0] = &in_dptr;
    arglist[1] = &out_dptr;
    arglist[2] = &n;

    //float ms;

    //CUevent eventStart;
    //CUevent eventEnd;
    //hipStream_t stream = 0;
    //hipSafeCall(cuEventCreate(&eventStart, CU_EVENT_BLOCKING_SYNC));
    //hipSafeCall(cuEventCreate(&eventEnd, CU_EVENT_BLOCKING_SYNC));

    //hipSafeCall(hipStream_tQuery(stream));
    //hipSafeCall(cuEventRecord(eventStart, stream));
    hipSafeCall(hipModuleLaunchKernel(kernel->GetHipFunction(), dimGrid.x, dimGrid.y,
		dimGrid.z, dimBlock.x, dimBlock.y, dimBlock.z, smemSize, stream, arglist,NULL));

    //hipSafeCall(cuCtxSynchronize());

    //hipSafeCall(hipStream_tQuery(stream));
    //hipSafeCall(cuEventRecord(eventEnd, stream));
    //hipSafeCall(cuEventSynchronize(eventEnd));
    //hipSafeCall(cuEventElapsedTime(&ms, eventStart, eventEnd));

    delete[] arglist;
}

int HipReducer::GetOutBufferSize()
{
	int numBlocks = 0;
	int temp = 0;
	getNumBlocksAndThreads(voxelCount, numBlocks, temp);
	return numBlocks;
}

void HipReducer::getNumBlocksAndThreads(int n, int &blocks, int &threads)
{    
    threads = (n < maxThreads*2) ? nextPow2((n + 1)/ 2) : maxThreads;
    blocks = (n + (threads * 2 - 1)) / (threads * 2);
    

    if (blocks > 2147483647) //Maximum of GTX Titan
    {
        printf("Grid size <%d> excceeds the device capability <%d>, set block size as %d (original %d)\n",
               blocks, 2147483647, threads*2, threads);

        blocks /= 2;
        threads *= 2;
    }

    blocks = min(maxBlocks, blocks);
}


unsigned int HipReducer::nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}

