#include "HipBasicKernel.h"


HipSub::HipSub(int aVolSize, hipStream_t aStream, Hip::HipContext* context)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1), 
	  gridSize(aVolSize / 32, aVolSize / 16, aVolSize)
{
	hipModule_t kernelModule = ctx->LoadModule("basicKernels.ptx");
	
	add = new HipKernel("add", kernelModule);
	sub = new HipKernel("sub", kernelModule);
	subCplx = new HipKernel("subCplx", kernelModule);
    subCplxMulVol = new HipKernel("subCplxMulVol", kernelModule);
	subCplx2 = new HipKernel("subCplx2", kernelModule);
}


void HipSub::runAddKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(add->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipSub::Add(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	runAddKernel(d_idata, d_odata);
}


void HipSub::runSubKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val)
{
	hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val;

    hipSafeCall(hipModuleLaunchKernel(sub->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipSub::Sub(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val)
{
	runSubKernel(d_idata, d_odata, val);
}


void HipSub::runSubCplxMulVolKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, HipDeviceVariable& divVal)
{
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();
    hipDeviceptr_t val_dptr = val.GetDevicePtr();
    hipDeviceptr_t divval_dptr = divVal.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val_dptr;
    arglist[4] = &divval_dptr;

    hipSafeCall(hipModuleLaunchKernel(subCplxMulVol->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipSub::SubCplxMulVol(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, HipDeviceVariable& divVal)
{
    runSubCplxMulVolKernel(d_idata, d_odata, val, divVal);
}

void HipSub::runSubCplxKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, float divVal)
{
	hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();
	hipDeviceptr_t val_dptr = val.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val_dptr;
    arglist[4] = &divVal;

    hipSafeCall(hipModuleLaunchKernel(subCplx->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipSub::SubCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, float divVal)
{
	runSubCplxKernel(d_idata, d_odata, val, divVal);
}


void HipSub::runSubCplxKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, HipDeviceVariable& divVal)
{
	hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();
	hipDeviceptr_t val_dptr = val.GetDevicePtr();
	hipDeviceptr_t divval_dptr = divVal.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val_dptr;
    arglist[4] = &divval_dptr;

    hipSafeCall(hipModuleLaunchKernel(subCplx2->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipSub::SubCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, HipDeviceVariable& divVal)
{
	runSubCplxKernel(d_idata, d_odata, val, divVal);
}


HipMakeCplxWithSub::HipMakeCplxWithSub(int aVolSize, hipStream_t aStream, Hip::HipContext* context)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1), 
	  gridSize(aVolSize / 32, aVolSize / 16, aVolSize)
{
	hipModule_t kernelModule = ctx->LoadModule("basicKernels.ptx");
	
	makeReal = new HipKernel("makeReal", kernelModule);
	makeCplxWithSub = new HipKernel("makeCplxWithSub", kernelModule);
	makeCplxWithSubSqr = new HipKernel("makeCplxWithSquareAndSub", kernelModule);
}


void HipMakeCplxWithSub::runRealKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(makeReal->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMakeCplxWithSub::MakeReal(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	runRealKernel(d_idata, d_odata);
}


void HipMakeCplxWithSub::runCplxWithSubKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val)
{
	hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val;

    hipSafeCall(hipModuleLaunchKernel(makeCplxWithSub->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMakeCplxWithSub::MakeCplxWithSub(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val)
{
	runCplxWithSubKernel(d_idata, d_odata, val);
}


void HipMakeCplxWithSub::runCplxWithSqrSubKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val)
{
	hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val;

    hipSafeCall(hipModuleLaunchKernel(makeCplxWithSubSqr->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMakeCplxWithSub::MakeCplxWithSqrSub(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val)
{
	runCplxWithSqrSubKernel(d_idata, d_odata, val);
}


HipBinarize::HipBinarize(int aVolSize, hipStream_t aStream, Hip::HipContext* context)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1), 
	  gridSize(aVolSize / 32, aVolSize / 16, aVolSize)
{
	hipModule_t kernelModule = ctx->LoadModule("basicKernels.ptx");
	
	binarize = new HipKernel("binarize", kernelModule);
}


void HipBinarize::runBinarizeKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(binarize->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipBinarize::Binarize(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	runBinarizeKernel(d_idata, d_odata);
}


HipMul::HipMul(int aVolSize, hipStream_t aStream, Hip::HipContext* context)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1), 
	  gridSize(aVolSize / 32, aVolSize / 16, aVolSize)
{
	hipModule_t kernelModule = ctx->LoadModule("basicKernels.ptx");
	
    mulVol = new HipKernel("mulVol", kernelModule);
    //mulVol_RealReal = new HipKernel("mulVol_RealReal", kernelModule);
    mulmulVol = new HipKernel("mulmulVol", kernelModule);
	mulVolCplx = new HipKernel("mulVolCplx", kernelModule);
    mul = new HipKernel("mul", kernelModule);
    mul_Real = new HipKernel("mul_Real", kernelModule);
}


void HipMul::runMulVolKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(mulVol->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMul::MulVol(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    runMulVolKernel(d_idata, d_odata);
}

void HipMul::runMulVol_RealRealKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(mulVol_RealReal->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMul::MulVol_RealReal(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    runMulVol_RealRealKernel(d_idata, d_odata);
}

void HipMul::runMulMulVolKernel(float val, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    //printf("Mark3");
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &volSize;
    arglist[1] = &val;
    arglist[2] = &in_dptr;
    arglist[3] = &out_dptr;
    //printf("Mark4");

    hipSafeCall(hipModuleLaunchKernel(mulmulVol->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMul::MulMulVol(float val, HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    //printf("Mark2");
    runMulMulVolKernel(val, d_idata, d_odata);
}



void HipMul::runMulVolCplxKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(mulVol->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMul::MulVolCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	runMulVolCplxKernel(d_idata, d_odata);
}


void HipMul::runMulKernel(float val, HipDeviceVariable& d_odata)
{
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &val;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(mul->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMul::Mul(float val, HipDeviceVariable& d_odata)
{
    runMulKernel(val, d_odata);
}


void HipMul::runMul_RealKernel(float val, HipDeviceVariable& d_odata)
{
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &val;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(mul_Real->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMul::Mul_Real(float val, HipDeviceVariable& d_odata)
{
    runMul_RealKernel(val, d_odata);
}


HipFFT::HipFFT(int aVolSize, hipStream_t aStream, Hip::HipContext* context)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1), 
	  gridSize(aVolSize / 32, aVolSize / 16, aVolSize)
{
	hipModule_t kernelModule = ctx->LoadModule("basicKernels.ptx");
	
	conv = new HipKernel("conv", kernelModule);
	correl = new HipKernel("correl", kernelModule);
	bandpass = new HipKernel("bandpass", kernelModule);
	bandpassFFTShift = new HipKernel("bandpassFFTShift", kernelModule);
	fftshiftReal = new HipKernel("fftshiftReal", kernelModule);
	fftshift = new HipKernel("fftshift", kernelModule);
	fftshift2 = new HipKernel("fftshift2", kernelModule);
	energynorm = new HipKernel("energynorm", kernelModule);
	energynorm_RC = new HipKernel("energynorm_RC", kernelModule);
	correlConvConv = new HipKernel("correlConvConv", kernelModule);
}


void HipFFT::runConvKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(conv->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT::Conv(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	runConvKernel(d_idata, d_odata);
}


void HipFFT::runCorrelConvConvKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& d_idata2, HipDeviceVariable& d_odata2, float val)
{
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();
    hipDeviceptr_t in2_dptr = d_idata2.GetDevicePtr();
    hipDeviceptr_t out2_dptr = d_odata2.GetDevicePtr();


    void** arglist = (void**)new void*[6];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &in2_dptr;
    arglist[4] = &out2_dptr;
    arglist[5] = &val;

    hipSafeCall(hipModuleLaunchKernel(correlConvConv->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT::CorrelConvConv(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& d_idata2, HipDeviceVariable& d_odata2, float val)
{
    runCorrelConvConvKernel(d_idata, d_odata, d_idata2, d_odata2, val);
}

void HipFFT::runCorrelKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(correl->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT::Correl(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
	runCorrelKernel(d_idata, d_odata);
}


void HipFFT::runBandpassFFTShiftKernel(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth)
{
	hipDeviceptr_t vol_dptr = d_vol.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &vol_dptr;
    arglist[2] = &rDown;
    arglist[3] = &rUp;
    arglist[4] = &smooth;

    hipSafeCall(hipModuleLaunchKernel(bandpassFFTShift->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT::BandpassFFTShift(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth)
{
	runBandpassFFTShiftKernel(d_vol, rDown, rUp, smooth);
}


void HipFFT::runBandpassKernel(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth)
{
	hipDeviceptr_t vol_dptr = d_vol.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &vol_dptr;
    arglist[2] = &rDown;
    arglist[3] = &rUp;
    arglist[4] = &smooth;

    hipSafeCall(hipModuleLaunchKernel(bandpass->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT::Bandpass(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth)
{
	runBandpassKernel(d_vol, rDown, rUp, smooth);
}


void HipFFT::runFFTShiftKernel(HipDeviceVariable& d_vol)
{
	hipDeviceptr_t vol_dptr = d_vol.GetDevicePtr();

    void** arglist = (void**)new void*[2];

    arglist[0] = &volSize;
    arglist[1] = &vol_dptr;

    hipSafeCall(hipModuleLaunchKernel(fftshift->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT::FFTShift(HipDeviceVariable& d_vol)
{
	runFFTShiftKernel(d_vol);
}


void HipFFT::runFFTShiftKernel2(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut)
{
	hipDeviceptr_t voli_dptr = d_volIn.GetDevicePtr();
	hipDeviceptr_t volo_dptr = d_volOut.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &voli_dptr;
    arglist[2] = &volo_dptr;

    hipSafeCall(hipModuleLaunchKernel(fftshift2->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT::FFTShift2(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut)
{
	runFFTShiftKernel2(d_volIn, d_volOut);
}


void HipFFT::runFFTShiftRealKernel(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut)
{
	hipDeviceptr_t voli_dptr = d_volIn.GetDevicePtr();
	hipDeviceptr_t volo_dptr = d_volOut.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &voli_dptr;
    arglist[2] = &volo_dptr;

    hipSafeCall(hipModuleLaunchKernel(fftshiftReal->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT::FFTShiftReal(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut)
{
	runFFTShiftRealKernel(d_volIn, d_volOut);
}


void HipFFT::runEnergyNorm_RCKernel(HipDeviceVariable& d_particle, HipDeviceVariable& d_partSqr, HipDeviceVariable& d_cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox, HipDeviceVariable& temp, HipDeviceVariable& ccMask)
{
	hipDeviceptr_t particle_dptr = d_particle.GetDevicePtr();
	hipDeviceptr_t partSqr_dptr = d_partSqr.GetDevicePtr();
	hipDeviceptr_t cccMap_dptr = d_cccMap.GetDevicePtr();
	hipDeviceptr_t energyRef_dptr = energyRef.GetDevicePtr();
	hipDeviceptr_t nVox_dptr = nVox.GetDevicePtr();
	hipDeviceptr_t temp_dptr = temp.GetDevicePtr();
    hipDeviceptr_t ccMask_dptr = ccMask.GetDevicePtr();

    void** arglist = (void**)new void*[8];

    arglist[0] = &volSize;
    arglist[1] = &particle_dptr;
    arglist[2] = &partSqr_dptr;
    arglist[3] = &cccMap_dptr;
    arglist[4] = &energyRef_dptr;
    arglist[5] = &nVox_dptr;
    arglist[6] = &temp_dptr;
    arglist[7] = &ccMask_dptr;

    hipSafeCall(hipModuleLaunchKernel(energynorm_RC->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT::EnergyNorm_RC(HipDeviceVariable& d_particle, HipDeviceVariable& d_partSqr, HipDeviceVariable& d_cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox, HipDeviceVariable& temp, HipDeviceVariable& ccMask)
{
	runEnergyNorm_RCKernel(d_particle, d_partSqr, d_cccMap, energyRef, nVox, temp, ccMask);
}


void HipFFT::runEnergyNormKernel(HipDeviceVariable& d_particle, HipDeviceVariable& d_partSqr, HipDeviceVariable& d_cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox, HipDeviceVariable& temp, HipDeviceVariable& ccMask)
{
	hipDeviceptr_t particle_dptr = d_particle.GetDevicePtr();
	hipDeviceptr_t partSqr_dptr = d_partSqr.GetDevicePtr();
	hipDeviceptr_t cccMap_dptr = d_cccMap.GetDevicePtr();
	hipDeviceptr_t energyRef_dptr = energyRef.GetDevicePtr();
	hipDeviceptr_t nVox_dptr = nVox.GetDevicePtr();
    hipDeviceptr_t temp_dptr = temp.GetDevicePtr();
    hipDeviceptr_t ccMask_dptr = ccMask.GetDevicePtr();

    void** arglist = (void**)new void*[8];

    arglist[0] = &volSize;
    arglist[1] = &particle_dptr;
    arglist[2] = &partSqr_dptr;
    arglist[3] = &cccMap_dptr;
    arglist[4] = &energyRef_dptr;
    arglist[5] = &nVox_dptr;
    arglist[6] = &temp_dptr;
    arglist[7] = &ccMask_dptr;

    hipSafeCall(hipModuleLaunchKernel(energynorm->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT::EnergyNorm(HipDeviceVariable& d_particle, HipDeviceVariable& d_partSqr, HipDeviceVariable& d_cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox, HipDeviceVariable& temp, HipDeviceVariable& ccMask)
{
	runEnergyNormKernel(d_particle, d_partSqr, d_cccMap, energyRef, nVox, temp, ccMask);
}


HipMax::HipMax(hipStream_t aStream, Hip::HipContext* context)
	: stream(aStream), ctx(context), blockSize(1, 1, 1), 
	  gridSize(1, 1, 1)
{
	hipModule_t kernelModule = ctx->LoadModule("basicKernels.ptx");
	
	max = new HipKernel("findmax", kernelModule);
}


void HipMax::runMaxKernel(HipDeviceVariable& maxVals, HipDeviceVariable& index, HipDeviceVariable& val, float rphi, float rpsi, float rthe)
{
	hipDeviceptr_t maxVals_dptr = maxVals.GetDevicePtr();
	hipDeviceptr_t index_dptr = index.GetDevicePtr();
	hipDeviceptr_t val_dptr = val.GetDevicePtr();

    void** arglist = (void**)new void*[6];

    arglist[0] = &maxVals_dptr;
    arglist[1] = &index_dptr;
    arglist[2] = &val_dptr;
    arglist[3] = &rphi;
    arglist[4] = &rpsi;
    arglist[5] = &rthe;

    hipSafeCall(hipModuleLaunchKernel(max->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMax::Max(HipDeviceVariable& maxVals, HipDeviceVariable& index, HipDeviceVariable& val, float rphi, float rpsi, float rthe)
{
	runMaxKernel(maxVals, index, val, rphi, rpsi, rthe);
}


HipWedgeNorm::HipWedgeNorm(int aVolSize, hipStream_t aStream, Hip::HipContext* context)
	: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1),
	gridSize(aVolSize / 32, aVolSize / 16, aVolSize)
{
	hipModule_t kernelModule = ctx->LoadModule("basicKernels.ptx");

	wedge = new HipKernel("wedgeNorm", kernelModule);
}


void HipWedgeNorm::runWedgeNormKernel(HipDeviceVariable& d_data, HipDeviceVariable& d_partdata, HipDeviceVariable& d_maxVal, int newMethod)
{
	hipDeviceptr_t in_dptr = d_data.GetDevicePtr();
	hipDeviceptr_t part_dptr = d_partdata.GetDevicePtr();
	hipDeviceptr_t out_dptr = d_maxVal.GetDevicePtr();

	void** arglist = (void**)new void*[5];

	arglist[0] = &volSize;
	arglist[1] = &in_dptr;
	arglist[2] = &d_partdata;
	arglist[3] = &out_dptr;
	arglist[4] = &newMethod;

	hipSafeCall(hipModuleLaunchKernel(wedge->GetHipFunction(), gridSize.x, gridSize.y,
		gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist, NULL));

	delete[] arglist;
}


void HipWedgeNorm::WedgeNorm(HipDeviceVariable& d_data, HipDeviceVariable& d_partdata, HipDeviceVariable& d_maxVal, int newMethod)
{
	runWedgeNormKernel(d_data, d_partdata, d_maxVal, newMethod);
}



/***************************************************************************************************************************

    Real to Complex Replacments

/***************************************************************************************************************************/


HipSub_RC::HipSub_RC(int aVolSize, hipStream_t aStream, Hip::HipContext* context)
    //: volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 32, 1), 
    //gridSize(aVolSize / 32, aVolSize / 32, aVolSize)

    //: volSize(aVolSize), stream(aStream), ctx(context), blockSize(16, 8, 8), 
    //gridSize(aVolSize*aVolSize*aVolSize / 1024, 1, 1)

    : volSize(aVolSize), stream(aStream), ctx(context), blockSize(1, 1, 1), 
    gridSize(aVolSize,aVolSize,aVolSize)
{
    hipModule_t kernelModule = ctx->LoadModule("basicKernels.ptx");
    
    // add = new HipKernel("add", kernelModule);
    sub_RC = new HipKernel("sub_RC", kernelModule);
    sqrsub_RC = new HipKernel("sqrsub_RC", kernelModule);
    subCplx_RC = new HipKernel("subCplx_RC", kernelModule);
    // subCplx2 = new HipKernel("subCplx2", kernelModule);
}

/*
void HipSub_RC::runAddKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(add->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipSub_RC::Add(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    runAddKernel(d_idata, d_odata);
}

*/

void HipSub_RC::runSub_RCKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val)
{
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val;

    hipSafeCall(hipModuleLaunchKernel(sub_RC->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipSub_RC::Sub_RC(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val)
{
    runSub_RCKernel(d_idata, d_odata, val);
}


void HipSub_RC::runSqrSub_RCKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val)
{
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[4];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val;

    hipSafeCall(hipModuleLaunchKernel(sqrsub_RC->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipSub_RC::SqrSub_RC(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, float val)
{
    runSqrSub_RCKernel(d_idata, d_odata, val);
}

/*

void HipSub_RC::runSubCplxKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, float divVal)
{
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();
    hipDeviceptr_t val_dptr = val.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val_dptr;
    arglist[4] = &divVal;

    hipSafeCall(hipModuleLaunchKernel(subCplx->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipSub_RC::SubCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, float divVal)
{
    runSubCplxKernel(d_idata, d_odata, val, divVal);
}

*/

void HipSub_RC::runSubCplx_RCKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, HipDeviceVariable& divVal)
{
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();
    hipDeviceptr_t val_dptr = val.GetDevicePtr();
    hipDeviceptr_t divval_dptr = divVal.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;
    arglist[3] = &val_dptr;
    arglist[4] = &divval_dptr;

    hipSafeCall(hipModuleLaunchKernel(subCplx_RC->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipSub_RC::SubCplx_RC(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata, HipDeviceVariable& val, HipDeviceVariable& divVal)
{
    runSubCplx_RCKernel(d_idata, d_odata, val, divVal);
}


HipMul_RC::HipMul_RC(int aVolSize, hipStream_t aStream, Hip::HipContext* context)
    // : volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1), 
    //   gridSize(aVolSize / 32, aVolSize / 16, aVolSize)

    : volSize(aVolSize), stream(aStream), ctx(context), blockSize(1, 1, 1), 
      gridSize(aVolSize/2+1, aVolSize, aVolSize)
{
    hipModule_t kernelModule = ctx->LoadModule("basicKernels.ptx");
    
    mulVol_RC = new HipKernel("mulVol_RC", kernelModule);
    // mulVolCplx = new HipKernel("mulVolCplx", kernelModule);
    mul_RC = new HipKernel("mul_RC", kernelModule);
    mul_Real = new HipKernel("mul_Real", kernelModule);

}


void HipMul_RC::runMulVol_RCKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(mulVol_RC->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMul_RC::MulVol_RC(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    runMulVol_RCKernel(d_idata, d_odata);
}


/*
void HipMul_RC::runMulVolCplxKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(mulVol->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMul_RC::MulVolCplx(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    runMulVolCplxKernel(d_idata, d_odata);
}

*/
void HipMul_RC::runMul_RCKernel(float val, HipDeviceVariable& d_odata)
{
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &val;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(mul_RC->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMul_RC::Mul_RC(float val, HipDeviceVariable& d_odata)
{
    runMul_RCKernel(val, d_odata);
}


void HipMul_RC::runMul_RealKernel(float val, HipDeviceVariable& d_odata)
{
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &val;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(mul_Real->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipMul_RC::Mul_Real(float val, HipDeviceVariable& d_odata)
{
    runMul_RealKernel(val, d_odata);
}



HipFFT_RC::HipFFT_RC(int aVolSize, hipStream_t aStream, Hip::HipContext* context)
    // : volSize(aVolSize), stream(aStream), ctx(context), blockSize(32, 16, 1), 
    //   gridSize(aVolSize / 32, aVolSize / 16, aVolSize)

    : volSize(aVolSize), stream(aStream), ctx(context), blockSize(1, 1, 1), 
      gridSize(aVolSize/2+1, aVolSize, aVolSize)
{
    hipModule_t kernelModule = ctx->LoadModule("basicKernels.ptx");
    
    conv_RC = new HipKernel("conv_RC", kernelModule);
    correl_RC = new HipKernel("correl_RC", kernelModule);
    /*
    bandpass = new HipKernel("bandpass", kernelModule);
    */
    bandpassFFTShift_RC = new HipKernel("bandpassFFTShift_RC", kernelModule);
    /*
    fftshiftReal = new HipKernel("fftshiftReal", kernelModule);
    fftshift = new HipKernel("fftshift", kernelModule);
    */
    //fftshift2_RC = new HipKernel("fftshift2_RC", kernelModule);
    energynorm_RC = new HipKernel("energynorm_RC", kernelModule);
}



void HipFFT_RC::runConv_RCKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(conv_RC->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT_RC::Conv_RC(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    runConv_RCKernel(d_idata, d_odata);
}


void HipFFT_RC::runCorrel_RCKernel(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    hipDeviceptr_t in_dptr = d_idata.GetDevicePtr();
    hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &in_dptr;
    arglist[2] = &out_dptr;

    hipSafeCall(hipModuleLaunchKernel(correl_RC->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT_RC::Correl_RC(HipDeviceVariable& d_idata, HipDeviceVariable& d_odata)
{
    runCorrel_RCKernel(d_idata, d_odata);
}


void HipFFT_RC::runBandpassFFTShift_RCKernel(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth)
{
    hipDeviceptr_t vol_dptr = d_vol.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &vol_dptr;
    arglist[2] = &rDown;
    arglist[3] = &rUp;
    arglist[4] = &smooth;

    hipSafeCall(hipModuleLaunchKernel(bandpassFFTShift_RC->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT_RC::BandpassFFTShift_RC(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth)
{
    runBandpassFFTShift_RCKernel(d_vol, rDown, rUp, smooth);
}

/*
void HipFFT_RC::runBandpassKernel(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth)
{
    hipDeviceptr_t vol_dptr = d_vol.GetDevicePtr();

    void** arglist = (void**)new void*[5];

    arglist[0] = &volSize;
    arglist[1] = &vol_dptr;
    arglist[2] = &rDown;
    arglist[3] = &rUp;
    arglist[4] = &smooth;

    hipSafeCall(hipModuleLaunchKernel(bandpass->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT_RC::Bandpass(HipDeviceVariable& d_vol, float rDown, float rUp, float smooth)
{
    runBandpassKernel(d_vol, rDown, rUp, smooth);
}


void HipFFT_RC::runFFTShiftKernel(HipDeviceVariable& d_vol)
{
    hipDeviceptr_t vol_dptr = d_vol.GetDevicePtr();

    void** arglist = (void**)new void*[2];

    arglist[0] = &volSize;
    arglist[1] = &vol_dptr;

    hipSafeCall(hipModuleLaunchKernel(fftshift->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT_RC::FFTShift(HipDeviceVariable& d_vol)
{
    runFFTShiftKernel(d_vol);
}

void HipFFT_RC::runFFTShift_RCKernel2(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut)
{
    hipDeviceptr_t voli_dptr = d_volIn.GetDevicePtr();
    hipDeviceptr_t volo_dptr = d_volOut.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &voli_dptr;
    arglist[2] = &volo_dptr;

    hipSafeCall(hipModuleLaunchKernel(fftshift2_RC->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT_RC::FFTShift2_RC(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut)
{
    runFFTShift_RCKernel2(d_volIn, d_volOut);
}

void HipFFT_RC::runFFTShiftRealKernel(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut)
{
    hipDeviceptr_t voli_dptr = d_volIn.GetDevicePtr();
    hipDeviceptr_t volo_dptr = d_volOut.GetDevicePtr();

    void** arglist = (void**)new void*[3];

    arglist[0] = &volSize;
    arglist[1] = &voli_dptr;
    arglist[2] = &volo_dptr;

    hipSafeCall(hipModuleLaunchKernel(fftshiftReal->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT_RC::FFTShiftReal(HipDeviceVariable& d_volIn, HipDeviceVariable& d_volOut)
{
    runFFTShiftRealKernel(d_volIn, d_volOut);
}
*/

void HipFFT_RC::runEnergyNorm_RCKernel(HipDeviceVariable& d_particle, HipDeviceVariable& d_partSqr, HipDeviceVariable& d_cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox, HipDeviceVariable& temp, HipDeviceVariable& ccMask)
{
    hipDeviceptr_t particle_dptr = d_particle.GetDevicePtr();
    hipDeviceptr_t partSqr_dptr = d_partSqr.GetDevicePtr();
    hipDeviceptr_t cccMap_dptr = d_cccMap.GetDevicePtr();
    hipDeviceptr_t energyRef_dptr = energyRef.GetDevicePtr();
    hipDeviceptr_t nVox_dptr = nVox.GetDevicePtr();
    hipDeviceptr_t temp_dptr = temp.GetDevicePtr();
    hipDeviceptr_t ccMask_dptr = ccMask.GetDevicePtr();

    void** arglist = (void**)new void*[8];

    arglist[0] = &volSize;
    arglist[1] = &particle_dptr;
    arglist[2] = &partSqr_dptr;
    arglist[3] = &cccMap_dptr;
    arglist[4] = &energyRef_dptr;
    arglist[5] = &nVox_dptr;
    arglist[6] = &temp_dptr;
    arglist[7] = &ccMask_dptr;

    hipSafeCall(hipModuleLaunchKernel(energynorm_RC->GetHipFunction(), gridSize.x, gridSize.y,
        gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));

    delete[] arglist;
}


void HipFFT_RC::EnergyNorm_RC(HipDeviceVariable& d_particle, HipDeviceVariable& d_partSqr, HipDeviceVariable& d_cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox, HipDeviceVariable& temp, HipDeviceVariable& ccMask)
{
    runEnergyNorm_RCKernel(d_particle, d_partSqr, d_cccMap, energyRef, nVox, temp, ccMask);
}
