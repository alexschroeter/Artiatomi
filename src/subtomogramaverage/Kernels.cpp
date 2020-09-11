#include "Kernels.h"

using namespace Hip;

/*
	* basicKernels+.cu
	*
	*/

SubCplxKernel::SubCplxKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("subCplx2", aModule, aGridDim, aBlockDim, 0)
{

}

SubCplxKernel::SubCplxKernel(hipModule_t aModule)
		: Hip::HipKernel("subCplx2", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float SubCplxKernel::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& sum, HipDeviceVariable& divVal)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		hipDeviceptr_t sum_dptr = sum.GetDevicePtr();
		hipDeviceptr_t divVal_dptr = divVal.GetDevicePtr();

		DevParamSubCplx rp;
		rp.size = size;
		rp.in_dptr = (float2*)input_dptr;
		rp.out_dptr = (float2*)output_dptr;
		rp.val_dptr = (float*)sum_dptr;
		rp.divval_dptr = (float*)divVal_dptr;

		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}


SubCplxKernel_RC::SubCplxKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("subCplx_RC", aModule, aGridDim, aBlockDim, 0)
{

}

SubCplxKernel_RC::SubCplxKernel_RC(hipModule_t aModule)
		: Hip::HipKernel("subCplx_RC", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float SubCplxKernel_RC::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& subval, HipDeviceVariable& divVal)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		hipDeviceptr_t subval_dptr = subval.GetDevicePtr();
		hipDeviceptr_t divVal_dptr = divVal.GetDevicePtr();

		DevParamSubCplx_RC rp;
		rp.size = size;
		rp.in_dptr = (float*)input_dptr;
		rp.out_dptr = (float*)output_dptr;
		rp.subval_dptr = (float*)subval_dptr;
		rp.divval_dptr = (float*)divVal_dptr;

		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}


FFTShiftRealKernel::FFTShiftRealKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("fftshiftReal", aModule, aGridDim, aBlockDim, 0)
{

}

FFTShiftRealKernel::FFTShiftRealKernel(hipModule_t aModule)
		: Hip::HipKernel("fftshiftReal", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float FFTShiftRealKernel::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();

		DevParamFFTShiftReal rp;
		rp.size = size;
		rp.in_dptr = (float*) input_dptr;
		rp.out_dptr = (float*) output_dptr;

		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}


MakeCplxWithSubKernel::MakeCplxWithSubKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("makeCplxWithSub", aModule, aGridDim, aBlockDim, 0)
{

}

MakeCplxWithSubKernel::MakeCplxWithSubKernel(hipModule_t aModule)
		: Hip::HipKernel("makeCplxWithSub", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float MakeCplxWithSubKernel::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, float val)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();

		DevParamMakeCplxWithSub rp;
		rp.size = size;
		rp.in_dptr = (float*) input_dptr;
		rp.out_dptr = (float2*) output_dptr;
		rp.val = val;

		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}

/* AS Simple MulVol Kernel */
MulVolKernel::MulVolKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("mulVol", aModule, aGridDim, aBlockDim, 0)
{

}

MulVolKernel::MulVolKernel(hipModule_t aModule)
		: Hip::HipKernel("mulVol", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float MulVolKernel::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();

		DevParamMulVol rp;
		rp.size = size;
		rp.in_dptr = (float*) input_dptr;
		rp.out_dptr = (float2*) output_dptr;

		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}

/* AS Simple MulVol Kernel Real2Real*/
MulVolKernel_RR::MulVolKernel_RR(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("mulVol_RR", aModule, aGridDim, aBlockDim, 0)
{

}

MulVolKernel_RR::MulVolKernel_RR(hipModule_t aModule)
		: Hip::HipKernel("mulVol_RR", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float MulVolKernel_RR::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();

		DevParamFFTShiftReal rp;
		rp.size = size;
		rp.in_dptr = (float*) input_dptr;
		rp.out_dptr = (float*) output_dptr;

		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}

/* AS Simple MulVol Kernel Real2Complex*/
MulVolKernel_RC::MulVolKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("mulVol_RC", aModule, aGridDim, aBlockDim, 0)
{

}

MulVolKernel_RC::MulVolKernel_RC(hipModule_t aModule)
		: Hip::HipKernel("mulVol_RC", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float MulVolKernel_RC::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();

		DevParamFFTShiftReal rp;
		rp.size = size;
		rp.in_dptr = (float*) input_dptr;
		rp.out_dptr = (float*) output_dptr;

		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}


BandpassFFTShiftKernel::BandpassFFTShiftKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("bandpassFFTShift", aModule, aGridDim, aBlockDim, 0)
{

}

BandpassFFTShiftKernel::BandpassFFTShiftKernel(hipModule_t aModule)
		: Hip::HipKernel("bandpassFFTShift", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float BandpassFFTShiftKernel::operator()(int size, HipDeviceVariable& input, float rDown, float rUp, float smooth)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();

		DevParamBandpassFFTShift rp;
		rp.size = size;
		rp.in_dptr = (float2*) input_dptr;
		//rp.out_dptr = (float*) output_dptr;
		rp.rDown = rDown;
		rp.rUp = rUp;
		rp.smooth = smooth;

		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}


BandpassFFTShiftKernel_RC::BandpassFFTShiftKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("bandpassFFTShift_RC", aModule, aGridDim, aBlockDim, 0)
{

}

BandpassFFTShiftKernel_RC::BandpassFFTShiftKernel_RC(hipModule_t aModule)
		: Hip::HipKernel("bandpassFFTShift_RC", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float BandpassFFTShiftKernel_RC::operator()(int size, HipDeviceVariable& input, float rDown, float rUp, float smooth)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		
		DevParamBandpassFFTShift rp;
		rp.size = size;
		rp.in_dptr = (float2*) input_dptr;
		//rp.out_dptr = (float*) output_dptr;
		rp.rDown = rDown;
		rp.rUp = rUp;
		rp.smooth = smooth;

		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}


MakeRealKernel::MakeRealKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("makeReal", aModule, aGridDim, aBlockDim, 0)
{

}

MakeRealKernel::MakeRealKernel(hipModule_t aModule)
		: Hip::HipKernel("makeReal", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float MakeRealKernel::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParamMakeReal rp;
		rp.size = size;
		rp.in_dptr = (float2*) input_dptr;
		rp.out_dptr = (float*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}


MulKernel::MulKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("mul", aModule, aGridDim, aBlockDim, 0)
{

}

MulKernel::MulKernel(hipModule_t aModule)
		: Hip::HipKernel("mul", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}


float MulKernel::operator()(int size, float val, HipDeviceVariable& output)
{
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParamMul rp;
		rp.size = size;
		rp.val = val;
		rp.out_dptr = (float2*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}

MulKernel_RC::MulKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("mul_RC", aModule, aGridDim, aBlockDim, 0)
{

}

MulKernel_RC::MulKernel_RC(hipModule_t aModule)
		: Hip::HipKernel("mul_RC", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float MulKernel_RC::operator()(int size, float val, HipDeviceVariable& output)
{
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParamMul rp;
		rp.size = size;
		rp.val = val;
		rp.out_dptr = (float2*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}

MulKernel_Real::MulKernel_Real(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("mul_Real", aModule, aGridDim, aBlockDim, 0)
{

}

MulKernel_Real::MulKernel_Real(hipModule_t aModule)
		: Hip::HipKernel("mul_Real", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float MulKernel_Real::operator()(int size, float val, HipDeviceVariable& output)
{
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParamMul_Real rp;
		rp.size = size;
		rp.val = val;
		rp.out_dptr = (float*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}

MakeCplxWithSqrSubKernel::MakeCplxWithSqrSubKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("makeCplxWithSquareAndSub", aModule, aGridDim, aBlockDim, 0)
{

}

MakeCplxWithSqrSubKernel::MakeCplxWithSqrSubKernel(hipModule_t aModule)
		: Hip::HipKernel("makeCplxWithSquareAndSub", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float MakeCplxWithSqrSubKernel::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, float val)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParamakeCplxWithSqrSub rp;
		rp.size = size;
		rp.in_dptr = (float*) input_dptr;
		rp.out_dptr = (float2*) output_dptr;
		rp.val = val;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}


SqrSubKernel_RC::SqrSubKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("sqrsub_RC", aModule, aGridDim, aBlockDim, 0)
{

}

SqrSubKernel_RC::SqrSubKernel_RC(hipModule_t aModule)
		: Hip::HipKernel("sqrsub_RC", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float SqrSubKernel_RC::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& h_sum, float val)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		hipDeviceptr_t h_sum_dptr = h_sum.GetDevicePtr();

		DevParaSub_RC rp;
		rp.size = size;
		rp.in_dptr = (float*) input_dptr;
		rp.out_dptr = (float*) output_dptr;
		rp.h_sum = (float*) h_sum_dptr;
		rp.val = val;

		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}


SubKernel_RC::SubKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("sub_RC", aModule, aGridDim, aBlockDim, 0)
{

}

SubKernel_RC::SubKernel_RC(hipModule_t aModule)
		: Hip::HipKernel("sub_RC", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float SubKernel_RC::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& h_sum, float val)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		hipDeviceptr_t h_sum_dptr = h_sum.GetDevicePtr();

		DevParaSub_RC rp;
		rp.size = size;
		rp.in_dptr = (float*) input_dptr;
		rp.out_dptr = (float*) output_dptr;
		rp.h_sum = (float*) h_sum_dptr;
		rp.val = val;

		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}



CorrelKernel::CorrelKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("correl", aModule, aGridDim, aBlockDim, 0)
{

}

CorrelKernel::CorrelKernel(hipModule_t aModule)
		: Hip::HipKernel("correl", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}


float CorrelKernel::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParaCorrel rp;
		rp.size = size;
		rp.in_dptr = (float2*) input_dptr;
		rp.out_dptr = (float2*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}


CorrelKernel_RC::CorrelKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("correl_RC", aModule, aGridDim, aBlockDim, 0)
{

}

CorrelKernel_RC::CorrelKernel_RC(hipModule_t aModule)
		: Hip::HipKernel("correl_RC", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float CorrelKernel_RC::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParaCorrel rp;
		rp.size = size;
		rp.in_dptr = (float2*) input_dptr;
		rp.out_dptr = (float2*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}



ConvKernel::ConvKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("conv", aModule, aGridDim, aBlockDim, 0)
{

}

ConvKernel::ConvKernel(hipModule_t aModule)
		: Hip::HipKernel("conv", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float ConvKernel::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParaConv rp;
		rp.size = size;
		rp.in_dptr = (float2*) input_dptr;
		rp.out_dptr = (float2*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}

ConvKernel_RC::ConvKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("conv_RC", aModule, aGridDim, aBlockDim, 0)
{

}

ConvKernel_RC::ConvKernel_RC(hipModule_t aModule)
		: Hip::HipKernel("conv_RC", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float ConvKernel_RC::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParaConv rp;
		rp.size = size;
		rp.in_dptr = (float2*) input_dptr;
		rp.out_dptr = (float2*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}

SubKernel::SubKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("substract", aModule, aGridDim, aBlockDim, 0)
{

}

SubKernel::SubKernel(hipModule_t aModule)
		: Hip::HipKernel("substract", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float SubKernel::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output, float val)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParaSub rp;
		rp.size = size;
		rp.in_dptr = (float*) input_dptr;
		rp.out_dptr = (float*) output_dptr;
		rp.val = val;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}



EnergyNormKernel::EnergyNormKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("energynorm", aModule, aGridDim, aBlockDim, 0)
{

}

EnergyNormKernel::EnergyNormKernel(hipModule_t aModule)
		: Hip::HipKernel("energynorm", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float EnergyNormKernel::operator()(int size, HipDeviceVariable& particle, HipDeviceVariable& partSqr, HipDeviceVariable& cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox)
{
		hipDeviceptr_t particle_dptr = particle.GetDevicePtr();
		hipDeviceptr_t partSqr_dptr = partSqr.GetDevicePtr();
		hipDeviceptr_t cccMap_dptr = cccMap.GetDevicePtr();
		hipDeviceptr_t energyRef_dptr = energyRef.GetDevicePtr();
		hipDeviceptr_t nVox_dptr = nVox.GetDevicePtr();
		
		DevParaEnergynorm rp;
		rp.size = size;
		rp.in_dptr = (float2*) particle_dptr;
		rp.out_dptr = (float2*) partSqr_dptr;
		rp.cccMap_dptr = (float2*) cccMap_dptr;
		rp.energyRef = (float*) energyRef_dptr;
		rp.nVox = (float*) nVox_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}



EnergyNormKernel_RC::EnergyNormKernel_RC(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("energynorm_RC", aModule, aGridDim, aBlockDim, 0)
{

}

EnergyNormKernel_RC::EnergyNormKernel_RC(hipModule_t aModule)
		: Hip::HipKernel("energynorm_RC", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float EnergyNormKernel_RC::operator()(int size, HipDeviceVariable& particle, HipDeviceVariable& partSqr, HipDeviceVariable& cccMap, HipDeviceVariable& energyRef, HipDeviceVariable& nVox, HipDeviceVariable& temp, HipDeviceVariable& ccMask)
{
		hipDeviceptr_t particle_dptr = particle.GetDevicePtr();
		hipDeviceptr_t partSqr_dptr = partSqr.GetDevicePtr();
		hipDeviceptr_t cccMap_dptr = cccMap.GetDevicePtr();
		hipDeviceptr_t energyRef_dptr = energyRef.GetDevicePtr();
		hipDeviceptr_t nVox_dptr = nVox.GetDevicePtr();
		hipDeviceptr_t temp_dptr = temp.GetDevicePtr();
		hipDeviceptr_t ccMask_dptr = ccMask.GetDevicePtr();
		
		DevParaEnergynorm_RC rp;
		rp.size = size;
		rp.in_dptr = (float*) particle_dptr;
		rp.out_dptr = (float*) partSqr_dptr;
		rp.cccMap_dptr = (float*) cccMap_dptr;
		rp.energyRef = (float*) energyRef_dptr;
		rp.nVox = (float*) nVox_dptr;
		rp.temp = (float2*) temp_dptr;
		rp.ccMask = (float*) ccMask_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}



FFTShift2Kernel::FFTShift2Kernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("fftshift2", aModule, aGridDim, aBlockDim, 0)
{

}

FFTShift2Kernel::FFTShift2Kernel(hipModule_t aModule)
		: Hip::HipKernel("fftshift2", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}

float FFTShift2Kernel::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParaFFTShift2 rp;
		rp.size = size;
		rp.in_dptr = (float2*) input_dptr;
		rp.out_dptr = (float2*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}




BinarizeKernel::BinarizeKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("binarize", aModule, aGridDim, aBlockDim, 0)
{

}


BinarizeKernel::BinarizeKernel(hipModule_t aModule)
		: Hip::HipKernel("binarize", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}


float BinarizeKernel::operator()(int size, HipDeviceVariable& input, HipDeviceVariable& output)
{
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParaBinarize rp;
		rp.size = size;
		rp.in_dptr = (float*) input_dptr;
		rp.out_dptr = (float*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}


MaxKernel::MaxKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("findmax", aModule, aGridDim, aBlockDim, 0)
{

}


MaxKernel::MaxKernel(hipModule_t aModule)
		: Hip::HipKernel("findmax", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{

}


float MaxKernel::operator()(HipDeviceVariable& maxVal, HipDeviceVariable& index, HipDeviceVariable& val, float rphi, float rpsi, float rthe)
{
		hipDeviceptr_t maxVal_dptr = maxVal.GetDevicePtr();
		hipDeviceptr_t index_dptr = index.GetDevicePtr();
		hipDeviceptr_t val_dptr = val.GetDevicePtr();  
		
		DevParaMax rp;
		rp.maxVals = (float*) maxVal_dptr;
		rp.index = (float*) index_dptr;
		rp.val = (float*) val_dptr;
		rp.rpsi = rpsi;
		rp.rphi = rphi;
		rp.rthe = rthe;

		float ms = Launch( &rp, sizeof(rp) );
		return ms;
}


/*
	* kernel.cu
	*
	*
	*/

Reducer::Reducer(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim)
		: Hip::HipKernel("_Z6reduceILj1EEv10DevParaSum", aModule, aGridDim, aBlockDim, 0)
{  
		hipSafeCall(hipModuleGetFunction(&sum512, mModule, "_Z6reduceILj512EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum256, mModule, "_Z6reduceILj256EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum128, mModule, "_Z6reduceILj128EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum64, mModule, "_Z6reduceILj64EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum32, mModule, "_Z6reduceILj32EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum16, mModule, "_Z6reduceILj16EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum8, mModule, "_Z6reduceILj8EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum4, mModule, "_Z6reduceILj4EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum2, mModule, "_Z6reduceILj2EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum1, mModule, "_Z6reduceILj1EEv10DevParaSum"));

		hipSafeCall(hipModuleGetFunction(&sumSqr512, mModule, "_Z9reduceSqrILj512EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr256, mModule, "_Z9reduceSqrILj256EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr128, mModule, "_Z6reduceILj128EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr64, mModule, "_Z9reduceSqrILj64EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr32, mModule, "_Z9reduceSqrILj32EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr16, mModule, "_Z9reduceSqrILj16EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr8, mModule, "_Z9reduceSqrILj8EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr4, mModule, "_Z9reduceSqrILj4EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr2, mModule, "_Z9reduceSqrILj2EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr1, mModule, "_Z9reduceSqrILj1EEv10DevParaSum"));

		hipSafeCall(hipModuleGetFunction(&sumSqrCplx512, mModule, "_Z13reduceSqrCplxILj512EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx256, mModule, "_Z13reduceSqrCplxILj256EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx128, mModule, "_Z13reduceSqrCplxILj128EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx64, mModule, "_Z13reduceSqrCplxILj64EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx32, mModule, "_Z13reduceSqrCplxILj32EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx16, mModule, "_Z13reduceSqrCplxILj16EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx8, mModule, "_Z13reduceSqrCplxILj8EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx4, mModule, "_Z13reduceSqrCplxILj4EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx2, mModule, "_Z13reduceSqrCplxILj2EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx1, mModule, "_Z13reduceSqrCplxILj1EEv17DevParaSumSqrCplx"));

		hipSafeCall(hipModuleGetFunction(&sumCplx512, mModule, "_Z10reduceCplxILj512EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx256, mModule, "_Z10reduceCplxILj256EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx128, mModule, "_Z10reduceCplxILj128EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx64, mModule, "_Z10reduceCplxILj64EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx32, mModule, "_Z10reduceCplxILj32EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx16, mModule, "_Z10reduceCplxILj16EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx8, mModule, "_Z10reduceCplxILj8EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx4, mModule, "_Z10reduceCplxILj4EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx2, mModule, "_Z10reduceCplxILj2EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx1, mModule, "_Z10reduceCplxILj1EEv14DevParaSumCplx"));

		hipSafeCall(hipModuleGetFunction(&maxIndexCplx512, mModule, "_Z12maxIndexCplxILj512EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx256, mModule, "_Z12maxIndexCplxILj256EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx128, mModule, "_Z12maxIndexCplxILj128EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx64, mModule, "_Z12maxIndexCplxILj64EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx32, mModule, "_Z12maxIndexCplxILj32EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx16, mModule, "_Z12maxIndexCplxILj16EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx8, mModule, "_Z12maxIndexCplxILj8EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx4, mModule, "_Z12maxIndexCplxILj4EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx2, mModule, "_Z12maxIndexCplxILj2EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx1, mModule, "_Z12maxIndexCplxILj1EEv19DevParaMaxIndexCplx"));

		hipSafeCall(hipModuleGetFunction(&maxIndex512, mModule, "_Z8maxIndexILj512EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex256, mModule, "_Z8maxIndexILj256EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex128, mModule, "_Z8maxIndexILj128EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex64, mModule, "_Z8maxIndexILj64EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex32, mModule, "_Z8maxIndexILj32EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex16, mModule, "_Z8maxIndexILj16EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex8, mModule, "_Z8maxIndexILj8EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex4, mModule, "_Z8maxIndexILj4EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex2, mModule, "_Z8maxIndexILj2EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex1, mModule, "_Z8maxIndexILj1EEv15DevParaMaxIndex"));
}

Reducer::Reducer(hipModule_t aModule)
		: Hip::HipKernel("_Z6reduceILj1EEv10DevParaSum", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0)
{  
		hipSafeCall(hipModuleGetFunction(&sum512, mModule, "_Z6reduceILj512EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum256, mModule, "_Z6reduceILj256EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum128, mModule, "_Z6reduceILj128EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum64, mModule, "_Z6reduceILj64EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum32, mModule, "_Z6reduceILj32EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum16, mModule, "_Z6reduceILj16EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum8, mModule, "_Z6reduceILj8EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum4, mModule, "_Z6reduceILj4EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum2, mModule, "_Z6reduceILj2EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sum1, mModule, "_Z6reduceILj1EEv10DevParaSum"));

		hipSafeCall(hipModuleGetFunction(&sumSqr512, mModule, "_Z9reduceSqrILj512EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr256, mModule, "_Z9reduceSqrILj256EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr128, mModule, "_Z6reduceILj128EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr64, mModule, "_Z9reduceSqrILj64EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr32, mModule, "_Z9reduceSqrILj32EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr16, mModule, "_Z9reduceSqrILj16EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr8, mModule, "_Z9reduceSqrILj8EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr4, mModule, "_Z9reduceSqrILj4EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr2, mModule, "_Z9reduceSqrILj2EEv10DevParaSum"));
		hipSafeCall(hipModuleGetFunction(&sumSqr1, mModule, "_Z9reduceSqrILj1EEv10DevParaSum"));

		hipSafeCall(hipModuleGetFunction(&sumSqrCplx512, mModule, "_Z13reduceSqrCplxILj512EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx256, mModule, "_Z13reduceSqrCplxILj256EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx128, mModule, "_Z13reduceSqrCplxILj128EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx64, mModule, "_Z13reduceSqrCplxILj64EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx32, mModule, "_Z13reduceSqrCplxILj32EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx16, mModule, "_Z13reduceSqrCplxILj16EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx8, mModule, "_Z13reduceSqrCplxILj8EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx4, mModule, "_Z13reduceSqrCplxILj4EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx2, mModule, "_Z13reduceSqrCplxILj2EEv17DevParaSumSqrCplx"));
		hipSafeCall(hipModuleGetFunction(&sumSqrCplx1, mModule, "_Z13reduceSqrCplxILj1EEv17DevParaSumSqrCplx"));

		hipSafeCall(hipModuleGetFunction(&sumCplx512, mModule, "_Z10reduceCplxILj512EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx256, mModule, "_Z10reduceCplxILj256EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx128, mModule, "_Z10reduceCplxILj128EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx64, mModule, "_Z10reduceCplxILj64EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx32, mModule, "_Z10reduceCplxILj32EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx16, mModule, "_Z10reduceCplxILj16EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx8, mModule, "_Z10reduceCplxILj8EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx4, mModule, "_Z10reduceCplxILj4EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx2, mModule, "_Z10reduceCplxILj2EEv14DevParaSumCplx"));
		hipSafeCall(hipModuleGetFunction(&sumCplx1, mModule, "_Z10reduceCplxILj1EEv14DevParaSumCplx"));

		hipSafeCall(hipModuleGetFunction(&maxIndexCplx512, mModule, "_Z12maxIndexCplxILj512EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx256, mModule, "_Z12maxIndexCplxILj256EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx128, mModule, "_Z12maxIndexCplxILj128EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx64, mModule, "_Z12maxIndexCplxILj64EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx32, mModule, "_Z12maxIndexCplxILj32EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx16, mModule, "_Z12maxIndexCplxILj16EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx8, mModule, "_Z12maxIndexCplxILj8EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx4, mModule, "_Z12maxIndexCplxILj4EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx2, mModule, "_Z12maxIndexCplxILj2EEv19DevParaMaxIndexCplx"));
		hipSafeCall(hipModuleGetFunction(&maxIndexCplx1, mModule, "_Z12maxIndexCplxILj1EEv19DevParaMaxIndexCplx"));

		hipSafeCall(hipModuleGetFunction(&maxIndex512, mModule, "_Z8maxIndexILj512EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex256, mModule, "_Z8maxIndexILj256EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex128, mModule, "_Z8maxIndexILj128EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex64, mModule, "_Z8maxIndexILj64EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex32, mModule, "_Z8maxIndexILj32EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex16, mModule, "_Z8maxIndexILj16EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex8, mModule, "_Z8maxIndexILj8EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex4, mModule, "_Z8maxIndexILj4EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex2, mModule, "_Z8maxIndexILj2EEv15DevParaMaxIndex"));
		hipSafeCall(hipModuleGetFunction(&maxIndex1, mModule, "_Z8maxIndexILj1EEv15DevParaMaxIndex"));
		}

float Reducer::sum(HipDeviceVariable& input, HipDeviceVariable& output, int sizeTot)
{
		float ms = 0;
		int blocks, threads;
		float gpu_result = 0;
		// bool needReadBack = true; // AS deprecated always true never changes

		getNumBlocksAndThreads(sizeTot, blocks, threads);
		runSumKernel(sizeTot, blocks, threads, input, output); // execute the kernel
		int s=blocks; // sum partial block sums on GPU

		while (s > 1)
		{
						int threads = 0, blocks = 0;
						getNumBlocksAndThreads(s, blocks, threads);

						runSumKernel(s, blocks, threads, output, output);
						s = (s + (threads*2-1)) / (threads*2);
		}

		if (s > 1)
				printf("Oops, not a power of 2?\n");

		/* AS deprecated always true never changes
		if (needReadBack){} 
		*/
		return ms;
}


void Reducer::runSumKernel(int size, int blocks, int threads, HipDeviceVariable& input, HipDeviceVariable& output)
{
				// when there is only one warp per block, we need to allocate two warps
				// worth of shared memory so that we don't index shared memory out of bounds
				int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

				SetBlockDimensions(threads, 1, 1);
				SetGridDimensions(blocks, 1, 1);
				SetDynamicSharedMemory(smemSize);
				switch (threads)
				{
								case 512:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj512EEv10DevParaSum"));
										mFunction = sum512; break;
								case 256:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj256EEv10DevParaSum"));
										mFunction = sum256; break;
								case 128:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj128EEv10DevParaSum"));
										mFunction = sum128; break;
								case 64:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj64EEv10DevParaSum"));
										mFunction = sum64; break;
								case 32:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj32EEv10DevParaSum"));
										mFunction = sum32; break;
								case 16:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj16EEv10DevParaSum"));
										mFunction = sum16; break;
								case  8:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj8EEv10DevParaSum"));
										mFunction = sum8; break;
								case  4:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj4EEv10DevParaSum"));
										mFunction = sum4; break;
								case  2:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj2EEv10DevParaSum"));
										mFunction = sum2; break;
								case  1:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj1EEv10DevParaSum"));
										mFunction = sum1; break;
				}
				
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParaSum rp;
		rp.size = size;
		rp.in_dptr = (float*) input_dptr;
		rp.out_dptr = (float*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
}


float Reducer::sumsqrcplx(HipDeviceVariable& input, HipDeviceVariable& output, int sizeTot)
{
		int blocks;
		int threads;
		float gpu_result = 0;
		bool needReadBack = true;

		gpu_result = 0;


		getNumBlocksAndThreads(sizeTot, blocks, threads);
		// execute the kernel
		runSumSqrCplxKernel(sizeTot, blocks, threads, input, output);

		// sum partial block sums on GPU
		int s=blocks;

		while (s > 1)
		{
						int threads = 0, blocks = 0;
						getNumBlocksAndThreads(s, blocks, threads);

						runSumKernel(s, blocks, threads, output, output);
						s = (s + (threads*2-1)) / (threads*2);
		}

		if (s > 1)
		{
		printf("Oops, not a power of 2?\n");
		}

		if (needReadBack)
		{

		}

		float ms = 0;
		return ms;
}




void Reducer::runSumSqrCplxKernel(int size, int blocks, int threads, HipDeviceVariable& input, HipDeviceVariable& output)
{
				// when there is only one warp per block, we need to allocate two warps
				// worth of shared memory so that we don't index shared memory out of bounds
				int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

				SetBlockDimensions(threads, 1, 1);
				SetGridDimensions(blocks, 1, 1);
				SetDynamicSharedMemory(smemSize);
				switch (threads)
				{
								case 512:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj512EEv10DevParaSum"));
										mFunction = sumSqrCplx512; break;
								case 256:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj256EEv10DevParaSum"));
										mFunction = sumSqrCplx256; break;
								case 128:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj128EEv10DevParaSum"));
										mFunction = sumSqrCplx128; break;
								case 64:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj64EEv10DevParaSum"));
										mFunction = sumSqrCplx64; break;
								case 32:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj32EEv10DevParaSum"));
										mFunction = sumSqrCplx32; break;
								case 16:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj16EEv10DevParaSum"));
										mFunction = sumSqrCplx16; break;
								case  8:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj8EEv10DevParaSum"));
										mFunction = sumSqrCplx8; break;
								case  4:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj4EEv10DevParaSum"));
										mFunction = sumSqrCplx4; break;
								case  2:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj2EEv10DevParaSum"));
										mFunction = sumSqrCplx2; break;
								case  1:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj1EEv10DevParaSum"));
										mFunction = sumSqrCplx1; break;
				}
				
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParaSumSqrCplx rp;
		rp.size = size;
		rp.in_dptr = (float2*) input_dptr;
		rp.out_dptr = (float*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
}


float Reducer::sumsqr(HipDeviceVariable& input, HipDeviceVariable& output, int sizeTot)
{
		//cout << "Summation running" << endl;

		int blocks;
		int threads;
		float gpu_result = 0;
		bool needReadBack = true;

		gpu_result = 0;


		getNumBlocksAndThreads(sizeTot, blocks, threads);
		// execute the kernel
		runSumSqrKernel(sizeTot, blocks, threads, input, output);

		// sum partial block sums on GPU
		int s=blocks;

		while (s > 1)
		{
						int threads = 0, blocks = 0;
						getNumBlocksAndThreads(s, blocks, threads);

						runSumKernel(s, blocks, threads, output, output);
						s = (s + (threads*2-1)) / (threads*2);
		}

		if (s > 1)
		{
		printf("Oops, not a power of 2?\n");
		}

		if (needReadBack)
		{

		}


		float ms = 0;
		//cout << "Summation ended" << endl;
		return ms;
}



void Reducer::runSumSqrKernel(int size, int blocks, int threads, HipDeviceVariable& input, HipDeviceVariable& output)
{
				// when there is only one warp per block, we need to allocate two warps
				// worth of shared memory so that we don't index shared memory out of bounds
				int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

				SetBlockDimensions(threads, 1, 1);
				SetGridDimensions(blocks, 1, 1);
				SetDynamicSharedMemory(smemSize);
				switch (threads)
				{
								case 512:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj512EEv10DevParaSum"));
										mFunction = sumSqr512; break;
								case 256:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj256EEv10DevParaSum"));
										mFunction = sumSqr256; break;
								case 128:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj128EEv10DevParaSum"));
										mFunction = sumSqr128; break;
								case 64:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj64EEv10DevParaSum"));
										mFunction = sumSqr64; break;
								case 32:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj32EEv10DevParaSum"));
										mFunction = sumSqr32; break;
								case 16:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj16EEv10DevParaSum"));
										mFunction = sumSqr16; break;
								case  8:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj8EEv10DevParaSum"));
										mFunction = sumSqr8; break;
								case  4:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj4EEv10DevParaSum"));
										mFunction = sumSqr4; break;
								case  2:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj2EEv10DevParaSum"));
										mFunction = sumSqr2; break;
								case  1:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj1EEv10DevParaSum"));
										mFunction = sumSqr1; break;
				}
				
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParaSumSqr rp;
		rp.size = size;
		rp.in_dptr = (float*) input_dptr;
		rp.out_dptr = (float*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
}


float Reducer::sumcplx(HipDeviceVariable& input, HipDeviceVariable& output, int sizeTot)
{
		int blocks;
		int threads;
		float gpu_result = 0;
		bool needReadBack = true;

		gpu_result = 0;

		getNumBlocksAndThreads(sizeTot, blocks, threads);
		// execute the kernel
		runSumCplxKernel(sizeTot, blocks, threads, input, output);

		// sum partial block sums on GPU
		int s=blocks;

		while (s > 1)
		{
						int threads = 0, blocks = 0;
						getNumBlocksAndThreads(s, blocks, threads);

						runSumKernel(s, blocks, threads, output, output);
						s = (s + (threads*2-1)) / (threads*2);
		}

		if (s > 1)
		{
		printf("Oops, not a power of 2?\n");
		}

		if (needReadBack)
		{

		}

		float ms = 0;
		return ms;
}


void Reducer::runSumCplxKernel(int size, int blocks, int threads, HipDeviceVariable& input, HipDeviceVariable& output)
{
				// when there is only one warp per block, we need to allocate two warps
				// worth of shared memory so that we don't index shared memory out of bounds
				int smemSize = (threads <= 32) ? 2 * threads * sizeof(float) : threads * sizeof(float);

				SetBlockDimensions(threads, 1, 1);
				SetGridDimensions(blocks, 1, 1);
				SetDynamicSharedMemory(smemSize);
				switch (threads)
				{
								case 512:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj512EEv10DevParaSum"));
										mFunction = sumCplx512; break;
								case 256:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj256EEv10DevParaSum"));
										mFunction = sumCplx256; break;
								case 128:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj128EEv10DevParaSum"));
										mFunction = sumCplx128; break;
								case 64:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj64EEv10DevParaSum"));
										mFunction = sumCplx64; break;
								case 32:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj32EEv10DevParaSum"));
										mFunction = sumCplx32; break;
								case 16:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj16EEv10DevParaSum"));
										mFunction = sumCplx16; break;
								case  8:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj8EEv10DevParaSum"));
										mFunction = sumCplx8; break;
								case  4:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj4EEv10DevParaSum"));
										mFunction = sumCplx4; break;
								case  2:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj2EEv10DevParaSum"));
										mFunction = sumCplx2; break;
								case  1:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj1EEv10DevParaSum"));
										mFunction = sumCplx1; break;
				}
				
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		
		DevParaSumCplx rp;
		rp.size = size;
		rp.in_dptr = (float2*) input_dptr;
		rp.out_dptr = (float*) output_dptr;
		
		float ms = Launch( &rp, sizeof(rp) );
}


float Reducer::maxindex(HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& d_index, int size)
{
		int blocks;
		int threads;
		float gpu_result = 0;
				bool needReadBack = true;

				gpu_result = 0;

		
				getNumBlocksAndThreads(size, blocks, threads);
				// execute the kernel
				runMaxIndexKernel(size, blocks, threads, input, output, d_index, false);

				// sum partial block sums on GPU
				int s=blocks;

				while (s > 1)
				{
								int threads = 0, blocks = 0;
								getNumBlocksAndThreads(s, blocks, threads);

								runMaxIndexKernel(s, blocks, threads, output, output, d_index, true);
				s = (s + (threads*2-1)) / (threads*2);
				}

				if (s > 1)
				{
				printf("Oops, not a power of 2?\n");

				}

				if (needReadBack)
				{

				}

}

float Reducer::maxindexcplx(HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& d_index, int size)
{
		int blocks;
		int threads;
		float gpu_result = 0;
		bool needReadBack = true;

		gpu_result = 0;

		getNumBlocksAndThreads(size, blocks, threads);
		// execute the kernel
		runMaxIndexCplxKernel(size, blocks, threads, input, output, d_index, false);

		// sum partial block sums on GPU
		int s=blocks;

		while (s > 1)
		{
						int threads = 0, blocks = 0;
						getNumBlocksAndThreads(s, blocks, threads);

						runMaxIndexKernel(s, blocks, threads, output, output, d_index, true);
		s = (s + (threads*2-1)) / (threads*2);
		}

		if (s > 1)
		{
				printf("Oops, not a power of 2?\n");
		}

		if (needReadBack)
		{

		}

}

void Reducer::runMaxIndexCplxKernel(int size, int blocks, int threads, HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& index, bool readIndex)
{
				// when there is only one warp per block, we need to allocate two warps
				// worth of shared memory so that we don't index shared memory out of bounds
				int smemSize = (threads <= 32) ? 4 * threads * sizeof(float) : 2 * threads * sizeof(float);

				SetBlockDimensions(threads, 1, 1);
				SetGridDimensions(blocks, 1, 1);
				SetDynamicSharedMemory(smemSize);
				switch (threads)
				{
								case 512:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj512EEv10DevParaSum"));
										mFunction = maxIndexCplx512; break;
								case 256:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj256EEv10DevParaSum"));
										mFunction = maxIndexCplx256; break;
								case 128:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj128EEv10DevParaSum"));
										mFunction = maxIndexCplx128; break;
								case 64:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj64EEv10DevParaSum"));
										mFunction = maxIndexCplx64; break;
								case 32:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj32EEv10DevParaSum"));
										mFunction = maxIndexCplx32; break;
								case 16:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj16EEv10DevParaSum"));
										mFunction = maxIndexCplx16; break;
								case  8:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj8EEv10DevParaSum"));
										mFunction = maxIndexCplx8; break;
								case  4:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj4EEv10DevParaSum"));
										mFunction = maxIndexCplx4; break;
								case  2:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj2EEv10DevParaSum"));
										mFunction = maxIndexCplx2; break;
								case  1:
										//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj1EEv10DevParaSum"));
										mFunction = maxIndexCplx1; break;
				}
				
		hipDeviceptr_t input_dptr = input.GetDevicePtr();
		hipDeviceptr_t output_dptr = output.GetDevicePtr();
		hipDeviceptr_t index_dptr = index.GetDevicePtr();

		DevParaMaxIndexCplx rp;
		rp.size = size;
		rp.in_dptr = (float2*) input_dptr;
		rp.out_dptr = (float*) output_dptr;
		rp.index = (int*) index_dptr;
		rp.readIndex = readIndex;
		
		float ms = Launch( &rp, sizeof(rp) );
}


void Reducer::runMaxIndexKernel(int size, int blocks, int threads, HipDeviceVariable& input, HipDeviceVariable& output, HipDeviceVariable& index, bool readIndex)
{
	// when there is only one warp per block, we need to allocate two warps
	// worth of shared memory so that we don't index shared memory out of bounds
	int smemSize = (threads <= 32) ? 4 * threads * sizeof(float) : 2 * threads * sizeof(float);

	SetBlockDimensions(threads, 1, 1);
	SetGridDimensions(blocks, 1, 1);
	SetDynamicSharedMemory(smemSize);
	switch (threads)
	{
		case 512:
			//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj512EEv10DevParaSum"));
			mFunction = maxIndex512; break;
		case 256:
			//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj256EEv10DevParaSum"));
			mFunction = maxIndex256; break;
		case 128:
			//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj128EEv10DevParaSum"));
			mFunction = maxIndex128; break;
		case 64:
			//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj64EEv10DevParaSum"));
			mFunction = maxIndex64; break;
		case 32:
			//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj32EEv10DevParaSum"));
			mFunction = maxIndex32; break;
		case 16:
			//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj16EEv10DevParaSum"));
			mFunction = maxIndex16; break;
		case  8:
			//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj8EEv10DevParaSum"));
			mFunction = maxIndex8; break;
		case  4:
			//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj4EEv10DevParaSum"));
			mFunction = maxIndex4; break;
		case  2:
			//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj2EEv10DevParaSum"));
			mFunction = maxIndex2; break;
		case  1:
			//hipSafeCall(hipModuleGetFunction(&mFunction, mModule, "_Z6reduceILj1EEv10DevParaSum"));
			mFunction = maxIndex1; break;
	}

	hipDeviceptr_t input_dptr = input.GetDevicePtr();
	hipDeviceptr_t output_dptr = output.GetDevicePtr();
	hipDeviceptr_t index_dptr = index.GetDevicePtr();

	DevParaMaxIndex rp;
	rp.size = size;
	rp.in_dptr = (float*) input_dptr;
	rp.out_dptr = (float*) output_dptr;
	rp.index = (int*) index_dptr;
	rp.readIndex = readIndex;

float ms = Launch( &rp, sizeof(rp) );
}


void getNumBlocksAndThreads(int n, int &blocks, int &threads)
{   
	static const int maxBlocks = 64;
	static const int maxThreads = 256;

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


unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}



const hipChannelFormatDesc float_desc = myhipCreateChannelDesc( 32, 0, 0, 0, hipChannelFormatKindFloat);
const hipChannelFormatDesc float2_desc = myhipCreateChannelDesc( 32, 32, 0, 0, hipChannelFormatKindFloat);

RotateKernel::RotateKernel(hipModule_t aModule, dim3 aGridDim, dim3 aBlockDim, int aVolSize, bool linearInterpolation)
	: Hip::HipKernel("rot3d", aModule, aGridDim, aBlockDim, 0),
	volSize(aVolSize),
	shiftTex(float_desc, aVolSize, aVolSize, aVolSize, 0),
	dataTex(float_desc, aVolSize, aVolSize, aVolSize, 0),
	dataTexCplx(float2_desc, aVolSize, aVolSize, aVolSize, 0),
	/*shiftTex(CU_AD_FORMAT_FLOAT, aVolSize, aVolSize, aVolSize, 1, 0),
	dataTex(CU_AD_FORMAT_FLOAT, aVolSize, aVolSize, aVolSize, 1, 0),
	dataTexCplx(CU_AD_FORMAT_FLOAT, aVolSize, aVolSize, aVolSize, 2, 0),*/
	mlinearInterpolation(linearInterpolation ? hipFilterModeLinear : hipFilterModePoint),
	oldphi(0), oldpsi(0), oldtheta(0)
	,shiftTexObj(&shiftTex, hipFilterModeLinear, hipAddressModeWrap, false)
	,dataTexObj(&dataTex, hipFilterModeLinear)
	,dataTexCplxObj(&dataTexCplx, mlinearInterpolation)
{
	hipSafeCall(hipModuleGetFunction(&rotVol, mModule, "rot3d"));
	hipSafeCall(hipModuleGetFunction(&shift, mModule, "shift"));
	hipSafeCall(hipModuleGetFunction(&rotVol_improved, mModule, "rot3d_soft_interpolate"));
	hipSafeCall(hipModuleGetFunction(&shiftrot3d, mModule, "shiftrot3d"));
	hipSafeCall(hipModuleGetFunction(&rotVolCplx, mModule, "rot3dCplx"));
}


RotateKernel::RotateKernel(hipModule_t aModule, int aVolSize)
	: Hip::HipKernel("rot3d", aModule, make_dim3(1,1,1), make_dim3(32, 8, 1), 0),
	volSize(aVolSize),
	shiftTex(float_desc, aVolSize, aVolSize, aVolSize, 0),
	dataTex(float_desc, aVolSize, aVolSize, aVolSize, 0),
	dataTexCplx(float2_desc, aVolSize, aVolSize, aVolSize, 0),
	/*shiftTex(CU_AD_FORMAT_FLOAT, aVolSize, aVolSize, aVolSize, 1, 0),
	dataTex(CU_AD_FORMAT_FLOAT, aVolSize, aVolSize, aVolSize, 1, 0),
	dataTexCplx(CU_AD_FORMAT_FLOAT, aVolSize, aVolSize, aVolSize, 2, 0),*/
	oldphi(0), oldpsi(0), oldtheta(0)
	,shiftTexObj(&shiftTex, hipFilterModeLinear, hipAddressModeWrap, false)
	,dataTexObj(&dataTex, mlinearInterpolation)
	,dataTexCplxObj(&dataTexCplx, mlinearInterpolation)
{
	hipSafeCall(hipModuleGetFunction(&rotVol, mModule, "rot3d"));
	hipSafeCall(hipModuleGetFunction(&shift, mModule, "shift"));
	hipSafeCall(hipModuleGetFunction(&rotVol_improved, mModule, "rot3d_soft_interpolate"));
	hipSafeCall(hipModuleGetFunction(&shiftrot3d, mModule, "shiftrot3d"));
	hipSafeCall(hipModuleGetFunction(&rotVolCplx, mModule, "rot3dCplx"));
}


void RotateKernel::SetTexture(HipDeviceVariable& d_idata)
{
	dataTex.CopyFromDeviceToArray(d_idata);
	// shiftTexObj = HipTextureObject3D(&dataTex);
}


void RotateKernel::SetTextureShift(HipDeviceVariable& d_idata)
{
	shiftTex.CopyFromDeviceToArray(d_idata);
}


void RotateKernel::SetTextureCplx(HipDeviceVariable& d_idata)
{
	dataTexCplx.CopyFromDeviceToArray(d_idata);
}



float RotateKernel::do_rotate(int size, HipDeviceVariable& input, float phi, float psi, float theta)
{
	/* FIXME calculation of rotation is the same for each loop it doesnt need to 
					be done for reference, mask and maskcc but just once*/
	mFunction = rotVol;

	float rotMat[3][3];
	float rotMat1[3][3];
	float rotMat2[3][3];

	computeRotMat(oldphi, oldpsi, oldtheta, rotMat1);
	computeRotMat(phi, psi, theta, rotMat2);
	multiplyRotMatrix(rotMat2, rotMat1, rotMat);

	hipDeviceptr_t input_dptr = input.GetDevicePtr();

	float3 rotmat0 = make_float3(rotMat[0][0], rotMat[0][1], rotMat[0][2]);
	float3 rotmat1 = make_float3(rotMat[1][0], rotMat[1][1], rotMat[1][2]);
	float3 rotmat2 = make_float3(rotMat[2][0], rotMat[2][1], rotMat[2][2]);

	DevParaRotate rp;
	rp.size = size;
	rp.in_dptr = (float*) input_dptr;
	rp.rotmat0 = rotmat0;
	rp.rotmat1 = rotmat1;
	rp.rotmat2 = rotmat2;
	rp.texture = dataTexObj.GetTexObject();

	float ms = Launch( &rp, sizeof(rp) );
	return ms;
}

float RotateKernel::do_rotate_improved(int size, HipDeviceVariable& input, float phi, float psi, float theta)
{
	/* FIXME calculation of rotation is the same for each loop it doesnt need to 
					be done for reference, mask and maskcc but just once*/
	mFunction = rotVol_improved;

	float rotMat[3][3];
	float rotMat1[3][3];
	float rotMat2[3][3];

	computeRotMat(oldphi, oldpsi, oldtheta, rotMat1);
	computeRotMat(phi, psi, theta, rotMat2);
	multiplyRotMatrix(rotMat2, rotMat1, rotMat);

	hipDeviceptr_t input_dptr = input.GetDevicePtr();

	float3 rotmat0 = make_float3(rotMat[0][0], rotMat[0][1], rotMat[0][2]);
	float3 rotmat1 = make_float3(rotMat[1][0], rotMat[1][1], rotMat[1][2]);
	float3 rotmat2 = make_float3(rotMat[2][0], rotMat[2][1], rotMat[2][2]);

	DevParaRotate rp;
	rp.size = size;
	rp.in_dptr = (float*) input_dptr;
	rp.rotmat0 = rotmat0;
	rp.rotmat1 = rotmat1;
	rp.rotmat2 = rotmat2;
	rp.texture = dataTexObj.GetTexObject();

	float ms = Launch( &rp, sizeof(rp) );
	return ms;
}


float RotateKernel::do_rotateCplx(int size, HipDeviceVariable& input, float phi, float psi, float theta)
{
	float ms = 0;
	// float rotMat[3][3];
	// float rotMat1[3][3];
	// float rotMat2[3][3];

	// computeRotMat(oldphi, oldpsi, oldtheta, rotMat1);
	// computeRotMat(phi, psi, theta, rotMat2);
	// multiplyRotMatrix(rotMat2, rotMat1, rotMat);

	// hipDeviceptr_t input_dptr = input.GetDevicePtr();
	// hipDeviceptr_t output_dptr = output.GetDevicePtr();

	// DevParaBinarize rp;
	// rp.size = size;
	// rp.in_dptr = (float*) input_dptr;
	// rp.rotmat0 = &rotMat0;
	// rp.rotmat1 = &rotMat1;
	// rp.rotmat2 = &rotMat2;
	// rp.texture = &(dataTexObj.GetTexObject());

	// arglist[0] = &rp;
	// //float ms = Launch(arglist);
	// float ms = Launch( &rp, sizeof(rp) );

	return ms;
}


float RotateKernel::do_shift(int size, HipDeviceVariable& d_odata, float3 shiftVal)
{
	mFunction = shift;

	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

/*	void** arglist = (void**)new void*[4];

	arglist[0] = &volSize;
	arglist[1] = &out_dptr;
	arglist[2] = &shiftVal;
	arglist[3] = &(shiftTexObj.GetTexObject());


	hipSafeCall(hipModuleLaunchKernel(shift->GetHipFunction(), gridSize.x, gridSize.y,
	gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));
*/

	DevParaShift rp;
	rp.size = size;
	rp.in_dptr = (float*) out_dptr;
	rp.shift = shiftVal;
	rp.texture = shiftTexObj.GetTexObject();

	float ms = Launch( &rp, sizeof(rp) );
	return ms;
}


float RotateKernel::do_shiftrot3d(int size, HipDeviceVariable& d_odata, float phi, float psi, float theta, float3 shiftVal)
{
	mFunction = shiftrot3d;

	float rotMat[3][3];
	float rotMat1[3][3];
	float rotMat2[3][3];

	computeRotMat(oldphi, oldpsi, oldtheta, rotMat1);
	computeRotMat(phi, psi, theta, rotMat2);
	multiplyRotMatrix(rotMat2, rotMat1, rotMat);

	float3 rotmat0 = make_float3(rotMat[0][0], rotMat[0][1], rotMat[0][2]);
	float3 rotmat1 = make_float3(rotMat[1][0], rotMat[1][1], rotMat[1][2]);
	float3 rotmat2 = make_float3(rotMat[2][0], rotMat[2][1], rotMat[2][2]);

	hipDeviceptr_t out_dptr = d_odata.GetDevicePtr();

/*	void** arglist = (void**)new void*[4];

	arglist[0] = &volSize;
	arglist[1] = &out_dptr;
	arglist[2] = &shiftVal;
	arglist[3] = &(shiftTexObj.GetTexObject());


	hipSafeCall(hipModuleLaunchKernel(shift->GetHipFunction(), gridSize.x, gridSize.y,
	gridSize.z, blockSize.x, blockSize.y, blockSize.z, 0, stream, arglist,NULL));
*/

	DevParaShiftRot3D rp;
	rp.size = size;
	rp.in_dptr = (float*) out_dptr;
	rp.rotmat0 = rotmat0;
	rp.rotmat1 = rotmat1;
	rp.rotmat2 = rotmat2;
	rp.shift = shiftVal;
	rp.texture = shiftTexObj.GetTexObject();

	float ms = Launch( &rp, sizeof(rp) );
	return ms;
}

void RotateKernel::SetOldAngles(float aPhi, float aPsi, float aTheta)
{
		oldphi = aPhi;
		oldpsi = aPsi;
		oldtheta = aTheta;
}


void RotateKernel::computeRotMat(float phi, float psi, float theta, float rotMat[3][3])
{
		int i, j;
		float sinphi, sinpsi, sintheta; /* sin of rotation angles */
		float cosphi, cospsi, costheta; /* cos of rotation angles */

		
		float angles[] = { 0, 30, 45, 60, 90, 120, 135, 150, 180, 210, 225, 240, 270, 300, 315, 330 };
		float angle_cos[16];
		float angle_sin[16];

		angle_cos[0]=1.0f;
		angle_cos[1]=sqrt(3.0f)/2.0f;
		angle_cos[2]=sqrt(2.0f)/2.0f;
		angle_cos[3]=0.5f;
		angle_cos[4]=0.0f;
		angle_cos[5]=-0.5f;
		angle_cos[6]=-sqrt(2.0f)/2.0f;
		angle_cos[7]=-sqrt(3.0f)/2.0f;
		angle_cos[8]=-1.0f;
		angle_cos[9]=-sqrt(3.0f)/2.0f;
		angle_cos[10]=-sqrt(2.0f)/2.0f;
		angle_cos[11]=-0.5f;
		angle_cos[12]=0.0f;
		angle_cos[13]=0.5f;
		angle_cos[14]=sqrt(2.0f)/2.0f;
		angle_cos[15]=sqrt(3.0f)/2.0f;
		angle_sin[0]=0.0f;
		angle_sin[1]=0.5f;
		angle_sin[2]=sqrt(2.0f)/2.0f;
		angle_sin[3]=sqrt(3.0f)/2.0f;
		angle_sin[4]=1.0f;
		angle_sin[5]=sqrt(3.0f)/2.0f;
		angle_sin[6]=sqrt(2.0f)/2.0f;
		angle_sin[7]=0.5f;
		angle_sin[8]=0.0f;
		angle_sin[9]=-0.5f;
		angle_sin[10]=-sqrt(2.0f)/2.0f;
		angle_sin[11]=-sqrt(3.0f)/2.0f;
		angle_sin[12]=-1.0f;
		angle_sin[13]=-sqrt(3.0f)/2.0f;
		angle_sin[14]=-sqrt(2.0f)/2.0f;
		angle_sin[15]=-0.5f;

		for (i=0, j=0 ; i<16; i++)
		{
				if (angles[i] == phi )
				{
						cosphi = angle_cos[i];
						sinphi = angle_sin[i];
						j = 1;
				}
		}

		if (j < 1)
		{
				phi = phi * (float)M_PI / 180.0f;
				cosphi=cos(phi);
				sinphi=sin(phi);
		}

		for (i=0, j=0 ; i<16; i++)
		{
				if (angles[i] == psi )
				{
						cospsi = angle_cos[i];
						sinpsi = angle_sin[i];
						j = 1;
				}
		}

		if (j < 1)
		{
				psi = psi * (float)M_PI / 180.0f;
				cospsi=cos(psi);
				sinpsi=sin(psi);
		}

		for (i=0, j=0 ; i<16; i++)
		{
				if (angles[i] == theta )
				{
							costheta = angle_cos[i];
							sintheta = angle_sin[i];
							j = 1;
				}
		}

		if (j < 1)
		{
				theta = theta * (float)M_PI / 180.0f;
				costheta=cos(theta);
				sintheta=sin(theta);
		}

//		*/
		/*
		const float scale = M_PI / 180.0;
		phi = phi * scale;
		psi = psi * scale;
		theta = theta * scale;

		cosphi = cos(phi);
		sinphi = sin(phi);

		costheta = cos(theta);
		sintheta = sin(theta);

		cospsi = cos(psi);
		sinpsi = sin(psi);

		/* calculation of rotation matrix */

		rotMat[0][0] = cospsi*cosphi-costheta*sinpsi*sinphi;
		rotMat[1][0] = sinpsi*cosphi+costheta*cospsi*sinphi;
		rotMat[2][0] = sintheta*sinphi;
		rotMat[0][1] = -cospsi*sinphi-costheta*sinpsi*cosphi;
		rotMat[1][1] = -sinpsi*sinphi+costheta*cospsi*cosphi;
		rotMat[2][1] = sintheta*cosphi;
		rotMat[0][2] = sintheta*sinpsi;
		rotMat[1][2] = -sintheta*cospsi;
		rotMat[2][2] = costheta;
}



void RotateKernel::multiplyRotMatrix(float m1[3][3], float m2[3][3], float out[3][3])
{
		out[0][0] = m1[0][0] * m2[0][0] + m1[1][0] * m2[0][1] + m1[2][0] * m2[0][2];
		out[1][0] = m1[0][0] * m2[1][0] + m1[1][0] * m2[1][1] + m1[2][0] * m2[1][2];
		out[2][0] = m1[0][0] * m2[2][0] + m1[1][0] * m2[2][1] + m1[2][0] * m2[2][2];
		out[0][1] = m1[0][1] * m2[0][0] + m1[1][1] * m2[0][1] + m1[2][1] * m2[0][2];
		out[1][1] = m1[0][1] * m2[1][0] + m1[1][1] * m2[1][1] + m1[2][1] * m2[1][2];
		out[2][1] = m1[0][1] * m2[2][0] + m1[1][1] * m2[2][1] + m1[2][1] * m2[2][2];
		out[0][2] = m1[0][2] * m2[0][0] + m1[1][2] * m2[0][1] + m1[2][2] * m2[0][2];
		out[1][2] = m1[0][2] * m2[1][0] + m1[1][2] * m2[1][1] + m1[2][2] * m2[1][2];
		out[2][2] = m1[0][2] * m2[2][0] + m1[1][2] * m2[2][1] + m1[2][2] * m2[2][2];
}
