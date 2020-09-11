#ifndef HIPKERNELS_H
#define HIPKERNELS_H

#include <hip/hip_runtime.h>
#include <string>

using namespace std;

namespace Hip
{
	//!  A wrapper class for a HIP Kernel.
	/*!
	  HipKernel manages all the functionality of a HIP Kernel.
	  \author Michael Kunz
	  \date   January 2010
	  \version 1.0
	*/
	//A wrapper class for a HIP Kernel.
	class HipKernel
	{
	private:

	protected:
		//HipContext* mCtx;
		hipModule_t mModule;
		hipFunction_t mFunction;
		uint mSharedMemSize;
		//int mParamOffset;
		dim3 mBlockDim;
		dim3 mGridDim;
		string mKernelName;

	public:
		//! HipKernel constructor
		/*!
			Loads a HIP Kernel with name \p aKernelName bound to the HipContext \p aCtx
			from the hipModule_t \p aModule.
			\param aKernelName Name of the kernel to load
			\param aModule The module to load the kernel from
			\param aCtx The HIP Context to use with this kernel.
		*/
		//HipKernel constructor
		HipKernel(string aKernelName, hipModule_t aModule/*, HipContext* aCtx*/);

		//! HipKernel constructor
		/*!
			Loads a HIP Kernel with name \p aKernelName bound to the HipContext \p aCtx
			from the hipModule_t \p aModule.
			\param aKernelName Name of the kernel to load
			\param aModule The module to load the kernel from
			\param aCtx The HIP Context to use with this kernel.
		*/
		//HipKernel constructor
		HipKernel(string aKernelName, hipModule_t aModule/*, HipContext* aCtx*/, dim3 aGridDim, dim3 aBlockDim, uint aDynamicSharedMemory);

		//! HipKernel destructor
		//HipKernel destructor
		~HipKernel();

		//! Set a constant variable value before kernel launch.
		/*!
			\param aName Name of the constant variable as defined in the .cpp kernel source file.
			\param aValue The value to set.
		*/
		//Set a constant variable value before kernel launch.
		void SetConstantValue(string aName, void* aValue);



		//! Manually reset the parameter offset counter
		//Manually reset the parameter offset counter
		// void ResetParameterOffset();

		//! Set the kernels thread block dimensions before first launch.
		/*!
			\param aX Block X dimension.
			\param aY Block Y dimension.
			\param aZ Block Z dimension.
		*/
		//Set the kernels thread block dimensions before first launch.
		void SetBlockDimensions(uint aX, uint aY, uint aZ);

		//! Set the kernels thread block dimensions before first launch.
		/*!
			\param aBlockDim Block dimensions.
		*/
		//Set the kernels thread block dimensions before first launch.
		void SetBlockDimensions(dim3 aBlockDim);

		//! Set the kernels block grid dimensions before first launch.
		/*!
			\param aX Grid X dimension.
			\param aY Grid Y dimension.
			\param aZ Grid Z dimension.
		*/
		//Set the kernels block grid dimensions before first launch.
		void SetGridDimensions(uint aX, uint aY, uint aZ);

		//! Set the kernels block grid dimensions before first launch.
		/*!
			\param aGridDim Grid dimensions.
		*/
		//Set the kernels block grid dimensions before first launch.
		void SetGridDimensions(dim3 aGridDim);

		//!Set the kernels block grid dimensions before first launch according to work load dimensions and block sizes
		//Set the kernels block grid dimensions before first launch according to work load dimensions and block sizes
		void SetWorkSize(dim3 aWorkSize);
		
		//!Set the kernels block grid dimensions before first launch according to work load dimensions and block sizes
		//Set the kernels block grid dimensions before first launch according to work load dimensions and block sizes
		void SetWorkSize(uint aX, uint aY, uint aZ);


		//! Set the dynamic amount of shared memory before first launch.
		/*!
			\param aSizeInBytes Size of shared memory in bytes.
		*/
		// Set the dynamic amount of shared memory before first launch.
		void SetDynamicSharedMemory(uint aSizeInBytes);

		//! Get the wrapped hipModule_t
		//Get the wrapped hipModule_t
		hipModule_t& GetHipModule();

		//! Get the wrapped hipFunction_t
		//Get the wrapped hipFunction_t
		hipFunction_t& GetHipFunction();


		//! Launches the kernel.
		/*!
			Before kernel launch all kernel parameters, constant variable values and block / grid dimensions must be set.
			\return kernel runtime in [ms]
		*/
		//Launches the kernel. Returns kernel runtime in [ms]
		virtual float operator()(int dummy, ...);

		float Launch( void* arglist[] ) const;
		float Launch( void* ArgObjectPtr, size_t size ) const;

	};
}
#endif //HIPKERNELS_H
