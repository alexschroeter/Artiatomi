#ifndef HIPKERNELBINARYS_H
#define HIPKERNELBINARYS_H

#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)
#include "hip_kernels/build/BackProjectionSquareOS.nvcc.h"
#include "hip_kernels/build/Compare.nvcc.h"
#include "hip_kernels/build/CopyToSquare.nvcc.h"
#include "hip_kernels/build/ForwardProjectionRayMarcher_TL.nvcc.h"
#include "hip_kernels/build/ctf.nvcc.h"
#include "hip_kernels/build/ForwardProjectionSlicer.nvcc.h"
#include "hip_kernels/build/wbpWeighting.nvcc.h"
#include "hip_kernels/build/NppEmulatorKernel.nvcc.h"
#endif

#if defined(__HIP_PLATFORM_HCC__) && !defined (__HIP_PLATFORM_NVCC__)
#include "hip_kernels/build/BackProjectionSquareOS.hcc.h"
#include "hip_kernels/build/Compare.hcc.h"
#include "hip_kernels/build/CopyToSquare.hcc.h"
#include "hip_kernels/build/ForwardProjectionRayMarcher_TL.hcc.h"
#include "hip_kernels/build/ctf.hcc.h"
#include "hip_kernels/build/ForwardProjectionSlicer.hcc.h"
#include "hip_kernels/build/wbpWeighting.hcc.h"
#include "hip_kernels/build/NppEmulatorKernel.hcc.h"
#endif

#endif
