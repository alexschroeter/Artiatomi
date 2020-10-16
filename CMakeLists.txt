cmake_minimum_required(VERSION 3.14)

project(Artiatomi)

set(CMAKE_BUILD_TYPE Release)

### HIP ###
if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

project(Artiatomi)

find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()

#set(CMAKE_CXX_STANDARD 14)
set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

set(EMSART_PATH "src/emsart" CACHE PATH "Path to EmSART source.")
set(STA_PATH "src/subtomogramaverage" CACHE PATH "Path to SubTomogramAverageMPI source.")


### Common ###
find_package(MPI REQUIRED)

set(ARTIATOMI_COMMON
    src/hip/HipBasicKernel.cpp
    src/hip/HipReducer.cpp
    src/io/File.cpp
    src/io/FileIOException.cpp
    src/io/FileReader.cpp
    src/io/FileWriter.cpp
    src/io/Image.cpp
    src/io/ImageConverterMethods.cpp
    src/hip/HipVariables.cpp
    src/config/ConfigExceptions.cpp
    src/hip/HipArrays.cpp
    src/hip/HipContext.cpp
    src/hip/HipDeviceProperties.cpp
    src/hip/HipException.cpp
    src/hip/HipKernel.cpp
    src/hip/HipTextures.cpp
)

set(LIB_COMMON
    ${MPI_CXX_LIBRARIES}
)

set(MY_HIPCC_OPTIONS)
set(MY_HCC_OPTIONS)
set(MY_NVCC_OPTIONS)


### SubtomogramAverage (Common) ###
set(STACOMMON_SRC
    ${ARTIATOMI_COMMON}
    ${STA_PATH}/Config.cpp
    ${STA_PATH}/MotiveList.cpp
    ${STA_PATH}/EMFile.cpp
    ${STA_PATH}/AvgProcess.cpp
    ${STA_PATH}/Kernels.cpp
)

include_directories(/opt/rocm/rocfft/include /usr/include/mpich /opt/rocm/opencl/include/ /usr/local/cuda/include)
link_directories(/usr/lib/x86_64-linux-gnu /opt/rocm/lib /usr/local/cuda/lib64/)


### STA ###
set(STA_BIN SubtomogramAverageMPI)
set(STA_SRC
    ${STACOMMON_SRC}
    ${STA_PATH}/SubTomogramAverageMPI.cpp
)

set_source_files_properties(${STA_SRC} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${STA_BIN} ${STA_SRC} HIPCC_OPTIONS ${MY_HIPCC_OPTIONS} HCC_OPTIONS ${MY_HCC_OPTIONS} NVCC_OPTIONS ${MY_NVCC_OPTIONS})
target_link_libraries(${STA_BIN} rocfft mpich clFFT OpenCL cuda cudart cufft)


### STA (AddParticles) ###
set(STAADDPARTICLES_BIN AddParticles)
set(STAADDPARTICLES_SRC
    ${STACOMMON_SRC}
    ${STA_PATH}/AddParticles.cpp
)

set_source_files_properties(${STAADDPARTICLES_SRC} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${STAADDPARTICLES_BIN} ${STAADDPARTICLES_SRC} HIPCC_OPTIONS ${MY_HIPCC_OPTIONS} HCC_OPTIONS ${MY_HCC_OPTIONS} NVCC_OPTIONS ${MY_NVCC_OPTIONS})
target_link_libraries(${STAADDPARTICLES_BIN} rocfft mpich clFFT OpenCL cuda cudart cufft)


### EmSART (Common) ###
set(EMSARTCOMMON_SRC
    src/emsart/CudaHelpers/CudaArrays.cpp 
    src/emsart/CudaHelpers/CudaContext.cpp 
    src/emsart/CudaHelpers/CudaDeviceProperties.cpp 
    src/emsart/CudaHelpers/CudaException.cpp 
    src/emsart/CudaHelpers/CudaKernel.cpp 
    src/emsart/CudaHelpers/CudaTextures.cpp 
    src/emsart/CudaHelpers/CudaVariables.cpp 
    src/emsart/CudaHelpers/CudaSurfaces.cpp 
    src/emsart/FileIO/CtfFile.cpp 
    src/emsart/FileIO/Dm3File.cpp 
    src/emsart/FileIO/Dm3FileTag.cpp 
    src/emsart/FileIO/Dm3FileTagDirectory.cpp 
    src/emsart/FileIO/Dm4File.cpp 
    src/emsart/FileIO/Dm4FileTag.cpp 
    src/emsart/FileIO/Dm4FileTagDirectory.cpp 
    src/emsart/FileIO/EmFile.cpp 
    src/emsart/FileIO/File.cpp 
    src/emsart/FileIO/FileIOException.cpp 
    src/emsart/FileIO/FileReader.cpp 
    src/emsart/FileIO/FileWriter.cpp 
    src/emsart/FileIO/ImageBase.cpp 
    src/emsart/FileIO/ImodFiducialFile.cpp 
    src/emsart/FileIO/MarkerFile.cpp 
    src/emsart/FileIO/MDocFile.cpp 
    src/emsart/FileIO/MotiveList.cpp 
    src/emsart/FileIO/MovieStack.cpp 
    src/emsart/FileIO/MRCFile.cpp 
    src/emsart/FileIO/SERFile.cpp 
    src/emsart/FileIO/ShiftFile.cpp 
    src/emsart/FileIO/SimpleFileList.cpp 
    src/emsart/FileIO/SingleFrame.cpp 
    src/emsart/FileIO/TIFFFile.cpp 
    src/emsart/FileIO/TiltSeries.cpp 
    src/emsart/FileIO/Volume.cpp 
    src/emsart/Minimization/levmar.cpp 
    src/emsart/FilterGraph/Matrix.cpp 
    src/emsart/FilterGraph/PointF.cpp 
    src/emsart/FilterGraph/FilterSize.cpp 
    src/emsart/FilterGraph/FilterROI.cpp 
    src/emsart/FilterGraph/FilterPoint2D.cpp 
    src/emsart/io/MPISource.cpp 
    src/emsart/io/FileSource.cpp 
    src/emsart/io/writeBMP.cpp 
    src/emsart/utils/Config.cpp 
    src/emsart/utils/CudaConfig.cpp 
    src/emsart/utils/ConfigExceptions.cpp 
    src/emsart/utils/log.cpp 
    src/emsart/utils/SimpleLogger.cpp 
    src/emsart/Kernels.cpp 
    src/emsart/Projection.cpp 
    src/emsart/Volume.cpp 
    src/emsart/Reconstructor.cpp 
)

include_directories(/opt/rocm/rocfft/include /usr/include/mpich /opt/rocm/opencl/include/ /usr/local/cuda/include)
link_directories(/usr/lib/x86_64-linux-gnu /opt/rocm/lib /usr/local/cuda/lib64/)


### EmSART ###
set(EMSART_BIN EmSART)
set(EMSART_SRC
    ${EMSARTCOMMON_SRC}
    src/emsart/EmSart.cpp
)

set_source_files_properties(${EMSART_SRC} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${EMSART_BIN} ${EMSART_SRC} HIPCC_OPTIONS ${MY_HIPCC_OPTIONS} HCC_OPTIONS ${MY_HCC_OPTIONS} NVCC_OPTIONS ${MY_NVCC_OPTIONS} )
target_link_libraries(${EMSART_BIN} rocfft mpich clFFT OpenCL cuda cudart cufft nppial nppist nppitc nppidei nppig npps)


### EmSARTRefine ###
set(EMSARTREFINE_BIN EmSARTRefine)
set(EMSARTREFINE_SRC
    ${EMSARTCOMMON_SRC}
    src/emsart/EmSartRefine.cpp
)

set_source_files_properties(${EMSARTREFINE_SRC} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${EMSARTREFINE_BIN} ${EMSARTREFINE_SRC} HIPCC_OPTIONS ${MY_HIPCC_OPTIONS} HCC_OPTIONS ${MY_HCC_OPTIONS} NVCC_OPTIONS ${MY_NVCC_OPTIONS})
target_link_libraries(${EMSARTREFINE_BIN} rocfft mpich clFFT OpenCL cuda cudart cufft nppial nppist nppitc nppidei nppig npps)
target_compile_definitions(${EMSARTREFINE_BIN} PRIVATE NO_OPENCL USE_MPI REFINE_MODE)


### EmSARTSubVolumes ###
set(EMSARTSUBVOL_BIN EmSARTSubVolumes)
set(EMSARTSUBVOL_SRC
    ${EMSARTCOMMON_SRC}
    src/emsart/EmSartSubVolumes.cpp
)

set_source_files_properties(${EMSARTSUBVOL_SRC} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${EMSARTSUBVOL_BIN} ${EMSARTSUBVOL_SRC} HIPCC_OPTIONS ${MY_HIPCC_OPTIONS} HCC_OPTIONS ${MY_HCC_OPTIONS} NVCC_OPTIONS ${MY_NVCC_OPTIONS})
target_link_libraries(${EMSARTSUBVOL_BIN} rocfft mpich clFFT OpenCL cuda cudart cufft nppial nppist nppitc nppidei nppig npps)
target_compile_definitions(${EMSARTSUBVOL_BIN} PUBLIC NO_OPENCL USE_MPI SUBVOLREC_MODE)
