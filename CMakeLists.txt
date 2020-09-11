cmake_minimum_required(VERSION 3.14)

set(CMAKE_BUILD_TYPE Release)

if(NOT DEFINED HIP_PATH)
    if(NOT DEFINED ENV{HIP_PATH})
        set(HIP_PATH "/opt/rocm/hip" CACHE PATH "Path to which HIP has been installed")
    else()
        set(HIP_PATH $ENV{HIP_PATH} CACHE PATH "Path to which HIP has been installed")
    endif()
endif()

set(CMAKE_MODULE_PATH "${HIP_PATH}/cmake" ${CMAKE_MODULE_PATH})

project(12_cmake)

find_package(HIP QUIET)
if(HIP_FOUND)
    message(STATUS "Found HIP: " ${HIP_VERSION})
else()
    message(FATAL_ERROR "Could not find HIP. Ensure that HIP is either installed in /opt/rocm/hip or the variable HIP_PATH is set to point to the right location.")
endif()

find_package(MPI REQUIRED)

set(MY_SOURCE_FILES
    src/subtomogramaverage/SubTomogramAverageMPI.cpp
    src/subtomogramaverage/AvgProcess.cpp
    src/subtomogramaverage/Kernels.cpp
    src/hip/HipBasicKernel.cpp
    #src/hip/HipRot.cpp
    src/hip/HipReducer.cpp
    src/io/EMFile.cpp
    src/io/File.cpp
    src/io/FileIOException.cpp
    src/io/FileReader.cpp
    src/io/FileWriter.cpp
    src/io/Image.cpp
    src/io/ImageConverterMethods.cpp
    src/io/MotiveList.cpp
    src/hip/HipVariables.cpp
    src/config/Config.cpp
    src/config/ConfigExceptions.cpp
    src/hip/HipArrays.cpp
    src/hip/HipContext.cpp
    src/hip/HipDeviceProperties.cpp
    src/hip/HipException.cpp
    src/hip/HipKernel.cpp
    src/hip/HipTextures.cpp)
set(MY_TARGET_NAME SubtomogramAverageMPI)
set(MY_HIPCC_OPTIONS )
set(MY_HCC_OPTIONS)
set(MY_NVCC_OPTIONS)

set(SUBTOMOGRAMAVERAGE_INC
        ${MPI_CXX_INCLUDE_DIRS}
        )

set(SUBTOMOGRAMAVERAGE_LIB
        ${MPI_CXX_LIBRARIES}
        )

include_directories(/opt/rocm/rocfft/include /usr/include/mpich /opt/rocm/opencl/include/ /usr/local/cuda/include)
link_directories(/usr/lib/x86_64-linux-gnu /opt/rocm/lib /usr/local/cuda/lib64/)
#target_link_libraries(rocfft mpich clFFT OpenCL cuda cudart cufft)


set_source_files_properties(${MY_SOURCE_FILES} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
hip_add_executable(${MY_TARGET_NAME} ${MY_SOURCE_FILES} HIPCC_OPTIONS ${MY_HIPCC_OPTIONS} HCC_OPTIONS ${MY_HCC_OPTIONS} NVCC_OPTIONS ${MY_NVCC_OPTIONS})
target_link_libraries(${MY_TARGET_NAME} rocfft mpich clFFT OpenCL cuda cudart cufft)

#target_include_directories(${SUBTOMOGRAMAVERAGE_INC})
#target_link_libraries(${SUBTOMOGRAMAVERAGE_LIB})
