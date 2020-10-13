##############################################################################
# Makefile for the Artiatomi Software Package
#
#
#
#
##############################################################################


##############################################################################
# Depending on the provided Hip Platform we will compile either 
# NVIDIA or AMD binaries
##############################################################################
#ifeq ($(HIP_PLATFORM), nvcc)
#CC=mpic++
#CFLAGS= -Wall -std=c++11 -Wno-unknown-pragmas
#else
CC=hipcc #-O3 -cl-no-signed-zeros #--ftz=true #-std=c++11 -ccbin=gcc -arch=sm_75 --machine 64 --use_fast_math #-DHIP_FAST_MATH
#CFLAGS+= -Wall -std=c++11 -Wno-unknown-pragmas
#MPI_COMPILE_FLAGS = $(shell mpic++ --showme:compile)
#MPI_LINK_FLAGS = $(shell mpic++ --showme:link)
#endif

##############################################################################
# Hipconfig provides a some CFLAGS
##############################################################################
CFLAGS += $(shell /opt/rocm/bin/hipconfig --cpp_config)

##############################################################################
# We include rocm, cuda and rocfft
##############################################################################
INCLUDES=-I/opt/rocm/rocfft/include -I/usr/include/mpich -I/usr/local/cuda/include

##############################################################################
# Linking library directories
##############################################################################
LIBDIRS=-L/usr/lib/x86_64-linux-gnu -L/opt/rocm/lib -L/usr/local/cuda/lib64/

##############################################################################
# Linking libraries
##############################################################################
LIBS=-lrocfft -lmpich -lcuda -lcudart -lcufft -lcudart -lnppial -lnppist -lnppitc -lnppidei -lnppig -lnpps

##############################################################################
# These are all the common sources
##############################################################################
COMMON= \
src/subtomogramaverage/Kernels.cpp \
src/io/EMFile.cpp \
src/io/File.cpp \
src/io/FileIOException.cpp \
src/io/FileReader.cpp \
src/io/FileWriter.cpp \
src/io/Image.cpp \
src/io/ImageConverterMethods.cpp \
src/io/MotiveList.cpp \
src/hip/HipVariables.cpp \
src/config/ConfigExceptions.cpp \
src/hip/HipArrays.cpp \
src/hip/HipContext.cpp \
src/hip/HipDeviceProperties.cpp \
src/hip/HipException.cpp \
src/hip/HipKernel.cpp \
src/hip/HipTextures.cpp #\

OBJECTS=$(COMMON:.cpp=.o)

COMMON2= \
src/emsart/CudaHelpers/CudaArrays.cpp \
src/emsart/CudaHelpers/CudaContext.cpp \
src/emsart/CudaHelpers/CudaDeviceProperties.cpp \
src/emsart/CudaHelpers/CudaException.cpp \
src/emsart/CudaHelpers/CudaKernel.cpp \
src/emsart/CudaHelpers/CudaTextures.cpp \
src/emsart/CudaHelpers/CudaVariables.cpp \
src/emsart/CudaHelpers/CudaSurfaces.cpp \
src/emsart/FileIO/CtfFile.cpp \
src/emsart/FileIO/Dm3File.cpp \
src/emsart/FileIO/Dm3FileTag.cpp \
src/emsart/FileIO/Dm3FileTagDirectory.cpp \
src/emsart/FileIO/Dm4File.cpp \
src/emsart/FileIO/Dm4FileTag.cpp \
src/emsart/FileIO/Dm4FileTagDirectory.cpp \
src/emsart/FileIO/EmFile.cpp \
src/emsart/FileIO/File.cpp \
src/emsart/FileIO/FileIOException.cpp \
src/emsart/FileIO/FileReader.cpp \
src/emsart/FileIO/FileWriter.cpp \
src/emsart/FileIO/ImageBase.cpp \
src/emsart/FileIO/ImodFiducialFile.cpp \
src/emsart/FileIO/MarkerFile.cpp \
src/emsart/FileIO/MDocFile.cpp \
src/emsart/FileIO/MotiveList.cpp \
src/emsart/FileIO/MovieStack.cpp \
src/emsart/FileIO/MRCFile.cpp \
src/emsart/FileIO/SERFile.cpp \
src/emsart/FileIO/ShiftFile.cpp \
src/emsart/FileIO/SimpleFileList.cpp \
src/emsart/FileIO/SingleFrame.cpp \
src/emsart/FileIO/TIFFFile.cpp \
src/emsart/FileIO/TiltSeries.cpp \
src/emsart/FileIO/Volume.cpp \
src/emsart/Minimization/levmar.cpp \
src/emsart/FilterGraph/Matrix.cpp \
src/emsart/FilterGraph/PointF.cpp \
src/emsart/FilterGraph/FilterSize.cpp \
src/emsart/FilterGraph/FilterROI.cpp \
src/emsart/FilterGraph/FilterPoint2D.cpp \
src/emsart/io/MPISource.cpp \
src/emsart/io/FileSource.cpp \
src/emsart/io/writeBMP.cpp \
src/emsart/utils/Config.cpp \
src/emsart/utils/CudaConfig.cpp \
src/emsart/utils/ConfigExceptions.cpp \
src/emsart/utils/log.cpp \
src/emsart/utils/SimpleLogger.cpp \
src/emsart/Kernels.cpp \
src/emsart/Projection.cpp \
src/emsart/Volume.cpp \
src/emsart/Reconstructor.cpp 

OBJECTS2=$(COMMON2:.cpp=.o)


SUBTOMOAVG_SRC= src/subtomogramaverage/Config.cpp src/subtomogramaverage/AvgProcess.cpp src/subtomogramaverage/SubTomogramAverageMPI.cpp
SUBTOMOAVG_OBJ=$(SUBTOMOAVG_SRC:.cpp=.o)
SUBTOMOAVG_EXE=bin/SubTomogramAverageMPI

ADDPARTICLES_SRC = src/subtomogramaverage/Config.cpp src/subtomogramaverage/AvgProcess.cpp src/subtomogramaverage/AddParticles.cpp
ADDPARTICLES_OBJ = $(ADDPARTICLES_SRC:.cpp=.o)
ADDPARTICLES_EXE = bin/AddParticles

EMSART_SRC = src/emsart/EmSart.cpp
EMSART_OBJ = $(EMSART_SRC:.cpp=.o)
EMSART_EXE = bin/EmSART

EMSARTRefine_SRC = src/emsart/EmSartRefine.cpp
EMSARTRefine_OBJ = $(EMSARTRefine_SRC:.cpp=.o)
EMSARTRefine_EXE = bin/EmSARTRefine

EMSARTSubVolumes_SRC = src/emsart/EmSartSubVolumes.cpp
EMSARTSubVolumes_OBJ = $(EMSARTSubVolumes_SRC:.cpp=.o)
EMSARTSubVolumes_EXE = bin/EmSARTSubVolumes

##############################################################################
# Default Makefile Target
##############################################################################

all: SubtomogramAverage AddParticles EmSART EmSARTRefine EmSARTSubVolumes
	@echo ""
	@echo "Your executables have been created in bin/"

SubtomogramAverage: $(SUBTOMOAVG_OBJ) $(OBJECTS) 
	mkdir -p bin
	$(CC) $(CFLAGS) $(SUBTOMOAVG_OBJ) $(OBJECTS) $(LIBDIRS) $(LIBS) -o $(SUBTOMOAVG_EXE)

AddParticles: $(ADDPARTICLES_OBJ) $(OBJECTS) 
	mkdir -p bin
	$(CC) $(CFLAGS) $(ADDPARTICLES_OBJ) $(OBJECTS) $(LIBDIRS) $(LIBS) -o $(ADDPARTICLES_EXE)

EmSART: EMSARTFLAG= 
EmSART: $(EMSART_OBJ) $(OBJECTS2)
	mkdir -p bin
	$(CC) $(CFLAGS) $(EMSART_OBJ) $(OBJECTS2) $(LIBDIRS) $(LIBS) -o $(EMSART_EXE)

# AS ToDo EmSART, EmSARTRefine and EmSARTSubVolumes need to rebuild Config.cpp and Reconstructor.cpp because they change depending on Environment Variable.
# Current Fix is a make clean and make EmSART...
EmSARTRefine: EMSARTFLAG=-DREFINE_MODE
EmSARTRefine: $(EMSARTRefine_OBJ) $(OBJECTS2)
	mkdir -p bin
	$(CC) $(EMSARTFLAG) $(CFLAGS) $(EMSARTRefine_OBJ) $(OBJECTS2) $(LIBDIRS) $(LIBS) -o $(EMSARTRefine_EXE)

EmSARTSubVolumes: EMSARTFLAG=-DSUBVOLREC_MODE
EmSARTSubVolumes: $(EMSARTSubVolumes_OBJ) $(OBJECTS2)
	mkdir -p bin
	$(CC) $(EMSARTFLAG) $(CFLAGS) $(EMSARTSubVolumes_OBJ) $(OBJECTS2) $(LIBDIRS) $(LIBS) -o $(EMSARTSubVolumes_EXE)

.cpp.o:
	$(CC) $(EMSARTFLAG) $(EXTRA) -c $(CFLAGS) $(INCLUDES) $< -o $@

distclean: clean
	rm -f $(ADDPARTICLES_EXE) $(SUBTOMOAVG_EXE)	$(EMSART_EXE) $(EMSARTRefine_EXE) $(EMSARTSubVolumes_EXE)

clean:
	rm -f $(OBJECTS) $(OBJECTS2)
	rm -f $(ADDPARTICLES_OBJ) $(SUBTOMOAVG_OBJ) $(EMSART_OBJ) $(EMSARTRefine_OBJ) $(EMSARTSubVolumes_OBJ)

