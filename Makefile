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
LIBS=-lrocfft -lmpich -lcuda -lcudart -lcufft

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
src/emsart/hip/HipMissedStuff.cpp \
src/emsart/hip/HipArrays.cpp \
src/emsart/hip/HipContext.cpp \
src/emsart/hip/HipDeviceProperties.cpp \
src/emsart/hip/HipException.cpp \
src/emsart/hip/HipKernel.cpp \
src/emsart/hip/HipTextures.cpp \
src/emsart/hip/HipVariables.cpp \
src/emsart/io/CtfFile.cpp \
src/emsart/io/Dm4File.cpp \
src/emsart/io/File.cpp \
src/emsart/io/ImageConverterMethods.cpp \
src/emsart/io/Dm4FileStack.cpp \
src/emsart/io/FileIOException.cpp \
src/emsart/io/MPISource.cpp \
src/emsart/io/Dm4FileTag.cpp \
src/emsart/io/FileReader.cpp \
src/emsart/io/MRCFile.cpp \
src/emsart/io/Dm4FileTagDirectory.cpp \
src/emsart/io/FileWriter.cpp \
src/emsart/io/MarkerFile.cpp \
src/emsart/io/EMFile.cpp \
src/emsart/io/Image.cpp \
src/emsart/io/writeBMP.cpp \
src/emsart/utils/Config.cpp \
src/emsart/utils/ConfigExceptions.cpp \
src/emsart/utils/Matrix.cpp \
src/emsart/utils/log.cpp \
src/emsart/Kernels.cpp \
src/emsart/Projection.cpp \
src/emsart/utils/SimpleLogger.cpp \
src/emsart/Volume.cpp \
src/emsart/Reconstructor.cpp \
src/emsart/NppEmulator.cpp

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

##############################################################################
# Default Makefile Target
##############################################################################

all: SubtomogramAverage AddParticles
	@echo ""
	@echo "Your executables have been created in bin/"

SubtomogramAverage: $(SUBTOMOAVG_OBJ) $(OBJECTS) 
	mkdir -p bin
	$(CC) $(CFLAGS) $(SUBTOMOAVG_OBJ) $(OBJECTS) $(LIBDIRS) $(LIBS) -o $(SUBTOMOAVG_EXE)

AddParticles: $(ADDPARTICLES_OBJ) $(OBJECTS) 
	mkdir -p bin
	$(CC) $(CFLAGS) $(ADDPARTICLES_OBJ) $(OBJECTS) $(LIBDIRS) $(LIBS) -o $(ADDPARTICLES_EXE)

EmSART: $(EMSART_OBJ) $(OBJECTS2) 
	mkdir -p bin
	$(CC) $(CFLAGS) $(EMSART_OBJ) $(OBJECTS2) $(LIBDIRS) $(LIBS) -o $(EMSART_EXE)

.cpp.o:
	$(CC) $(EXTRA) -c $(CFLAGS) $(INCLUDES) $< -o $@

distclean: clean
	rm -f $(ADDPARTICLES_EXE) $(SUBTOMOAVG_EXE)	$(EMSART_EXE)

clean:
	rm -f $(OBJECTS)
	rm -f $(ADDPARTICLES_OBJ) $(SUBTOMOAVG_OBJ) $(EMSART_OBJ)

