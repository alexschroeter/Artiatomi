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
src/subtomogramaverage/SubTomogramAverageMPI.cpp \
src/subtomogramaverage/AvgProcess.cpp \
src/io/EMFile.cpp \
src/io/File.cpp \
src/io/FileIOException.cpp \
src/io/FileReader.cpp \
src/io/FileWriter.cpp \
src/io/Image.cpp \
src/io/ImageConverterMethods.cpp \
src/io/MotiveList.cpp \
src/hip/HipVariables.cpp \
src/config/Config.cpp \
src/config/ConfigExceptions.cpp \
src/hip/HipArrays.cpp \
src/hip/HipContext.cpp \
src/hip/HipDeviceProperties.cpp \
src/hip/HipException.cpp \
src/hip/HipKernel.cpp \
src/hip/HipTextures.cpp #\
src/subtomogramaverage/SubTomogramAverageMPI.cpp

COMMON2= \
src/subtomogramaverage/Kernels.cpp \
src/subtomogramaverage/AddParticles.cpp \
src/subtomogramaverage/AvgProcess.cpp \
src/io/EMFile.cpp \
src/io/File.cpp \
src/io/FileIOException.cpp \
src/io/FileReader.cpp \
src/io/FileWriter.cpp \
src/io/Image.cpp \
src/io/ImageConverterMethods.cpp \
src/io/MotiveList.cpp \
src/hip/HipVariables.cpp \
src/config/Config.cpp \
src/config/ConfigExceptions.cpp \
src/hip/HipArrays.cpp \
src/hip/HipContext.cpp \
src/hip/HipDeviceProperties.cpp \
src/hip/HipException.cpp \
src/hip/HipKernel.cpp \
src/hip/HipTextures.cpp

OBJECTS=$(COMMON:.cpp=.o)
OBJECTS2=$(COMMON2:.cpp=.o)

# unused
SOURCE+=$(EXECUTABLE:.cpp=.o)

EXECUTABLE=bin/SubTomogramAverageMPI


##############################################################################
# Default Makefile Target
##############################################################################
#all: SubTomogramAverageMPI #Speedtest

all: $(SOURCES) $(EXECUTABLE)

SubtomogramAverage: $(OBJECTS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(OBJECTS) $(LIBDIRS) $(LIBS) -o bin/SubTomogramAverageMPI

AddParticles: $(OBJECTS2)
	mkdir -p bin
	$(CC) $(CFLAGS) $(OBJECTS2) $(LIBDIRS) $(LIBS) -o bin/AddParticles

$(EXECUTABLE): $(OBJECTS)
	mkdir -p bin
	$(CC) $(CFLAGS) $(OBJECTS) $(LIBDIRS) $(LIBS) -o $@


.cpp.o:
	$(CC) $(EXTRA) -c $(CFLAGS) $(INCLUDES) $< -o $@


clean:
	rm -f $(OBJECTS)
	rm -f $(SOURCE)
	rm -f src/subtomogramaverage/SubTomogramAverageMPI.o
	rm -f src/subtomogramaverage/SubTomogramAverageMPIBench.o
	rm -f src/subtomogramaverage/Speedtest.o
	rm -f src/subtomogramaverage/AvgProcess.o
	rm -f bin/SubTomogramAverageMPI
#	rm -f bin/Speedtest
	
