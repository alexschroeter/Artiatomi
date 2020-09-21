# SubTomogramAveraging

This is a port of the Subtomogram Average MPI to HIP allowing it to run on Nvidia and AMD hardware. 

## Creating a binary

### Binary kernels
One of the first steps is to create the GPU-kernel binary.

> cd src/hip_kernels/
> ./compileHIPKernels<AMD|NVIDIA>.sh

### Make
Currently only android has the required HIP libraries installed. You can use the MakefileAndroid to compile the binary. With HIP_PLATFORM you can set for which platform the binary should be build.

> HIP_PLATFORM=<nvcc|hcc> make -f MakefileAndroid

You can also set additional parameters at compile time like the verboseness of the program like this

> EXTRA="-DVERBOSE=10" ...

## Running

The configuration now has an additional parameter 'AveragingType' which allows to switch between different versions of the averaging process at runtime.

| Averaging Type | Description |
| --- | --- |
| OriginalBinary | Using the same binaries with minimal changes to port them to HIP |
| OriginalHIP | For ease of developement we removed all the different classes for kernels. |
| C2C | This version has all the improvements to quality and bugfixes but still uses the Complex to Complex FFT. This version might be useful if at some point the filters are not point symetric anymore |
| R2C | Default This version has all the improvements to quality and bugfixes and is also performance improved. |
| PhaseCorrelation | Alignemnt using phase correlation |


