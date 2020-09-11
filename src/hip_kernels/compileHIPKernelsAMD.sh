#@echo off
#echo Current directory:
#echo .
#if "%VCSETUPDONE%"=="TRUE" (
#	ECHO Skip VC setup...
#) else (
#	ECHO Setup VC environment...
#	call "C:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" amd64
#	set VCSETUPDONE="TRUE"
#)

#set NVCCEXE="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\bin\nvcc.exe"
#set INCLUDE="C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v8.0\include"

#INCLUDE="/usr/local/cuda-8.0/include"
OUTPUT='.'
INPUT='.'

SOURCES=" basicKernels.cpp kernel.cpp "


OLD_PLATFORM=$HIP_PLATFORM


export HIP_PLATFORM=hcc
#OPTIONS='-ffast-math -finline-functions'
#OPTIONS='-fno-fast-math'
#'-cl-opt-disable'

echo "current platform " $OLD_PLATFORM
echo "compile for platform " $HIP_PLATFORM

echo $1

if [ $# -gt 0 ] && [ $1 == '-D' ] ;then
echo "DEBUG Modus!"
OPTIONS="$OPTIONS -g -G -lineinfo"
fi

#rm *.ptx >&/dev/null

for hipFile in $SOURCES
do
    echo compiling file $hipFile ...
    mkdir -p build
    name=$(expr "$hipFile" : "\(.*\).cpp")
    rm build/$name.hcc.h build/$name.hcc.code >&/dev/null
#    hipcc  --genco  -o build/$name.hcc.code $name.cpp --flags=\'-fno-fast-math -fno-reciprocal-math \'
    hipcc  --genco --targets gfx906 -o build/$name.hcc.code $name.cpp --flags=\'-ffast-math -freciprocal-math \'
    bin2c -p 0 --name Kernel$name  build/$name.hcc.code > build/$name.hcc.h  
    rm build/$name.hcc.code >&/dev/null
done

export HIP_PLATFORM=$OLD_PLATFORM
