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

SOURCES="BackProjectionSquareOS.cu Compare.cu CopyToSquare.cu ctf.cu ForwardProjectionRayMarcher_TL.cu ForwardProjectionSlicer.cu Textures.cu wbpWeighting.cu"

OLD_PLATFORM=$HIP_PLATFORM

export HIP_PLATFORM=nvcc
# AS Todo enable first options for optimal performance

#OPTIONS='-arch compute_35 --machine 64 --use_fast_math'
#OPTIONS='-arch compute_35 --machine 64'

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
    name=$(expr "$hipFile" : "\(.*\).cu")
    rm build/$name.nvcc.h build/$name.nvcc.code >&/dev/null
    hipcc $OPTIONS --ptx -c -o build/$name.nvcc.code $name.cu
    bin2c -p 0 --name Kernel$name  build/$name.nvcc.code > $name.cu.h  
    rm build/$name.nvcc.code >&/dev/null
done

export HIP_PLATFORM=$OLD_PLATFORM