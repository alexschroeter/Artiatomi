#include "MPISource.h"
#include "../utils/Config.h"

MPISource::MPISource(int aDimX, int aDimY, int aDimZ, float aPixelSize)
	: ProjectionSource(""), _projectionCache(0)
{
	DimX = aDimX;
	DimY = aDimY;
	DimZ = aDimZ;

	AllocArrays(DimZ);
	//printf("6\n");

	for (int i = 0; i < DimZ; i++)
	{
		PixelSize[i] = aPixelSize;
	}
}

MPISource::~MPISource()
{
	/*if (_projectionCache)
        for (int i = 0; i < DimZ; i++)
            if (_projectionCache[i])
                delete[] _projectionCache[i];*/
}

bool MPISource::OpenAndRead()
{
	return true;
}

bool MPISource::OpenAndWrite()
{
	return true;
}

FileDataType_enum MPISource::GetDataType()
{
	return FDT_FLOAT;
}

void MPISource::SetDataType(FileDataType_enum aType)
{
	
}

size_t MPISource::GetDataSize()
{
	return 0;
}

char* MPISource::GetData()
{
	return _data;
}

char* MPISource::GetProjection(uint aIndex)
{
	return NULL;
}

float* MPISource::GetProjectionFloat(uint aIndex)
{
	return NULL;
}

float* MPISource::GetProjectionInvertFloat(uint aIndex)
{
	return NULL;
}

void MPISource::ReadHeaderInfo()
{}

void MPISource::WriteInfoToHeader()
{}
