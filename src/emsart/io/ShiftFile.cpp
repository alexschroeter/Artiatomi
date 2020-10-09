#include "ShiftFile.h"


ShiftFile::ShiftFile(string aFileName)
	: EMFile(aFileName)
{
	OpenAndRead();
	ReadHeaderInfo();
}

ShiftFile::ShiftFile(string aFileName, int aProjectionCount, int aMotiveCount)
	: EMFile(aFileName)
{
	DimX = aMotiveCount;
	DimY = aProjectionCount;
	DimZ = 2;
	_fileHeader.DimX = DimX;
	_fileHeader.DimY = DimY;
	_fileHeader.DimZ = DimZ;
	
	SetDataType(FDT_FLOAT);
	_fileHeader.DataType = EMDATATYPE_FLOAT;
	
	_data = new char[aProjectionCount * aMotiveCount * 2 * sizeof(float)]; //x and y coordinate
	memset(_data, 0, aProjectionCount * aMotiveCount * 2 * sizeof(float));
}

float* ShiftFile::GetData()
{
	return (float*)EMFile::GetData();
}

int ShiftFile::GetMotiveCount()
{
	return _fileHeader.DimX;
}

int ShiftFile::GetProjectionCount()
{
	return _fileHeader.DimY;
}

float2 ShiftFile::operator() (const int aProjection, const int aMotive)
{
	float2 erg;

	float* fdata = (float*)_data;
	erg.x = -fdata[0 * DimX * DimY + aProjection * DimX + aMotive];
	erg.y = -fdata[1 * DimX * DimY + aProjection * DimX + aMotive];
	return erg;
}

void ShiftFile::SetValue(const int aProjection, const int aMotive, float2 aVal)
{
	float* fdata = (float*)_data;
	fdata[0 * DimX * DimY + aProjection * DimX + aMotive] = aVal.x;
	fdata[1 * DimX * DimY + aProjection * DimX + aMotive] = aVal.y;
}
