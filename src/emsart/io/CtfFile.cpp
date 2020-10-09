#include "CtfFile.h"

CtfFile::CtfFile(std::string aFileName)
	: EMFile(aFileName)
{
	OpenAndRead();
	ReadHeaderInfo();
}

float CtfFile::GetMinDefocus(uint index)
{
	float* fdata = (float*)_data;
	return fdata[1 * DimX + index];
}

float CtfFile::GetMaxDefocus(uint index)
{
	float* fdata = (float*)_data;
	return fdata[2 * DimX + index];
}

float CtfFile::GetAstigmatismAngle(uint index)
{
	float* fdata = (float*)_data;
	return fdata[4 * DimX + index] * 180.0f / M_PI;
}