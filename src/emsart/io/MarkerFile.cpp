#include "MarkerFile.h"

MarkerFile::MarkerFile(string aFileName, int aRefMarker)
	: EMFile(aFileName), mRefMarker(aRefMarker)
{
	OpenAndRead();
	ReadHeaderInfo();
}

float* MarkerFile::GetData()
{
	return (float*)EMFile::GetData();
}

int MarkerFile::GetMarkerCount()
{
	return _fileHeader.DimZ;
}

int MarkerFile::GetProjectionCount()
{
	int count = 0;
	for (int i = 0; i < _fileHeader.DimY; i++)
		if((*this)(MFI_X_Coordinate, i, mRefMarker) > 0 && (*this)(MFI_Y_Coordinate, i, mRefMarker) > 0) count++;
	return count;
}

bool MarkerFile::CheckIfProjIndexIsGood(const int index)
{
	return ((*this)(MFI_X_Coordinate, index, mRefMarker) > 0 && (*this)(MFI_Y_Coordinate, index, mRefMarker) > 0);
}

float& MarkerFile::operator() (const MarkerFileItem_enum aItem, const int aProjection, const int aMarker)
{
	float* fdata = (float*) _data;
//    if (aItem == MFI_RotationPsi) {
//        fdata[aMarker * DimX * DimY + aProjection * DimX + aItem] = 0;}
//    if (aItem == MFI_X_Shift) {
//        fdata[aMarker * DimX * DimY + aProjection * DimX + aItem] = 0;}
//    if (aItem == MFI_Y_Shift) {
//        fdata[aMarker * DimX * DimY + aProjection * DimX + aItem] = 0;}

	return fdata[aMarker * DimX * DimY + aProjection * DimX + aItem];
}
