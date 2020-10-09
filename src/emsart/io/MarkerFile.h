#ifndef MARKERFILE_H
#define MARKERFILE_H

#include "IODefault.h"
#include "EMFile.h"


using namespace std;

//! Definition of marker file items
enum MarkerFileItem_enum
{
	MFI_TiltAngle = 0,
	MFI_X_Coordinate,
	MFI_Y_Coordinate,
	MFI_DevOfMark,
	MFI_X_Shift,
	MFI_Y_Shift,
	MFI_X_MeanShift,
	MFI_MagnifiactionX=8,
	MFI_MagnifiactionY=8,
	MFI_RotationPsi=9
};

//! Represents a marker file stored in EM-file format.
/*!
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class MarkerFile : private EMFile
{
protected:
	int mRefMarker;
public:
	//! Creates a new MarkerFile instance. The file is directly read from file.
	MarkerFile(string aFileName, int aRefMarker);

	//! Returns the number of markers in the marker file.
	int GetMarkerCount();
	
	//! Returns the number of projections in the marker file.
	int GetProjectionCount();

	//! Returns a pointer to the inner data array.
	float* GetData();

	//! Returns a reference to value with index (\p aItem, \p aProjection, \p aMarker).
	float& operator() (const MarkerFileItem_enum aItem, const int aProjection, const int aMarker);

	bool CheckIfProjIndexIsGood(const int index);
};

#endif