#ifndef SHIFTFILE_H
#define SHIFTFILE_H

#include "IODefault.h"
#include "EMFile.h"


using namespace std;

//! Represents a shift file stored in EM-file format.
/*!
\author Michael Kunz
\date   September 2011
\version 1.0
*/
class ShiftFile : public EMFile
{
protected:
public:
	//! Creates a new ShiftFile instance. The file is directly read from file.
	ShiftFile(string aFileName);

	//! Creates a new ShiftFile instance. The file ist not yet created.
	ShiftFile(string aFileName, int aProjectionCount, int aMotiveCount);

	//! Returns the number of markers in the marker file.
	int GetMotiveCount();

	//! Returns the number of projections in the marker file.
	int GetProjectionCount();

	//! Returns a pointer to the inner data array.
	float* GetData();

	//! Returns value with index (\p aProjection, \p aMotive).
	float2 operator() (const int aProjection, const int aMotive);

	//! Returns value with index (\p aProjection, \p aMotive).
	void SetValue(const int aProjection, const int aMotive, float2 aVal);
};

#endif