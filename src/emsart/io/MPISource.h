#ifndef MPISOURCE_H
#define MPISOURCE_H

#include "ProjectionSource.h"

using namespace std;

//!  MRCFile represents a *.mrc file in memory and maps contained information to the default internal Image format.
/*!
	MRCFile gives access to header infos, volume data and single projections.
	\author Michael Kunz
	\date   October 2012
	\version 2.0
*/
class MPISource : public ProjectionSource
{
private:
	float** _projectionCache;

public:
	//! Creates a new MRCFile instance. The file name is only set internally; the file itself keeps untouched.
	MPISource(int DimX, int DimY, int DimZ, float aPixelSize);
	~MPISource();

	//! Opens the file File#mFileName and reads the entire content.
	/*!
		\throw FileIOException
	*/
	bool OpenAndRead();

	//! Converts from MRC data type enum to internal data type
	/*!
		MRCFile::GetDataType dows not take into account if the data type is unsigned or signed as the
		MRC file format cannot distinguish them.
	*/
	FileDataType_enum GetDataType();

	void SetDataType(FileDataType_enum aType);

	//! Returns the size of the data block. If the header is not yet read, it will return 0.
	size_t GetDataSize();

	//! Returns the inner data pointer.
	char* GetData();

	//! Reads the projection with index \p aIndex from file and returns it's pointer.
	char* GetProjection(uint aIndex);

	//! Reads the projection converted to float with index \p aIndex from file and returns it's pointer.
	float* GetProjectionFloat(uint aIndex);

	//! Reads the projection converted to float with index \p aIndex from file and returns it's pointer. Values are inverted.
	float* GetProjectionInvertFloat(uint aIndex);

	void ReadHeaderInfo();
	void WriteInfoToHeader();
	bool OpenAndWrite();
};

#endif
