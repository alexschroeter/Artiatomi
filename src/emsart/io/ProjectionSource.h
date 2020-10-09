#ifndef PROJECTIONSOURCE_H
#define PROJECTIONSOURCE_H

#include "IODefault.h"
#include "FileReader.h"
#include "FileWriter.h"
#include "Image.h"

using namespace std;

//!  MRCFile represents a *.mrc file in memory and maps contained information to the default internal Image format.
/*!
	MRCFile gives access to header infos, volume data and single projections.
	\author Michael Kunz
	\date   October 2012
	\version 1.0
*/
class ProjectionSource : public FileReader, public FileWriter, public Image
{
private:


public:
	ProjectionSource(string aFileName, const Image& aImage) : FileReader(aFileName), FileWriter(aFileName), Image(aImage) {};
	ProjectionSource(string aFileName) : FileReader(aFileName), FileWriter(aFileName) {};

	//! Opens the file File#mFileName and reads the entire content.
	/*!
		\throw FileIOException
	*/
	virtual bool OpenAndRead() = 0;

	//! Converts from MRC data type enum to internal data type
	/*!
		MRCFile::GetDataType dows not take into account if the data type is unsigned or signed as the
		MRC file format cannot distinguish them.
	*/
	virtual FileDataType_enum GetDataType() = 0;

	//! Returns the size of the data block. If the header is not yet read, it will return 0.
	virtual size_t GetDataSize() = 0;

	//! Returns the inner data pointer.
	virtual char* GetData() = 0;

	//! Reads the projection with index \p aIndex from file and returns it's pointer.
	virtual char* GetProjection(uint aIndex) = 0;

	//! Reads the projection converted to float with index \p aIndex from file and returns it's pointer.
	virtual float* GetProjectionFloat(uint aIndex) = 0;

	//! Reads the projection converted to float with index \p aIndex from file and returns it's pointer. Values are inverted.
	virtual float* GetProjectionInvertFloat(uint aIndex) = 0;
};

#endif
