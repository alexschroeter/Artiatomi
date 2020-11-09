#ifndef EMFILE_H
#define EMFILE_H

#include "../io/IODefault.h"
#include "../io/emHeader.h"
#include "../io/FileReader.h"
#include "../io/FileWriter.h"
#include "../io/Image.h"

using namespace std;

//!  EMFile represents a *.em file in memory and maps contained information to the default internal Image format.
/*!
	EMFile gives access to header infos, volume data and single projections.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class EMFile : public FileReader, public FileWriter, public Image
{
protected:
	EmHeader _fileHeader;
	uint _GetDataTypeSize(EmDataType_Enum aDataType);
	uint _dataStartPosition;

public:
	//! Creates a new EMFile instance. The file name is only set internally; the file itself keeps untouched.
	EMFile(string aFileName);
	//! Creates a new MRCFile instance and copies information from an existing Image. The file name is only set internally; the file itself keeps untouched.
	EMFile(string aFileName, const Image& aImage);

	EMFile* CreateEMFile(string aFileNameBase, int index);

	//! Opens the file File#mFileName and reads the entire content.
	/*!
		\throw FileIOException
	*/
	bool OpenAndRead();

	//! Opens the file File#mFileName and reads only the file header.
	/*!
		\throw FileIOException
	*/
	bool OpenAndReadHeader();

	//! Opens the file File#mFileName and writes the entire content.
	/*!
		\throw FileIOException
	*/
	bool OpenAndWrite();

	//! Converts from em data type enum to internal data type
	/*!
		EMFile::SetDataType dows not take into account if the data type is unsigned or signed as the
		EM file format cannot distinguish them.
	*/
	FileDataType_enum GetDataType();

	//! Converts from internal data type enum to em data type and sets the flag in the file header
	/*!
		EMFile::SetDataType dows not take into account if the data type is unsigned or signed as the
		EM file format cannot distinguish them.
	*/
	void SetDataType(FileDataType_enum aType);

	//! Returns the size of the data block. If the header is not yet read, it will return 0.
	size_t GetDataSize();

	//! Creates a copy for this em-file from \p aHeader.
	void SetFileHeader(EmHeader& aHeader);

	//! Returns a reference to the inner em file header.
	EmHeader& GetFileHeader();

	//! Sets the inner data pointer to \p aData.
	void SetData(char* aData);
	//! Returns the inner data pointer.
	char* GetData();

	//! Reads the projection with index \p aIndex from file and returns it's pointer.
	char* GetProjection(uint aIndex);

	//! Reads the information from the em file header and stores in general Image format.
	void ReadHeaderInfo();
	//! Reads the information from the general Image format and stores it in the em file header.
	void WriteInfoToHeader();

};

void emwrite(string aFileName, float* data, int width, int height);
void emread(string aFileName, float*& data, int& width, int& height);
void emwrite(string aFileName, float* data, int width, int height, int depth);
void emread(string aFileName, float*& data, int& width, int& height, int &depth);

#endif