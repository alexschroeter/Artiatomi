#ifndef DM4FILESTACK_H
#define DM4FILESTACK_H

#include "ProjectionSource.h"
#include "Dm4File.h"

//!  Dm4FileStack represents a tilt series in gatan's dm4 format.
/*!
	Dm4FileStack reads all projections with the same name base into one volume in memory.
	The provided file name must end with file index "0000".
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class Dm4FileStack : public ProjectionSource
{
private:
	bool fexists(std::string filename);
	std::string GetStringFromInt(int aInt);
	std::string GetFileNameFromIndex(int aIndex, std::string aFileName);
	int CountFilesInStack(std::string aFileName);
	int _fileCount;
	int _firstIndex;

	std::vector<Dm4File*> _dm4files;

public:
	Dm4FileStack(std::string aFileName);
	virtual ~Dm4FileStack();

	bool OpenAndRead();
	FileDataType_enum GetDataType();
	size_t GetDataSize();
	char* GetData();
	char* GetProjection(uint aIndex);
	float* GetProjectionFloat(uint aIndex);
	float* GetProjectionInvertFloat(uint aIndex);
	void ReadHeaderInfo();
	void WriteInfoToHeader();
	bool OpenAndWrite();
	void SetDataType(FileDataType_enum aType);

};

#endif
