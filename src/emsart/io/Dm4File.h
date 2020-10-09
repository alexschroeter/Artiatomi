#ifndef DM4FILE_H
#define DM4FILE_H

//#include "Default.h"
#include "FileReader.h"
#include "Image.h"
#include "Dm4FileTagDirectory.h"

//!  Dm4File represents a gatan *.dm4 file in memory and maps contained information to the default internal Image format.
/*!
	Dm4File gives access to header infos, image data.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class Dm4File : public FileReader, public Image
{
public:
	Dm4File(std::string filename);
	virtual ~Dm4File();

	uint version;
	ulong64 fileSize;

	Dm4FileTagDirectory* root;

	uint GetPixelDepthInBytes();
	uint GetImageSizeInBytes();
	char* GetImageData();
	uint GetImageDimensionX();
	uint GetImageDimensionY();
	char* GetThumbnailData();
	uint GetThumbnailDimensionX();
	uint GetThumbnailDimensionY();
	float GetPixelSizeX();
	float GetPixelSizeY();
	float GetExposureTime();
	string GetAcquisitionDate();
	string GetAcquisitionTime();
	int GetCs();
	int GetVoltage();
	int GetMagnification();
	float GetTiltAngle(int aIndex);
	float GetTiltAngle();
	bool OpenAndRead();
	bool OpenAndReadThumbnail();
	FileDataType_enum GetDataType();
	void ReadHeaderInfo();
	void WriteInfoToHeader();
	/*bool Test();
	bool Test2(Dm4FileTagDirectory* dir, int& count, string& ort);*/

private:
	Dm4FileTagDirectory* GetImageDataDir();
	Dm4FileTagDirectory* GetImageTagsDir();
	Dm4FileTagDirectory* GetThumbnailDataDir();
	Dm4FileTagDirectory* GetThumbnailTagsDir();

	friend class Dm4FileStack;
};

#endif
