#ifndef FILEWRITER_H
#define FILEWRITER_H

#include "IODefault.h"
#include "File.h"

using namespace std;

//!  FileWriter provides endianess independent file write methods. 
/*!
	FileWriter is an abstract class.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class FileWriter : public File
{
public:
	//! Creates a new FileWriter instance for file \p aFileName. File endianess is set to little Endian.
	FileWriter(string aFileName);
	//! Creates a new FileWriter instance for file \p aFileName. File endianess is set to \p aIsLittleEndian.
	FileWriter(string aFileName, bool aIsLittleEndian);
	//! Creates a new FileWriter instance for file stream \p aStream. File endianess is set to little Endian.
	FileWriter(fstream* aStream);
	//! Creates a new FileWriter instance for file stream \p aStream. File endianess is set to \p aIsLittleEndian.
	FileWriter(fstream* aStream, bool aIsLittleEndian);
	
	virtual bool OpenAndWrite() = 0;
	virtual void SetDataType(FileDataType_enum aType) = 0;

protected:	
	void WriteBE(ulong64& aX);
	void WriteLE(ulong64& aX);
	void WriteBE(uint& aX);
	void WriteLE(uint& aX);
	void WriteBE(ushort& aX);
	void WriteLE(ushort& aX);
	void Write(uchar& aX);
	void WriteBE(long64& aX);
	void WriteLE(long64& aX);
	void WriteBE(int& aX);
	void WriteLE(int& aX);
	void WriteBE(short& aX);
	void WriteLE(short& aX);
	void Write(char& aX);
	void WriteBE(double& aX);
	void WriteLE(double& aX);
	void WriteBE(float& aX);
	void WriteLE(float& aX);
	void Write(char* aX, uint aCount);

	bool OpenWrite();
	void CloseWrite();
};

#endif