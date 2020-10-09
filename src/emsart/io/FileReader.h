#ifndef FILEREADER_H
#define FILEREADER_H

#include "IODefault.h"
#include "File.h"

using namespace std;

//!  FileReader provides endianess independent file read methods. 
/*!
	FileReader is an abstract class.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class FileReader : public File
{
public:
	//! Creates a new FileReader instance for file \p aFileName. File endianess is set to little Endian.
	FileReader(string aFileName);
	//! Creates a new FileReader instance for file \p aFileName. File endianess is set to \p aIsLittleEndian.
	FileReader(string aFileName, bool aIsLittleEndian);
	//! Creates a new FileReader instance for file stream \p aStream. File endianess is set to little Endian.
	FileReader(fstream* aStream);
	//! Creates a new FileReader instance for file stream \p aStream. File endianess is set to \p aIsLittleEndian.
	FileReader(fstream* aStream, bool aIsLittleEndian);
	
	virtual bool OpenAndRead() = 0;
	virtual FileDataType_enum GetDataType() = 0;
protected:
	long64 ReadI8LE();
	long64 ReadI8BE();
	int ReadI4LE();
	int ReadI4BE();
	short ReadI2LE();
	short ReadI2BE();
	char ReadI1();
	ulong64 ReadUI8LE();
	ulong64 ReadUI8BE();
	uint ReadUI4LE();
	uint ReadUI4BE();
	ushort ReadUI2LE();
	ushort ReadUI2BE();
	uchar ReadUI1();
	float ReadF4LE();
	float ReadF4BE();
	double ReadF8LE();
	double ReadF8BE();
	string ReadStr(int aCount);
	string ReadStrUTF(int aCount);
	bool OpenRead();
	void CloseRead();

};

#endif