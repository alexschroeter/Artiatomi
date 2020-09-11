#ifndef FILE_H
#define FILE_H

#include "IODefault.h"
#include "FileIOException.h"

using namespace std;

//!  File is a very simple class wrapping a file stream. 
/*!
	File only wrapps a file stream and provides a filename and endian-swap methods.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class File
{
protected:
	fstream* mFile;
	string mFileName;

	bool mIsLittleEndian;

	void Endian_swap(ushort& x);
	void Endian_swap(uint& x);
	void Endian_swap(ulong64& x);
	void Endian_swap(short& x);
	void Endian_swap(int& x);
	void Endian_swap(long64& x);
	void Endian_swap(double& x);
	void Endian_swap(float& x);

public:
	//! Creates a new instance of File with name \p aFileName and endianess \p aIsLittleEndian.
	File(string aFileName, bool aIsLittleEndian);
	//! Creates a new instance of File with filestram given by \p aStream and endianess \p aIsLittleEndian.
	File(fstream* aStream, bool aIsLittleEndian);
	virtual ~File();


};

#endif