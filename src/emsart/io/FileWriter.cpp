#include "FileWriter.h"

FileWriter::FileWriter(string aFileName)
  : File(aFileName, true)
{

}

FileWriter::FileWriter(string aFileName, bool aIsLittleEndian)
  : File(aFileName, aIsLittleEndian)
{

}
FileWriter::FileWriter(fstream* aStream)
  : File(aStream, true)
{

}

FileWriter::FileWriter(fstream* aStream, bool aIsLittleEndian)
  : File(aStream, aIsLittleEndian)
{

}

bool FileWriter::OpenWrite(bool append)
{
  if (append)
    {
      mFile->open(mFileName.c_str(), ios_base::out | ios_base::binary | ios_base::in);
    }
  else
    {
      mFile->open(mFileName.c_str(), ios_base::out | ios_base::binary);
    }
  return mFile->is_open() && mFile->good();
}

void FileWriter::CloseWrite()
{
  mFile->close();
}

bool FileWriter::OpenAndWriteAppend(char * aData, uint aCount)
{
  bool ok = true;
  mFile->open(mFileName.c_str(), fstream::out | fstream::binary | fstream::app);
  ok = mFile->is_open() && mFile->good();
	
  if (!ok) 
    return ok;

  //	size_t pos1 = mFile->tellp();
  //mFile->seekp(0, ios_base::end); 
  //	size_t pos2 = mFile->tellp();

  mFile->write(aData, aCount);
  //	size_t pos3 = mFile->tellp();

  ok = mFile->is_open() && mFile->good();
  mFile->close();
	
  return ok;
}


void FileWriter::WriteBE(ulong64& aX)
{	
  if (mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 8);
}

void FileWriter::WriteLE(ulong64& aX)
{
  if (!mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 8);
}

void FileWriter::WriteBE(uint& aX)
{
  if (mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 4);
}

void FileWriter::WriteLE(uint& aX)
{
  if (!mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 4);
}

void FileWriter::WriteBE(ushort& aX)
{
  if (mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 2);
}

void FileWriter::WriteLE(ushort& aX)
{
  if (!mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 2);
}

void FileWriter::Write(uchar& aX)
{
  mFile->write((char*)&aX, 1);
}

void FileWriter::WriteBE(long64& aX)
{
  if (mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 8);
}

void FileWriter::WriteLE(long64& aX)
{
  if (!mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 8);
}

void FileWriter::WriteBE(int& aX)
{
  if (mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 4);
}

void FileWriter::WriteLE(int& aX)
{
  if (!mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 4);
}

void FileWriter::WriteBE(short& aX)
{
  if (mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 2);
}

void FileWriter::WriteLE(short& aX)
{
  if (!mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 2);
}

void FileWriter::Write(char& aX)
{
  mFile->write((char*)&aX, 1);
}

void FileWriter::WriteBE(double& aX)
{
  if (mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 8);
}

void FileWriter::WriteLE(double& aX)
{
  if (!mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 8);
}

void FileWriter::WriteBE(float& aX)
{
  if (mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 4);
}

void FileWriter::WriteLE(float& aX)
{
  if (!mIsLittleEndian) Endian_swap(aX);

  mFile->write((char*)&aX, 4);
}

void FileWriter::Write(char* aX, uint aCount)
{
  mFile->write(aX, aCount);
}
