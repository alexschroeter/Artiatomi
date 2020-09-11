#include "FileIOException.h"

FileIOException::FileIOException()
	: mFileName(), mMessage()
{

}

FileIOException::~FileIOException() throw()
{

}

FileIOException::FileIOException(string aMessage)
	: mFileName(), mMessage(aMessage)
{

}

FileIOException::FileIOException(string aFileName, string aMessage)
	: mFileName(aFileName), mMessage(aMessage)
{

}

const char* FileIOException::what() const throw()
{
	string str = GetMessage();

	char* cstr = new char [str.size()+1];
	strcpy (cstr, str.c_str());

	return cstr;
}

string FileIOException::GetMessage() const throw()
{
	if (mFileName.length() == 0 && mMessage.length() == 0)
		return "FileIOException";
	if (mFileName.length() == 0 && mMessage.length() > 0)
		return mMessage;

	stringstream ss;
	ss << "Could not access file '";
	ss << mFileName << "'. " << mMessage << endl;
	return ss.str();
}