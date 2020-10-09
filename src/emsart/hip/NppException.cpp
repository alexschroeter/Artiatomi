#if defined(__HIP_PLATFORM_NVCC__) && !defined(__HIP_PLATFORM_HCC__)

#include <hip/hip_runtime.h>
#include "HipException.h"

namespace Hip
{

HipException::HipException()
	: mFileName(), mMessage(), mLine(0)
{

}

HipException::~HipException() throw()
{

}

HipException::HipException(string aMessage)
	: mFileName(), mMessage(aMessage), mLine()
{

}

HipException::HipException(string aFileName, int aLine, string aMessage, hipError_t aErr)
	: mFileName(aFileName), mMessage(aMessage), mLine(aLine), mErr(aErr)
{

}

const char* HipException::what() const throw()
{
	string str = GetMessage();

	char* cstr = new char [str.size()+1];
	strcpy (cstr, str.c_str());

	return cstr;
}

string HipException::GetMessage() const
{
	if (mFileName.length() == 0)
		return mMessage;

	stringstream ss;
	ss << "HIP Driver API error = ";
	ss << mErr << " from file " << mFileName << ", line " << mLine << ": " << mMessage << ".";
	return ss.str();
}






HipfftException::HipfftException()
	: mFileName(), mMessage(), mLine(0)
{

}

HipfftException::~HipfftException() throw()
{

}

HipfftException::HipfftException(string aMessage)
	: mFileName(), mMessage(aMessage), mLine()
{

}

HipfftException::HipfftException(string aFileName, int aLine, string aMessage, hipfftResult aErr)
	: mFileName(aFileName), mMessage(aMessage), mLine(aLine), mErr(aErr)
{

}

const char* HipfftException::what() const throw()
{
	string str = GetMessage();

	char* cstr = new char [str.size()+1];
	strcpy (cstr, str.c_str());

	return cstr;
}

string HipfftException::GetMessage() const
{
	if (mFileName.length() == 0)
		return mMessage;

	stringstream ss;
	ss << "HIP HIPFFT error = ";
	ss << mErr << " from file " << mFileName << ", line " << mLine << ": " << mMessage << ".";
	return ss.str();
}






NppException::NppException()
	: mFileName(), mMessage(), mLine(0)
{

}

NppException::~NppException() throw()
{

}

NppException::NppException(string aMessage)
	: mFileName(), mMessage(aMessage), mLine()
{

}

NppException::NppException(string aFileName, int aLine, string aMessage, NppStatus aErr)
	: mFileName(aFileName), mMessage(aMessage), mLine(aLine), mErr(aErr)
{

}

const char* NppException::what() const throw()
{
	string str = GetMessage();

	char* cstr = new char [str.size()+1];
	strcpy (cstr, str.c_str());

	return cstr;
}

string NppException::GetMessage() const
{
	if (mFileName.length() == 0)
		return mMessage;

	stringstream ss;
	ss << "CUDA NPP error = ";
	ss << mErr << " from file " << mFileName << ", line " << mLine << ": " << mMessage << ".";
	return ss.str();
}



} // namespace


#endif
