#ifndef FILEIOEXCEPETION_H
#define FILEIOEXCEPETION_H

#include "IODefault.h"
#include <sstream>

using namespace std;

//!  An exception thrown while accessing Files. 
/*!
	FileIOException is thrown, if a file cannot be accessed as intended.
	\author Michael Kunz
	\date   September 2011
	\version 1.0
*/
class FileIOException : public exception
{
protected:
	string mFileName;
	string mMessage;
	
public:
	FileIOException();

	~FileIOException() throw();

	
	//! FileIOException constructor
	/*!
		\param aMessage Ecxeption message
	*/
	//FileIOException constructor
	FileIOException(string aMessage);

	//! FileIOException constructor
	/*!
		\param aFileName Name of the file provoking the exception
		\param aMessage Ecxeption message
	*/
	//FileIOException constructor
	FileIOException(string aFileName, string aMessage);
		
	//! Returns "FileIOException"
	//Returns "FileIOException"
	virtual const char* what() const throw();
		
	//! Returns an error message
	//Returns an error message
	virtual string GetMessage() const throw();
};

#endif