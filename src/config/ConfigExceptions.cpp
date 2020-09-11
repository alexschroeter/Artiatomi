#include "ConfigExceptions.h"

namespace Configuration
{
	ConfigException::ConfigException()
	{

	}

	ConfigException::~ConfigException() throw()
	{

	}

	const char* ConfigException::what() const throw()
	{
		return "ConfigException";
	}

	string ConfigException::GetMessage()
	{
		return "ConfigException";
	}

	ConfigValueException::ConfigValueException()
		:mConfigFile(), mConfigEntry(), mType()
	{
	
	}
	
	ConfigValueException::~ConfigValueException() throw()
	{

	}

	ConfigValueException::ConfigValueException(string aConfigFile, string aConfigEntry, string aType)
		:mConfigFile(aConfigFile), mConfigEntry(aConfigEntry), mType(aType)
	{
	
	}

	const char* ConfigValueException::what() const throw()
	{
		string str = GetMessage();

		char* cstr = new char [str.size()+1];
		strcpy (cstr, str.c_str());

		return cstr;
	}

	string ConfigValueException::GetMessage() const throw()
	{
		string retVal = "The value for property '";
		retVal += mConfigEntry + "' in file '" + mConfigFile + "' doesn't match it's type. It should be of type '";
		retVal += mType + "'.";
		return retVal;
	}

	void ConfigValueException::setValue(string aConfigFile, string aConfigEntry, string aType)
	{
		mConfigFile = aConfigFile;
		mConfigEntry = aConfigEntry;
		mType = aType;
	}

	ConfigPropertyException::ConfigPropertyException()
		:mConfigFile(), mConfigEntry()
	{
	
	}

	ConfigPropertyException::~ConfigPropertyException() throw()
	{
	
	}

	ConfigPropertyException::ConfigPropertyException(string aConfigFile, string aConfigEntry)
		:mConfigFile(aConfigFile), mConfigEntry(aConfigEntry)
	{
	
	}

	const char* ConfigPropertyException::what() const throw()
	{
		string str = GetMessage();

		char* cstr = new char [str.size()+1];
		strcpy (cstr, str.c_str());

		return cstr;
	}

	string ConfigPropertyException::GetMessage() const throw()
	{
		string retVal = "The property '";
		retVal += mConfigEntry + "' is missing in file '" + mConfigFile + "'.";
		return retVal;
	}

	void ConfigPropertyException::setValue(string aConfigFile, string aConfigEntry)
	{
		mConfigFile = aConfigFile;
		mConfigEntry = aConfigEntry;
	}


	ConfigFileException::ConfigFileException()
		:mConfigFile()
	{
	
	}

	ConfigFileException::~ConfigFileException() throw()
	{
	
	}

	ConfigFileException::ConfigFileException(string aConfigFile)
		:mConfigFile(aConfigFile)
	{
	
	}

	const char* ConfigFileException::what() const throw()
	{
		string str = GetMessage();

		char* cstr = new char [str.size()+1];
		strcpy (cstr, str.c_str());

		return cstr;
	}

	string ConfigFileException::GetMessage() const throw()
	{
		string retVal = "Cannot read the configuration file '";
		retVal += mConfigFile + "'.";
		return retVal;
	}

	void ConfigFileException::setValue(string aConfigFile)
	{
		mConfigFile = aConfigFile;
	}
}
