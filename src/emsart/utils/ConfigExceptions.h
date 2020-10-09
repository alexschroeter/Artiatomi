#ifndef CONFIGEXCEPTIONS_H
#define CONFIGEXCEPTIONS_H

#include "UtilsDefault.h"

using namespace std;

namespace Configuration
{
	//! Baseclass for exceptions occuring while processing a configuration file
	//Baseclass for exceptions occuring while processing a configuration file
	class ConfigException: public exception
	{
	protected:

	public:
		ConfigException();

		~ConfigException() throw();

		virtual const char* what() const throw();

		virtual string GetMessage() const;
	};
	
	//! Thrown when a value in the configuration file has the wrong value / type
	//Thrown when a value in the configuration file has the wrong value / type
	class ConfigValueException: public ConfigException
	{
		private:
			string mConfigFile, mConfigEntry, mType;

		public:
			ConfigValueException();

			~ConfigValueException() throw();

			ConfigValueException(string aConfigFile, string aConfigEntry, string aType);

			virtual const char* what() const throw();

			void setValue(string aConfigFile, string aConfigEntry, string aType);

			virtual string GetMessage() const throw();
	};
	
	//! Thrown when a missing property is fetched
	//Thrown when a missing property is fetched
	class ConfigPropertyException: public ConfigException
	{
		private:
			string mConfigFile, mConfigEntry;

		public:
			ConfigPropertyException();

			~ConfigPropertyException() throw();

			ConfigPropertyException(string aConfigFile, string aConfigEntry);

			virtual const char* what() const throw();

			void setValue(string aConfigFile, string aConfigEntry);

			virtual string GetMessage() const throw();
	};
	
	//! Thrown when the configuration file can't be read
	//Thrown when the configuration file can't be read
	class ConfigFileException: public ConfigException
	{
		private:
			string mConfigFile;

		public:
			ConfigFileException();

			~ConfigFileException() throw();

			ConfigFileException(string aConfigFile);

			virtual const char* what() const throw();

			void setValue(string aConfigFile);

			virtual string GetMessage() const throw();
	};
} //end namespace Configuration
#endif //CONFIGEXCEPTIONS_H
