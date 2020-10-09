#ifndef SIMPLELOGGER_H
#define SIMPLELOGGER_H

#include "UtilsDefault.h"
#include "../io/FileIOException.h"
#include <ctime>

using namespace std;

class SimpleLogger
{
public:
	enum SimpleLogLevel { LOG_QUIET, LOG_ERROR, LOG_INFO, LOG_DEBUG };
	SimpleLogger(string aFilename, SimpleLogLevel aLevel, bool aOff);
	~SimpleLogger();

	friend SimpleLogger& operator<<(SimpleLogger& logger, const SimpleLogger::SimpleLogLevel& level);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const char* val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const string& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const int& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, bool val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const unsigned int& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const float& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const double& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const dim3& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const int3& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const int4& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const float3& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const float4& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, const FileDataType_enum& val);
	friend SimpleLogger& operator<<(SimpleLogger& logger, ostream& (*f)(std::ostream&));

private:
	void printLevel();
	string _filename;
	SimpleLogLevel _level;
	SimpleLogLevel _currentLevel;
	ofstream _stream;
	bool _off;
	bool _isNewLine;
};

SimpleLogger& operator<<(SimpleLogger& logger, const SimpleLogger::SimpleLogLevel& level);
SimpleLogger& operator<<(SimpleLogger& logger, const char* val);
SimpleLogger& operator<<(SimpleLogger& logger, const string& val);
SimpleLogger& operator<<(SimpleLogger& logger, bool val);
SimpleLogger& operator<<(SimpleLogger& logger, const int& val);
SimpleLogger& operator<<(SimpleLogger& logger, const unsigned int& val);
SimpleLogger& operator<<(SimpleLogger& logger, const float& val);
SimpleLogger& operator<<(SimpleLogger& logger, const double& val);
SimpleLogger& operator<<(SimpleLogger& logger, const dim3& val);
SimpleLogger& operator<<(SimpleLogger& logger, const int3& val);
SimpleLogger& operator<<(SimpleLogger& logger, const int4& val);
SimpleLogger& operator<<(SimpleLogger& logger, const float3& val);
SimpleLogger& operator<<(SimpleLogger& logger, const float4& val);
SimpleLogger& operator<<(SimpleLogger& logger, const FileDataType_enum& val);
SimpleLogger& operator<<(SimpleLogger& logger, ostream& (*f)(std::ostream&) );

#endif