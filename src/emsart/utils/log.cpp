#include "log.h"

#ifdef _DEBUG
LogLevel logLevel = LOG_DEBUG;
#else
LogLevel logLevel = LOG_INFO;
#endif

