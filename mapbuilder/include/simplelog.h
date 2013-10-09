#ifndef SIMPLELOG_H
#define SIMPLELOG_H

#include <stdio.h>
#include <stdarg.h>
#include <string>

#include <mrpt/synch.h>
#include <mrpt/system.h>
#include <mrpt/base.h>

#define LOGPRINTF(...) {	                            \
	SimpleLogger *logger = SimpleLogger::getInstance(); \
	mrpt::synch::CCriticalSectionLocker lock( &logger->mutex );      \
	logger->silent(true);                               \
	logger->logtimestamp();                       \
	logger->logf("%s:%s@%d ",__FILE__,__FUNCTION__,__LINE__);  \
	logger->silent(false);      \
	logger->logf(__VA_ARGS__);  \
}


//[RR] drop-in replacements for LOG4CXX_INFO etc...  s/LOG4CXX_/SIMPLELOGGER_/g;
#define SIMPLELOGGER_PREFIX(prefix,...) {             \
	SimpleLogger *logger = SimpleLogger::getInstance(); \
	mrpt::synch::CCriticalSectionLocker lock( &logger->mutex );      \
	logger->logtimestamp();      \
	logger->logf(prefix);        \
	logger->logf(__VA_ARGS__);   \
	logger->logf("\n");          \
}
#define ROS_INFO(...)  SIMPLELOGGER_PREFIX(" [INFO] ", __VA_ARGS__)
#define SIMPLELOGGER_INFO(...)  SIMPLELOGGER_PREFIX(" [INFO] ", __VA_ARGS__)
#define SIMPLELOGGER_WARN(...)  SIMPLELOGGER_PREFIX(" [WARN] ", __VA_ARGS__)
#define SIMPLELOGGER_ERROR(...) SIMPLELOGGER_PREFIX(" [ERROR] ",__VA_ARGS__)

#define SIMPLELOGGER_DEBUG(...) SIMPLELOGGER_PREFIX(" [DEBUG] ",__VA_ARGS__)
//#define SIMPLELOGGER_DEBUG(...)

//#define SIMPLELOGGER_DEBUG_MORE(...) SIMPLELOGGER_PREFIX(" [DEBUG] ",__VA_ARGS__)
#define SIMPLELOGGER_DEBUG_MORE(...)



class SimpleLogger {
public:
	SimpleLogger();
	~SimpleLogger();

	static SimpleLogger* getInstance();

	void silent(bool status);
	void logtimestamp();
	void logf(const char *pstring, ...);
	void logf( std::string str );

	mrpt::synch::CCriticalSection mutex;

protected:
	FILE *m_flog;
	bool m_silent;
	static SimpleLogger* s_logger;
	//unsigned long m_startTime;
   	mrpt::system::TTimeStamp start;

};

#endif
