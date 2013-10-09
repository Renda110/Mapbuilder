#include "simplelog.h"
#include <mrpt/system.h>

#include <string>


SimpleLogger::SimpleLogger()
{
	m_silent = false;
    /*
        get process name
        TODO RR breaks with batch file or service?
        std::string cmd = GetCommandLineA();
        using process name instead:
        char module_name[MAX_PATH];

        int ret = GetModuleFileName(0, (LPTSTR)module_name, MAX_PATH);

        if (ret == 0)
{
            sprintf(module_name,"Failed.Fail");
        }

        std::string cmd = std::string( module_name );

        int dot = cmd.find_first_of('.');
        int start = cmd.rfind('\\',dot);

        std::string name = cmd.substr(start+1,dot-(start+1));
        std::string path = std::string( Configuration::getInstance()->getLogPath() );
        std::string filename = path + "\\" + name + ".txt";

        mrpt::system::createDirectory( path );
    */

	std::string filename = "mapbuilder.txt";
	m_flog = fopen( filename.c_str(),"a");

    if (m_flog == NULL)
    {
		printf(">>>>>>>>>>>>>>>>>>>>>>>Failed to open %s!<<<<<<<<<<<<<<<<<<<<<<<<\n",filename.c_str());
	}

	start = mrpt::system::now();//timeGetTime();
}

SimpleLogger::~SimpleLogger()
{
	if (m_flog)
    {
		fclose(m_flog);
    }
}

SimpleLogger* SimpleLogger::getInstance()
{
	if (!s_logger)
    {
		s_logger = new SimpleLogger;
    }

	return s_logger;
}

void SimpleLogger::silent( bool status )
{
	m_silent = status;
}

void SimpleLogger::logf( const char *pstring, ... )
{
	va_list vargs;
	va_list vargs2;
	va_start(vargs, pstring);
	va_copy(vargs2, vargs);

	if (!m_silent)
    {
		vprintf(pstring, vargs);
    }

    if (m_flog)
    {
		vfprintf(m_flog,pstring,vargs2);
    }

    va_end(vargs);

	/* TODO: Figure out why this crashes!*/
	//RR the mutex should fix it?
	if (m_flog)
    {
		fflush(m_flog);
    }
}

void SimpleLogger::logf( std::string str )
{
	logf( str.c_str() );
}

void SimpleLogger::logtimestamp()
{
    mrpt::system::TTimeStamp diff = mrpt::system::now() - start;
//	DWORD tDiff = timeGetTime() - m_startTime;
	logf("[%6d]",diff/10000);
}

SimpleLogger* SimpleLogger::s_logger = 0;
