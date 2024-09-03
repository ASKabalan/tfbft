/**
 * @file logger.hpp
 * @version 0.4.0
 * @brief Logger for C++ with timestamp, name, and configurable options via TRACE environment variable.
 *
 * Environment variable:
 * - TRACE: Enables trace for specific logger names or tags, and accepts additional parameters:
 *   - `-v`: Enables verbose logging.
 *   - `-o FOLDER`: Sets the output directory for log files.
 *
 * Example usage:
 * @code
 * #include "logger.hpp"
 *
 * int main() {
 *     Logger logger("CUD");
 *
 *     StartTraceInfo(logger) << "This is an info message" << '\n';
 *     StartTraceVerbose(logger) << "This is a verbose message" << '\n';
 *
 *     return 0;
 * }
 * @endcode
 *
 * Logger for C++
 * configurable via the TRACE environment variable.
 *
 * Author: Wassim KABALAN
 */

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <string>

#ifdef MPI_VERSION
#include <mpi.h>
#endif

class Logger {
public:
    Logger(const std::string &name)
        : name(name), traceInfo(false), traceVerbose(false), traceToConsole(true) {
        static const char *traceEnv = std::getenv("TRACE");
        if (traceEnv != nullptr) {
            std::string traceString = traceEnv;
            parseTraceString(traceString);
            if (traceString.find(name) != std::string::npos) {
                traceInfo = true;
            }
        }
    }

    Logger &startTraceInfo() {
        if (traceInfo || traceVerbose) {
            std::ostringstream ss;
            addTimestamp(ss);
            ss << "[INFO] ";
            ss << "[" << name << "] ";
            output(ss.str());
        }
        return *this;
    }

    Logger &startTraceVerbose() {
        if (traceVerbose) {
            std::ostringstream ss;
            addTimestamp(ss);
            ss << "[VERB] ";
            ss << "[" << name << "] ";
            output(ss.str());
        }
        return *this;
    }

    template <typename T>
    Logger &operator<<(const T &value) {
        if (traceInfo || traceVerbose) {
            std::ostringstream ss;
            ss << value;
            output(ss.str());
        }
        return *this;
    }

    Logger &operator<<(std::ostream &(*manipulator)(std::ostream &)) {
        if (traceInfo || traceVerbose) {
            std::ostringstream ss;
            ss << manipulator;
            output(ss.str());
        }
        return *this;
    }

    bool getTraceInfo() const { return traceInfo; }
    bool getTraceVerbose() const { return traceVerbose; }

    void addStackTrace() {
        if (traceInfo || traceVerbose) {
            std::ostringstream ss;
            ss << "Call stack:" << std::endl;

            const int max_frames = 64;
            void *frame_ptrs[max_frames];
            int num_frames = backtrace(frame_ptrs, max_frames);
            char **symbols = backtrace_symbols(frame_ptrs, num_frames);

            if (symbols == nullptr) {
                output("Error retrieving backtrace symbols.\n");
                return;
            }

            for (int i = 0; i < num_frames; ++i) {
                ss << symbols[i] << std::endl;
            }

            free(symbols);
            output(ss.str());
        }
    }

private:
    void addTimestamp(std::ostringstream &stream) {
        auto now = std::chrono::system_clock::now();
        auto timePoint = std::chrono::system_clock::to_time_t(now);
        auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
                                now.time_since_epoch()) %
                            1000;

        std::tm tm;
#ifdef _WIN32
        localtime_s(&tm, &timePoint);
#else
        localtime_r(&timePoint, &tm);
#endif

        stream << "[" << tm.tm_year + 1900 << "/"
               << tm.tm_mon + 1 << "/"
               << tm.tm_mday << " "
               << tm.tm_hour << ":"
               << tm.tm_min << ":"
               << tm.tm_sec << ":"
               << milliseconds.count() << "] ";
    }

    void output(const std::string &message) {
        if (traceToConsole) {
            std::cout << message;
        } else {
            std::ostringstream filename;
            std::string rankStr = rank >= 0 ? "_" + std::to_string(rank) : "";
            filename << outputDir << "/Trace_" << name << rankStr << ".log";

            std::ofstream outfile(filename.str(), std::ios::app);
            if (outfile.is_open()) {
                outfile << message;
                outfile.close();
            }
        }
    }

    void parseTraceString(const std::string &traceString) {
        std::istringstream iss(traceString);
        std::string token;
        while (iss >> token) {
            if (token == "-v") {
                traceVerbose = true;
                traceInfo = true; // Verbose implies info tracing.
            } else if (token == "-o") {
                if (iss >> token) {
                    outputDir = token;
                    traceToConsole = false;
                }
            }
        }
    }

    std::string name;
    std::string outputDir;
    bool traceInfo;
    bool traceVerbose;
    bool traceToConsole;
    int rank = -1;
};

#define StartTraceInfo(logger) \
    if (logger.getTraceInfo()) \
    logger.startTraceInfo()

#define TraceInfo(logger) \
    if (logger.getTraceInfo()) \
    logger

#define PrintStack(logger) \
    if (logger.getTraceInfo()) \
    logger.addStackTrace()

#define StartTraceVerbose(logger) \
    if (logger.getTraceVerbose()) \
    logger.startTraceVerbose()

#define TraceVerbose(logger) \
    if (logger.getTraceVerbose()) \
    logger

#endif // LOGGER_HPP

/*
Example usage:

#include "logger.hpp"

int main() {
    Logger logger("CUD");

    StartTraceInfo(logger) << "This is an info message" << '\n';
    StartTraceVerbose(logger) << "This is a verbose message" << '\n';

    return 0;
}
*/
