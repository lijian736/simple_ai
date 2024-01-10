#ifndef _H_SIMPLE_AI_UTILS_LOGGER_H_
#define _H_SIMPLE_AI_UTILS_LOGGER_H_

#include <stdio.h>

#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>

#include "common/common.h"
#include "msg_queue.h"
#include "utils.h"

namespace simple_ai {
namespace utils {

#define LOG_LEVEL_NONE 0x00000000    /* none    */
#define LOG_LEVEL_ERROR 0x00000001   /* error   */
#define LOG_LEVEL_WARNING 0x00000002 /* warning */
#define LOG_LEVEL_INFO 0x00000004    /* info    */
#define LOG_LEVEL_DEBUG 0x00000008   /* debug   */
#define LOG_LEVEL_TRACE 0x00000010   /* trace   */
#define LOG_LEVEL_VERBOSE 0x00000020 /* verbose */

class Logger;
Logger *g_pLogger;
int g_log_level;

#define LOG_INFO(x, ...)                                                                                           \
    do {                                                                                                           \
        if (g_pLogger && (g_log_level >= LOG_LEVEL_INFO)) g_pLogger->info(__FILE__, __LINE__, (x), ##__VA_ARGS__); \
    } while (0)

#define LOG_WARNING(x, ...)                                             \
    do {                                                                \
        if (g_pLogger && (g_log_level >= LOG_LEVEL_WARNING))            \
            g_pLogger->warning(__FILE__, __LINE__, (x), ##__VA_ARGS__); \
    } while (0)

#define LOG_ERROR(x, ...)                                                                                            \
    do {                                                                                                             \
        if (g_pLogger && (g_log_level >= LOG_LEVEL_ERROR)) g_pLogger->error(__FILE__, __LINE__, (x), ##__VA_ARGS__); \
    } while (0)

#define LOG_DEBUG(x, ...)                                                                                            \
    do {                                                                                                             \
        if (g_pLogger && (g_log_level >= LOG_LEVEL_DEBUG)) g_pLogger->debug(__FILE__, __LINE__, (x), ##__VA_ARGS__); \
    } while (0)

#define LOG_TRACE(x, ...)                                                                                            \
    do {                                                                                                             \
        if (g_pLogger && (g_log_level >= LOG_LEVEL_TRACE)) g_pLogger->trace(__FILE__, __LINE__, (x), ##__VA_ARGS__); \
    } while (0)

#define LOG_VERBOSE(x, ...)                                             \
    do {                                                                \
        if (g_pLogger && (g_log_level >= LOG_LEVEL_VERBOSE))            \
            g_pLogger->verbose(__FILE__, __LINE__, (x), ##__VA_ARGS__); \
    } while (0)

struct LogTask {
    LogTask(const char *filename, int linenumber, const char *msgType, const char *strMessage, int msgSize)
        : file_name(filename), line_number(linenumber), msg_type(msgType), message(strMessage, msgSize) {}

    std::string file_name;
    int line_number;
    std::string msg_type;
    std::string message;
};

using LogTaskPtr = std::shared_ptr<LogTask>;

/**
 * @brief A simple logger
 */
class Logger : public SimpleMessageQueue<LogTaskPtr, Logger> {
public:
    Logger(std::size_t max_capacity = 10240);
    virtual ~Logger();

    /**
     * @brief initialize
     *
     * @param prefix_name the file name prefix string
     * @param log_directory the file directory
     * @param is_daily whether file name container daily date info
     * @param file_max_size each file max size, in kb
     * @param file_count max file count
     * @return Status
     */
    Status initialize(const std::string &prefix_name, const std::string &log_directory, bool is_daily,
                      uint32_t file_max_size, uint32_t file_count);
    Status uninitialize();

    void info(const char *filename, int line, const char *format, ...);
    void debug(const char *filename, int line, const char *format, ...);
    void warning(const char *filename, int line, const char *format, ...);
    void error(const char *filename, int line, const char *format, ...);
    void trace(const char *filename, int line, const char *format, ...);
    void verbose(const char *filename, int line, const char *format, ...);

    void handle_msg(const LogTaskPtr &msg);

private:
    /**
     * @brief write log
     *
     * @param filename source file name
     * @param linenumber code line number
     * @param msg_type msg type [E],[W],[I],[D],[V], are error, warning, info, debug, verbose
     * @param message string to write to log
     */
    void write_log(const char *filename, int linenumber, const char *msg_type, const char *message);

    SIMPLE_AI_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Logger);

private:
    bool m_initialized{false};
    std::once_flag m_init_flag;

    // log file prefix
    std::string m_prefix_name;
    // log file directory
    std::filesystem::path m_log_directory;

    // if the log file name contains daily date info
    bool m_is_daily{true};

    // file size, in kb
    uint32_t m_max_file_size{4096};
    // file count
    uint32_t m_max_file_count{100};

    // 1 to 31
    int m_day{0};

    // current file sequence
    int m_seq{0};

    // file path
    std::filesystem::path m_log_file_path;

    // file
    FILE *m_file{nullptr};

    // the printed log size
    std::size_t m_printed_size{0};

    std::mutex m_log_mutex;
};

void set_logger_level(int level) { g_log_level = level; }
bool init_logger(const std::string &prefix, const std::string &directory) {
    auto exist = file_exist(directory);
    if (!exist) {
        bool ret = create_directory_recursively(directory);
        if (!ret) {
            return false;
        }
    }

    Logger *pLogger = new Logger(8000);
    if (pLogger) {
        auto ret = pLogger->initialize(prefix, directory, false, 1024 * 10, 10);
        if (!ret.is_ok()) {
            return false;
        }
    }
    g_pLogger = pLogger;

    return true;
}

void release_logger(){
    if(g_pLogger){
        delete g_pLogger;
        g_pLogger = nullptr;
    }
}

}    // namespace utils
}    // namespace simple_ai

#endif