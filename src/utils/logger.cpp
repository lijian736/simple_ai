#include "utils/logger.h"

#include <stdarg.h>
#include <string.h>
#include <sys/types.h>
#include <time.h>

#include <new>
#include <sstream>
#include <thread>

#include "utils/utils.h"

using namespace simple_ai::common;

namespace simple_ai {
namespace utils {

Logger::Logger(std::size_t max_capacity) : SimpleMessageQueue(max_capacity) {}

Logger::~Logger() { uninitialize(); }

Status Logger::initialize(const std::string &prefix_name, const std::string &log_directory, bool is_daily,
                          uint32_t file_max_size, uint32_t file_count) {
    if (!file_exist(log_directory)) {
        return Status(StatusCode::FAIL, "directory does not exist: " + log_directory);
    }

    std::call_once(m_init_flag, [&] {
        if (prefix_name.length() > 128) {
            m_prefix_name = prefix_name.substr(0, 128);
        } else {
            m_prefix_name = prefix_name;
        }

        m_log_directory = log_directory;
        m_is_daily = is_daily;
        m_max_file_size = std::min<uint32_t>(file_max_size, 1024 * 1024);
        m_max_file_count = std::min<uint32_t>(file_count, 100);

        start();
        m_initialized = true;
    });

    return Status::ok();
}

Status Logger::uninitialize() {
    if (!m_initialized) {
        return Status::ok();
    }

    m_initialized = false;

    stop();

    if (nullptr != m_file) {
        fclose(m_file);
        m_file = nullptr;
        m_printed_size = 0;
    }

    return Status::ok();
}

void Logger::handle_msg(const LogTaskPtr &msg) {
    write_log(msg->file_name.c_str(), msg->line_number, msg->msg_type.c_str(), msg->message.c_str());
}

void Logger::write_log(const char *filename, int linenumber, const char *msgtype, const char *message) {
    time_t now = time(NULL);

    struct tm *tmnow;
    struct tm tmtmp;
    tmnow = localtime_r(&now, &tmtmp);
    if (nullptr == tmnow) {
        return;
    }

    const char *pFileName = filename;
    pFileName = strrchr(filename, '/');
    if (nullptr == pFileName) {
        pFileName = filename;
    } else {
        ++pFileName;
    }

    char szNow[32] = {0};
    strftime(szNow, sizeof(szNow), "[%y-%m-%d %H:%M:%S]", tmnow);

    if ((m_is_daily && m_day != tmnow->tm_mday) || m_printed_size > m_max_file_size * 1024 || !m_file) {
        if (m_is_daily && m_day != tmnow->tm_mday) {
            m_seq = 0;
        }

        bool bLoop = false;
        std::filesystem::path log_file;
        do {
            char szFileName[256] = {0};
            m_seq = m_seq % m_max_file_count + 1;
            if (m_is_daily) {
                sprintf(szFileName, "%s-%d-%.2d-%.2d.%d.log", m_prefix_name.c_str(), tmnow->tm_year + 1900,
                        tmnow->tm_mon + 1, tmnow->tm_mday, m_seq);
            } else {
                sprintf(szFileName, "%s.%d.log", m_prefix_name.c_str(), m_seq);
            }

            log_file = m_log_directory;
            log_file /= szFileName;
            if (m_log_file_path.empty()) {
                if (bLoop) {
                    break;
                }

                FILE *pFile = fopen(log_file.c_str(), "rb");
                if (nullptr == pFile) {
                    break;
                } else {
                    fclose(pFile);
                    pFile = nullptr;
                }

                if (m_seq == m_max_file_count) {
                    bLoop = true;
                }
            }
        } while (m_log_file_path.empty());

        m_log_file_path = log_file;
        if (nullptr != m_file) {
            fclose(m_file);
            m_file = nullptr;
            m_printed_size = 0;
        }

        m_file = fopen(m_log_file_path.c_str(), "wb");
        if (nullptr != m_file) {
            setvbuf(m_file, nullptr, _IONBF, 0);
            m_printed_size = 0;
            m_day = tmnow->tm_mday;
        }
    }

    if (m_file) {
        int size = fprintf(m_file, "%s%s %s(%d):\t\t%s\r\n", szNow, msgtype, pFileName, linenumber, message);
        fflush(m_file);
        if (size > 0) {
            m_printed_size += size;
        }
    }
}

void Logger::info(const char *filename, int line, const char *format, ...) {
    char szLog[1024] = {0};

    va_list a_list;
    va_start(a_list, format);
    int nLog = vsnprintf(szLog, 1000, format, a_list);
    va_end(a_list);

    if (nLog > 0) {
        put(std::make_shared<LogTask>(filename, line, "[I]", szLog, nLog));
    }
}

void Logger::debug(const char *filename, int line, const char *format, ...) {
    char szLog[1024] = {0};

    va_list a_list;
    va_start(a_list, format);
    int nLog = vsnprintf(szLog, 1000, format, a_list);
    va_end(a_list);

    if (nLog > 0) {
        put(std::make_shared<LogTask>(filename, line, "[D]", szLog, nLog));
    }
}

void Logger::warning(const char *filename, int line, const char *format, ...) {
    char szLog[1024] = {0};

    va_list a_list;
    va_start(a_list, format);
    int nLog = vsnprintf(szLog, 1000, format, a_list);
    va_end(a_list);

    if (nLog > 0) {
        put(std::make_shared<LogTask>(filename, line, "[W]", szLog, nLog));
    }
}

void Logger::error(const char *filename, int line, const char *format, ...) {
    char szLog[1024] = {0};

    va_list a_list;
    va_start(a_list, format);
    int nLog = vsnprintf(szLog, 1000, format, a_list);
    va_end(a_list);

    if (nLog > 0) {
        put(std::make_shared<LogTask>(filename, line, "[E]", szLog, nLog));
    }
}

void Logger::trace(const char *filename, int line, const char *format, ...) {
    char szLog[1024] = {0};

    va_list a_list;
    va_start(a_list, format);
    int nLog = vsnprintf(szLog, 1000, format, a_list);
    va_end(a_list);

    if (nLog > 0) {
        put(std::make_shared<LogTask>(filename, line, "[T]", szLog, nLog));
    }
}

void Logger::verbose(const char *filename, int line, const char *format, ...) {
    char szLog[1024] = {0};

    va_list a_list;
    va_start(a_list, format);
    int nLog = vsnprintf(szLog, 1000, format, a_list);
    va_end(a_list);

    if (nLog > 0) {
        put(std::make_shared<LogTask>(filename, line, "[V]", szLog, nLog));
    }
}

}    // namespace utils
}    // namespace simple_ai