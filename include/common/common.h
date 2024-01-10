#ifndef _H_SIMPLE_AI_COMMON_COMMON_H_
#define _H_SIMPLE_AI_COMMON_COMMON_H_

#include <iostream>
#include <string>

namespace simple_ai {
namespace common {

/**
 * @brief Status code
 *
 */
enum class StatusCode : uint8_t {
    OK,
    FAIL,
    FILE_NOT_FOUND,
    OUT_OF_MEMORY,
    INVALID_MODEL,
    INVALID_PARAM,
    RUNTIME_ERROR,
    NOT_IMPLEMENTED,
    THREAD_ERROR
};

constexpr const char* statuscode_to_string(StatusCode status) {
    switch (status) {
        case StatusCode::OK:
            return "OK";
        case StatusCode::FAIL:
            return "FAIL";
        case StatusCode::FILE_NOT_FOUND:
            return "FILE_NOT_FOUND";
        case StatusCode::OUT_OF_MEMORY:
            return "OUT_OF_MEMORY";
        case StatusCode::INVALID_PARAM:
            return "INVALID_PARAM";
        case StatusCode::INVALID_MODEL:
            return "INVALID_MODEL";
        case StatusCode::RUNTIME_ERROR:
            return "RUNTIME_ERROR";
        case StatusCode::NOT_IMPLEMENTED:
            return "NOT_IMPLEMENTED";
        case StatusCode::THREAD_ERROR:
            return "THREAD_ERROR";
        default:
            return "UNKNOWN ERROR";
    }
}

inline std::ostream& operator<<(std::ostream& out, const StatusCode& code) {
    return out << statuscode_to_string(code);
}

class Status final {
public:
    Status() = default;
    Status(const Status&) = default;
    Status(Status&&) = default;
    ~Status() = default;
    Status& operator=(Status&&) = default;
    Status& operator=(const Status&) = default;

    Status(StatusCode code, const std::string& msg);
    Status(StatusCode code, const char* msg);
    Status(StatusCode code);

    bool operator==(const Status& rhs) const;
    bool operator!=(const Status& rhs) const;

    StatusCode code() const;
    const std::string& message() const;

    bool is_ok() const;
    std::string to_string() const;

    static Status ok();

private:
    StatusCode m_code{StatusCode::OK};
    std::string m_message;
};

inline std::ostream& operator<<(std::ostream& out, const Status& status) { return out << status.to_string(); }

}    // namespace common

using common::Status;

// MACRO definitions for disabled copy/assign/move member functions
#define SIMPLE_AI_DISALLOW_COPY(Type) Type(const Type&) = delete
#define SIMPLE_AI_DISALLOW_ASSIGNMENT(Type) Type& operator=(const Type&) = delete
#define SIMPLE_AI_DISALLOW_MOVE(Type) \
    Type(Type&&) = delete;            \
    Type& operator=(Type&&) = delete

#define SIMPLE_AI_DISALLOW_COPY_AND_ASSIGNMENT(Type) \
    SIMPLE_AI_DISALLOW_COPY(Type);                   \
    SIMPLE_AI_DISALLOW_ASSIGNMENT(Type)

#define SIMPLE_AI_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Type) \
    SIMPLE_AI_DISALLOW_COPY_AND_ASSIGNMENT(Type);         \
    SIMPLE_AI_DISALLOW_MOVE(Type)

}    // namespace simple_ai

#endif