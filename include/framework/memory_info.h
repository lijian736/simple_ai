#ifndef _H_SIMPLE_AI_FRAMEWORK_MEMORY_INFO_H_
#define _H_SIMPLE_AI_FRAMEWORK_MEMORY_INFO_H_

#include <algorithm>
#include <cstdint>
#include <sstream>
#include <string>

#include "common/hash_combine.h"
#include "common_defines.h"
#include "device.h"

using namespace simple_ai::common;

namespace simple_ai {
namespace framework {

/**
 * @brief The memory information
 *
 */
struct MemoryInfo {
    MemoryInfo() = default;

    MemoryInfo(const std::string& name_p, AllocatorType alloc_p, Device device_p = Device(), int id_p = 0,
               MemoryType mem_type_p = MemoryType::DEFAULT)
        : name(name_p), id(id_p), mem_type(mem_type_p), alloc_type(alloc_p), device(device_p) {}

    bool operator<(const MemoryInfo& rhs) const {
        if (alloc_type != rhs.alloc_type) {
            return alloc_type < rhs.alloc_type;
        }

        if (mem_type != rhs.mem_type) {
            return mem_type < rhs.mem_type;
        }

        if (id != rhs.id) {
            return id < rhs.id;
        }

        if (device != rhs.device) {
            return device < rhs.device;
        }

        return name < rhs.name;
    }

    size_t hash() const {
        auto h = std::hash<int>()(static_cast<int>(alloc_type));
        hash_combine(mem_type, h);
        hash_combine(id, h);
        hash_combine(device, h);
        hash_combine<std::string>(name, h);

        return h;
    }

    std::string to_string() const {
        std::ostringstream ostr;
        ostr << "MemoryInfo:["
             << "name:" << name << " id:" << id << " MemoryType:" << mem_type << " AllocatorType:" << alloc_type << " "
             << device.to_string() << "]";
        return ostr.str();
    }

public:
    std::string name;
    int id = -1;
    MemoryType mem_type = MemoryType::DEFAULT;
    AllocatorType alloc_type = AllocatorType::INVALID;
    Device device;
};

inline bool operator==(const MemoryInfo& left, const MemoryInfo& right) {
    return left.mem_type == right.mem_type && left.alloc_type == right.alloc_type && left.id == right.id &&
           left.device == right.device && left.name == right.name;
}

inline bool operator!=(const MemoryInfo& left, const MemoryInfo& right) { return !(left == right); }

inline std::ostream& operator<<(std::ostream& out, const MemoryInfo& info) { return out << info.to_string(); }

}    // namespace framework
}    // namespace simple_ai

namespace std {

template <>
struct hash<simple_ai::framework::MemoryInfo> {
    size_t operator()(const simple_ai::framework::MemoryInfo& info) const { return info.hash(); };
};

}    // namespace std

#endif