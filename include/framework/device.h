#ifndef _H_SIMPLE_AI_FRAMEWORK_DEVICE_H_
#define _H_SIMPLE_AI_FRAMEWORK_DEVICE_H_

#include <cstdint>
#include <sstream>

#include "common/hash_combine.h"

using namespace simple_ai::common;

namespace simple_ai {
namespace framework {

/**
 * @brief A physical device
 */
struct Device {
    using DeviceType = int8_t;    // device type
    using DeviceId = int16_t;     // device id

    // pre-defined device types.
    static const DeviceType CPU = 0;
    static const DeviceType GPU = 1;    // Nvidia or AMD
    static const DeviceType NPU = 2;    //

    Device(DeviceType device_type, DeviceId device_id) : m_device_type(device_type), m_device_id(device_id) {}

    Device() : Device(CPU, 0) {}

    DeviceType device_type() const { return m_device_type; }

    DeviceId device_id() const { return m_device_id; }

    std::string to_string() const {
        std::ostringstream oss;
        oss << "Device:["
            << "DeviceType:" << static_cast<int>(m_device_type) << " DeviceId:" << m_device_id << "]";
        return oss.str();
    }

    size_t hash() const {
        auto hash = std::hash<int>()(m_device_type);
        hash_combine(m_device_id, hash);
        return hash;
    }

    bool operator<(const Device& rhs) const {
        if (m_device_type != rhs.m_device_type) {
            return m_device_type < rhs.m_device_type;
        }

        return m_device_id < rhs.m_device_id;
    }

private:
    // device type
    DeviceType m_device_type;
    // device id
    DeviceId m_device_id;
};

inline bool operator==(const Device& left, const Device& right) {
    return left.device_id() == right.device_id() && left.device_type() == right.device_type();
}

inline bool operator!=(const Device& left, const Device& right) { return !(left == right); }

inline std::ostream& operator<<(std::ostream& out, const Device& dev) { return out << dev.to_string(); }

}    // namespace framework
}    // namespace simple_ai

namespace std {
template <>
struct hash<simple_ai::framework::Device> {
    size_t operator()(const simple_ai::framework::Device& dev) const { return dev.hash(); }
};

}    // namespace std

#endif