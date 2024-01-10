#ifndef _H_SIMPLE_AI_FRAMEWORK_COMMON_DEFINES_H_
#define _H_SIMPLE_AI_FRAMEWORK_COMMON_DEFINES_H_

#include <algorithm>
#include <iostream>
#include <unordered_map>

namespace simple_ai {
namespace framework {

/**
 * @brief Memory types for allocated memory.
 */
enum class MemoryType : int {
    DEFAULT    // The default memory type
};

/**
 * @brief Memory allocator type
 */
enum class AllocatorType : int {
    DEVICE,    // the device allocator
    ARENA,     // the arena allocator
    INVALID    // invalid allocator
};

/**
 * @brief The primitive data type
 *
 */
enum class PrimitiveDataType : int {
    FLOAT32,
    FLOAT16,
    INT8,
    UINT8,
    INT16,
    UINT16,
    INT32,
    UINT32,
    INT64,
    UINT64,
    UNKNOWN
};

inline size_t size_of_datatype(PrimitiveDataType dt) {
    static std::unordered_map<PrimitiveDataType, size_t> kPrimitiveSize = {
        {PrimitiveDataType::FLOAT32, 4}, {PrimitiveDataType::FLOAT16, 2}, {PrimitiveDataType::INT8, 1},
        {PrimitiveDataType::UINT8, 1},   {PrimitiveDataType::INT16, 2},   {PrimitiveDataType::UINT16, 2},
        {PrimitiveDataType::INT32, 4},   {PrimitiveDataType::UINT32, 4},  {PrimitiveDataType::INT64, 8},
        {PrimitiveDataType::UINT64, 8},  {PrimitiveDataType::UNKNOWN, 0}};

    return kPrimitiveSize[dt];
}

inline std::ostream& operator<<(std::ostream& out, const MemoryType& type) {
    if (type == MemoryType::DEFAULT) {
        out << "DEFAULT";
    }

    return out;
}

inline std::ostream& operator<<(std::ostream& out, const AllocatorType& type) {
    if (type == AllocatorType::DEVICE) {
        out << "DEVICE";
    } else if (type == AllocatorType::ARENA) {
        out << "ARENA";
    } else if (type == AllocatorType::INVALID) {
        out << "INVALID";
    }

    return out;
}

}    // namespace framework
}    // namespace simple_ai

#endif