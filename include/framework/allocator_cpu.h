#ifndef _H_SIMPLE_AI_FRAMEWORK_ALLOCATOR_CPU_H_
#define _H_SIMPLE_AI_FRAMEWORK_ALLOCATOR_CPU_H_

#include <cstdint>
#include <sstream>
#include <string>

#include "allocator.h"
#include "allocator_stats.h"
#include "device.h"
#include "memory_info.h"

namespace simple_ai {
namespace framework {

/**
 * @brief The cpu allocator
 *
 */
class CPUAllocator : public IAllocator {
public:
    explicit CPUAllocator(const MemoryInfo& memory_info) : IAllocator(memory_info) {}

    CPUAllocator() : IAllocator(MemoryInfo("CPU", AllocatorType::DEVICE)) {}

    virtual void* alloc(size_t size) override;
    virtual void free(void* p) override;

    virtual AllocatorStats stats() const override;
};

}    // namespace framework
}    // namespace simple_ai

#endif