#ifndef _H_SIMPLE_AI_FRAMEWORK_ALLOCATOR_INTERFACE_H_
#define _H_SIMPLE_AI_FRAMEWORK_ALLOCATOR_INTERFACE_H_

#include <cstdint>
#include <sstream>
#include <string>

#include "allocator_stats.h"
#include "device.h"
#include "memory_info.h"

namespace simple_ai {
namespace framework {

/**
 * @brief The memory allocator
 */
class IAllocator {
public:
    enum class Type {
        CPU,        // CPU allocator
        DEFAULT,    // CPU allocator as the default allocator
        INVALID     // Invalid allocator
    };

public:
    explicit IAllocator(const MemoryInfo& info) : m_memory_info(info) {}
    virtual ~IAllocator() = default;

    /**
     * @brief allocate a piece of memory with `size`
     *
     * @param size the memory size
     * @return void* the pointer to the allocated memory
     */
    virtual void* alloc(size_t size) = 0;

    /**
     * @brief free the memory
     *
     * @param ptr the memory pointer
     */
    virtual void free(void* ptr) = 0;

    /**
     * @brief get the allocator statistics
     *
     * @return AllocatorStats statistics info
     */
    virtual AllocatorStats stats() const = 0;

    /**
     * @brief get the memory info of this allocator
     *
     * @return const MemoryInfo&
     */
    const MemoryInfo& info() const { return m_memory_info; }

    /**
     * @brief Allocate memory for an array which has `item_num` numbers, each `item_size` bytes
     *
     * @param item_num item counter
     * @param item_size item size in bytes
     * @return void*
     */
    void* alloc_array(size_t item_num, size_t item_size) { return alloc(item_num * item_size); }

    /**
     * @brief Allocate memory for an array which has `item_num` numbers, each `item_size` bytes
     *
     * @tparam alignment
     * @param item_num item counter
     * @param item_size item size in bytes
     * @return void*
     */
    template <size_t alignment>
    void* alloc_array_aligned(size_t item_num, size_t item_size) {
        size_t len = calc_aligned_mem_size(item_num * item_size, alignment);
        return alloc(len);
    }

    /**
     * @brief Calculate the alignment memory size
     *
     * @param size the original memory size
     * @param alignment the alignment size, it MUST be power of 2
     * @return size_t the aligned memory size
     */
    static size_t calc_aligned_mem_size(size_t size, size_t alignment);

private:
    // the memory info
    MemoryInfo m_memory_info;
};

}    // namespace framework
}    // namespace simple_ai

#endif