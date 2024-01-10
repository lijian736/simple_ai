#include "framework/allocator_cpu.h"

#include <stdlib.h>

namespace {
constexpr size_t kPreferredAlignment = 64;
}

namespace simple_ai {
namespace framework {

void* CPUAllocator::alloc(size_t size) {
    void* ptr;
    int ret = posix_memalign(&ptr, kPreferredAlignment, size);
    if (ret == 0) {
        return ptr;
    } else {
        return nullptr;
    }
}

void CPUAllocator::free(void* p) { std::free(p); }

AllocatorStats CPUAllocator::stats() const { return AllocatorStats(); }

}    // namespace framework
}    // namespace simple_ai