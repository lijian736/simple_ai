#include "framework/allocator.h"

namespace simple_ai {
namespace framework {

size_t IAllocator::calc_aligned_mem_size(size_t size, size_t alignment) {
    if (alignment == 0) {
        return size;
    } else {
        size_t alignment_mask = alignment - 1;
        return (size + alignment_mask) & (~alignment_mask);
    }
}

}    // namespace framework
}    // namespace simple_ai