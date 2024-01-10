#ifndef _H_SIMPLE_AI_FRAMEWORK_ALLOCATOR_STATS_H_
#define _H_SIMPLE_AI_FRAMEWORK_ALLOCATOR_STATS_H_

#include <cstdint>
#include <sstream>
#include <string>

namespace simple_ai {
namespace framework {

/**
 * @brief Runtime statistics for an allocator
 *
 */
struct AllocatorStats {
    int64_t num_allocs{0};               // number of allocations.
    int64_t bytes_in_use{0};             // number of bytes in use.
    int64_t total_allocated_bytes{0};    // the total number of allocated bytes by the allocator.
    int64_t max_bytes_in_use{0};         // the maximum bytes in use.
    int64_t max_alloc_size{0};           // the max single allocation.
    int64_t bytes_limit{0};              // The upper limit what the allocator can allocate, if such a limit
                                         // is known. Certain allocator may return 0 to indicate the limit is unknown.

    void clear() {
        this->num_allocs = 0;
        this->bytes_in_use = 0;
        this->max_bytes_in_use = 0;
        this->max_alloc_size = 0;
        this->bytes_limit = 0;
        this->total_allocated_bytes = 0;
    }

    std::string to_string() const {
        std::ostringstream ss;
        ss << "Limit:                    " << this->bytes_limit << std::endl
           << "InUse:                    " << this->bytes_in_use << std::endl
           << "TotalAllocated:           " << this->total_allocated_bytes << std::endl
           << "MaxInUse:                 " << this->max_bytes_in_use << std::endl
           << "NumAllocs:                " << this->num_allocs << std::endl
           << "MaxAllocSize:             " << this->max_alloc_size << std::endl;
        return ss.str();
    }
};

inline std::ostream& operator<<(std::ostream& out, const AllocatorStats& stat) { return out << stat.to_string(); }

}    // namespace framework
}    // namespace simple_ai

#endif