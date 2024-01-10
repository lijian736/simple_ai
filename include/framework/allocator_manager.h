#ifndef _H_SIMPLE_AI_FRAMEWORK_ALLOCATOR_MANAGER_H_
#define _H_SIMPLE_AI_FRAMEWORK_ALLOCATOR_MANAGER_H_

#include "allocator.h"
#include "common/common.h"

#include <memory>
#include <unordered_map>
#include <mutex>

namespace simple_ai {
namespace framework {

class AllocatorManager {
public:
    ~AllocatorManager() = default;

    static AllocatorManager* instance();

    IAllocator* get_allocator(IAllocator::Type type);

private:
    SIMPLE_AI_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(AllocatorManager);
    AllocatorManager() = default;

private:
    //allocator type map. key: the allocator type, value: allocator pointer
    std::unordered_map<IAllocator::Type, std::unique_ptr<IAllocator>> m_allocator_map;
    std::mutex m_mutex;
};

}    // namespace framework
}    // namespace simple_ai

#endif