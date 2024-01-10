#include "framework/allocator_manager.h"

#include "framework/allocator_cpu.h"

namespace simple_ai {
namespace framework {

AllocatorManager* AllocatorManager::instance() {
    static AllocatorManager instance;
    return &instance;
}

IAllocator* AllocatorManager::get_allocator(IAllocator::Type type) {
    std::unique_lock<std::mutex> lock(m_mutex);

    auto result = m_allocator_map.emplace(type, nullptr);
    if (result.second) {
        if (type == IAllocator::Type::DEFAULT || type == IAllocator::Type::CPU) {
            result.first->second = std::make_unique<CPUAllocator>();
        }
    }

    return result.first->second.get();
}

}    // namespace framework
}    // namespace simple_ai