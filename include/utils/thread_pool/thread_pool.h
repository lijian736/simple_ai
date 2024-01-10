#ifndef _H_SIMPLE_AI_UTILS_THREAD_POOL_THREAD_POOL_H_
#define _H_SIMPLE_AI_UTILS_THREAD_POOL_THREAD_POOL_H_

#include <functional>
#include <memory>
#include <thread>

#include "common/common.h"

namespace simple_ai {
namespace utils {
namespace thread_pool {

/**
 * @brief A thread pool interface
 */
class IThreadPool {
public:
    IThreadPool() = default;
    virtual ~IThreadPool() = default;

    /**
     * @brief Submit a callable job to run
     *
     * @param run a callable job
     */
    virtual void schedule(std::function<void()> run) = 0;

    /**
     * @brief Cancel the threads that have been enqueued.
     * The current running thread will be run until it's task is finished.
     */
    virtual void cancel() {}

    /**
     * @brief Get the number of threads in the pool
     *
     * @return int the thread number
     */
    virtual int num_threads() const = 0;

    /**
     * @brief Get the current thread index between 0 and num_threads() -1.
     * 
     * 
     * @return int the current thread index in the pool. -1 specifies the current
     * thread is not in the pool
     */
    virtual int current_thead_index() const = 0;

private:
    SIMPLE_AI_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(IThreadPool);
};

}    // namespace thread_pool
}    // namespace utils
}    // namespace simple_ai

#endif