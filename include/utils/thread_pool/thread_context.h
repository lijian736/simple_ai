#ifndef _H_SIMPLE_AI_UTILS_THREAD_POOL_THREAD_CONTEXT_H_
#define _H_SIMPLE_AI_UTILS_THREAD_POOL_THREAD_CONTEXT_H_

#include <functional>
#include <memory>
#include <thread>

namespace simple_ai {
namespace utils {
namespace thread_pool {

/**
 * @brief the thread context
 */
struct ThreadContext {
    /**
     * @brief the thread task
     */
    struct Task {
        std::function<void()> func;
    };

    /**
     * @brief the context thread class
     */
    class ContextThread {
    public:
        /**
         * @brief Constructor
         *
         * @param run the thread run function
         */
        ContextThread(std::function<void()> run) : m_thread(std::move(run)) {}
        ~ContextThread() { m_thread.join(); }

    private:
        std::thread m_thread;
    };

    /**
     * @brief Create a context thread
     *
     * @param run the thread run function
     * @return ContextThread* the thread pointer
     */
    std::shared_ptr<ContextThread> create_thread(std::function<void()> run) {
        return std::make_shared<ContextThread>(std::move(run));
    }

    /**
     * @brief Create a task
     *
     * @param f the task execute function
     * @return Task the created task object
     */
    Task create_task(std::function<void()> f) { return Task{.func = std::move(f)}; }

    /**
     * @brief Execute the task
     *
     * @param task the task object
     */
    void execute_task(const Task& task) { task.func(); }
};

}    // namespace thread_pool
}    // namespace utils
}    // namespace simple_ai

#endif