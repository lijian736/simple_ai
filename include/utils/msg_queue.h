#ifndef _H_SIMPLE_AI_UTILS_MESSAGE_QUEUE_H_
#define _H_SIMPLE_AI_UTILS_MESSAGE_QUEUE_H_

#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

#include "common/common.h"

namespace simple_ai {
namespace utils {

/**
 * @brief A Simple Message Queue
 *
 */
template <typename T, typename S>
class SimpleMessageQueue {
public:
    /**
     * @brief Constructor
     *
     * @param max_count the maximux capacity of the queue
     */
    SimpleMessageQueue(std::size_t max_count = 2048) : m_max_queue_length(max_count) {}
    virtual ~SimpleMessageQueue() = default;

    /**
     * @brief Add message item to the queue.
     * if the queue if full, the queue will ignore the item
     *
     * @param msg message item
     * @return true
     * @return false the queue is full
     */
    bool put(T&& msg) {
        std::unique_lock<std::mutex> lock(this->m_mutex);
        // add item
        if (m_msg_queue.size() < m_max_queue_length) {
            m_msg_queue.emplace_back(std::forward<T>(msg));
            if (m_msg_queue.size() == 1) {
                this->m_condi.notify_one();
            }

            return true;
        } else {
            return false;
        }
    }

protected:
    /**
     * @brief start the message queue loop
     */
    void start() {
        std::call_once(m_once_flag, [this] {
            m_running_thread = std::thread(std::bind(&SimpleMessageQueue::run, this));
            m_is_running = true;
        });
    }

    /**
     * @brief stop the message queue loop
     */
    void stop() {
        if (m_is_running) {
            m_is_running = false;
            this->m_condi.notify_one();
            m_running_thread.join();
        }
    }

    /**
     * @brief if the queue is empty, this function will blocked.
     */
    void wait_for_item() {
        std::unique_lock<std::mutex> lock(this->m_mutex);

        this->m_condi.wait(lock, [this] {
            if (!m_is_running || !m_msg_queue.empty()) {
                return true;
            } else {
                return false;
            }
        });
    }

    void run() {
        while (m_is_running) {
            wait_for_item();
            if (!m_is_running) {
                break;
            }

            auto msg = m_msg_queue.front();
            m_msg_queue.pop_front();
            static_cast<S*>(this)->handle_msg(msg);

            // std::this_thread::yield();
        }
    }

private:
    SIMPLE_AI_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SimpleMessageQueue);

private:
    // the max queue length
    std::size_t m_max_queue_length;

    // is the message queue running?
    std::atomic<bool> m_is_running{false};

    std::thread m_running_thread;

    // the inner queue
    std::deque<T> m_msg_queue;

    std::mutex m_mutex;
    std::condition_variable m_condi;
    std::once_flag m_once_flag;
};

}    // namespace utils
}    // namespace simple_ai

#endif