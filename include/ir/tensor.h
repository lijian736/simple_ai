#ifndef _H_SIMPLE_AI_IR_TENSOR_H_
#define _H_SIMPLE_AI_IR_TENSOR_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "common/common.h"
#include "framework/allocator.h"
#include "framework/common_defines.h"
#include "framework/memory_info.h"
#include "tensor_shape.h"

using namespace simple_ai::framework;

namespace simple_ai {
namespace ir {

class Tensor {
public:
    Tensor() = default;
    Tensor(const std::string& name) : m_name(name){};
    virtual ~Tensor();

    Tensor(Tensor&& rhs);
    Tensor& operator=(Tensor&& rhs);

    /**
     * @brief Initialize a tensor with given primitive data type, shape, pre-allocated memory and memory info.
     * This function does NOT check if the preallocated buffer `p_data` has enough room for the shape.
     *
     * @param data_type the tensor elements primitive data type
     * @param shape the tensor shape
     * @param p_data a pre-allocated buffer. nullptr if the shape is empty. Tensor does NOT own the data and will not
     * delete it
     * @param memory_info memory info of p_data
     * @param offset offset in bytes to start of tensor within p_data
     * @return Status
     */
    Status init(PrimitiveDataType data_type, const TensorShape& shape, void* p_data, const MemoryInfo& memory_info,
                std::ptrdiff_t offset = 0);

    /**
     * @brief Initialize a tensor which allocates and owns the buffer required for the specified shape.
     *
     * @param data_type primitive data type of tensor elements
     * @param shape tensor shape
     * @param allocator allocator to create and free buffer.
     * @return Status
     */
    Status init(PrimitiveDataType data_type, const TensorShape& shape, IAllocator* allocator);

    /**
     * @brief release buffer
     */
    void release_buffer();

    PrimitiveDataType data_type() const { return m_data_type; }

    const TensorShape& shape() const { return m_shape; }
    TensorShape& shape() { return m_shape; }

    const std::string& name() { return m_name; }

    const MemoryInfo memory_info() const { return m_memory_info; }

    template <typename T>
    T* data_as() {
        return reinterpret_cast<T*>(static_cast<char*>(m_p_data) + m_byte_offset);
    }

    void* data_raw() { return static_cast<char*>(m_p_data) + m_byte_offset; }
    const void* data_raw() const { return static_cast<char*>(m_p_data) + m_byte_offset; }

    std::ptrdiff_t byte_offset() const { return m_byte_offset; }
    void set_byte_offset(std::ptrdiff_t byte_offset) { m_byte_offset = byte_offset; }

    /**
     * @brief Calculate the required storage room for the tensor
     *
     * @param data_type primitive data type
     * @param shape tensor shape
     * @param size output parameter. the bytes length
     * @return Status
     */
    static Status calc_storage_size(PrimitiveDataType data_type, const TensorShape& shape, size_t& size);

private:
    SIMPLE_AI_DISALLOW_COPY_AND_ASSIGNMENT(Tensor);

private:
    // the primitive data type
    PrimitiveDataType m_data_type;
    // the tensor shape
    TensorShape m_shape;
    // the tensor memory info
    MemoryInfo m_memory_info;

    // the tensor name
    std::string m_name;

    // the raw data pointer
    void* m_p_data{nullptr};
    // the byte offset relative to m_p_data;
    std::ptrdiff_t m_byte_offset{0};

    // if m_allocator is nullptr, the tensor does NOT own the buffer. otherwise tensor will
    // use the m_allocator to release the buffer when tensor is destructed.
    IAllocator* m_allocator{nullptr};
};

}    // namespace ir
}    // namespace simple_ai

#endif
