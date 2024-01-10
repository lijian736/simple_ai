#include "ir/tensor.h"

#include <algorithm>

namespace simple_ai {
namespace ir {

Tensor::Tensor(Tensor&& rhs)
    : m_data_type(std::move(rhs.m_data_type)),
      m_shape(std::move(rhs.m_shape)),
      m_memory_info(std::move(rhs.m_memory_info)),
      m_byte_offset(std::move(rhs.m_byte_offset)),
      m_allocator(std::move(rhs.m_allocator)),
      m_p_data(std::move(rhs.m_p_data)),
      m_name(std::move(rhs.m_name)) {
    rhs.m_allocator = nullptr;
    rhs.m_p_data = nullptr;
    rhs.m_byte_offset = 0;
}

Tensor::~Tensor() { release_buffer(); }

Tensor& Tensor::operator=(Tensor&& rhs) {
    if (this != &rhs) {
        release_buffer();
    }

    m_data_type = std::move(rhs.m_data_type);
    m_shape = std::move(rhs.m_shape);
    m_memory_info = std::move(rhs.m_memory_info);
    m_byte_offset = std::move(rhs.m_byte_offset);
    m_allocator = std::move(rhs.m_allocator);
    m_p_data = std::move(rhs.m_p_data);
    m_name = std::move(rhs.m_name);

    rhs.m_allocator = nullptr;
    rhs.m_p_data = nullptr;
    rhs.m_byte_offset = 0;

    return *this;
}

void Tensor::release_buffer() {
    if (m_allocator && m_p_data) {
        m_allocator->free(m_p_data);
    }
}

Status Tensor::calc_storage_size(PrimitiveDataType data_type, const TensorShape& shape, size_t& size) {
    int64_t shape_size = shape.element_num();
    if (shape_size < 0) {
        return Status(StatusCode::FAIL, "invalid tensor shape");
    }

    if (shape_size == 0) {
        size = 0;
        return Status::ok();
    }

    size = size_of_datatype(data_type) * shape_size;
    return Status::ok();
}

Status Tensor::init(PrimitiveDataType data_type, const TensorShape& shape, void* p_data, const MemoryInfo& memory_info,
                    std::ptrdiff_t offset) {
    release_buffer();

    m_data_type = data_type;
    m_shape = shape;
    m_memory_info = memory_info;
    m_p_data = p_data;
    m_byte_offset = offset;

    m_allocator = nullptr;

    return Status::ok();
}

Status Tensor::init(PrimitiveDataType data_type, const TensorShape& shape, IAllocator* allocator) {
    release_buffer();

    m_data_type = data_type;
    m_shape = shape;
    m_memory_info = allocator->info();
    m_byte_offset = 0;
    m_allocator = allocator;

    size_t len = 0;
    auto status = calc_storage_size(data_type, shape, len);
    if (status.is_ok() && len > 0) {
        m_p_data = allocator->alloc(len);
    } else {
        m_p_data = nullptr;
    }

    return Status::ok();
}

}    // namespace ir
}    // namespace simple_ai