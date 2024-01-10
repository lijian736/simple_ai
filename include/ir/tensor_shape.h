#ifndef _H_SIMPLE_AI_IR_TENSOR_SHAPE_H_
#define _H_SIMPLE_AI_IR_TENSOR_SHAPE_H_

#include <memory>
#include <string>
#include <vector>

#include "common/common.h"

namespace simple_ai {
namespace ir {

/**
 * @brief The tensor shape
 *
 */
class TensorShape final {
public:
    TensorShape() = default;
    ~TensorShape() = default;

    TensorShape(const TensorShape&) = default;
    TensorShape(TensorShape&&) = default;
    TensorShape& operator=(TensorShape&&) = default;
    TensorShape& operator=(const TensorShape&) = default;

    bool operator==(const TensorShape& rhs) const;
    bool operator!=(const TensorShape& rhs) const;

    int64_t operator[](size_t index) const { return m_dims[index]; }
    int64_t& operator[](size_t index) { return m_dims[index]; }

    size_t dims_num() const { return m_dims.size(); }
    void set_dims_num(size_t num) { m_dims.resize(num); }

    /**
     * @brief check if this tensor is a scalar.
     * if the dims_num() return 0, it is a scalar.
     * if the dims_num() return 1, and only 1 element in dim 0, it is a scalar.
     *
     * @return true
     * @return false
     */
    bool is_scalar() const;

    const std::vector<int64_t>& dims() const { return m_dims; }
    std::vector<int64_t>& dims() { return m_dims; }

    void add_dim(int64_t dim) { m_dims.emplace_back(dim); }
    void set_dims(const std::vector<int64_t>& dims){ m_dims = dims;}

    /**
     * @brief get the elements number
     *
     * @return int64_t
     */
    int64_t element_num() const;

    std::string to_string() const;

private:
    // the dimensions
    std::vector<int64_t> m_dims;
};

inline std::ostream& operator<<(std::ostream& out, const TensorShape& shape) { return out << shape.to_string(); }

}    // namespace ir
}    // namespace simple_ai

#endif
