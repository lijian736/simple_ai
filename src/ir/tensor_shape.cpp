#include "ir/tensor_shape.h"

#include <algorithm>
#include <sstream>

namespace simple_ai {
namespace ir {

bool TensorShape::operator==(const TensorShape& rhs) const {
    return std::equal(m_dims.begin(), m_dims.end(), rhs.m_dims.begin(), rhs.m_dims.end());
}

bool TensorShape::operator!=(const TensorShape& rhs) const { return !(*this == rhs); }

int64_t TensorShape::element_num() const {
    int64_t result = 0;
    if (m_dims.size() > 0) {
        result = m_dims[0];
    }

    for (size_t i = 1; i < m_dims.size(); ++i) {
        result *= m_dims[i];
    }

    return result;
}

bool TensorShape::is_scalar() const {
    size_t len = m_dims.size();
    return len == 0 || (len == 1 && m_dims[0] == 1);
}

std::string TensorShape::to_string() const {
    std::ostringstream oss;

    oss << "{";
    bool first = true;
    for (auto dim : m_dims) {
        if (!first) {
            oss << ",";
        }

        oss << dim;
        first = false;
    }
    oss << "}";

    return oss.str();
}

}    // namespace ir
}    // namespace simple_ai