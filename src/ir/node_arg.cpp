#include "ir/node_arg.h"

namespace simple_ai {
namespace ir {

NodeArg::NodeArg(const std::string& name) : m_name(name), m_data_type(PrimitiveDataType::UNKNOWN) {}

NodeArg::NodeArg(const std::string& name, PrimitiveDataType data_type, const TensorShape& shape)
    : m_name(name), m_data_type(data_type), m_shape(shape) {}

NodeArg::NodeArg(const NodeArg& rhs) : m_name(rhs.m_name), m_data_type(rhs.m_data_type), m_shape(rhs.m_shape) {}

NodeArg::NodeArg(NodeArg&& rhs)
    : m_name(std::move(rhs.m_name)), m_data_type(std::move(rhs.m_data_type)), m_shape(std::move(rhs.m_shape)) {}

const std::string& NodeArg::name() const { return m_name; }

const PrimitiveDataType NodeArg::data_type() const { return m_data_type; }

const TensorShape& NodeArg::shape() const { return m_shape; }

bool NodeArg::operator==(const NodeArg& rhs) const {
    return m_name == rhs.m_name && m_data_type == rhs.m_data_type && m_shape == rhs.m_shape;
}

bool NodeArg::operator!=(const NodeArg& rhs) const { return (*this) != rhs; }

void NodeArg::set_shape(const TensorShape& shape){
    m_shape = shape;
}

}    // namespace ir
}    // namespace simple_ai