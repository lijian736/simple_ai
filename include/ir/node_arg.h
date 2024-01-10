#ifndef _H_SIMPLE_AI_IR_NODE_ARG_H_
#define _H_SIMPLE_AI_IR_NODE_ARG_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "common/common.h"
#include "framework/common_defines.h"
#include "tensor_shape.h"

using namespace simple_ai::framework;

namespace simple_ai {
namespace ir {

/**
 * @brief Node argument to a node, for node inputs and node outputs.
 * including argument name, argument primitive data type and shape.
 */
class NodeArg {
public:
    NodeArg(const std::string& name);
    NodeArg(const std::string& name, PrimitiveDataType data_type, const TensorShape& shape);
    NodeArg(const NodeArg&);
    NodeArg(NodeArg&&);
    NodeArg& operator=(NodeArg&&) = default;
    NodeArg& operator=(const NodeArg&) = default;
    ~NodeArg() = default;

    bool operator==(const NodeArg& rhs) const;
    bool operator!=(const NodeArg& rhs) const;

    const std::string& name() const;
    const PrimitiveDataType data_type() const;
    const TensorShape& shape() const;

    void set_shape(const TensorShape& shape);

private:
    std::string m_name;
    PrimitiveDataType m_data_type;
    TensorShape m_shape;
};

}    // namespace ir
}    // namespace simple_ai

#endif