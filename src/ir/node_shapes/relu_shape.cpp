#include "ir/node_shapes/relu_shape.h"

#include "ir/node.h"

namespace simple_ai {
namespace ir {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu
std::string ReluShapeInfer::node_type() const { return "Relu"; }

Status ReluShapeInfer::infer(const std::string& node_name, const std::vector<NodeArg*>& inputs,
                             const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes,
                             std::vector<NodeArg*>& outputs) {
    (void)attributes;

    if (inputs.size() != 1 || outputs.size() != 1) {
        std::ostringstream oss;
        oss << "Node: Relu[" << node_name << "], Invalid input size: " << inputs.size()
            << " or output size: " << outputs.size();
        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    outputs[0]->set_shape(inputs[0]->shape());

    return Status::ok();
}

}    // namespace ir
}    // namespace simple_ai