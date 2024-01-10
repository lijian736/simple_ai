#include "ir/node_shapes/flatten_shape.h"

#include "ir/node.h"
#include "ir/node_utils.h"

namespace simple_ai {
namespace ir {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Flatten

std::string FlattenShapeInfer::node_type() const { return "Flatten"; }

Status FlattenShapeInfer::infer(const std::string& node_name, const std::vector<NodeArg*>& inputs,
             const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes,
             std::vector<NodeArg*>& outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        std::ostringstream oss;
        oss << "Node: Flatten[" << node_name << "], Invalid input size: " << inputs.size()
            << " or output size: " << outputs.size();
        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    int64_t axis = utils::get_attr_or_default<int64_t>("axis", 1, attributes);
    int64_t axis_tmp = axis;

    const auto& input_shape = inputs[0]->shape();
    int64_t rank = static_cast<int64_t>(input_shape.dims_num());

    // The value for axis must be in the range [-r, r], where r is the rank of the input tensor
    if (axis < 0) {
        axis += rank;
    }

    if (axis < 0 || axis > rank) {
        std::ostringstream oss;
        oss << "Node: Flatten[" << node_name << "], Invalid axis: " << axis_tmp;
        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    int64_t dim1 = 1;
    int64_t dim2 = 1;
    for (int i = 0; i < axis; ++i) {
        dim1 *= input_shape[i];
    }

    for (int i = axis; i < rank; ++i) {
        dim2 *= input_shape[i];
    }

    TensorShape output_shape;
    output_shape.add_dim(dim1);
    output_shape.add_dim(dim2);

    outputs[0]->set_shape(output_shape);

    return Status::ok();
}

}    // namespace ir
}    // namespace simple_ai