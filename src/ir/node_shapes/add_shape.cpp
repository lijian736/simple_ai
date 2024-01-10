
#include "ir/node_shapes/add_shape.h"

#include <sstream>

#include "ir/node.h"

namespace simple_ai {
namespace ir {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Add
// https://github.com/onnx/onnx/blob/main/docs/Broadcasting.md

std::string AddShapeInfer::node_type() const { return "Add"; }

Status AddShapeInfer::infer(const std::string& node_name, const std::vector<NodeArg*>& inputs,
                            const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes,
                            std::vector<NodeArg*>& outputs) {
    (void)attributes;

    if (inputs.size() != 2 || outputs.size() != 1) {
        std::ostringstream oss;
        oss << "Node: Add[" << node_name << "], Invalid input size: " << inputs.size()
            << " or output size: " << outputs.size();
        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    const auto& shape1 = inputs[0]->shape();
    const auto& shape2 = inputs[1]->shape();

    TensorShape out_shape;

    if (shape1.dims_num() >= shape2.dims_num()) {
        std::vector<int64_t> out_dims(shape1.dims());

        auto start1 = shape1.dims_num() - shape2.dims_num();
        for (int i = 0; i < static_cast<int>(shape2.dims_num()); ++i) {
            auto d1 = shape1[start1 + i];
            auto d2 = shape2[i];
            if (d1 == 1 || d2 == 1 || d1 == d2) {
                out_dims[start1 + i] = std::max<int64_t>(d1, d2);
            } else {
                std::ostringstream oss;
                oss << "Node: Add[" << node_name << "], input1 shape: " << shape1.to_string()
                    << " input2 shape: " << shape2.to_string();
                return Status(StatusCode::INVALID_PARAM, oss.str());
            }
        }

        out_shape.set_dims(out_dims);
    } else {
        std::vector<int64_t> out_dims(shape2.dims());

        auto start2 = shape2.dims_num() - shape1.dims_num();
        for (int i = 0; i < static_cast<int>(shape1.dims_num()); ++i) {
            auto d2 = shape2[start2 + i];
            auto d1 = shape1[i];
            if (d1 == 1 || d2 == 1 || d1 == d2) {
                out_dims[start2 + i] = std::max<int64_t>(d1, d2);
            } else {
                std::ostringstream oss;
                oss << "Node: Add[" << node_name << "], input1 shape: " << shape1.to_string()
                    << " input2 shape: " << shape2.to_string();
                return Status(StatusCode::INVALID_PARAM, oss.str());
            }
        }

        out_shape.set_dims(out_dims);
    }

    outputs[0]->set_shape(out_shape);

    return Status::ok();
}

}    // namespace ir
}    // namespace simple_ai
