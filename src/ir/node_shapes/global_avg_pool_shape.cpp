#include "ir/node_shapes/global_avg_pool_shape.h"

#include "ir/node.h"

namespace simple_ai {
namespace ir {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#GlobalAveragePool
std::string GlobalAveragePoolShapeInfer::node_type() const { return "GlobalAveragePool"; }

Status GlobalAveragePoolShapeInfer::infer(
    const std::string& node_name, const std::vector<NodeArg*>& inputs,
    const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes, std::vector<NodeArg*>& outputs) {
    (void)attributes;

    if (inputs.size() != 1 || outputs.size() != 1) {
        std::ostringstream oss;
        oss << "Node: GlobalAveragePool[" << node_name << "], Invalid input size: " << inputs.size()
            << " or output size: " << outputs.size();
        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    // for image: (N x C x H x W)
    // for non image: (N x C x D1 x D2 ... Dn)
    // where N is the batch size,  C is the number of channels
    const auto& input_shape = inputs[0]->shape();
    size_t dim_num = input_shape.dims_num();
    if (dim_num < 2) {
        std::ostringstream oss;
        oss << "Node: GlobalAveragePool[" << node_name << "], too few input dimensions: " << dim_num;
        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    // The output tensor has the same rank as the input.
    // The first two dimensions of output shape are the same as the input (N x C), while the other dimensions are
    // all 1.
    TensorShape output_shape;
    output_shape.add_dim(input_shape[0]);
    output_shape.add_dim(input_shape[1]);
    for (int i = 2; i < dim_num; ++i) {
        output_shape.add_dim(1);
    }

    outputs[0]->set_shape(output_shape);
    return Status::ok();
}

}    // namespace ir
}    // namespace simple_ai