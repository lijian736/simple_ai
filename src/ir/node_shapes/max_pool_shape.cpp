#include "ir/node_shapes/max_pool_shape.h"

#include <algorithm>
#include <unordered_set>

#include "ir/node.h"
#include "ir/node_utils.h"

namespace simple_ai {
namespace ir {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#MaxPool
std::string MaxPoolShapeInfer::node_type() const { return "MaxPool"; }

Status MaxPoolShapeInfer::infer(const std::string& node_name, const std::vector<NodeArg*>& inputs,
                                const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes,
                                std::vector<NodeArg*>& outputs) {
    if (inputs.size() != 1 || outputs.size() != 1) {
        std::ostringstream oss;
        oss << "Node: MaxPool[" << node_name << "], not implemented or invalid input size: " << inputs.size()
            << " or output size: " << outputs.size();
        return Status(StatusCode::NOT_IMPLEMENTED, oss.str());
    }

    std::string auto_pad = utils::get_attr_or_default<std::string>("auto_pad", "NOTSET", attributes);
    int64_t ceil_mode = utils::get_attr_or_default<int64_t>("ceil_mode", 0, attributes);
    std::vector<int64_t> dilations = utils::get_attrs_or_default<int64_t>("dilations", {}, attributes);
    std::vector<int64_t> kernel_shape = utils::get_attrs_or_default<int64_t>("kernel_shape", {}, attributes);
    std::vector<int64_t> pads = utils::get_attrs_or_default<int64_t>("pads", {}, attributes);
    int64_t storage_order = utils::get_attr_or_default<int64_t>("storage_order", 0, attributes);
    std::vector<int64_t> strides = utils::get_attrs_or_default<int64_t>("strides", {}, attributes);

    // for image case are (N x C x H x W), where N is the batch size, C is the number of channels, and H and W are
    // the height and the width of the data.
    // For non image case, the dimensions are in the form of (N x C x D1 x D2
    // ... Dn), where N is the batch size.
    const auto& input_shape = inputs[0]->shape();
    size_t dim_num = input_shape.dims_num();

    if (dim_num < 2) {
        std::ostringstream oss;
        oss << "Node: MaxPool[" << node_name << "], invalid input dimensions length: " << dim_num;

        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    // check the node's attributes
    // auto_pad is a DEPRECATED attribute, skip it
    if (auto_pad != "NOTSET") {
        std::ostringstream oss;
        oss << "Node: MaxPool[" << node_name
            << "], auto_pad is a DEPRECATED attribute, not supported now. auto_pad value: " << auto_pad;

        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    size_t kernel_size = kernel_shape.size();
    // check the kernel shape
    if (kernel_size > dim_num) {
        return Status(StatusCode::INVALID_PARAM, "Invalid kernel_shape");
    }

    // check the dilations
    if (dilations.size() != kernel_size) {
        if (dilations.size() != 0) {
            return Status(StatusCode::INVALID_PARAM, "Invalid dilations");
        } else {
            dilations.assign(kernel_size, 1);
        }
    }

    // check the pads
    if (pads.size() % 2 != 0) {
        return Status(StatusCode::INVALID_PARAM, "Invalid pads");
    }

    if (pads.size() / 2 == kernel_size) {
        bool pads_invalid = std::any_of(pads.cbegin(), pads.cend(), [](int64_t pad) { return pad < 0; });
        if (pads_invalid) {
            return Status(StatusCode::INVALID_PARAM, "Invalid pads");
        }
    } else if (pads.size() == 0) {
        pads.assign(kernel_size * 2, 0);
    } else {
        return Status(StatusCode::INVALID_PARAM, "Invalid pads");
    }

    // 0 is row major. skip this attribute.
    (void)storage_order;

    // check the strides
    if (strides.size() != kernel_size) {
        if (strides.size() != 0) {
            return Status(StatusCode::INVALID_PARAM, "Invalid strides");
        } else {
            strides.assign(kernel_size, 1);
        }
    }

    TensorShape output_shape;
    output_shape.set_dims_num(dim_num);

    // now, compute the output shape
    for (size_t i = 0; i < dim_num - kernel_size; ++i) {
        output_shape[i] = input_shape[i];
    }

    int j = 0;
    for (size_t i = dim_num - kernel_size; i < dim_num; ++i) {
        int64_t dim = 0;
        int64_t tmp1 = input_shape[i] + pads[j] + pads[j + pads.size() / 2] - dilations[j] * (kernel_shape[j] - 1) - 1;
        int64_t tmp2 = tmp1 / strides[i];
        if (ceil_mode) {
            dim = (tmp2 * strides[i] == tmp1) ? (tmp2 + 1) : (tmp2 + 2);
        } else {
            // floor
            dim = tmp2 + 1;
        }

        output_shape[i] = dim;
        ++j;
    }

    outputs[0]->set_shape(output_shape);

    return Status::ok();
}

}    // namespace ir
}    // namespace simple_ai