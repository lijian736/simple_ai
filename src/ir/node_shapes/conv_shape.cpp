#include "ir/node_shapes/conv_shape.h"

#include "ir/node.h"
#include "ir/node_utils.h"

namespace simple_ai {
namespace ir {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Conv

std::string ConvShapeInfer::node_type() const { return "Conv"; }

Status ConvShapeInfer::infer(const std::string& node_name, const std::vector<NodeArg*>& inputs,
                             const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes,
                             std::vector<NodeArg*>& outputs) {
    if (inputs.size() < 2 || outputs.size() != 1) {
        std::ostringstream oss;
        oss << "Node: Conv[" << node_name << "], invalid input size: " << inputs.size()
            << " or output size: " << outputs.size();
        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    std::string auto_pad = utils::get_attr_or_default<std::string>("auto_pad", "NOTSET", attributes);
    std::vector<int64_t> dilations = utils::get_attrs_or_default<int64_t>("dilations", {}, attributes);
    int64_t group = utils::get_attr_or_default<int64_t>("group", 1, attributes);
    std::vector<int64_t> kernel_shape = utils::get_attrs_or_default<int64_t>("kernel_shape", {}, attributes);
    std::vector<int64_t> pads = utils::get_attrs_or_default<int64_t>("pads", {}, attributes);
    std::vector<int64_t> strides = utils::get_attrs_or_default<int64_t>("strides", {}, attributes);

    if (auto_pad != "NOTSET") {
        std::ostringstream oss;
        oss << "Node: Conv[" << node_name << "], auto_pad attribute is not supported now. auto_pad value: " << auto_pad;

        return Status(StatusCode::NOT_IMPLEMENTED, oss.str());
    }

    if (group <= 1) {
        std::ostringstream oss;
        oss << "Node: Conv[" << node_name << "], group convolution is not supported now. group attribute: " << group;

        return Status(StatusCode::NOT_IMPLEMENTED, oss.str());
    }

    // for the 2D image, dimensions is (N x C x H x W), where N is the batch size, C is the number of channels, and
    // H and W are the height and width. Otherwise the size is (N x C x D1 x D2 ... x Dn)
    const auto& input_shape = inputs[0]->shape();
    const auto& weight_shape = inputs[1]->shape();

    size_t input_dim_num = input_shape.dims_num();
    size_t weight_dim_num = weight_shape.dims_num();

    if (input_dim_num < 2) {
        std::ostringstream oss;
        oss << "Node: Conv[" << node_name << "], invalid input dimensions length: " << input_dim_num;

        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    if (weight_dim_num < 2 || input_dim_num != weight_dim_num) {
        std::ostringstream oss;
        oss << "Node: Conv[" << node_name << "], invalid weight dimensions length: " << weight_dim_num;

        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    size_t kernel_size = kernel_shape.size();
    // the kernel shape is empty. get its info from the weigths
    if (kernel_size == 0) {
        kernel_shape = weight_shape.dims();
        kernel_shape.erase(kernel_shape.begin(), kernel_shape.begin() + 2);
        kernel_size = kernel_shape.size();
    }

    // check the kernel shape
    if (kernel_size > input_dim_num) {
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

    // check the strides
    if (strides.size() != kernel_size) {
        if (strides.size() != 0) {
            return Status(StatusCode::INVALID_PARAM, "Invalid strides");
        } else {
            strides.assign(kernel_size, 1);
        }
    }

    // check the groups, skip now.
    (void)group;

    // the bias exists
    if (inputs.size() == 3) {
        const auto& bias_shape = inputs[2]->shape();
    }

    TensorShape out_shape;
    out_shape.add_dim(input_shape[0]);     // batch
    out_shape.add_dim(kernel_shape[0]);    // output channel

    for (size_t i = 0; i < kernel_size; ++i) {
        int64_t dim =
            (input_shape[i + 2] + pads[i] + pads[i + pads.size() / 2] - dilations[i] * (kernel_shape[i] - 1) - 1) /
                strides[i] +
            1;

        out_shape.add_dim(dim);
    }

    outputs[0]->set_shape(out_shape);
    return Status::ok();
}

}    // namespace ir
}    // namespace simple_ai