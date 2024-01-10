#include "ir/node_shapes/gemm_shape.h"

#include "ir/node.h"
#include "ir/node_utils.h"

namespace simple_ai {
namespace ir {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Gemm

std::string GemmShapeInfer::node_type() const { return "Gemm"; }

Status GemmShapeInfer::infer(const std::string& node_name, const std::vector<NodeArg*>& inputs,
                             const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes,
                             std::vector<NodeArg*>& outputs) {
    if (inputs.size() < 2 || outputs.size() != 1) {
        std::ostringstream oss;
        oss << "Node: Gemm[" << node_name << "], invalid input size: " << inputs.size()
            << " or output size: " << outputs.size();
        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    const auto& mat_A_shape = inputs[0]->shape();
    const auto& mat_B_shape = inputs[1]->shape();

    size_t a_dim_num = mat_A_shape.dims_num();
    size_t b_dim_num = mat_B_shape.dims_num();
    if (a_dim_num != 2 || b_dim_num != 2) {
        std::ostringstream oss;
        oss << "Node: Gemm[" << node_name << "], invalid dims of inputs. Matrix A: " << a_dim_num
            << " Matrix B: " << b_dim_num;
        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    int64_t transA = utils::get_attr_or_default<int64_t>("transA", 0, attributes);
    int64_t transB = utils::get_attr_or_default<int64_t>("transB", 0, attributes);

    int64_t m_a = 0;
    int64_t k_a = 0;
    int64_t k_b = 0;
    int64_t n_b = 0;

    if (transA) {
        k_a = mat_A_shape[0];
        m_a = mat_A_shape[1];
    } else {
        m_a = mat_A_shape[0];
        k_a = mat_A_shape[1];
    }

    if (transB) {
        n_b = mat_B_shape[0];
        k_b = mat_B_shape[1];
    } else {
        k_b = mat_B_shape[0];
        n_b = mat_B_shape[1];
    }

    if (k_a != k_b) {
        std::ostringstream oss;
        oss << "Node: Gemm[" << node_name << "], mismatch for A dim1 and B dim0";
        return Status(StatusCode::INVALID_PARAM, oss.str());
    }

    // the matrix C exists
    if (inputs.size() == 3) {
        const auto& mat_C_shape = inputs[2]->shape();
        size_t c_dim_num = mat_C_shape.dims_num();
        bool valid_c_shape = (c_dim_num == 2 && (mat_C_shape[0] == m_a || mat_C_shape[0] == 1) &&
                              (mat_C_shape[1] == n_b || mat_C_shape[1] == 1)) ||
                             (c_dim_num == 1 && (mat_C_shape[0] == 1 || mat_C_shape[0] == n_b));
        if (!valid_c_shape) {
            std::ostringstream oss;
            oss << "Node: Gemm[" << node_name << "], invalid matrix C dimensions";
            return Status(StatusCode::INVALID_PARAM, oss.str());
        }
    }

    TensorShape out_shape;
    out_shape.add_dim(m_a);
    out_shape.add_dim(n_b);

    outputs[0]->set_shape(out_shape);

    return Status::ok();
}

}    // namespace ir
}    // namespace simple_ai