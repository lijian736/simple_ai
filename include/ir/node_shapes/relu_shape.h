#ifndef _H_SIMPLE_AI_IR_NODE_SHAPES_RELU_H_
#define _H_SIMPLE_AI_IR_NODE_SHAPES_RELU_H_

#include "ir/node.h"

namespace simple_ai {
namespace ir {

// https://github.com/onnx/onnx/blob/main/docs/Operators.md#Relu
class ReluShapeInfer : public IShapeInfer {
public:
    virtual std::string node_type() const override;

    virtual Status infer(const std::string& node_name, const std::vector<NodeArg*>& inputs,
                         const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes,
                         std::vector<NodeArg*>& outputs) override;
};

}    // namespace ir
}    // namespace simple_ai

#endif