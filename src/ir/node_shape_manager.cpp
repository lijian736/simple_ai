#include "ir/node_shape_manager.h"

#include "ir/node_shapes/add_shape.h"
#include "ir/node_shapes/conv_shape.h"
#include "ir/node_shapes/flatten_shape.h"
#include "ir/node_shapes/gemm_shape.h"
#include "ir/node_shapes/global_avg_pool_shape.h"
#include "ir/node_shapes/max_pool_shape.h"
#include "ir/node_shapes/relu_shape.h"

namespace simple_ai {
namespace ir {

NodeShapeManager* NodeShapeManager::instance() {
    static NodeShapeManager instance;
    return &instance;
}

template <typename T>
void NodeShapeManager::register_node_infer() {
    auto infer = std::make_unique<T>();

    auto ret = m_node_infer_map.emplace(infer->node_type(), nullptr);
    if (ret.second) {
        ret.first->second = std::move(infer);
    }
}

void NodeShapeManager::register_all_infer() {
    register_node_infer<ConvShapeInfer>();
    register_node_infer<GemmShapeInfer>();
    register_node_infer<ReluShapeInfer>();
    register_node_infer<MaxPoolShapeInfer>();
    register_node_infer<GlobalAveragePoolShapeInfer>();
    register_node_infer<FlattenShapeInfer>();
    register_node_infer<AddShapeInfer>();
}

IShapeInfer* NodeShapeManager::get_shape_infer(const std::string& node_type) {
    auto iter = m_node_infer_map.find(node_type);
    if (iter != m_node_infer_map.end()) {
        iter->second.get();
    }

    return nullptr;
}

}    // namespace ir
}    // namespace simple_ai