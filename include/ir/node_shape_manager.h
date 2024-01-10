#ifndef _H_SIMPLE_AI_IR_NODE_SHAPE_MANAGER_H_
#define _H_SIMPLE_AI_IR_NODE_SHAPE_MANAGER_H_

#include <mutex>

#include "common/common.h"
#include "framework/common_defines.h"
#include "node.h"
#include "node_utils.h"

using namespace simple_ai::framework;

namespace simple_ai {
namespace ir {

class NodeShapeManager {
public:
    ~NodeShapeManager() = default;
    static NodeShapeManager* instance();

    /**
     * @brief Get the shape infer object
     *
     * @param node_type the node type
     * @return IShapeInfer* nullptr if the node type exist in the manager.
     */
    IShapeInfer* get_shape_infer(const std::string& node_type);

    /**
     * @brief register all node shape infers
     *
     */
    void register_all_infer();

private:
    SIMPLE_AI_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(NodeShapeManager);
    NodeShapeManager() = default;

    template <typename T>
    void register_node_infer();

private:
    std::unordered_map<std::string, std::unique_ptr<IShapeInfer>> m_node_infer_map;
    std::once_flag m_init_flag;
};

}    // namespace ir
}    // namespace simple_ai

#endif