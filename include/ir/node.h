#ifndef _H_SIMPLE_AI_IR_NODE_H_
#define _H_SIMPLE_AI_IR_NODE_H_

#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "common/common.h"
#include "node_arg.h"
#include "node_attribute.h"

namespace simple_ai {
namespace ir {

class Graph;
class Node;
class Edge;
struct EdgeCompare;

/**
 * @brief The edge which links nodes in the graph.
 *
 */
class Edge {
public:
    /**
     * @brief Constructor
     *
     * @param other_node the other node this edge links
     * @param src_arg_index the node argument index of source node of this edge
     * @param dst_arg_index the node argument index of destination node of this edge
     */
    Edge(const Node& other_node, int src_arg_index, int dst_arg_index)
        : m_other_node(other_node), m_src_arg_index(src_arg_index), m_dst_arg_index(dst_arg_index) {}

    const Node& other_node() const { return m_other_node; }
    int src_arg_index() const { return m_src_arg_index; }
    int dst_arg_index() const { return m_dst_arg_index; }

private:
    // the other node relative to the current node which owns this Edge
    const Node& m_other_node;
    // the source NodeArg index
    int m_src_arg_index;
    // the destination NodeArg index
    int m_dst_arg_index;
};

struct EdgeCompare {
    bool operator()(const Edge& lhs, const Edge& rhs) const;
};

using EdgeSet = std::set<Edge, EdgeCompare>;

/**
 * @brief Node shape infer interface
 *
 */
class IShapeInfer {
public:
    IShapeInfer() = default;
    virtual ~IShapeInfer() = default;

    /**
     * @brief Get the node type
     *
     * @return const std::string&
     */
    virtual std::string node_type() const = 0;

    /**
     * @brief  do shap infer
     *
     * @param node_name the node name
     * @param inputs the node inputs args
     * @param attributes the node attributes
     * @param outputs input/output parameter. the node output args
     * @return Status
     */
    virtual Status infer(const std::string& node_name, const std::vector<NodeArg*>& inputs,
                         const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes,
                         std::vector<NodeArg*>& outputs) = 0;
};

/**
 * @brief The node in a computation graph which is a DAG structure;
 *
 */
class Node {
public:
    Node() = default;
    virtual ~Node() = default;

    /**
     * @brief Constructor
     *
     * @param id the unique id for this node
     * @param graph the gragh which holds this node
     */
    Node(int id, const Graph& graph);

    /**
     * @brief initialize the node
     *
     * @param name node name
     * @param type node type
     * @param domain node domain
     * @param description node description
     * @param input_args node input args
     * @param output_args node output args
     * @param attributes node attributes
     */
    void init(const std::string& name, const std::string& type, const std::string& domain,
              const std::string& description, const std::vector<NodeArg*>& input_args,
              const std::vector<NodeArg*>& output_args,
              std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>&& attributes);

    int id() const { return m_id; }

    const std::string& name() const { return m_name; }
    const std::string& type() const { return m_type; }

    const std::vector<NodeArg*>& input_args() const;
    const std::vector<NodeArg*>& output_args() const;

    const EdgeSet& input_edges() const;
    const EdgeSet& output_edges() const;

    void add_input_edge(const Edge& edge);
    void add_output_edge(const Edge& edge);

    void remove_input_edge(const Edge& edge);
    void remove_output_edge(const Edge& edge);

    Status infer_shape(IShapeInfer* infer);

private:
    const Graph& m_graph;
    // the unique id
    int m_id;

    // node name
    std::string m_name;
    // node type
    std::string m_type;
    // node domain
    std::string m_domain;
    // node description
    std::string m_desc;
    // node inputs arguments
    std::vector<NodeArg*> m_input_args;
    // node outputs arguments
    std::vector<NodeArg*> m_output_args;
    // node attributes
    std::unordered_map<std::string, std::unique_ptr<NodeAttribute>> m_attributes;

    // the edges for nodes that produce inputs to this Node
    EdgeSet m_input_edges;

    // the edges for nodes that consume outputs from this Node
    EdgeSet m_output_edges;
};

}    // namespace ir
}    // namespace simple_ai

#endif