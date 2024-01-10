#ifndef _H_SIMPLE_AI_IR_GRAPH_H_
#define _H_SIMPLE_AI_IR_GRAPH_H_

#include <memory>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "common/common.h"
#include "node.h"
#include "node_arg.h"
#include "tensor.h"

namespace simple_ai {
namespace ir {

class Model;

/**
 * @brief A graph defines the computational logic of a model and is comprised of a parameterized
 * list of nodes that form a directed acyclic graph based on their inputs and outputs.
 *
 */
class Graph {
public:
    virtual ~Graph() = default;

    Graph(const Model& model);

    /**
     * @brief When all the args, nodes, initializers are ready, initialize the graph state,
     * including graph inputs, graph outputs
     *
     * @return Status
     */
    Status initialize();

    /**
     * @brief add graph input name
     *
     * @param name the input name
     */
    void add_input_name(const std::string& name);

    /**
     * @brief add graph output name
     *
     * @param name the output name
     */
    void add_output_name(const std::string& name);

    /**
     * @brief add an initializer to this graph.
     * if the initializer has already existed in this graph, it will be
     * replaced with the new one
     *
     * @param tensor the initializer
     */
    void add_initializer(std::unique_ptr<Tensor>&& tensor);

    /**
     * @brief Check if the initialize with the `name` exist in the graph
     *
     * @param name the initializer name
     * @return true
     * @return false
     */
    bool has_initializer(const std::string& name);

    /**
     * @brief Add an ir node
     *
     * @param node the ir node
     */
    void add_node(std::unique_ptr<Node>&& node);

    /**
     * @brief Get or create the node arg with name
     *
     * @param name the unique node arg name
     * @param node_arg the node arg
     * @return NodeArg*
     */
    NodeArg* get_or_create_nodearg(const std::string& name, const NodeArg& node_arg);

    /**
     * @brief Get the node arg
     *
     * @param name the node arg name
     * @return NodeArg* nullptr if the arg does not exist. otherwise return its pointer
     */
    NodeArg* get_nodearg(const std::string& name);

    /**
     * @brief Get the nodes in the graph
     * 
     * @return const std::vector<std::unique_ptr<Node>>& 
     */
    const std::vector<std::unique_ptr<Node>>& get_nodes() const;

    /**
     * @brief Get the nodes of the graph in topological order
     * 
     * @return const std::vector<Node*>& 
     */
    const std::vector<Node*>& get_topological_nodes() const;

    /**
     * @brief construct the topological structure of this graph, ensure that the graph is valid, initialized,
     * and be able to be executed.
     *
     * 1. Node name and node output's name should be unique.
     * 2. Graph is must be a DAG(directed acyclic graph) and nodes must be in topological sort.
     *
     * @return Status
     */
    Status construct_topology();

private:
    /**
     * @brief initialize the graphs's inputs, initializers and outputs
     *
     * @return Status
     */
    Status init_inputs_initializers_outputs();

    /**
     * @brief inputs name should be unique
     *
     * @return Status
     */
    Status check_inputs_initializers_names();

    /**
     * @brief node name and node outputs should be unique
     *
     * @return Status
     */
    Status check_no_duplicate_names();

    /**
     * @brief build the nodes connections
     *
     * @return Status
     */
    Status build_nodes_connections();

    /**
     * @brief remove a node
     *
     * @param id the node id
     * @return Status
     */
    Status remove_node(int id);

    /**
     * @brief Add an edge between the source node and destination node
     *
     * @param src_node_id source node id
     * @param dest_node_id destination node id
     * @param src_arg_index source arg index
     * @param dst_arg_index destination arg index
     *
     * @return Status
     */
    Status add_edge(int src_node_id, int dest_node_id, int src_arg_index, int dst_arg_index);

    /**
     * @brief Remove an edge between the source node and destination node
     *
     * @param src_node_id source node id
     * @param dest_node_id destination node id
     * @param src_arg_index source arg index
     * @param dst_arg_index destination arg index
     * @return Status
     */
    Status remove_edge(int src_node_id, int dest_node_id, int src_arg_index, int dst_arg_index);

    /**
     * @brief intialize the node args, connect its producer and consumer nodes
     *
     * @return Status
     */
    Status init_node_arg_to_connected_nodes();

    /**
     * @brief do topological sort
     *
     * @return Status if the graph is not a DAG, return fail
     */
    Status topological_sort();

    /**
     * @brief do node input/output shape inference
     *
     * @return Status
     */
    Status infer_shape();

    /**
     * @brief clean up unused initializers and node args
     * 
     * @return Status 
     */
    Status clean_unused_initializers_args();

private:
    /**
     * @brief the Topological Context help to build the topology of this graph.
     *
     */
    struct TopologyContext {
        TopologyContext(const Graph& owner) : m_graph{owner} {}

        // inputs and initializers names
        std::unordered_set<std::string_view> inputs_and_initializers;
        // outputs. key: output argument name, value: the output arg's attached node and the argument index in the
        // node's outputs
        std::unordered_map<std::string_view, std::pair<Node*, int>> output_args;
        // node name to node id. key: node name, value: node id
        std::unordered_map<std::string_view, int> node_name_to_id;

        void clear() {
            inputs_and_initializers.clear();
            output_args.clear();
            node_name_to_id.clear();
        }

    private:
        const Graph& m_graph;
    };

private:
    // the model which the graph belongs to
    const Model& m_model;

    // the graph inputs names
    std::vector<std::string> m_inputs_name;

    // the graph outputs names
    std::vector<std::string> m_outputs_name;

    // the nodes in the graph
    std::vector<std::unique_ptr<Node>> m_nodes;

    // key: the initializer tensor name, value: the initializer tensor unique pointer
    std::unordered_map<std::string, std::unique_ptr<Tensor>> m_initializer_map;

    // key: node arg name, value: the NodeArg unique pointer
    std::unordered_map<std::string, std::unique_ptr<NodeArg>> m_nodearg_map;

    // the graph inputs, including the initializers which are treated as inputs to the graph
    std::vector<NodeArg*> m_inputs_include_initializer;

    // the graph inputs, excluding the initializers which are treated as inputs to the graph
    std::vector<NodeArg*> m_inputs_exclude_initializer;

    // the graph outputs
    std::vector<NodeArg*> m_outputs;

    // the graph overridable initializers. some initializers be treated as the graph inputs,
    // in this case, the initializers can be override by the user input
    std::vector<NodeArg*> m_overridable_initializers;

    // the nodes in topological order
    std::vector<Node*> m_topological_nodes;

    // node arg to its producer node, key: node arg name, value: producer node id
    std::unordered_map<std::string, int> m_node_arg_to_producer_node;

    // node arg to its consumer nodes, key: node arg name, value: consumer node id set
    std::unordered_map<std::string, std::unordered_set<int>> m_node_arg_to_consumer_nodes;

    // the topology context
    TopologyContext m_topology_context{*this};
};

}    // namespace ir
}    // namespace simple_ai

#endif