#include "ir/graph.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stack>
#include <unordered_set>

#include "ir/model.h"
#include "ir/node_arg.h"
#include "ir/node_attribute.h"
#include "ir/node_shape_manager.h"
#include "ir/tensor.h"
#include "utils/utils.h"

using namespace simple_ai::utils;
using namespace simple_ai::common;

namespace simple_ai {
namespace ir {

Graph::Graph(const Model& model) : m_model(model) {}

void Graph::add_input_name(const std::string& name) { m_inputs_name.emplace_back(name); }
void Graph::add_output_name(const std::string& name) { m_outputs_name.emplace_back(name); }

void Graph::add_initializer(std::unique_ptr<Tensor>&& tensor) {
    auto ret = m_initializer_map.emplace(tensor->name(), nullptr);
    if (ret.second) {
        // create new initializer tensor
        ret.first->second = std::move(tensor);
    } else {
        // override the existed initializer tensor
        ret.first->second = std::move(tensor);
    }
}

bool Graph::has_initializer(const std::string& name) { return m_initializer_map.find(name) != m_initializer_map.end(); }

NodeArg* Graph::get_or_create_nodearg(const std::string& name, const NodeArg& node_arg) {
    auto insert_result = m_nodearg_map.emplace(name, nullptr);
    if (insert_result.second) {
        insert_result.first->second = std::make_unique<NodeArg>(node_arg);
    }
    return insert_result.first->second.get();
}

NodeArg* Graph::get_nodearg(const std::string& name) {
    auto it = m_nodearg_map.find(name);
    if (it != m_nodearg_map.end()) {
        return it->second.get();
    }

    return nullptr;
}

void Graph::add_node(std::unique_ptr<Node>&& node) { m_nodes.emplace_back(std::move(node)); }

const std::vector<Node*>& Graph::get_topological_nodes() const { return m_topological_nodes; }

const std::vector<std::unique_ptr<Node>>& Graph::get_nodes() const { return m_nodes; }

Status Graph::initialize() {
    m_inputs_include_initializer.clear();
    m_inputs_exclude_initializer.clear();
    m_outputs.clear();
    m_overridable_initializers.clear();

    // graph initializers map. key: node arg name, value: node arg pointer
    std::unordered_map<std::string, NodeArg*> graph_initializers;
    // graph inputs map. key: node arg name, value: node arg pointer
    std::unordered_map<std::string, NodeArg*> graph_inputs;
    // graph all node outputs. key: node arg name, value: node arg pointer
    std::unordered_map<std::string, NodeArg*> nodes_outputs;

    // get initializers of the graph
    for (auto& pair : m_initializer_map) {
        auto arg = get_nodearg(pair.first);
        graph_initializers.insert({pair.first, arg});
    }

    // get graph inputs
    for (auto& name : m_inputs_name) {
        auto node_arg = get_nodearg(name);
        graph_inputs.insert({name, node_arg});
        m_inputs_include_initializer.emplace_back(node_arg);
        if (graph_initializers.end() == graph_initializers.find(name)) {
            m_inputs_exclude_initializer.emplace_back(node_arg);
        }
    }

    // get all nodes outputs
    for (auto& node : m_nodes) {
        const auto& output_args = node->output_args();
        for (auto& out : output_args) {
            nodes_outputs.insert({out->name(), out});
        }
    }

    // set graph outputs. graph outptus are nodes's outputs, initializers or graph inputs.
    for (auto& name : m_outputs_name) {
        auto it_node = nodes_outputs.find(name);

        // the graph output is some node's output
        if (it_node != nodes_outputs.end()) {
            m_outputs.emplace_back(it_node->second);
        } else {
            auto it_initializer = graph_initializers.find(name);
            // the graph output is an initializer
            if (it_initializer != graph_initializers.end()) {
                m_outputs.emplace_back(it_initializer->second);
            } else {
                auto it_input = graph_inputs.find(name);
                // the graph output is graph's some input
                if (it_input != graph_inputs.end()) {
                    m_outputs.emplace_back(it_input->second);
                } else {
                    // error
                    std::ostringstream oss;
                    oss << "Invalid graph. graph's output [" << name << "] does not exist in the graph";
                    return Status(StatusCode::INVALID_MODEL, oss.str());
                }
            }
        }
    }

    // set overridable initializers
    auto it_include = m_inputs_include_initializer.cbegin();
    const auto end_include = m_inputs_include_initializer.cend();
    auto it_exclude = m_inputs_exclude_initializer.cbegin();
    const auto end_exclude = m_inputs_exclude_initializer.cend();

    while (it_include != end_include) {
        if (it_exclude != end_exclude && *it_include == *it_exclude) {
            ++it_include;
            ++it_exclude;
            continue;
        }
        m_overridable_initializers.push_back(*it_include);
        ++it_include;
    }

    return Status::ok();
}

Status Graph::construct_topology() {
    // Step 1. initialize and check inputs, initializers and outputs, ensure their names are unique
    auto ret = init_inputs_initializers_outputs();
    if (!ret.is_ok()) {
        return ret;
    }
    // Step 2. build connections between nodes in this graph
    ret = build_nodes_connections();
    if (!ret.is_ok()) {
        return ret;
    }
    // Step 3. topological sort, ensure the graph is a DAG
    ret = topological_sort();
    if (!ret.is_ok()) {
        return ret;
    }
    // Step 4. type/shape inference.
    ret = infer_shape();
    if (!ret.is_ok()) {
        return ret;
    }

    // Step 5. clean up
    m_topology_context.clear();
    ret = clean_unused_initializers_args();
    if (!ret.is_ok()) {
        return ret;
    }

    return Status::ok();
}

Status Graph::init_inputs_initializers_outputs() {
    auto ret = check_inputs_initializers_names();
    if (!ret.is_ok()) {
        return ret;
    }

    ret = check_no_duplicate_names();
    if (!ret.is_ok()) {
        return ret;
    }

    return Status::ok();
}

Status Graph::check_inputs_initializers_names() {
    std::unordered_set<std::string_view>& inputs_and_initializers = m_topology_context.inputs_and_initializers;

    // check the input (excluding the initializers) names
    for (auto& item : m_inputs_exclude_initializer) {
        auto ret = inputs_and_initializers.insert(item->name());
        if (!ret.second) {
            std::ostringstream oss;
            oss << "Duplicate input name: " << item->name();
            return Status(StatusCode::INVALID_MODEL, oss.str());
        }
    }

    // check the initializer names
    for (auto& item : m_initializer_map) {
        inputs_and_initializers.insert(item.first);
    }

    return Status::ok();
}

Status Graph::check_no_duplicate_names() {
    auto& inputs_and_initializers = m_topology_context.inputs_and_initializers;
    auto& output_args = m_topology_context.output_args;
    auto& node_name_to_id = m_topology_context.node_name_to_id;

    output_args.clear();
    node_name_to_id.clear();

    for (auto& item : m_nodes) {
        // check node name shoule be unique
        auto& node_name = item->name();
        if (!node_name.empty() && node_name_to_id.find(node_name) != node_name_to_id.end()) {
            std::ostringstream oss;
            oss << "Node name is not unique: " << node_name;
            return Status(StatusCode::INVALID_MODEL, oss.str());
        }

        node_name_to_id[node_name] = item->id();

        // node output's name should be unique
        int output_index = -1;
        for (auto& output : item->output_args()) {
            ++output_index;
            auto& output_name = output->name();
            if (!output_name.empty()) {
                if (inputs_and_initializers.count(output_name)) {
                    std::ostringstream oss;
                    oss << "Node output name is same to some input/initializer: " << output_name;
                    return Status(StatusCode::INVALID_MODEL, oss.str());
                }

                auto ret = output_args.insert({output_name, {item.get(), output_index}});
                if (!ret.second) {
                    std::ostringstream oss;
                    oss << "Node output name is not unique: " << output_name;
                    return Status(StatusCode::INVALID_MODEL, oss.str());
                }
            }
        }
    }

    return Status::ok();
}

Status Graph::build_nodes_connections() {
    std::vector<int> unused_nodes_id;

    for (auto& node : m_nodes) {
        const auto& inputs = node->input_args();
        if (!inputs.empty()) {
            int input_arg_index = -1;
            for (auto& input_arg : inputs) {
                ++input_arg_index;

                const auto& input_name = input_arg->name();
                if (input_name.empty()) {
                    continue;
                }

                auto it_output_arg = m_topology_context.output_args.find(input_name);
                if (it_output_arg != m_topology_context.output_args.end()) {
                    // the input to this node is an output from the previous node.
                    // build connections between this node and the node producing the output
                    auto& output_node = it_output_arg->second.first;
                    add_edge(output_node->id(), node->id(), it_output_arg->second.second, input_arg_index);
                } else {
                    // the input to this node should be a graph input or initializer
                    if (m_topology_context.inputs_and_initializers.find(input_name) ==
                        m_topology_context.inputs_and_initializers.end()) {
                        std::ostringstream oss;
                        oss << "Invalid mode. Node input [" << input_name
                            << " is not a graph input, initializer, or output of a previous node";
                        return Status(StatusCode::INVALID_MODEL, oss.str());
                    }
                }
            }
        } else if (node->output_args().empty()) {
            // the node has no inputs and outputs. remove it.
            unused_nodes_id.emplace_back(node->id());
        }
    }

    // the node has no inputs and outputs. remove it.
    for (auto& id : unused_nodes_id) {
        auto status = remove_node(id);
        if (!status.is_ok()) {
            return status;
        }
    }

    auto ret = init_node_arg_to_connected_nodes();
    if (!ret.is_ok()) {
        return ret;
    }

    return Status::ok();
}

Status Graph::remove_node(int id) {
    auto it_node = std::find_if(m_nodes.begin(), m_nodes.end(), [id](const auto& node) { return node->id() == id; });
    if (it_node == m_nodes.end()) {
        return Status::ok();
    }

    // ensure the node has no outputs
    if ((*it_node)->output_edges().size() != 0) {
        std::ostringstream oss;
        oss << "Remove node fail. the node has " << (*it_node)->output_edges().size() << " output edges";
        return Status(StatusCode::FAIL, oss.str());
    }

    // remove the input edges of this node
    auto& input_edges = (*it_node)->input_edges();
    for (auto& input_edge : input_edges) {
        remove_edge(input_edge.other_node().id(), id, input_edge.src_arg_index(), input_edge.dst_arg_index());
    }

    // erase the node
    m_nodes.erase(it_node);

    return Status::ok();
}

Status Graph::add_edge(int src_node_id, int dest_node_id, int src_arg_index, int dst_arg_index) {
    // find source node
    auto it_node_src = std::find_if(m_nodes.begin(), m_nodes.end(),
                                    [src_node_id](const auto& node) { return node->id() == src_node_id; });
    if (it_node_src == m_nodes.end()) {
        std::ostringstream oss;
        oss << "node not found, nod id: " << src_node_id;
        return Status(StatusCode::FAIL, oss.str());
    }

    // find destination node
    auto it_node_dst = std::find_if(m_nodes.begin(), m_nodes.end(),
                                    [dest_node_id](const auto& node) { return node->id() == dest_node_id; });
    if (it_node_dst == m_nodes.end()) {
        std::ostringstream oss;
        oss << "node not found, nod id: " << dest_node_id;
        return Status(StatusCode::FAIL, oss.str());
    }

    // check the arg index
    if (src_arg_index < 0 || dst_arg_index < 0) {
        return Status(StatusCode::FAIL, "invalid arg index");
    }

    NodeArg* src_arg = nullptr;
    NodeArg* dst_arg = nullptr;
    if (static_cast<size_t>(src_arg_index) < (*it_node_src)->output_args().size()) {
        src_arg = (*it_node_src)->output_args()[src_arg_index];
    } else {
        return Status(StatusCode::FAIL, "invalid source arg index");
    }

    if (static_cast<size_t>(dst_arg_index) < (*it_node_dst)->input_args().size()) {
        dst_arg = (*it_node_dst)->input_args()[dst_arg_index];
    } else {
        return Status(StatusCode::FAIL, "invalid destination arg index");
    }

    if (src_arg != dst_arg) {
        if (*src_arg != *dst_arg) {
            return Status(StatusCode::FAIL, "Argument type mismatch");
        }
    }

    (*it_node_src)->add_output_edge(Edge(*(it_node_dst->get()), src_arg_index, dst_arg_index));
    (*it_node_dst)->add_input_edge(Edge(*(it_node_src->get()), src_arg_index, dst_arg_index));

    return Status::ok();
}

Status Graph::remove_edge(int src_node_id, int dest_node_id, int src_arg_index, int dst_arg_index) {
    // find source node
    auto it_node_src = std::find_if(m_nodes.begin(), m_nodes.end(),
                                    [src_node_id](const auto& node) { return node->id() == src_node_id; });
    if (it_node_src == m_nodes.end()) {
        std::ostringstream oss;
        oss << "node not found, nod id: " << src_node_id;
        return Status(StatusCode::FAIL, oss.str());
    }

    // find destination node
    auto it_node_dst = std::find_if(m_nodes.begin(), m_nodes.end(),
                                    [dest_node_id](const auto& node) { return node->id() == dest_node_id; });
    if (it_node_dst == m_nodes.end()) {
        std::ostringstream oss;
        oss << "node not found, nod id: " << dest_node_id;
        return Status(StatusCode::FAIL, oss.str());
    }

    // check the arg index
    if (src_arg_index < 0 || dst_arg_index < 0) {
        return Status(StatusCode::FAIL, "invalid arg index");
    }

    NodeArg* src_arg = nullptr;
    NodeArg* dst_arg = nullptr;
    if (static_cast<size_t>(src_arg_index) < (*it_node_src)->output_args().size()) {
        src_arg = (*it_node_src)->output_args()[src_arg_index];
    } else {
        return Status(StatusCode::FAIL, "invalid source arg index");
    }

    if (static_cast<size_t>(dst_arg_index) < (*it_node_dst)->input_args().size()) {
        dst_arg = (*it_node_dst)->input_args()[dst_arg_index];
    } else {
        return Status(StatusCode::FAIL, "invalid destination arg index");
    }

    if (src_arg != dst_arg) {
        return Status(StatusCode::FAIL, "Argument mismatch when removing edge");
    }

    (*it_node_dst)->remove_input_edge(Edge((*(*it_node_src).get()), src_arg_index, dst_arg_index));
    (*it_node_src)->remove_output_edge(Edge((*(*it_node_dst).get()), src_arg_index, dst_arg_index));

    return Status::ok();
}

Status Graph::init_node_arg_to_connected_nodes() {
    m_node_arg_to_producer_node.clear();
    m_node_arg_to_consumer_nodes.clear();

    for (auto& node : m_nodes) {
        for (auto& input : node->input_args()) {
            m_node_arg_to_consumer_nodes[input->name()].insert(node->id());
        }

        for (auto& output : node->output_args()) {
            m_node_arg_to_producer_node.insert({output->name(), node->id()});
        }
    }

    return Status::ok();
}

Status Graph::topological_sort() {
    m_topological_nodes.clear();

    std::unordered_set<int> downstream_nodes;    // downstream nodes id of the current node
    std::unordered_set<int> nodes_visited;       // the nodes id visited
    std::unordered_set<int> nodes_added;         // the nodes id added
    std::stack<int> nodes_stack;                 // the nodes id stack for depth-first search

    // add the graph input to topological nodes list
    std::for_each(m_nodes.cbegin(), m_nodes.cend(), [&](const auto& node) {
        auto& input_edges = node->input_edges();
        if (input_edges.size() == 0) {
            m_topological_nodes.emplace_back(node.get());
            nodes_visited.insert(node->id());
            nodes_added.insert(node->id());
        }
    });

    // find all the leaf nods
    for (auto it = m_nodes.begin(); it != m_nodes.end(); ++it) {
        if ((*it)->output_edges().empty()) {
            nodes_stack.push((*it)->id());
        }
    }

    while (!nodes_stack.empty()) {
        auto& current_node = nodes_stack.top();
        nodes_stack.pop();

        auto current_node_ptr = std::find_if(m_nodes.cbegin(), m_nodes.cend(), [current_node](const auto& item) {
            if (item->id() == current_node) {
                return true;
            } else {
                return false;
            }
        });

        if (current_node_ptr == m_nodes.cend()) {
            continue;
        }

        if (nodes_added.find(current_node) != nodes_added.end()) {
            continue;
        }

        if (nodes_visited.find(current_node) != nodes_visited.end()) {
            m_topological_nodes.emplace_back(current_node_ptr->get());
            nodes_added.insert(current_node);
            downstream_nodes.erase(current_node);
            continue;
        }

        nodes_visited.insert(current_node);
        downstream_nodes.insert(current_node);

        nodes_stack.push(current_node);

        for (auto it_input = (*current_node_ptr)->input_edges().cbegin();
             it_input != (*current_node_ptr)->input_edges().cend(); ++it_input) {
            auto other_node_id = it_input->other_node().id();
            if (downstream_nodes.find(other_node_id) != downstream_nodes.end()) {
                return Status(StatusCode::INVALID_MODEL, "The graph is not a DAG");
            }

            if (nodes_visited.find(other_node_id) == nodes_visited.end()) {
                nodes_stack.push(other_node_id);
            }
        }
    }

    if (m_topological_nodes.size() != m_nodes.size()) {
        return Status(StatusCode::INVALID_MODEL, "the graph is not a DAG");
    }

    return Status::ok();
}

Status Graph::clean_unused_initializers_args() {
    std::unordered_set<const NodeArg*> used_node_args;

    // the graph inputs(exclude initializers) must be in used node args set
    std::for_each(m_inputs_exclude_initializer.cbegin(), m_inputs_exclude_initializer.cend(),
                  [&used_node_args](const NodeArg* item) { used_node_args.insert(item); });

    // the grapn overridable initializers must be in used node args set
    std::for_each(m_overridable_initializers.cbegin(), m_overridable_initializers.cend(),
                  [&used_node_args](const NodeArg* item) { used_node_args.insert(item); });

    // the graph outputs must be in used node args set
    std::for_each(m_outputs.cbegin(), m_outputs.cend(),
                  [&used_node_args](const NodeArg* item) { used_node_args.insert(item); });

    // the nodes inputs must be in used node args set
    for (const auto& node : m_nodes) {
        for (const auto& input_arg : node->input_args()) {
            used_node_args.insert(input_arg);
        }
    }

    std::vector<std::string> erase_initializers;
    for (const auto& pair : m_initializer_map) {
        const std::string& init_name = pair.first;
        const auto* tmp_node_arg = get_nodearg(init_name);
        if (tmp_node_arg == nullptr) {
            std::ostringstream oss;
            oss << "Can't find the initializer: " << init_name;
            return Status(StatusCode::FAIL, oss.str());
        }

        if (used_node_args.find(tmp_node_arg) == used_node_args.cend()) {
            erase_initializers.emplace_back(init_name);
        }
    }

    // erase unused initialziers
    std::for_each(erase_initializers.cbegin(), erase_initializers.cend(), [this](const std::string& name) {
        auto iter = m_initializer_map.find(name);
        if (iter != m_initializer_map.end()) {
            m_initializer_map.erase(iter);
        }
    });

    // clear unused node args

    // the nodes outputs
    for (const auto& node : m_nodes) {
        for (const auto& output_arg : node->output_args()) {
            used_node_args.insert(output_arg);
        }
    }

    for (auto it = m_nodearg_map.begin(); it != m_nodearg_map.end();) {
        const auto* current_node_arg = it->second.get();
        const auto& node_arg_name = it->first;

        if (!node_arg_name.empty() && used_node_args.find(current_node_arg) == used_node_args.cend()) {
            it = m_nodearg_map.erase(it);
        } else {
            ++it;
        }
    }

    return Status::ok();
}

Status Graph::infer_shape() {
    for (auto& node : m_topological_nodes) {
        IShapeInfer* infer = NodeShapeManager::instance()->get_shape_infer(node->type());
        if(!infer){
            std::ostringstream oss;
            oss << "Infer object for node: " << node->type() << "[" << node->name() << "]" << " not found";
            return Status(StatusCode::FAIL, oss.str());
        }
        auto ret = node->infer_shape(infer);
        if(!ret.is_ok()){
            return ret;
        }
    }

    return Status::ok();
}

}    // namespace ir
}    // namespace simple_ai