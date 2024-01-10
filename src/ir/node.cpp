#include "ir/node.h"
#include "ir/graph.h"

using namespace simple_ai::common;

namespace simple_ai {
namespace ir {

bool EdgeCompare::operator()(const Edge& lhs, const Edge& rhs) const {
    if (lhs.other_node().id() == rhs.other_node().id()) {
        if (lhs.src_arg_index() == rhs.src_arg_index()) {
            return lhs.dst_arg_index() < rhs.dst_arg_index();
        }
        return lhs.src_arg_index() < rhs.src_arg_index();
    }
    return lhs.other_node().id() < rhs.other_node().id();
}

Node::Node(int id, const Graph& graph) : m_id(id), m_graph(graph) {}

void Node::init(const std::string& name, const std::string& type, const std::string& domain,
                const std::string& description, const std::vector<NodeArg*>& input_args,
                const std::vector<NodeArg*>& output_args,
                std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>&& attributes) {
    m_name = name;
    m_type = type;
    m_domain = domain;
    m_desc = description;
    m_input_args = input_args;
    m_output_args = output_args;
    m_attributes = std::move(attributes);
}

const std::vector<NodeArg*>& Node::input_args() const { return m_input_args; }

const std::vector<NodeArg*>& Node::output_args() const { return m_output_args; }

const EdgeSet& Node::input_edges() const { return m_input_edges; }

const EdgeSet& Node::output_edges() const { return m_output_edges; }

void Node::add_input_edge(const Edge& edge) { m_input_edges.insert(edge); }

void Node::add_output_edge(const Edge& edge) { m_output_edges.insert(edge); }

void Node::remove_input_edge(const Edge& edge) { m_input_edges.erase(edge); }

void Node::remove_output_edge(const Edge& edge) { m_output_edges.erase(edge); }

Status Node::infer_shape(IShapeInfer* infer){
    return infer->infer(m_name, m_input_args, m_attributes, m_output_args);
}

}    // namespace ir
}    // namespace simple_ai