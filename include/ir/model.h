#ifndef _H_SIMPLE_AI_IR_MODEL_H_
#define _H_SIMPLE_AI_IR_MODEL_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "common/common.h"
#include "graph.h"

namespace simple_ai {
namespace ir {

/**
 * @brief the model intermediate representation class
 *
 */
class Model {
public:
    Model() = default;
    virtual ~Model() = default;

    void set_graph(std::unique_ptr<Graph>&& graph);
    Graph* get_graph() const;

    int64_t get_ir_version() const;
    void set_ir_version(int64_t ir_version);

    std::string get_producer_name() const;
    void set_producer_name(const std::string& producer_name);

    std::string get_producer_version() const;
    void set_producer_version(const std::string& producer_version);

    std::string get_domain() const;
    void set_domain(const std::string& domain);

    int64_t get_model_version() const;
    void set_model_version(int64_t model_version);

    std::string get_doc_string() const;
    void set_doc_string(const std::string& doc_string);

    const std::unordered_map<std::string, std::string>& get_metadata() const;
    void set_metadata(const std::unordered_map<std::string, std::string>& meta_map);

    const std::unordered_map<std::string, int64_t>& get_domain_version() const;
    void set_domain_version(const std::unordered_map<std::string, int64_t>& dom_ver_map);

private:
    int64_t m_ir_version;
    std::string m_producer_name;
    std::string m_producer_version;
    std::string m_domain;
    int64_t m_model_version;
    std::string m_doc_string;

    // the meta data prop map
    std::unordered_map<std::string, std::string> m_metadata_map;

    // the opset map
    std::unordered_map<std::string, int64_t> m_domain_version;

    // the graph
    std::unique_ptr<Graph> m_graph;
};

}    // namespace ir
}    // namespace simple_ai

#endif