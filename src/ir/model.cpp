#include "ir/model.h"

using namespace simple_ai::common;

namespace simple_ai {
namespace ir {

void Model::set_graph(std::unique_ptr<Graph>&& graph) { m_graph = std::move(graph); }

Graph* Model::get_graph() const { return m_graph.get(); }

int64_t Model::get_ir_version() const { return m_ir_version; }

void Model::set_ir_version(int64_t ir_version) { m_ir_version = ir_version; }

std::string Model::get_producer_name() const { return m_producer_name; }

void Model::set_producer_name(const std::string& producer_name) { m_producer_name = producer_name; }

std::string Model::get_producer_version() const { return m_producer_version; }

void Model::set_producer_version(const std::string& producer_version) { m_producer_version = producer_version; }

std::string Model::get_domain() const { return m_domain; }

void Model::set_domain(const std::string& domain) { m_domain = domain; }

int64_t Model::get_model_version() const { return m_model_version; }

void Model::set_model_version(int64_t model_version) { m_model_version = model_version; }

std::string Model::get_doc_string() const { return m_doc_string; }

void Model::set_doc_string(const std::string& doc_string) { m_doc_string = doc_string; }

const std::unordered_map<std::string, std::string>& Model::get_metadata() const { return m_metadata_map; }

void Model::set_metadata(const std::unordered_map<std::string, std::string>& meta_map) { m_metadata_map = meta_map; }

const std::unordered_map<std::string, int64_t>& Model::get_domain_version() const { return m_domain_version; }

void Model::set_domain_version(const std::unordered_map<std::string, int64_t>& dom_ver_map) {
    m_domain_version = dom_ver_map;
}

}    // namespace ir
}    // namespace simple_ai