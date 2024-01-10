#ifndef _H_SIMPLE_AI_IR_NODE_ATTRIBUTE_H_
#define _H_SIMPLE_AI_IR_NODE_ATTRIBUTE_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "common/common.h"
#include "framework/common_defines.h"
#include "tensor.h"

using namespace simple_ai::framework;

namespace simple_ai {
namespace ir {

/**
 * @brief node attribute data type
 *
 */
enum class NodeAttributeType {
    INT64,     // int64_t
    FLOAT,     // float
    STRING,    // string
    TENSOR,    // tensor

    INT64_ARRAY,     // int64_t array
    FLOAT_ARRAY,     // float array
    STRING_ARRAY,    // string array
    TENSOR_ARRAY,    // tensor array

    INVALID
};

/**
 * @brief Node attributes
 */
class NodeAttribute {
public:
    NodeAttribute(const std::string& name, const NodeAttributeType& type) : m_name(name), m_type(type) {}

    /**
     * @brief Get the node attribute data type
     *
     * @return NodeAttributeType
     */
    NodeAttributeType type() const { return m_type; }

    void set_float(float f) { m_float = f; }
    float get_float() const { return m_float; }

    void set_int64(int64_t i) { m_int64 = i; }
    int64_t get_int64() const { return m_int64; }

    void set_string(const std::string& str) { m_string = str; }
    const std::string& get_string() const { return m_string; }

    void set_tensor(std::unique_ptr<Tensor>&& tensor) { m_tensor = std::move(tensor); }
    Tensor* get_tensor() const { return m_tensor.get(); }

    void add_float(float f) { m_floats.emplace_back(f); }
    const std::vector<float>& get_floats() const { return m_floats; }

    void add_int64(int64_t i) { m_int64s.emplace_back(i); }
    const std::vector<int64_t>& get_int64s() const { return m_int64s; }

    void add_string(const std::string& str) { m_strings.emplace_back(str); }
    const std::vector<std::string>& get_strings() const { return m_strings; }

    void add_tensor(std::unique_ptr<Tensor>&& tensor) { m_tensors.emplace_back(std::move(tensor)); }
    const std::vector<std::unique_ptr<Tensor>>& get_tensors() const { return m_tensors; }

private:
    // name of the attribute
    std::string m_name;

    // attribyte type
    NodeAttributeType m_type;

    // data section
    float m_float;
    int64_t m_int64;
    std::string m_string;
    std::unique_ptr<Tensor> m_tensor;

    std::vector<float> m_floats;
    std::vector<int64_t> m_int64s;
    std::vector<std::string> m_strings;
    std::vector<std::unique_ptr<Tensor>> m_tensors;
    // end data section
};

}    // namespace ir
}    // namespace simple_ai

#endif