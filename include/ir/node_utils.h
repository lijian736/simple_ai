#ifndef _H_SIMPLE_AI_IR_NODE_UTILS_H_
#define _H_SIMPLE_AI_IR_NODE_UTILS_H_

#include <string>
#include <unordered_map>

#include "common/common.h"
#include "node_attribute.h"

namespace simple_ai {
namespace ir {
namespace utils {

/**
 * @brief Get a single attribute
 *
 * @tparam T
 * @param name the attribute name
 * @param value output parameter. the returned attribute value
 * @param attributes the attributes
 * @return Status if the attribute does NOT exist in the `attributes` or the attribute data type mismatch, return fail.
 */
template <typename T>
Status get_attr(const std::string& name, T* value,
                const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes);

template <>
inline Status get_attr<float>(const std::string& name, float* value,
                       const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes) {
    auto iter = attributes.find(name);
    if (iter == attributes.end() || iter->second->type() != NodeAttributeType::FLOAT) {
        return Status(StatusCode::FAIL);
    }

    *value = iter->second->get_float();
    return Status::ok();
}

template <>
inline Status get_attr<int64_t>(const std::string& name, int64_t* value,
                         const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes) {
    auto iter = attributes.find(name);
    if (iter == attributes.end() || iter->second->type() != NodeAttributeType::INT64) {
        return Status(StatusCode::FAIL);
    }

    *value = iter->second->get_int64();
    return Status::ok();
}

template <>
inline Status get_attr<std::string>(const std::string& name, std::string* value,
                             const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes) {
    auto iter = attributes.find(name);
    if (iter == attributes.end() || iter->second->type() != NodeAttributeType::STRING) {
        return Status(StatusCode::FAIL);
    }

    *value = iter->second->get_string();
    return Status::ok();
}

/**
 * @brief Get the attrs vector
 *
 * @tparam T
 * @param name the attribute name
 * @param values output parameter. the returned attributes
 * @param attributes the attributes
 * @return Status
 */
template <typename T>
Status get_attrs(const std::string& name, std::vector<T>& values,
                 const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes);

template <>
inline Status get_attrs<float>(const std::string& name, std::vector<float>& values,
                        const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes) {
    auto iter = attributes.find(name);
    if (iter == attributes.end() || iter->second->type() != NodeAttributeType::FLOAT_ARRAY) {
        return Status(StatusCode::FAIL);
    }

    values = iter->second->get_floats();
    return Status::ok();
}

template <>
inline Status get_attrs<int64_t>(const std::string& name, std::vector<int64_t>& values,
                          const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes) {
    auto iter = attributes.find(name);
    if (iter == attributes.end() || iter->second->type() != NodeAttributeType::INT64_ARRAY) {
        return Status(StatusCode::FAIL);
    }

    values = iter->second->get_int64s();
    return Status::ok();
}

template <>
inline Status get_attrs<std::string>(const std::string& name, std::vector<std::string>& values,
                              const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes) {
    auto iter = attributes.find(name);
    if (iter == attributes.end() || iter->second->type() != NodeAttributeType::STRING_ARRAY) {
        return Status(StatusCode::FAIL);
    }

    values = iter->second->get_strings();
    return Status::ok();
}

/**
 * @brief Get the attr or default value
 *
 * @tparam T
 * @param name the attribute name
 * @param default_value the default attribute value
 * @param attributes the attributes
 * @return T if the attribute does NOT exist in the `attributes` or the attribute data type mismatch, return the default
 * value. otherwise return the attribute value
 */
template <typename T>
inline T get_attr_or_default(const std::string& name, const T& default_value,
                      const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes) {
    T tmp;
    return get_attr<T>(name, &tmp, attributes).is_ok() ? tmp : default_value;
}

/**
 * @brief Get the attr or default value
 *
 * @tparam T
 * @param name the attribute name
 * @param value output parameter. the attribute value
 * @param default_value the default attribute value
 * @param attributes the attributes
 */
template <typename T>
inline void get_attr_or_default(const std::string& name, T* value, const T& default_value,
                         const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes) {
    if (!get_attr<T>(name, value, attributes).is_ok()) {
        *value = default_value;
    }
}

template <typename T>
inline std::vector<T> get_attrs_or_default(const std::string& name, const std::vector<T>& default_value,
                                    const std::unordered_map<std::string, std::unique_ptr<NodeAttribute>>& attributes) {
    std::vector<T> tmp;
    return get_attrs<T>(name, tmp, attributes).is_ok() ? tmp : default_value;
}

}    // namespace utils
}    // namespace ir
}    // namespace simple_ai

#endif