#ifndef _H_SIMPLE_AI_COMMON_HASH_COMBINE_H_
#define _H_SIMPLE_AI_COMMON_HASH_COMBINE_H_

#include <algorithm>

namespace simple_ai {
namespace common {

/**
 * @brief Combine hash value `seed` with hash value `value`, updating `seed` in place.
 *
 * @param value the hash value
 * @param seed the hash value, it will be updated when this function invoked.
 */
inline void hash_combine_with_hash_value(size_t value, size_t& seed) {
    seed ^= value + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

/**
 * @brief Combine hash value `seed` with the hash value `value`, updating `seed` in place.
 *
 * @tparam T the type
 * @tparam Hash the T hash value template
 * @param value the value
 * @param seed the seed which will be updated
 */
template <typename T, typename Hash = std::hash<T>>
inline void hash_combine(const T& value, size_t& seed) {
    hash_combine_with_hash_value(Hash{}(value), seed);
}

}    // namespace common
}    // namespace simple_ai

#endif