#ifndef _H_SIMPLE_AI_UTILS_UTILS_H_
#define _H_SIMPLE_AI_UTILS_UTILS_H_

#include <string>

namespace simple_ai {
namespace utils {
/**
 * @brief trim the string start
 *
 * @param str input/output parameter
 */
void trim_start(std::string& str);

/**
 * @brief trim the string end
 *
 * @param str input/output parameter
 */
void trim_end(std::string& str);

/**
 * @brief trim the string
 *
 * @param str input/output parameter
 */
void trim(std::string& str);

/**
 * @brief string ends with the suffix
 *
 * @param str the string
 * @param suffix suffix
 * @return true
 * @return false
 */
bool ends_with(const std::string& str, const std::string& suffix);

/**
 * @brief check if the file path exists
 *
 * @param file_path the file path
 * @return true
 * @return false
 */
bool file_exist(const std::string& file_path);

/**
 * @brief Create a directory recursively
 *
 * @param folder_path the folder path
 * @return true
 * @return false
 */
bool create_directory_recursively(const std::string& folder_path);

}    // namespace utils
}    // namespace simple_ai

#endif