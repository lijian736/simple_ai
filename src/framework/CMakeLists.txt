aux_source_directory(. SRC_LIST)

#add include folder
include_directories("${CMAKE_SOURCE_DIR}/include")

add_library(framework SHARED ${SRC_LIST})
target_link_libraries(framework PRIVATE common utils)