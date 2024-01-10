aux_source_directory(. SRC_LIST)

#add include folder
include_directories("${CMAKE_SOURCE_DIR}/include")

add_library(backend SHARED ${SRC_LIST})
target_link_libraries(backend PRIVATE common)

add_subdirectory(tvm)