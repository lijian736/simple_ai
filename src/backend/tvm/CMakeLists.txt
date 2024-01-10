aux_source_directory(. SRC_LIST)

#add include folder
include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/third_party/tvm/include")
include_directories("${CMAKE_SOURCE_DIR}/third_party/tvm/3rdparty/dlpack/include")

add_library(backend_tvm SHARED ${SRC_LIST})
target_link_libraries(backend_tvm PRIVATE common)