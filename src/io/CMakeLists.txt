find_package(Protobuf 3 REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

aux_source_directory(. SRC_LIST)

#add include folder
include_directories("${CMAKE_SOURCE_DIR}/include")
include_directories("${CMAKE_SOURCE_DIR}/src/onnx_proto")

add_library(io SHARED ${SRC_LIST})
target_link_libraries(io PRIVATE common ir framework onnx_proto ${Protobuf_LIBRARIES})