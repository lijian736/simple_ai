find_package(Protobuf 3 REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

set(SRC_LIST ./onnx.proto3.pb.cc)

add_library(onnx_proto SHARED ${SRC_LIST})
target_link_libraries(onnx_proto ${Protobuf_LIBRARIES})