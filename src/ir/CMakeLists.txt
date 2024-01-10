aux_source_directory(. SRC_LIST)
aux_source_directory(./node_shapes SHAPES_SRC_LIST)

#add include folder
include_directories("${CMAKE_SOURCE_DIR}/include")

add_library(ir SHARED ${SRC_LIST} ${SHAPES_SRC_LIST})
target_link_libraries(ir PRIVATE common)