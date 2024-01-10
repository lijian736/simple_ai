aux_source_directory(. SRC_LIST)

#add include folder
include_directories("${CMAKE_SOURCE_DIR}/include")

add_library(common SHARED ${SRC_LIST})