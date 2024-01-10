# CMake minimal version
cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# Project name
project(simple_ai C CXX)

# C++ 17
set(CMAKE_CXX_STANDARD 17)
# C++ 17 required
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Generate compile_commands.json
if(NOT CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# Fetch google test
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
FetchContent_MakeAvailable(googletest)

# Add source folder
add_subdirectory(src/common)
add_subdirectory(src/utils)
add_subdirectory(src/framework)
add_subdirectory(src/onnx_proto)
add_subdirectory(src/ir)
add_subdirectory(src/io)
# add_subdirectory(src/backend)

# Add unit test folder
add_subdirectory(tests)