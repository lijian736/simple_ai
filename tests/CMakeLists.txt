enable_testing()

include(GoogleTest)

include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/src/onnx_proto)

function(SIMPLE_AI_TESTS name file)
    add_executable(${name} ${CMAKE_CURRENT_SOURCE_DIR}/${file})
    foreach(arg IN LISTS ARGN)
        target_link_libraries(${name} PRIVATE ${arg})
    endforeach()
    
    target_link_libraries(${name} PRIVATE gtest gtest_main)
    gtest_discover_tests(${name})
endfunction()

SIMPLE_AI_TESTS(test_common  "common/test_common.cpp" "common")
SIMPLE_AI_TESTS(test_utils   "utils/test_utils.cpp"   "utils")
SIMPLE_AI_TESTS(test_logger  "utils/test_logger.cpp"  "common" "utils")
SIMPLE_AI_TESTS(test_ir      "ir/test_ir.cpp"         "common" "utils" "ir" "io")
