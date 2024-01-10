#include <gtest/gtest.h>

#include <string>

#include "utils/utils.h"

TEST(UtilsTest, String) {
    using namespace simple_ai;
    using namespace simple_ai::utils;

    std::string str1 = "  abc";
    trim_start(str1);
    EXPECT_EQ(str1, "abc");

    std::string str2 = "abc   ";
    trim_end(str2);
    EXPECT_EQ(str2, "abc");

    std::string str3 = "  abc   ";
    trim(str3);
    EXPECT_EQ(str3, "abc");

    std::string str4 = "abc test";
    EXPECT_TRUE(ends_with(str4, "test"));

    std::string str5 = "abc test.";
    EXPECT_FALSE(ends_with(str5, "test"));
}

TEST(UtilsTest, Directory) {
    using namespace simple_ai;
    using namespace simple_ai::utils;

    auto exist = file_exist("./test_utils");
    EXPECT_TRUE(exist);

    exist = file_exist("./logabc");
    EXPECT_FALSE(exist);

    create_directory_recursively("./abc/def");
    exist = file_exist("./abc");
    EXPECT_TRUE(exist);

    exist = file_exist("./abc/def");
    EXPECT_TRUE(exist);

    exist = file_exist("./abc/def/cat.txt");
    EXPECT_FALSE(exist);
}