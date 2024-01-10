#include <gtest/gtest.h>

#include <string>

#include "common/common.h"

TEST(CommonTest, StatusCode) {
    using namespace simple_ai;
    using namespace simple_ai::common;

    Status status1;
    EXPECT_TRUE(status1.is_ok());
    EXPECT_TRUE(Status::ok().is_ok());

    Status status2(StatusCode::FILE_NOT_FOUND, "file removed");

    std::string error_info("no model");
    Status status3(StatusCode::INVALID_MODEL, error_info);

    Status status4(StatusCode::FAIL);

    EXPECT_FALSE(status2 == status3);

    Status status5(status3);
    EXPECT_TRUE(status5 == status3);
    EXPECT_TRUE(status5 == status5);

    EXPECT_TRUE(status4.code() == StatusCode::FAIL);
    EXPECT_TRUE(status3.message() == error_info);

    EXPECT_TRUE(status3.to_string() == "INVALID_MODEL:no model");

    Status status6;
    status6 = status3;
    EXPECT_TRUE(status6 == status3);
    
    std::cout << status3 << std::endl;
}