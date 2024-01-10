#include <gtest/gtest.h>

#include <string>
#include <thread>

#include "utils/logger.h"
#include "utils/utils.h"

using namespace simple_ai;
using namespace simple_ai::utils;

TEST(UtilsTest, LoggerTest) {
    using namespace simple_ai;
    using namespace simple_ai::utils;

    set_logger_level(LOG_LEVEL_VERBOSE);
    auto ret = init_logger("simple_ai", "./log");
    EXPECT_TRUE(ret);

    LOG_INFO("Program begins.......");

    for (int i = 0; i < 10000; i++) {
        LOG_INFO("%d", i);
    }

    LOG_INFO("Program ends.......");

    std::this_thread::sleep_for(std::chrono::seconds(1));
    release_logger();
}