#include "buffalo/misc/log.hpp"

int BuffaloLogger::global_logging_level_ = 2;

BuffaloLogger::BuffaloLogger() 
{
    spdlog::set_pattern("[%^%-8l%$] %Y-%m-%d %H:%M:%S %v");
    logger_ = spdlog::default_logger();
}

std::shared_ptr<spdlog::logger>& BuffaloLogger::get_logger() {
    return logger_;
}

void BuffaloLogger::set_log_level(int level) {
    global_logging_level_ = level;
    switch(level) {
        case 0: spdlog::set_level(spdlog::level::off); break;
        case 1: spdlog::set_level(spdlog::level::warn); break;
        case 2: spdlog::set_level(spdlog::level::info); break;
        case 3: spdlog::set_level(spdlog::level::debug); break;
        default: spdlog::set_level(spdlog::level::trace); break;
    }
}

int BuffaloLogger::get_log_level() {
    return global_logging_level_;
}
