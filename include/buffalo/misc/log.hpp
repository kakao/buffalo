#pragma once

#define SPDLOG_EOL ""
#define SPDLOG_TRACE_ON
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define INFO(x, ...) logger_->info("[{}:{}] " x "\n", __FILENAME__, __LINE__, __VA_ARGS__);
#define DEBUG(x, ...) logger_->debug("[{}:{}] " x "\n", __FILENAME__, __LINE__, __VA_ARGS__);
#define WARN(x, ...) logger_->warn("[{}:{}] " x "\n", __FILENAME__, __LINE__, __VA_ARGS__);
#define CRITICAL(x, ...) logger_->critical("[{}:{}] " x "\n", __FILENAME__, __LINE__, __VA_ARGS__);

#define INFO0(x) logger_->info("[{}:{}] " x "\n", __FILENAME__, __LINE__);
#define DEBUG0(x) logger_->debug("[{}:{}] " x "\n", __FILENAME__, __LINE__);
#define WARN0(x) logger_->warn("[{}:{}] " x "\n", __FILENAME__, __LINE__);
#define CRITICAL0(x) logger_->critical("[{}:{}] " x "\n", __FILENAME__, __LINE__);


class BuffaloLogger
{
public:
    BuffaloLogger() {
        spdlog::set_pattern("[%^%-8l%$] [%Y-%m-%d %H:%M:%S] %v");
        logger_ = spdlog::default_logger();
        lvl_ = 1;
    }

    std::shared_ptr<spdlog::logger>& get_logger() {
        return logger_;
    }

    void set_log_level(int level) {
        lvl_ = level;
        switch(level) {
            case 0: spdlog::set_level(spdlog::level::off); break;
            case 1: spdlog::set_level(spdlog::level::info); break;
            case 2: spdlog::set_level(spdlog::level::debug); break;
            default: spdlog::set_level(spdlog::level::trace); break;
        }
    }

    int get_log_level() {
        return lvl_;
    }

private:
    int lvl_;
    std::shared_ptr<spdlog::logger> logger_;
};
