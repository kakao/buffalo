#pragma once

#define SPDLOG_EOL ""
#define SPDLOG_TRACE_ON
#include "spdlog/spdlog.h"
#include "spdlog/sinks/stdout_color_sinks.h"

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define INFO(x, ...) logger_->info("[{}:{}] " x "\n", __FILENAME__, __LINE__, __VA_ARGS__);
#define DEBUG(x, ...) logger_->debug("[{}:{}] " x "\n", __FILENAME__, __LINE__, __VA_ARGS__);
#define WARN(x, ...) logger_->warn("[{}:{}] " x "\n", __FILENAME__, __LINE__, __VA_ARGS__);
#define TRACE(x, ...) logger_->trace("[{}:{}] " x "\n", __FILENAME__, __LINE__, __VA_ARGS__);
#define CRITICAL(x, ...) logger_->critical("[{}:{}] " x "\n", __FILENAME__, __LINE__, __VA_ARGS__);

#define INFO0(x) logger_->info("[{}:{}] " x "\n", __FILENAME__, __LINE__);
#define DEBUG0(x) logger_->debug("[{}:{}] " x "\n", __FILENAME__, __LINE__);
#define WARN0(x) logger_->warn("[{}:{}] " x "\n", __FILENAME__, __LINE__);
#define TRACE0(x) logger_->trace("[{}:{}] " x "\n", __FILENAME__, __LINE__);
#define CRITICAL0(x) logger_->critical("[{}:{}] " x "\n", __FILENAME__, __LINE__);

class BuffaloLogger
{
public:
    BuffaloLogger();
    std::shared_ptr<spdlog::logger>& get_logger();
    void set_log_level(int level);
    int get_log_level();

private:
    static int global_logging_level_;
    std::shared_ptr<spdlog::logger> logger_;
};
