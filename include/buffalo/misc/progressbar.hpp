#pragma once

#include <string>
#include <bits/stdc++.h>

#include "log.h"

using namespace std;
using namespace spdlog;

double time_diff(timespec beg, timespec end){
    return end.tv_sec - beg.tv_sec + (end.tv_nsec - beg.tv_nsec) / 1e+9;
}

class ProgressBar 
{
public:
    ProgressBar(size_t total) :
        ProgressBar(total, "")
    {
        level_ = spdlog::level::info;
    }

    ProgressBar(size_t total, string prefix) :
        total_(total),
        prefix_(prefix),
        bar_width_(30),
        current_(0)
    {
        level_ = spdlog::level::info;
        logger_ = BuffaloLogger().get_logger();
        clock_gettime(CLOCK_REALTIME, &start_t_);
    }

    void set_log_level(spdlog::level::level_enum lvl) {
        level_ = lvl;
    }

    void reset(size_t total, string prefix) {
        clock_gettime(CLOCK_REALTIME, &start_t_);
        total_ = total;
        prefix_ = prefix;
    }

    void update(int delta) {
        current_ += delta;
        sink();
    }

    void set(size_t step) {
        current_ = step;
        sink();
    }

    void end() {
        current_ = total_;
        sink(false);
    }

private:

    void sink(bool without_eol=true) {
        timespec now;
        clock_gettime(CLOCK_REALTIME, &now);
        double progress = (double)current_ / (double)total_;

        string msg;
        if (not prefix_.empty())
            msg += prefix_ + ": ";
        msg += "[";

        int pos = (int)(bar_width_ * progress);
        for(int i=0; i < bar_width_; ++i) {
            if (i <= pos) 
                msg += "=";
            else
                msg += " ";
        }
        msg += "]";

        double passed = time_diff(start_t_, now);
        double eta = (passed / progress) * (1.0 - progress);
        if ( without_eol ) {
            logger_->log(level_, "{} {:6.2f} secs remains\r", msg, eta);
        }
        else {
            logger_->log(level_, "{} {:6.2f} secs elapsed\n", msg, passed);
        }
    }

private:
    size_t total_;
    timespec start_t_;
    string prefix_;
    int bar_width_;

    size_t current_;
    spdlog::level::level_enum level_;
    std::shared_ptr<spdlog::logger> logger_;
};
