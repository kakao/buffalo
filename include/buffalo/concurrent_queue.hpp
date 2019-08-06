#pragma once

#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
using namespace std;

template <typename T>
class Queue
{
public:
    T pop() 
    {
        unique_lock<mutex> mlock(mutex_);
        while(queue_.empty())
        {
            cond_.wait(mlock);
        }
        queue_size_ -= 1;
        auto val = queue_.front();
        queue_.pop();
        cond2_.notify_one();
        return val;
    }

    int pop(T& item)
    {
        unique_lock<mutex> mlock(mutex_);
        while (queue_.empty())
        {
            cond_.wait(mlock);
        }
        queue_size_ -= 1;
        item = queue_.front();
        queue_.pop();
        cond2_.notify_one();
        return queue_size_;
    }

    int push(const T& item)
    {
        unique_lock<mutex> mlock(mutex_);
        while(max_size_ != -1 && max_size_ <= queue_size_)
        {
            cond2_.wait(mlock);
        }
        queue_.push(item);
        queue_size_ += 1;
        mlock.unlock();
        cond_.notify_one();
        return queue_size_;
    }

    void set_max_size(int max_size)
    {
        max_size_ = max_size;
    }

    int get_max_size()
    {
        return max_size_;
    }

    int get_size()
    {
        return queue_size_;
    }

    Queue() : queue_size_(0), max_size_(-1) {}
    Queue(const Queue&) = delete;
    Queue& operator=(const Queue&) = delete;

private:
    queue<T> queue_;
    mutex mutex_;
    condition_variable cond_, cond2_;
    int queue_size_;
    int max_size_;
};
