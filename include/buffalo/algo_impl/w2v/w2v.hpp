#pragma once
#include <vector>
#include <string>
#include <random>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#include "buffalo/algo.hpp"
#include "buffalo/concurrent_queue.hpp"

using namespace std;
using namespace Eigen;

static const int EXP_TABLE_SIZE = 1000;

namespace w2v {

struct job_t;
struct progress_t;


class CW2V : public Algorithm {
public:
    CW2V();
    ~CW2V();

    void release();
    bool init(string opt_path);
    bool parse_option(string out_path);

    void initialize_model(
            Map<FactorTypeRowMajor>& L0,
            int32_t* index,
            uint32_t* scale,
            int32_t* dist,
            int64_t total_word_count);

    void build_exp_table();

    void launch_workers();

    void add_jobs(
            int start_x,
            int next_x,
            int64_t* indptr,
            Map<VectorXi>& sequences);

    void worker(int worker_id);

    double update_parameter(Map<FactorTypeRowMajor>& L0,
                            Array<float, 1, Dynamic, RowMajor>& work,
                            double alpha,
                            int input_word_idx,
                            vector<int>& negatives,
                            bool comtpue_loss);

    void progress_manager();
    
    double join();


private:
    Json opt_;
    float *L0_data_;
    int L0_rows_, L0_cols_;
    uint32_t *scale_;
    int32_t *index_, *dist_;
    FactorTypeRowMajor L1_;
    double alpha_;
    double total_processed_;
    vector<double> processed_;
    int D_;

    float exp_table_[EXP_TABLE_SIZE];

    vector<thread> workers_;
    thread* progress_manager_;
    Queue<job_t> job_queue_;
    Queue<progress_t> progress_queue_;
};

}
