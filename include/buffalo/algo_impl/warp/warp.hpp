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

namespace warp {
class CWARP : public SGDAlgorithm {
public:
    CWARP();
    ~CWARP();

    void release();
    bool init(string opt_path);
    bool parse_option(string opt_path);

    void initialize_model(
            float* P, int32_t P_rows,
            float* Q, int32_t Q_rows,
            float* Qb,
            int64_t num_total_samples);

    void add_jobs(
            int start_x,
            int next_x,
            int64_t* indptr,
            int32_t* positives);
    void set_cumulative_table(int64_t* cum_table, int size);

    void worker(int worker_id);
    void launch_workers();
    void wait_until_done();
    double join();
    void update_parameters();

    double compute_loss(
            int32_t num_loss_samples,
            int32_t* users,
            int32_t* positives,
            int32_t* negatives);

    double distance(size_t p, size_t q);

private:
    int max_trial_;
    int64_t* cum_table_;
    int cum_table_size_;
    vector<float> warp_loss_table;

};

}
