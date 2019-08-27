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

namespace bpr {

struct job_t;
struct progress_t;

class CBPRMF : public Algorithm {
public:
    CBPRMF();
    ~CBPRMF();

    void release();
    bool init(string opt_path);
    bool parse_option(string out_path);

    void build_exp_table();

    void launch_workers();

    void initialize_model(
            float* P, int32_t P_rows,
            float* Q, int32_t Q_rows,
            float* Qb,
            int64_t num_total_samples);

    void set_cumulative_table(int64_t* cum_table, int size);

    void initialize_adam_optimizer();

    void initialize_sgd_optimizer();

    void add_jobs(
            int start_x,
            int next_x,
            int64_t* indptr,
            int32_t* positives);

    void worker(int worker_id);

    void wait_until_done();

    double compute_loss(
            int32_t num_loss_samples,
            int32_t* users,
            int32_t* positives,
            int32_t* negatives);

    void update_adam(
            FactorTypeRowMajor& grad,
            FactorTypeRowMajor& momentum,
            FactorTypeRowMajor& velocity,
            int i,
            double beta1,
            double beta2);

    void progress_manager();

    double join();

    void update_parameters();

    double distance(size_t p, size_t q);


private:
    Json opt_;
    Map<FactorTypeRowMajor> P_, Q_, Qb_;
    int iters_;
    double lr_;
    string optimizer_;
    double total_processed_;


    int64_t* cum_table_;
    int cum_table_size_;
    vector<int> P_samples_per_coordinates_;
    vector<int> Q_samples_per_coordinates_;
    FactorTypeRowMajor gradP_, gradQ_, gradQb_;
    FactorTypeRowMajor momentumP_, momentumQ_, momentumQb_;
    FactorTypeRowMajor velocityP_, velocityQ_, velocityQb_;

    float exp_table_[EXP_TABLE_SIZE];

    vector<thread> workers_;
    thread* progress_manager_;
    Queue<job_t> job_queue_;
    Queue<progress_t> progress_queue_;
};

}
