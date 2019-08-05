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
            Map<MatrixXf>& P,
            Map<MatrixXf>& Q,
            Map<MatrixXf>& Qb,
            int64_t num_total_samples);

    void set_cumulative_table(int64_t* cum_table, int size);

    void initialize_adam_optimizer();

    void initialize_sgd_optimizer();

    void add_jobs(
            int start_x,
            int next_x,
            int64_t* indptr,
            Map<VectorXi>& positives);

    void worker(int worker_id);

    int get_negative_sample(unordered_set<int>& seen);

    double compute_loss(
            Map<VectorXi>& users,
            Map<VectorXi>& positives,
            Map<VectorXi>& negatives);

    void update_adam(
            FactorType& grad,
            FactorType& momentum,
            FactorType& velocity,
            int i,
            double beta1,
            double beta2);

    void progress_manager();

    double join();

    void update_parameters();

    double distance(size_t p, size_t q);


private:
    Json opt_;
    float *P_data_, *Q_data_, *Qb_data_;
    int P_rows_, P_cols_, Q_rows_, Q_cols_;
    int iters_;
    double lr_;
    string optimizer_;
    double total_processed_;

    mt19937 rng_;

    int64_t* cum_table_;
    int cum_table_size_;
    vector<int> P_samples_per_coordinates_;
    vector<int> Q_samples_per_coordinates_;
    FactorType gradP_, gradQ_, gradQb_;
    FactorType momentumP_, momentumQ_, momentumQb_;
    FactorType velocityP_, velocityQ_, velocityQb_;

    float exp_table_[EXP_TABLE_SIZE];

    vector<thread> workers_;
    thread* progress_manager_;
    Queue<job_t> job_queue_;
    Queue<progress_t> progress_queue_;
};

}
