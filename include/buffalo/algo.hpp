#pragma once
#include <omp.h>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/IterativeSolvers>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/concurrent_queue.hpp"

using namespace std;
using namespace json11;
using namespace Eigen;

typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixType;
typedef RowVectorXf VectorType;

typedef Matrix<float, Dynamic, Dynamic, ColMajor> FactorType;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> FactorTypeRowMajor;

struct job_t
{
    int size;
    double alpha;
    vector<vector<int>> samples;

    job_t() : size(0), alpha(0.0)
    {
    }

    void add(const vector<int>& sample)
    {
        samples.push_back(sample);
        size += (int)sample.size();
    }

    int total_samples()
    {
        return size;
    }
};


struct progress_t
{
    int num_sents;
    int num_processed_samples;
    int num_total_samples;
    double loss;
    progress_t(int a, int b, int c) :
        num_sents(a),
        num_processed_samples(b),
        num_total_samples(c) {
    }

    progress_t(int a, int b, int c, double l) :
        num_sents(a),
        num_processed_samples(b),
        num_total_samples(c),
        loss(l) {
    }
};



class Algorithm  // Algorithm? Logic? Learner?
{
public:
    Algorithm();

    virtual ~Algorithm() {}
    virtual bool init(string opt_path) = 0;
    virtual bool parse_option(string opt_path) = 0;

    bool parse_option(string opt_path, Json& j);
    void _leastsquare(Map<MatrixType>& X, int idx, MatrixType& A, VectorType& y);

public:
    char optimizer_code_ = 0;
    int num_cg_max_iters_ = 3;
    float cg_tolerance_ = 1e-10;
    float eps_ = 1e-10;
    std::shared_ptr<spdlog::logger> logger_;
};

class SGDAlgorithm : public Algorithm
{
public:
    SGDAlgorithm();
    ~SGDAlgorithm();

    bool init(string opt_path);

public:
    void release();
    void initialize_model(
        float* P, int32_t P_rows,
        float* Q, int32_t Q_rows,
        float* Qb,
        int64_t num_total_samples);

    void launch_workers();

    void initialize_adam_optimizer();
    void initialize_sgd_optimizer();
    void progress_manager();
    void add_jobs(
        int start_x,
        int next_x,
        int64_t* indptr,
        int32_t* positives);
    void update_parameters();
    void update_adam(
        FactorTypeRowMajor& grad,
        FactorTypeRowMajor& momentum,
        FactorTypeRowMajor& velocity, int i, double beta1, double beta2);
    void update_adagrad(FactorTypeRowMajor& grad,
        FactorTypeRowMajor& velocity, int i);
    void wait_until_done();
    double join();

public:
    virtual void worker(int worker_id) = 0;
    Json opt_;
    Map<FactorTypeRowMajor> P_, Q_, Qb_;

    int iters_;
    double lr_;
    string optimizer_;
    double total_processed_;

    vector<int> P_samples_per_coordinates_;
    vector<int> Q_samples_per_coordinates_;
    FactorTypeRowMajor gradP_, gradQ_, gradQb_;
    FactorTypeRowMajor momentumP_, momentumQ_, momentumQb_;
    FactorTypeRowMajor velocityP_, velocityQ_, velocityQb_;
    vector<thread> workers_;
    thread* progress_manager_;
    Queue<job_t> job_queue_;
    Queue<progress_t> progress_queue_;

};