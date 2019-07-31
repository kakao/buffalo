#pragma once
#include <vector>
#include <string>
#include <random>
#include <unordered_set>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#include "buffalo/algo.hpp"
using namespace std;
using namespace Eigen;


namespace bpr {


class CBPRMF : public Algorithm {
public:
    CBPRMF();
    ~CBPRMF();

    void release();
    bool init(string opt_path);
    bool parse_option(string out_path);

    void set_factors(
            Map<MatrixXf>& P,
            Map<MatrixXf>& Q,
            Map<MatrixXf>& Qb);

    void set_cumulative_table(int64_t* cum_table, int size);

    void initialize_adam_optimizer();

    void initialize_sgd_optimizer();

    double partial_update(int start_x,
                          int next_x,
                          int64_t* indptr,
                          Map<VectorXi>& positives);

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

    double update_parameters();

    double distance(size_t p, size_t q);


private:
    Json opt_;
    float *P_data_, *Q_data_, *Qb_data_;
    int P_rows_, P_cols_, Q_rows_, Q_cols_;
    int iters_;
    int64_t total_samples_;
    double lr_;
    string optimizer_;

    mt19937 rng_;

    int64_t* cum_table_;
    int cum_table_size_;
    vector<int> P_samples_per_coordinates_;
    vector<int> Q_samples_per_coordinates_;
    FactorType gradP_, gradQ_, gradQb_;
    FactorType momentumP_, momentumQ_, momentumQb_;
    FactorType velocityP_, velocityQ_, velocityQb_;
};

}
