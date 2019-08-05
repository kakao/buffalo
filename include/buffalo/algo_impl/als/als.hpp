#pragma once
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#include "buffalo/algo.hpp"
using namespace std;
using namespace Eigen;


namespace als {


class CALS : public Algorithm {
public:
    CALS();
    ~CALS();

    void release();
    bool init(string opt_path);
    bool parse_option(string out_path);

    void initialize_model(
            Map<FactorTypeRowMajor>& P,
            Map<FactorTypeRowMajor>& Q);
    void precompute(int axis);
    double partial_update(int start_x,
                          int next_x,
                          int64_t* indptr,
                          Map<VectorXi>& keys,
                          Map<VectorXf>& vals,
                          int axis);

private:
    Json opt_;
    float* P_data_, * Q_data_;
    int P_rows_, P_cols_, Q_rows_, Q_cols_;
    FactorType FF_;
};

}
