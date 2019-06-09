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

    bool init(string opt_path);
    bool parse_option(string out_path);

    void set_factors(Map<MatrixXf>& P, Map<MatrixXf>& Q);

private:
    Json opt_;
    float* P_data_, * Q_data_;
    int P_rows_, P_cols_, Q_rows_, Q_cols_;
};

}
