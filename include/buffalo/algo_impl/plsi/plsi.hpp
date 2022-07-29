#pragma once
#include <string>
#include <random>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#include "buffalo/algo.hpp"

using namespace std;
using namespace Eigen;


namespace plsi {


class CPLSI : public Algorithm {
public:
    CPLSI();
    ~CPLSI();

    bool init(string opt_path);
    bool parse_option(string opt_path);
    void swap();
    void reset();
    void release();
    void normalize(float alpha1, float alpha2);
    void initialize_model(float* P, int P_rows, float* Q, int Q_rows);
    float partial_update(int start_x, int next_x, int64_t* indptrs, int32_t* keys, float* vals);

private:
    Json opt_;
    Map<MatrixType> P_, Q_;
    MatrixType P, Q;
    int d_, num_workers_, seed_;
};

}
