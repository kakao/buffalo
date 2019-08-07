#pragma once
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#include "buffalo/algo.hpp"
using namespace std;
using namespace Eigen;


typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixType;
typedef RowVectorXf VectorType;

namespace cfr {


class CCFR : public Algorithm {
public:
    CCFR(int dim, float alpha, float l);
    ~CCFR();

    void set_embedding(float* data, int size, string obj_type);
    void precompute(string obj_type);
    double partial_update_user(int start_x, int next_x,
                               int64_t* indptrs, int* keys, float* vals);
    double partial_update_item(int start_x, int next_x,
                               int64_t* indptrs_u, int* keys_u, float* vals_u,
                               int64_t* indptrs_c, int* keys_c, float* vals_c);
    double partial_update_context(int start_x, int next_x,
                                  int64_t* indptrs, int* keys, float* vals);

private:
    int dim_;
    float alpha_, l_;
    Map<MatrixType> U_, I_, C_;
    MatrixType U2_, I2_;
    Map<VectorType> Ib_, Cb_;
};

}
