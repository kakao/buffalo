#pragma once
#include <string>
#include <cstdio>
#include <fstream>
#include <streambuf>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#include "buffalo/algo.hpp"
using namespace std;
using namespace Eigen;


namespace cfr {


class CCFR : public Algorithm {
public:
    CCFR();
    ~CCFR();

    bool init(string opt_path);
    bool parse_option(string opt_path);
    void set_embedding(float* data, int size, string obj_type);
    void precompute(string obj_type);
    double partial_update_user(int start_x, int next_x,
                               int64_t* indptrs, int32_t* keys, float* vals);
    double partial_update_item(int start_x, int next_x,
                               int64_t* indptrs_u, int32_t* keys_u, float* vals_u,
                               int64_t* indptrs_c, int32_t* keys_c, float* vals_c);
    double partial_update_context(int start_x, int next_x,
                                  int64_t* indptrs, int32_t* keys, float* vals);

private:
    Json opt_;
    int dim_, num_threads_;
    float alpha_, l_, reg_u_, reg_i_, reg_c_;
    bool compute_loss_;
    Map<MatrixType> U_, I_, C_;
    MatrixType FF_;
    Map<VectorType> Ib_, Cb_;
};

}
