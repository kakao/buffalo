#pragma once
#include <fstream>
#include <utility>

#include "json11.hpp"

struct cublasContext;

namespace cuda_als{

using namespace json11;

class CuALS{
public:
    CuALS();
    ~CuALS();

    bool init(std::string opt_path); 
    bool parse_option(std::string opt_path, Json& j);

    void initialize_model(
            float* P, int P_rows,
            float* Q, int Q_rows);
    void precompute(int axis);
    void synchronize(int axis, bool device_to_host);
    int get_vdim();
    std::pair<double, double> partial_update(int start_x, 
            int next_x,
            int* indptr,
            int* keys,
            float* vals,
            int axis);
public:
    Json opt_;
    float *hostP_, *hostQ_, *devP_, *devQ_, *devFF_;
    cublasContext* blas_handle_;
    int dim_, vdim_, num_cg_max_iters_, P_rows_, Q_rows_;
    float alpha_, reg_u_, reg_i_, cg_tolerance_, eps_;
    bool compute_loss_, adaptive_reg_;
};

} // namespace cuda_als
