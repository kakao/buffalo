#pragma once
#include <fstream>
#include "json11.hpp"
#include "buffalo/misc/log.hpp"

struct cublasContext;

namespace cuda_als{

using namespace json11;

class CuALS{
public:
    CuALS();
    ~CuALS();

    void set_options(bool compute_loss, int dim, int num_cg_max_iters, 
            float alpha, float reg_u, 
            float reg_i, float cg_tolerance, float eps);
    bool init(std::string opt_path); 
    bool parse_option(std::string opt_path, Json& j);

    void initialize_model(
            float* P, int P_rows,
            float* Q, int Q_rows);
    void precompute(int axis);
    void synchronize(int axis, bool device_to_host);
    int get_vdim();
    float partial_update(int start_x, 
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
    bool compute_loss_;
    std::shared_ptr<spdlog::logger> logger_;
};

} // namespace cuda_als
