#pragma once

struct cublasContext;

namespace cuda_als{

class CuALS{
public:
    CuALS();
    ~CuALS();
    
    void set_options(bool compute_loss, int dim, int num_cg_max_iters, 
            float alpha, float reg_u, 
            float reg_i, float cg_tolerance, float eps);
    void initialize_model(
            float* P, int P_rows,
            float* Q, int Q_rows);
    void precompute(int axis);
    void synchronize(int axis);
    int get_vdim();
    float partial_update(int start_x, 
            int next_x,
            int* indptr,
            int* keys,
            float* vals,
            int axis);
public:
    float *hostP_, *hostQ_, *devP_, *devQ_, *devFF_;
    cublasContext* blas_handle_;
    int dim_, vdim_, num_cg_max_iters_, P_rows_, Q_rows_;
    float alpha_, reg_u_, reg_i_, cg_tolerance_, eps_;
    bool compute_loss_;
};

} // namespace cuda_als
