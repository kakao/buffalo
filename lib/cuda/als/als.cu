#include <unistd.h>

#include <cuda_runtime.h>
#include "cublas_v2.h"

#include "buffalo/cuda/utils.cuh"
#include "buffalo/cuda/als/als.hpp"


namespace cuda_als{

using std::invalid_argument;
using namespace cuda_buffalo;

__global__ void least_squares_cg_kernel(const int dim, const int vdim, 
        const int rows, const int op_rows, 
        float* P, const float* Q, const float* FF, float* losses,
        const int start_x, const int next_x,
        const int* indptr, const int* keys, const float* vals, 
        const float alpha, const float reg, const float cg_tolerance,
        const int num_cg_max_iters, const bool compute_loss,
        const float eps){
    extern __shared__ float shared_memory[];
    float* Ap = &shared_memory[0];
    float* r = &shared_memory[vdim];
    float* p = &shared_memory[2*vdim];

    for (int row=blockIdx.x; row<next_x-start_x; row+=gridDim.x){
        float* _P = &P[(row+start_x)*vdim];

        if (indptr[row] == indptr[row + 1]) {
            _P[threadIdx.x] = 0;
            continue;
        }

        float tmp = -_P[threadIdx.x] * reg;
        // not necessary to compute vdim times
        for (int d=0; d<dim; ++d)
            tmp -= _P[d] * FF[d * vdim + threadIdx.x];
        
        for (int idx=indptr[row]; idx<indptr[row+1]; ++idx){
            const float* _Q = &Q[keys[idx] * vdim];
            const float v = vals[idx];
            float _dot = dot(_P, _Q);
            if (compute_loss and threadIdx.x == 0){
                float err = v - _dot;
                losses[blockIdx.x] += err * err;
            }
            tmp += (1 + alpha * v * (1 - _dot)) * _Q[threadIdx.x];
        }
        p[threadIdx.x] = r[threadIdx.x] = tmp;

        float rsold = dot(r, r);
        // early stopping
        if (rsold < cg_tolerance)
            continue;

        // iterate cg
        for (int it=0; it<num_cg_max_iters; ++it){
            Ap[threadIdx.x] = reg * p[threadIdx.x];
            for (int d=0; d<dim; ++d){
                Ap[threadIdx.x] += p[d] * FF[d * vdim + threadIdx.x];
            }
            for (int idx=indptr[row]; idx<indptr[row+1]; ++idx){
                const float* _Q = &Q[keys[idx] * vdim];
                const float v = vals[idx];
                float _dot = dot(p, _Q);
                Ap[threadIdx.x] += v * alpha * _dot * _Q[threadIdx.x];
            }
            float alpha = rsold / (dot(p, Ap) + eps);
            _P[threadIdx.x] += alpha * p[threadIdx.x];
            r[threadIdx.x] -= alpha * Ap[threadIdx.x];
            float rsnew = dot(r, r);
            if (rsnew < cg_tolerance) break;
            p[threadIdx.x] = r[threadIdx.x] + (rsnew / (rsold + eps)) * p[threadIdx.x];
            rsold = rsnew;
            __syncthreads();
        }
        
        if (isnan(rsold)){
            if (threadIdx.x == 0)
                printf("Warning NaN detected in row %d of %d\n", row, rows);
            _P[threadIdx.x] = 0.0;
        }
    }
}

CuALS::CuALS(){}

CuALS::~CuALS(){
    // destructor
    CHECK_CUDA(cudaFree(devP_));
    CHECK_CUDA(cudaFree(devQ_));
    CHECK_CUDA(cudaFree(devFF_));
    devP_ = nullptr, devQ_ = nullptr, devFF_ = nullptr;
    hostP_ = nullptr, hostQ_ = nullptr;
    CHECK_CUBLAS(cublasDestroy(blas_handle_));
}

void CuALS::set_options(bool compute_loss, int dim, int num_cg_max_iters, 
        float alpha, float reg_u, float reg_i, float cg_tolerance, float eps){
    // set options
    compute_loss_ = compute_loss;
    dim_ = dim, num_cg_max_iters_ = num_cg_max_iters;
    alpha_ = alpha, reg_u_  = reg_u, reg_i_ = reg_i;
    cg_tolerance_ = cg_tolerance; eps_ = eps;
    
    // virtual dimension
    vdim_ = (dim_ / WARP_SIZE) * WARP_SIZE;
    if (dim_ % WARP_SIZE > 0) vdim_ += WARP_SIZE;
    CHECK_CUDA(cudaMalloc(&devFF_, sizeof(float)*vdim_*vdim_));
    CHECK_CUBLAS(cublasCreate(&blas_handle_));
}

bool CuALS::parse_option(std::string opt_path, Json& j){
    std::ifstream in(opt_path.c_str());
    if (not in.is_open()) {
        INFO("File not exists: {}", opt_path);
        return false;
    }

    std::string str((std::istreambuf_iterator<char>(in)),
               std::istreambuf_iterator<char>());
    std::string err_cmt;
    auto _j = Json::parse(str, err_cmt);
    if (not err_cmt.empty()) {
        INFO("Failed to parse: {}", err_cmt);
        return false;
    }
    j = _j;
    return true;
}

bool CuALS::init(std::string opt_path){
    // parse options
    bool ok = parse_option(opt_path, opt_);
    if (ok){
        // set options
        compute_loss_ = opt_["compute_loss"].bool_value();
        
        dim_ = opt_["d"].int_value();
        num_cg_max_iters_ = opt_["num_cg_max_iters"].int_value();
         
        alpha_ = opt_["alpha"].number_value();
        reg_u_ = opt_["reg_u"].number_value();
        reg_i_ = opt_["reg_i"].number_value();
        cg_tolerance_ = opt_["cg_tolerance"].number_value();
        eps_ = opt_["eps"].number_value();
        
        // virtual dimension
        vdim_ = (dim_ / WARP_SIZE) * WARP_SIZE;
        if (dim_ % WARP_SIZE > 0) vdim_ += WARP_SIZE;
        CHECK_CUDA(cudaMalloc(&devFF_, sizeof(float)*vdim_*vdim_));
        CHECK_CUBLAS(cublasCreate(&blas_handle_));
    }
    return ok;
}

void CuALS::initialize_model(
        float* P, int P_rows,
        float* Q, int Q_rows){
    // initialize parameters and send to gpu memory
    hostP_ = P;
    hostQ_ = Q;
    P_rows_ = P_rows;
    Q_rows_ = Q_rows;
    CHECK_CUDA(cudaMalloc(&devP_, sizeof(float)*P_rows_*vdim_));
    CHECK_CUDA(cudaMemcpy(devP_, hostP_, sizeof(float)*P_rows_*vdim_, 
               cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&devQ_, sizeof(float)*Q_rows_*vdim_));
    CHECK_CUDA(cudaMemcpy(devQ_, hostQ_, sizeof(float)*Q_rows_*vdim_, 
               cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaDeviceSynchronize());
}

void CuALS::precompute(int axis){
    // precompute FF using cublas
    int op_rows = axis == 0? Q_rows_: P_rows_;
    float* opF = axis == 0? devQ_: devP_;
    float alpha = 1.0, beta = 0.0;
    CHECK_CUBLAS(cublasSgemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                 vdim_, vdim_, op_rows, &alpha, 
                 opF, vdim_, opF, vdim_, &beta, devFF_, vdim_));
    CHECK_CUDA(cudaDeviceSynchronize());
}

void CuALS::synchronize(int axis, bool device_to_host){
    // synchronize parameters between cpu memory and gpu memory
    float* devF = axis == 0? devP_: devQ_;
    float* hostF = axis == 0? hostP_: hostQ_;
    int rows = axis == 0? P_rows_: Q_rows_;
    if (device_to_host){
        CHECK_CUDA(cudaMemcpy(hostF, devF, sizeof(float)*rows*vdim_, 
                   cudaMemcpyDeviceToHost));
    } else{
        CHECK_CUDA(cudaMemcpy(devF, hostF, sizeof(float)*rows*vdim_, 
                   cudaMemcpyHostToDevice));
    }
    CHECK_CUDA(cudaDeviceSynchronize());
}

int CuALS::get_vdim(){
    return vdim_;
}

float CuALS::partial_update(int start_x, 
        int next_x,
        int* indptr,
        int* keys,
        float* vals,
        int axis){
    int devId;
    CHECK_CUDA(cudaGetDevice(&devId));
    int mp_cnt;
    CHECK_CUDA(cudaDeviceGetAttribute(&mp_cnt, cudaDevAttrMultiProcessorCount, devId));
    int block_cnt = 128 * mp_cnt;
    int thread_cnt = vdim_;
    size_t shared_memory_size = sizeof(float) * (3 * vdim_);
    int rows = axis == 0? P_rows_: Q_rows_;
    int op_rows = axis == 0? Q_rows_: P_rows_;
    float* P = axis == 0? devP_: devQ_;
    float* Q = axis == 0? devQ_: devP_;
    float reg = axis == 0? reg_u_: reg_i_;
    bool compute_loss = compute_loss_ and axis == 1;
    
    // copy data to gpu memory
    int sz1 = next_x - start_x;
    int sz2 = indptr[sz1];
    int *_indptr, *_keys;
    float* _vals;
    CHECK_CUDA(cudaMalloc(&_indptr, sizeof(int)*(sz1+1)));
    CHECK_CUDA(cudaMemcpy(_indptr, indptr, sizeof(int)*(sz1+1), 
                cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&_keys, sizeof(int)*sz2));
    CHECK_CUDA(cudaMemcpy(_keys, keys, sizeof(int)*sz2, 
                cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMalloc(&_vals, sizeof(float)*sz2));
    CHECK_CUDA(cudaMemcpy(_vals, vals, sizeof(float)*sz2, 
                cudaMemcpyHostToDevice));

    // allocate memory for measuring losses
    float* host_losses = (float*) malloc(sizeof(float)*block_cnt);
    for (size_t i=0; i<block_cnt; ++i)
        host_losses[i] = 0;
    float* device_losses;
    CHECK_CUDA(cudaMalloc(&device_losses, sizeof(float)*block_cnt));
    CHECK_CUDA(cudaMemcpy(device_losses, host_losses, sizeof(float)*block_cnt, 
               cudaMemcpyHostToDevice));
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // compute least square
    least_squares_cg_kernel<<<block_cnt, thread_cnt, shared_memory_size>>>(
            dim_, vdim_, rows, op_rows, P, Q, devFF_, device_losses, start_x, next_x,
            _indptr, _keys, _vals, alpha_, reg, 
            cg_tolerance_, num_cg_max_iters_, compute_loss, eps_);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // accumulate losses
    CHECK_CUDA(cudaMemcpy(host_losses, device_losses, sizeof(float)*block_cnt, 
               cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaDeviceSynchronize());
    float loss = 0;
    for (size_t i=0; i<block_cnt; ++i)
        loss += host_losses[i];

    // free memory
    free(host_losses);
    CHECK_CUDA(cudaFree(_indptr));
    CHECK_CUDA(cudaFree(_keys));
    CHECK_CUDA(cudaFree(_vals));
    CHECK_CUDA(cudaFree(device_losses));

    return loss;
}

} // namespace cuda_als
