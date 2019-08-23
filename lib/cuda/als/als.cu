#include "buffalo/cuda/als/als.cuh"


namespace cuda_als{

using std::invalid_argument;

__device__ void least_squares_cg_kernel(const int dim, const int vdim, 
        const int rows, const int op_rows, 
        float* P, const float* Q, const float* FF, float* loss,
        const int* indptr, const int* keys, const float* vals, 
        const float alpha, const float reg, const float cg_tolerance,
        const int num_cg_max_iters, const bool compute_loss,
        const float eps){

    extern __shared__ float shared_memory[];
    float* Ap = &shared_memory[0];
    float* r = &shared_memory[vdim];
    float* p = &shared_memory[2*vdim];
    float* l = &shared_memory[3*vdim];
    for (int idx=threadIdx.x; idx<4*vdim; idx+=blockDim.x)
        Ap[idx] = 0;
    __syncthreads();

    for (int row=blockIdx.x; row<rows; idx+=gridDim.x){
        float* _P = &P[row*vdim];

        if (indptr[row] == indptr[row + 1]) {
            _P[threadIdx.x] = 0;
            continue;
        }

        float tmp = -_P[threadIdx.x] * reg;
        for (int d=0; d<dim; ++d)
            tmp -= _P[d] * FF[d * vdim + threadIdx.x];
        
        for (int idx=indptr[row]; idx<indptr[row+1]; ++idx){
            const float* _Q = &Q[keys[idx] * vdim];
            const float v = vals[idx];
            if (compute_loss){
                float _th_dot = _P[threadIdx.x] * _Q[threadIdx.x];
                l[threadIdx.x] += (v - _th_dot) * (v - _th_dot);
            }
            float _dot = dot(_P, _Q);
            tmp += (1 + alpha * v * (1 - _dot)) * _Q[threadIdx.x];
        }
        p[threadIdx.x] = r[threadIdx.x] = tmp;

        float rsold = dot(r, r);
        if (rsold < cg_tolerance) continue;


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
            __syncthreads();
        }

        if (isnan(rsold)){
            if (threadIdx.x == 0)
                printf("Warning NaN detected in row %d of %d\n", row, rows);
            _P[threadIdx.x] = 0.0;
        }
        if (compute_loss){
            _loss += dot(l, l);
            if (threadIdx.x == 0)
                atomicAdd(loss, _loss);
        }
    }
}

CuALS::CuALS(){}

CuALS::~CuALS(){
    CHECK_CUDA(cudaFree(devFF_));
    CHECK_CUDA(cudaFree(devP_));
    CHECK_CUDA(cudaFree(devQ_));
    devP_ = nullptr, devQ_ = nullptr, devFF_ = nullptr;
    hostP_ = nullptr, hostQ_ = nullptr;
}

void CuALS::set_options(bool compute_loss, int dim, int num_cg_max_iters, 
        float alpha, float reg_u, float reg_i, float cg_tolerance, float eps){
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

void initialize_model(
        float* P, int P_rows,
        float* Q, int Q_rows){
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
}

void precompute(int axis){
    int op_rows = axis == 0? Q_rows_: P_rows_;
    float* opF = axis == 0? devQ_: devP_;
    float alpha = 1.0, beta = 0.0;
    CHECK_CUBLAS(cublasSgemm(blas_handle_, CUBLAS_OP_N, CUBLAS_OP_T,
                vdim_, vdim_, op_rows, &alpha, 
                opF, vdim_, opF, vdim, &beta, devFF_, vdim_));
}

void synchronize(int axis){
    float* devF = axis == 0? devP_: devQ_;
    float* hostF = axis == 0? hostP_: hostQ_;
    int rows = axis == 0? P_rows_: Q_rows_;
    CHECK_CUDA(cudaMemcpy(hostF, devF, sizeof(float)*rows*vdim_, 
                cudaMemcpyDeviceToHost));
}

int get_vdim(){
    return vdim_;
}

float partial_update(int start_x, 
        int next_x,
        int* indptr,
        int* keys,
        float* vals,
        int axis);

    int devId;
    CHECK_CUDA(cudaGetDevice(&devId));
    int mp_cnt;
    CHECK_CUDA(cudaDeviceGetAttribute(&mp_cnt, cudaDevAttrMultiProcessorCount, devId));
    int block_cnt = 128 * mp_cnt;
    int thread_cnt = dim_;
    size_t shared_memory_size = sizeof(float) * (4 * factors);
    float loss = 0.0;
    int rows = axis == 0? P_rows_: Q_rows_;
    int op_rows = axis == 0? Q_rows_: P_rows_;
    float* P = axis == 0? devP_: devQ_;
    float* Q = axis == 0? devQ_: devP_;
    float reg = axis == 0? reg_u_: reg_i_;
    bool compute_loss = compute_loss_ and axis == 1;
    least_square_cg_kernel<<<block_cnt, thread_cnt, shared_memory_size>>>(
            dim_, vdim_, rows_, op_rows, P, Q, devFF_, &loss, indptr, keys, vals,
           alpha_, reg_, cg_tolerance_, num_cg_max_iters_, compute_loss, eps_);
    return loss;
}
