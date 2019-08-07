#include <string>
#include <numeric>
#include <fstream>
#include <iostream>
#include <streambuf>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/algo_impl/als/als.hpp"


namespace cfr {


CCFR::CCFR()
{}

CCFR::~CCFR()
{
    new (&U_) Map<MatrixType>(nullptr, 0, 0);
    new (&I_) Map<MatrixType>(nullptr, 0, 0);
    new (&C_) Map<MatrixType>(nullptr, 0, 0);
    new (&Ib_) Map<VectorType>(nullptr, 0);
    new (&Cb_) Map<VectorType>(nullptr, 0);
    FF_.resize(0, 0);
}

bool CCFR::init(string opt_path) {
    bool ok = Algorithm::parse_option(opt_path, opt_);
    if (ok) {
        num_threads_ = opt_["num_workers"].int_value();
        dim_ = opt_["d"].int_value();
        alpha_ = opt_["alpha"].float_value();
        l_ = opt_["l"].float_value();
        reg_u_ = opt_["reg_u"].float_value();
        reg_i_ = opt_["reg_i"].float_value();
        reg_c_ = opt_["reg_c"].float_value();
        FF_.resize(dim_, dim_);
    }
    return ok;
}

void CCFR::set_embedding(float* data, int size, string obj_type) {
    if (obj_type == "user")
        new (&U_) Map<MatrixType>(data, size, dim_);
    else if (obj_type == "item")
        new (&I_) Map<MatrixType>(data, size, dim_);
    else if (obj_type == "context")
        new (&C_) Map<MatrixType>(data, size, dim_);
    else if (obj_type == "item_bias")
        new (&Ib_) Map<VectorType>(data, size);
    else if (obj_type == "context_bias")
        new (&Ib_) Map<VectorType>(data, size);
    DEBUG("{} setted (size: {})", obj_type, size);
}


void CALS::precompute(string obj_type)
{
    if (obj_type == "user") FF_ = P_.transpose() * P_;
    else if (obj_type == "item") FF_ = Q_.transpose() * Q_;
}

void CCFR::partial_update_user(int start_x, int next_x,
        int64_t* indptr, int* keys, float* vals)
{
    if( (next_x - start_x) == 0) {
        WARN0("No data to process");
        return;
    }

    int end_loop = next_x - start_x;
    const int64_t shifted = indptr[0];
    #pragma omp parallel
    {
        int _thread = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 4)
        for (int i=0; i<end_loop; ++i)
        {
            const int x = start_x + i;
            // assume that shifted index is not so big
            const int beg = indptr[i] - shifted;
            const int end = indptr[i+1] - shifted;
            const int data_size = end - beg;
            if (data_size == 0) {
                TRACE("No data exists for {}", x);
                continue;
            }

            MatrixType A(dim_, dim_);
            MatrixType Fs(data_size, dim_);
            VectorType vs(data_size);
            VectorType y(dim_);
            
            A = FF_;
            for (int idx=0, it = beg; it < end; ++it, ++idx) {
                const int& c = keys[it];
                const float& v = vals[it];
                Fs.row(idx) = I_.row(c);
                vs(idx) = v * alpha; 
            }

            VectorType y = (1 + vs) * Fs; 
            A += Fs.transpose() * (Fs.array().colwise() * vs.transpose().array());
            
            // multiplicate relative weight over item-context relation
            A.noalias() *= l_;
            y.noalias() *= l_;
            
            for (int d=0; d < dim_; ++d)
                A(d, d) += reg_u_;
            _leastsquare(U_.row(x), A, y);

        }
    }
}

double CCFR::partial_update_item(int start_x, int next_x,
        int64_t* indptr_u, int* keys_u, float* vals_u,
        int64_t* indptr_c, int* keys_c, float* vals_c)
{
    if( (next_x - start_x) == 0) {
        WARN0("No data to process");
        return;
    }
    
    double losses(num_threads_);
    int end_loop = next_x - start_x;
    const int64_t shifted_u = indptr_u[0];
    const int64_t shifted_c = indptr_c[0];

    #pragma omp parallel
    {
        int _thread = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 4)
        for (int i=0; i<end_loop; ++i)
        {
            const int x = start_x + i;
            
            MatrixType A(dim_, dim_);
            VectorType y(dim_); 
            
            const int beg_u = indptr_u[i] - shifted_u;
            const int end_u = indptr_u[i+1] - shifted_u;
            const int data_size_u = end_u - beg_u;
            
            const int beg_c = indptr_c[i] - shifted_c;
            const int end_c = indptr_c[i+1] - shifted_c;
            const int data_size_c = end_c - beg_c;
            
            if (data_size_u == 0 and data_size_c == 0) {
                TRACE("No data exists for {}", u);
                continue;
            }

            // computing for the part of user-item relation
            A = FF_;
            MatrixType Fs(data_size_u, dim_);
            VectorType vs(data_size_u);
            float loss = compute_loss_? (I_.row(x) * FF_).dot(I_.row(x)): 0;
            for (int idx=0, it = beg_u; it < end_u; ++it, ++idx) {
                const int& c = keys_u[it];
                const float& v = vals_u[it];
                Fs.row(idx) = U_.row(c);
                vs(idx) = v * alpha;
                if (compute_loss_){
                    float dot = I_.row(x).dot(U_.row(c));
                    loss += (-dot * dot + (1 + vs(idx)) * (dot - 1) * (dot - 1))
                }
            }
            losses[_thread] += loss * l_;
            VectorType y = (1 + vs) * Fs; 
            A += Fs.transpose() * (Fs.array().colwise() * vs.transpose().array());
            
            // multiplicate relative weight over item-context relation
            A.noalias() *= l_;
            y.noalias() *= l_;
            // computing for the part of item-context relation
            Fs.resize(data_size_c, dim_);
            vs.resize(data_size_c);
            
            loss = 0.0
            for (int idx=0, it = beg_c; it < end_c; ++it, ++idx) {
                const int& c = keys_c[it];
                const float& v = vals_c[it];
                Fs.row(idx) = C_.row(c);
                vs(idx) = v - Ib_(x) - Cb_(c); 
                if (compute_loss_){
                    float err = v - I_.row(x).dot(C_.row(c)) - Ib_(x) - Cb_(c);
                    loss += err * err;
                }
            }
            losses[_thread] += loss;
            A += Fs.transpose() * Fs;
            y += vs * Fs;
            
            for (int d=0; d < dim_; ++d)
                A(d, d) += reg_i_;
            
            _leastsquare(I_.row(x), A, y);
            // update bias
            float b = 0;
            for (int idx=0, it = beg_c; it < end_c; ++it, ++idx) {
                const int& c = keys_c[it];
                const float& v = vals_c[it];
                b += (v - I_.row(x).dot(C_.row(c)) - Cb_(c));
            }
            Ib_(x) = b / (data_size_c + 1e-10);
        }
    }
    return accumulate(losses.begin(), losses.end(), 0.0);
}

void CCFR::partial_update_context(int start_x, int next_x,
        int64_t* indptr, int* keys, float* vals)
{
    if( (next_x - start_x) == 0) {
        WARN0("No data to process");
        return;
    }

    int end_loop = next_x - start_x;
    const int64_t shifted = indptr[0];
    #pragma omp parallel
    {
        int _thread = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 4)
        for (int i=0; i<end_loop; ++i)
        {
            const int x = start_x + i;
            
            MatrixType A(dim_, dim_);
            VectorType y(dim_); 
            
            const int beg = indptr[i] - shifted;
            const int end = indptr[i+1] - shifted;
            const int data_size = end - beg;
            
            if (data_size == 0) {
                TRACE("No data exists for {}", x);
                continue;
            }

            MatrixType Fs(data_size, dim_);
            VectorType vs(data_size);
            
            for (int idx=0, it = beg; it < end; ++it, ++idx) {
                const int& c = keys[it];
                const float& v = vals[it];
                Fs.row(idx) = I_.row(c);
                vs(idx) = v - Cb_(x) - Ib_(c); 
            }
            A = Fs.transpose() * Fs;
            y += vs * Fs;

            for (int d=0; d < dim_; ++d)
                A(d, d) += reg_c_;
            _leastsquare(C_.row(x), A, y);
            // update bias
            float b = 0;
            for (int idx=0, it = beg; it < end; ++it, ++idx) {
                const int& c = keys[it];
                const float& v = vals[it];
                b += (v - C_.row(x).dot(I_.row(c)) - Ib_(c));
            }
            Cb_(x) = b / (data_size + 1e-10);

        }
    }
}

void _leastsquare(VectorType& x, MatrixType& A, VectorType& y){
    switch(optimizer_code_){
        case 0:
            x.noalias() = A.llt().solve(y);
            break;
        case 1:
            x.noalias() = A.ldlt().solve(y);
            break;
        case 2:
            VectorType rold = y - x * A;
            VectorType p = rold;
            for (int i=0; i<num_cg_max_iters_; ++i){
                float rold_norm = r.dot(r);
                float alpha = rold_norm / (p * A).dot(p);
                x.noalias() += alpha * p;
                r.noalias() -= alpha * (p * A);
                float rnew_norm = r.dot(r);
                if (rnew_norm < cg_tolerance_)
                    break;
                float beta = rnew_norm / rold_norm;
                p.noalias() = r + beta * p;
            }
            break;
        case 3:
            ConjugateGradient<MatrixTeyp, Lower|Upper> cg;
            cg.setMaxIteration(num_iteration_for_cg_).compute(A);
            x.noalias() = cg.solve(y);
            break;
        default:
            break;
    }
}

} // namespace cfr
