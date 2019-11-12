#include <string>
#include <numeric>
#include <fstream>
#include <iostream>
#include <streambuf>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/algo_impl/cfr/cfr.hpp"


namespace cfr {

CCFR::CCFR():
    U_(nullptr, 0, 0), I_(nullptr, 0, 0), C_(nullptr, 0, 0),
    Ib_(nullptr, 0), Cb_(nullptr, 0){}

CCFR::~CCFR()
{
    new (&U_) Map<MatrixType>(nullptr, 0, 0);
    new (&I_) Map<MatrixType>(nullptr, 0, 0);
    new (&C_) Map<MatrixType>(nullptr, 0, 0);
    new (&Ib_) Map<VectorType>(nullptr, 0);
    new (&Cb_) Map<VectorType>(nullptr, 0);
    FF_.resize(0, 0);
}

// implementation of inherited virtual functions
bool CCFR::init(string opt_path){
    bool ok = parse_option(opt_path);
    if (ok){

        // int parameters
        dim_ = opt_["d"].int_value();
        num_threads_ = opt_["num_workers"].int_value();
        num_cg_max_iters_ = opt_["num_cg_max_iters"].int_value();
        alpha_ = opt_["alpha"].number_value();
        l_ = opt_["l"].number_value();

        // floating number parameters
        cg_tolerance_ = opt_["cg_tolerance_"].number_value();
        eps_ = opt_["eps"].number_value();
        reg_u_ = opt_["reg_u"].number_value();
        reg_i_ = opt_["reg_i"].number_value();
        reg_c_ = opt_["reg_c"].number_value();

        // boolean parameters
        compute_loss_ = opt_["compute_loss"].bool_value();

        // get optimizer
        string optimizer = opt_["optimizer"].string_value();
        if (optimizer == "llt") optimizer_code_ = 0;
        else if (optimizer == "ldlt") optimizer_code_ = 1;
        else if (optimizer == "manual_cg") optimizer_code_ = 2;
        else if (optimizer == "eigen_cg") optimizer_code_ = 3;
        else if (optimizer == "eigen_bicg") optimizer_code_ = 4;
        else if (optimizer == "eigen_gmres") optimizer_code_ = 5;
        else if (optimizer == "eigen_dgmres") optimizer_code_ = 6;
        else if (optimizer == "eigen_minres") optimizer_code_ = 7;

        FF_.resize(dim_, dim_);
    }
    return ok;
}

// implementation of inherited virtual functions
bool CCFR::parse_option(string opt_path){
    return Algorithm::parse_option(opt_path, opt_);
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
        new (&Cb_) Map<VectorType>(data, size);
    DEBUG("{} setted (size: {})", obj_type, size);
}


void CCFR::precompute(string obj_type)
{
    omp_set_num_threads(num_threads_);
    if (obj_type == "user") FF_ = U_.transpose() * U_;
    else if (obj_type == "item") FF_ = I_.transpose() * I_;
}

double CCFR::partial_update_user(int start_x, int next_x,
        int64_t* indptr, int32_t* keys, float* vals)
{
    if( (next_x - start_x) == 0) {
        WARN0("No data to process");
        return 0.0;
    }

    int end_loop = next_x - start_x;
    const int64_t shifted = start_x == 0? 0: indptr[start_x - 1];
    omp_set_num_threads(num_threads_);
    vector<double> losses(num_threads_, 0.0);

    #pragma omp parallel
    {
        int _thread = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 4)
        for (int i=0; i<end_loop; ++i)
        {
            const int x = start_x + i;
            // assume that shifted index can be represented by size_t
            const size_t beg = x == 0? 0: indptr[x - 1] - shifted;
            const size_t end = indptr[x] - shifted;
            const size_t data_size = end - beg;
            if (data_size == 0) {
                TRACE("No data exists for {}", x);
                continue;
            }

            MatrixType A(dim_, dim_);
            MatrixType Fs(data_size, dim_);
            VectorType coeff(data_size);
            VectorType y(dim_);

            A = FF_;
            for (size_t idx=0, it = beg; it < end; ++it, ++idx) {
                const int& c = keys[it];
                const float& v = vals[it];
                Fs.row(idx) = I_.row(c);
                coeff(idx) = v * alpha_;
            }
            MatrixType _Fs = Fs.array().colwise() * coeff.transpose().array();
            A.noalias() += Fs.transpose() * _Fs;
            coeff.array() += 1;
            y.noalias() = coeff * Fs;

            // multiplicate relative weight over item-context relation
            A.array() *= l_;
            y.array() *= l_;

            for (int d=0; d < dim_; ++d)
                A(d, d) += reg_u_;
            _leastsquare(U_, x, A, y);
            if (compute_loss_)
                losses[_thread] += U_.row(x).dot(U_.row(x));
        }
    }
    return reg_u_ * accumulate(losses.begin(), losses.end(), 0.0);
}

double CCFR::partial_update_item(int start_x, int next_x,
        int64_t* indptr_u, int32_t* keys_u, float* vals_u,
        int64_t* indptr_c, int32_t* keys_c, float* vals_c)
{
    if( (next_x - start_x) == 0) {
        WARN0("No data to process");
        return 0.0;
    }

    omp_set_num_threads(num_threads_);
    vector<double> losses(num_threads_, 0.0);
    int end_loop = next_x - start_x;
    const int64_t shifted_u = start_x == 0? 0: indptr_u[start_x - 1];
    const int64_t shifted_c = start_x == 0? 0: indptr_c[start_x - 1];
    #pragma omp parallel
    {
        int _thread = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 4)
        for (int i=0; i<end_loop; ++i)
        {
            const int x = start_x + i;

            MatrixType A(dim_, dim_);
            VectorType y(dim_);

            const size_t beg_u = x == 0? 0: indptr_u[x - 1] - shifted_u;
            const size_t end_u = indptr_u[x] - shifted_u;
            const size_t data_size_u = end_u - beg_u;

            const size_t beg_c = x == 0? 0: indptr_c[x - 1] - shifted_c;
            const size_t end_c = indptr_c[x] - shifted_c;
            const size_t data_size_c = end_c - beg_c;
            if (data_size_u == 0 and data_size_c == 0) {
                TRACE("No data exists for {}", x);
                continue;
            }

            // computing for the part of user-item relation
            A = FF_;
            MatrixType Fs(data_size_u, dim_);
            VectorType coeff(data_size_u);
            float loss = compute_loss_? (I_.row(x) * FF_).dot(I_.row(x)): 0;
            for (size_t idx=0, it = beg_u; it < end_u; ++it, ++idx) {
                const int& c = keys_u[it];
                const float& v = vals_u[it];
                Fs.row(idx) = U_.row(c);
                coeff(idx) = v * alpha_;
                if (compute_loss_){
                    float dot = I_.row(x).dot(U_.row(c));
                    loss += (-dot * dot + (1 + coeff(idx)) * (dot - 1) * (dot - 1));
                }
            }
            if (compute_loss_)
                losses[_thread] += loss * l_;
            MatrixType _Fs = Fs.array().colwise() * coeff.transpose().array();
            A.noalias() += Fs.transpose() * _Fs;
            coeff.array() += 1;
            y.noalias() = coeff * Fs;


            // multiplicate relative weight over item-context relation
            A.array() *= l_;
            y.array() *= l_;
            // computing for the part of item-context relation
            Fs.resize(data_size_c, dim_);
            coeff.resize(data_size_c);

            loss = 0.0;
            for (size_t idx=0, it=beg_c; it < end_c; ++it, ++idx) {
                const int& c = keys_c[it];
                const float& v = vals_c[it];
                Fs.row(idx) = C_.row(c);
                coeff(idx) = v - Ib_(x) - Cb_(c);
                if (compute_loss_){
                    float err = v - I_.row(x).dot(C_.row(c)) - Ib_(x) - Cb_(c);
                    loss += err * err;
                }
            }
            if (compute_loss_){
                losses[_thread] += loss;
                losses[_thread] += reg_i_ * I_.row(x).dot(I_.row(x));
            }
            A.noalias() += Fs.transpose() * Fs;
            y.noalias() += coeff * Fs;

            for (int d=0; d < dim_; ++d)
                A(d, d) += reg_i_;

            _leastsquare(I_, x, A, y);

            // update bias
            float b = 0;
            for (size_t idx=0, it = beg_c; it < end_c; ++it, ++idx) {
                const int& c = keys_c[it];
                const float& v = vals_c[it];
                b += (v - I_.row(x).dot(C_.row(c)) - Cb_(c));
            }
            Ib_(x) = b / (data_size_c + 1e-10);
        }
    }
    return accumulate(losses.begin(), losses.end(), 0.0);
}

double CCFR::partial_update_context(int start_x, int next_x,
        int64_t* indptr, int32_t* keys, float* vals)
{
    if( (next_x - start_x) == 0) {
        WARN0("No data to process");
        return 0.0;
    }
    omp_set_num_threads(num_threads_);
    vector<double> losses(num_threads_, 0.0);
    int end_loop = next_x - start_x;
    const int64_t shifted = start_x == 0? 0: indptr[start_x - 1];
    #pragma omp parallel
    {
        int _thread = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 4)
        for (int i=0; i<end_loop; ++i)
        {
            const int x = start_x + i;

            MatrixType A(dim_, dim_);
            VectorType y(dim_);

            const size_t beg = x == 0? 0: indptr[x - 1] - shifted;
            const size_t end = indptr[x] - shifted;
            const size_t data_size = end - beg;

            if (data_size == 0) {
                TRACE("No data exists for {}", x);
                continue;
            }

            MatrixType Fs(data_size, dim_);
            VectorType coeff(data_size);

            for (size_t idx=0, it = beg; it < end; ++it, ++idx) {
                const int& c = keys[it];
                const float& v = vals[it];
                Fs.row(idx) = I_.row(c);
                coeff(idx) = v - Cb_(x) - Ib_(c);
            }
            A = Fs.transpose() * Fs;
            y = coeff * Fs;

            for (int d=0; d < dim_; ++d)
                A(d, d) += reg_c_;
            if (compute_loss_)
                losses[_thread] += C_.row(x).dot(C_.row(x));
            _leastsquare(C_, x, A, y);
            // update bias
            float b = 0;
            for (size_t idx=0, it = beg; it < end; ++it, ++idx) {
                const int& c = keys[it];
                const float& v = vals[it];
                b += (v - C_.row(x).dot(I_.row(c)) - Ib_(c));
            }
            Cb_(x) = b / (data_size + 1e-10);

        }
    }
    return reg_c_ * accumulate(losses.begin(), losses.end(), 0.0);
}


} // namespace cfr
