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
    setNbThreads(num_threads_);
    if (obj_type == "user") FF_ = P_.transpose() * P_;
    else if (obj_type == "item") FF_ = Q_.transpose() * Q_;
    setNbThreads(1);
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
            int x = start_x + i;
            const int u = x;
            // assume that shifted index is not so big
            const int beg = indptr[i] - shifted;
            const int end = indptr[i+1] - shifted;
            const int data_size = end - beg;
            if (data_size == 0) {
                TRACE("No data exists for {}", u);
                continue;
            }

            MatrixType m = FF_;
            MatrixType Is(data_size, D);
            VectorType vs(data_size);

            Fxy.resize(D, 1);
            Fxy.setZero();
            for (int idx=0, it = beg; it < end; ++it, ++idx) {
                const int& c = keys[it - shifted];
                const float& v = vals[it - shifted];
                Fs.row(idx) = Q.row(c);
                vs(idx) = v * alpha; 
            }

            VectorType y = (1 + vs) * Fs; 
            m += Fs.transpose() * (Fs.array().colwise() * vs.transpose().array());
            float ada_reg = adaptive_reg ? (float)data_size : 1.0;
            for (int d=0; d < D; ++d)
                m(d, d) += (reg * ada_reg);

            if (use_conjugate_gradient) {
                cg[worker_id].setMaxIterations(num_iteration_for_cg).compute(m);
                P.row(u).noalias() = cg[worker_id].solve(Fxy);
            } else{
                P.row(u).noalias() = m.ldlt().solve(Fxy);
            }
        }
    }

    double err = accumulate(errs.begin(), errs.end(), 0.0);
    return err;
}

}
