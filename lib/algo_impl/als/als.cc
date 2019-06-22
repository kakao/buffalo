#include <string>
#include <numeric>
#include <fstream>
#include <iostream>
#include <streambuf>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/algo_impl/als/als.hpp"


namespace als {


CALS::CALS()
{

}

CALS::~CALS()
{
    P_data_ = Q_data_ = nullptr;
}

void CALS::release()
{
}

bool CALS::init(string opt_path) {
    bool ok = true;
    ok = ok & parse_option(opt_path);
    if (ok) {
        int num_workers = opt_["num_workers"].int_value();
        omp_set_num_threads(num_workers);
        int d = opt_["d"].int_value();
        FF_.resize(d, d);
    }
    return ok;
}


bool CALS::parse_option(string opt_path) {
    bool ok = Algorithm::parse_option(opt_path, opt_);
    return ok;
}

void CALS::set_factors(Map<MatrixXf>& _P, Map<MatrixXf>& _Q) {
    decouple(_P, &P_data_, P_rows_, P_cols_);
    decouple(_Q, &Q_data_, Q_rows_, Q_cols_);

    Map<MatrixXf> P(P_data_, P_rows_, P_cols_);
    Map<MatrixXf> Q(Q_data_, Q_rows_, Q_cols_);

    DEBUG("P({} x {}) Q({} x {}) setted", P.rows(), P.cols(), Q.rows(), Q.cols());
}


void CALS::precompute(int axis)
{
    Map<MatrixXf> Q(Q_data_, Q_rows_, Q_cols_);   // Q = Q.transpose();
    Map<MatrixXf> P(P_data_, P_rows_, P_cols_);   // P = P.transpose();
    if (axis == 0) { // rowwise
        FF_ = Q.transpose() * Q;
    } else  { // colwise
        FF_ = P.transpose() * P;
    }
}

double CALS::partial_update(
        int start_x,
        int next_x,
        int64_t* indptr,
        Map<VectorXi>& keys,
        Map<VectorXf>& vals,
        int axis)
{
    if( (next_x - start_x) == 0) {
        // WARN0("No data to process");
        return 0.0;
    }

    float reg = 0.0;
    float* P_data = P_data_, *Q_data = Q_data_;
    int P_rows = P_rows_, P_cols = P_cols_, Q_rows = Q_rows_, Q_cols = Q_cols_;
    if (axis == 0) {
        reg = opt_["reg_u"].number_value();
    }
    else { // if (axis == 1) {
        reg = opt_["reg_i"].number_value();
        P_data = Q_data_; P_rows = Q_rows_; Q_cols = Q_cols_;
        Q_data = P_data_; Q_rows = P_rows_; Q_cols = P_cols_;
    }

    Map<MatrixXf> P(P_data, P_rows, P_cols),
                  Q(Q_data, Q_rows, Q_cols);

    int D = opt_["d"].int_value();
    int num_workers = opt_["num_workers"].int_value();
    bool adaptive_reg = opt_["adaptive_reg"].bool_value();
    bool evaluation_on_learning = opt_["evaluation_on_learning"].bool_value();
    float alpha = opt_["alpha"].number_value();

    vector<float> errs(num_workers, 0.0);
    int end_loop = next_x - start_x;
    const int64_t shifted = start_x == 0 ? 0 : indptr[start_x - 1];
    #pragma omp parallel
    {
        int worker_id = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 4)
        for (int i=0; i < end_loop; ++i)
        {
            int x = start_x + i;
            const int u = x;
            int64_t beg = x == 0 ? 0 : indptr[x - 1];
            int64_t end = indptr[x];
            int64_t data_size = end - beg;
            if (data_size == 0) {
                DEBUG("No data exists for {}", u);
                continue;
            }

            FactorType FiF(D, D); FiF.setZero();
            FactorType m(D, D); m.setZero();
            FactorType Fs(data_size, D);
            FactorType Fs2(data_size, D);

            Matrix<float, Dynamic, 1> Fxy;
            Fxy.resize(D, 1);
            Fxy.setZero();
            for (int64_t idx=0, it = beg; it < end; ++it, ++idx) {
                const int& c = keys[it - shifted];
                const float& v = vals[it - shifted];
                Fs.row(idx) = v * Q.row(c);
                Fs2.row(idx) = Q.row(c);
                Fxy.noalias() += (Q.row(c) * (1.0 + v * alpha));
            }
            FiF = Fs.transpose() * Fs2 * alpha;
            m = FF_ + FiF;
            float ada_reg = adaptive_reg ? (float)data_size : 1.0;
            for (int d=0; d < D; ++d)
                m(d, d) += (reg * ada_reg);

            P.row(u).noalias() = m.ldlt().solve(Fxy);

            if (evaluation_on_learning and axis == 1) {  // for only on item side
                for (int64_t it=beg; it < end; ++it) {
                    const int& c = keys[it - shifted];
                    const float& v = vals[it - shifted];
                    float p = P.row(u) * Q.row(c).transpose();
                    double error = v - p;
                    errs[worker_id] += error * error;
                }
            }
        }
    }

    double err = accumulate(errs.begin(), errs.end(), 0.0);
    return err;
}

}
