#include <string>
#include <fstream>
#include <streambuf>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/algo_impl/als/als.hpp"

typedef Matrix<float, Dynamic, Dynamic, RowMajor> FactorType;


namespace als {


CALS::CALS()
{

}

CALS::~CALS()
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
    Map<MatrixXf> Q(Q_data_, Q_rows_, Q_cols_);
    Map<MatrixXf> P(P_data_, P_rows_, P_cols_);
    if (axis == 0) { // rowwise
        FF_ = Q.transpose() * Q;
    } else  { // colwise
        FF_ = P.transpose() * P;
    }
}

void CALS::partial_update(
        Map<VectorXi>& indptr,
        Map<VectorXi>& rows,
        Map<VectorXi>& keys,
        Map<VectorXf>& vals,
        int axis)
{
    float reg = 0.0;
    Map<MatrixXf> P(P_data_, P_rows_, P_cols_),
                  Q(Q_data_, Q_rows_, Q_cols_);
    if (axis == 0) {  // row-wise
        reg = opt_["reg_u"].number_value();
    } else { // col-wise
        swap(P, Q);
        reg = opt_["reg_i"].number_value();
    }

    float alpha = opt_["alpha"].number_value();
    bool adaptive_reg = opt_["adaptive_reg"].bool_value();
    int num_workers = opt_["num_workers"].int_value();
    int D = opt_["d"].int_value();
    vector<float> errs(num_workers, 0.0);

    #pragma omp parallel
    {
        // int worker_id = omp_get_thread_num();
        #pragma omp for schedule(dynamic, 4)
        for (int i=0; i < indptr.size(); ++i)
        {
            const int u = rows[i];
            const size_t beg = i == 0 ? 0 : indptr[i - 1];
            const size_t end = indptr[i];
            int data_size = end - beg;

            FactorType FiF(D, D); FiF.setZero();
            FactorType m(D, D); m.setZero();
            FactorType Fs(data_size, D);
            FactorType Fs2(data_size, D);

            Matrix<float, Dynamic, 1> Fxy;
            Fxy.resize(D, 1);
            Fxy.setZero();
            for (size_t idx=0, it = beg; it < end; ++it, ++idx) {
                const int& c = keys[it];
                const float& v = vals[it];
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
        }
    }
}

}
