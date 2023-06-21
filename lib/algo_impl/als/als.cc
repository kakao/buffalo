#include <ostream>
#include <string>
#include <numeric>
#include <fstream>
#include <iostream>
#include <streambuf>

#include "buffalo/algo.hpp"
#include "json11.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/algo_impl/als/als.hpp"


namespace als {
CALS::CALS() :
    P_(nullptr, 0, 0), Q_(nullptr, 0, 0)
{ }

CALS::~CALS()
{
    new (&P_) Map<FactorTypeRowMajor>(nullptr, 0, 0);
    new (&Q_) Map<FactorTypeRowMajor>(nullptr, 0, 0);
}

void CALS::release()
{
    FF_.resize(0, 0);
}

bool CALS::init(string opt_path) {
    bool ok = true;
    ok = ok & parse_option(opt_path);
    if (ok) {
        int num_workers = opt_["num_workers"].int_value();
        omp_set_num_threads(num_workers);
        int d = opt_["d"].int_value();
        FF_.resize(d, d);


        // get coefficient for optimizer
        eps_ = opt_["eps"].number_value();
        cg_tolerance_ = opt_["cg_tolerance"].number_value();
        num_cg_max_iters_ = opt_["num_cg_max_iters"].int_value();
        // get optimizer
        string optimizer = opt_["optimizer"].string_value();
        if (d >= 128) optimizer = "ialspp";
        if (optimizer == "llt") {
            optimizer_code_ = 0;
        } else if (optimizer == "ldlt") {
            optimizer_code_ = 1;
        } else if (optimizer == "manual_cg") {
            optimizer_code_ = 2;
        } else if (optimizer == "eigen_cg") {
            optimizer_code_ = 3;
        } else if (optimizer == "eigen_bicg") {
            optimizer_code_ = 4;
        } else if (optimizer == "eigen_gmres") {
            optimizer_code_ = 5;
        } else if (optimizer == "eigen_dgmres") {
            optimizer_code_ = 6;
        } else if (optimizer == "eigen_minres") {
            optimizer_code_ = 7;
        } else if (optimizer == "ialspp") {
            use_ialspp_ = true;
            optimizer_code_ = 8;
        }
    }
    return ok;
}

bool CALS::parse_option(string opt_path) {
    bool ok = Algorithm::parse_option(opt_path, opt_);
    return ok;
}

void CALS::initialize_model(float* P, int P_rows, float* Q, int Q_rows) {
    int d = opt_["d"].int_value();
    new (&P_) Map<FactorTypeRowMajor>(P, P_rows, d);
    new (&Q_) Map<FactorTypeRowMajor>(Q, Q_rows, d);

    DEBUG("P({} x {}) Q({} x {}) set",
            P_.rows(), P_.cols(), Q_.rows(), Q_.cols());
}


void CALS::precompute(int axis)
{
    if (axis == 0) { // rowwise
        FF_ = Q_.transpose() * Q_;
    } else { // colwise
        FF_ = P_.transpose() * P_;
    }
}

pair<double, double> CALS::partial_update(int start_x,
                                          int next_x,
                                          int64_t* indptr,
                                          int32_t* keys,
                                          float* vals,
                                          int axis) {
    if (use_ialspp_)
        return _partial_update_ialspp(start_x, next_x, indptr, keys, vals, axis);
    else
        return _partial_update(start_x, next_x, indptr, keys, vals, axis);
}

pair<double, double> CALS::_partial_update(
        int start_x,
        int next_x,
        int64_t* indptr,
        int32_t* keys,
        float* vals,
        int axis)
{
    if ((next_x - start_x) == 0) {
        // WARN0("No data to process");
        return make_pair(0.0, 0.0);
    }

    float reg = 0.0;
    int P_rows = P_.rows(),
        P_cols = P_.cols(),
        Q_rows = Q_.rows(),
        Q_cols = Q_.cols();

    if (axis == 0) {
        reg = opt_["reg_u"].number_value();
    } else { // if (axis == 1) {
        reg = opt_["reg_i"].number_value();
        swap(P_rows, Q_rows);
        swap(P_cols, Q_cols);
    }
    Map<FactorTypeRowMajor>& P = axis == 0 ? P_ : Q_;
    Map<FactorTypeRowMajor>& Q = axis == 0 ? Q_ : P_;

    int D = opt_["d"].int_value();
    int num_workers = opt_["num_workers"].int_value();
    bool adaptive_reg = opt_["adaptive_reg"].bool_value();
    bool compute_loss_on_training = opt_["compute_loss_on_training"].bool_value();
    float alpha = opt_["alpha"].number_value();

    omp_set_num_threads(num_workers);

    vector<double> loss_nume(num_workers, 0.0);
    vector<double> loss_deno(num_workers, 0.0);
    int end_loop = next_x - start_x;
    const int64_t shifted = start_x == 0 ? 0 : indptr[start_x - 1];
    #pragma omp parallel num_threads(num_workers)
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
                TRACE("No data exists for {}", u);
                continue;
            }

            FactorTypeRowMajor FiF(D, D);
            FactorTypeRowMajor m(D, D);
            FiF.setZero();
            m.setZero();
            FactorTypeRowMajor Fs(data_size, D);
            FactorTypeRowMajor Fs2(data_size, D);

            VectorType Fxy(D);
            Fxy.setZero();

            // compute loss on negative samples (only item side)
            if (compute_loss_on_training && axis == 1) {
                loss_nume[worker_id] += P.row(u).dot(P.row(u) * FF_);
                loss_deno[worker_id] += Q.rows();
            }

            for (int64_t idx=0, it = beg; it < end; ++it, ++idx) {
                const int c = keys[it - shifted];
                const float v = vals[it - shifted];
                Fs.row(idx) = v * Q.row(c);
                Fs2.row(idx) = Q.row(c);
                Fxy.noalias() += (Q.row(c) * (1.0 + v * alpha));
                // taking positive samples into account for computing loss (only item side)
                if (compute_loss_on_training && axis == 1) {
                    float dot = P.row(u).dot(Q.row(c));
                    loss_nume[worker_id] -= dot * dot;
                    loss_nume[worker_id] += (dot - 1) * (dot - 1) * (1.0 + v * alpha);
                    loss_deno[worker_id] += v * alpha;
                }
            }
            FiF = Fs.transpose() * Fs2 * alpha;
            m = FF_ + FiF;
            float ada_reg = adaptive_reg ? (float)data_size : 1.0;
            // compute loss on regularizatio term (both user and item side)
            if (compute_loss_on_training) {
                loss_nume[worker_id] += ada_reg * reg * P.row(u).dot(P.row(u));
            }
            for (int d=0; d < D; ++d)
                m(d, d) += (reg * ada_reg);

            _leastsquare(P, u, m, Fxy);
        }
    }
    return make_pair(accumulate(loss_nume.begin(), loss_nume.end(), 0.0),
            accumulate(loss_deno.begin(), loss_deno.end(), 0.0));
}

pair<double, double> CALS::_partial_update_ialspp(
        int start_x,
        int next_x,
        int64_t* indptr,
        int32_t* keys,
        float* vals,
        int axis)
{
    if ((next_x - start_x) == 0) {
        // WARN0("No data to process");
        return make_pair(0.0, 0.0);
    }

    float reg = 0.0;
    int P_rows = P_.rows(),
        P_cols = P_.cols(),
        Q_rows = Q_.rows(),
        Q_cols = Q_.cols();

    if (axis == 0) {
        reg = opt_["reg_u"].number_value();
    } else { // if (axis == 1) {
        reg = opt_["reg_i"].number_value();
        swap(P_rows, Q_rows);
        swap(P_cols, Q_cols);
    }
    Map<FactorTypeRowMajor>& P = axis == 0 ? P_ : Q_;
    Map<FactorTypeRowMajor>& Q = axis == 0 ? Q_ : P_;
    int D = opt_["d"].int_value();
    int num_workers = opt_["num_workers"].int_value();
    bool adaptive_reg = opt_["adaptive_reg"].bool_value();
    bool compute_loss_on_training = opt_["compute_loss_on_training"].bool_value();
    const float alpha = opt_["alpha"].number_value();
    int block_size_ = min(D, opt_["block_size"].int_value());
    omp_set_num_threads(num_workers);

    vector<double> loss_nume(num_workers, 0.0);
    vector<double> loss_deno(num_workers, 0.0);
    int end_loop = next_x - start_x;
    const int64_t shifted = start_x == 0 ? 0 : indptr[start_x - 1];
    FactorType Identity = FactorType::Identity(block_size_, block_size_);

    // build Y_ui
    vector<float> Yui(indptr[end_loop - 1]);

    #pragma omp parallel for schedule(dynamic, 16)
    for (int i=0; i < end_loop; ++i) {
        int x = start_x + i;
        const int u = x;
        int64_t beg = x == 0 ? 0 : indptr[x - 1];
        int64_t end = indptr[x];
        for (int64_t idx=0, it = beg; it < end; ++it, ++idx) {
            const int col = keys[it - shifted];
            Yui[it - shifted] = (P.row(u) * Q.row(col).transpose()).value();
        }
    }


    for (int block_beg = 0; block_beg < D; block_beg += block_size_) {
        int block_size = block_size_;
        if (block_beg + block_size >= D) {
            block_size = D - block_beg;
            Identity.conservativeResize(block_size, block_size);
        }

        const auto& block_q = Q.block(0, block_beg, Q.rows(), block_size);
        const FactorType &gramian = FF_.block(0, block_beg, D, block_size);
        const FactorType A = gramian.block(block_beg, 0, block_size, block_size) + Identity * reg;
        #pragma omp parallel
        {
            int worker_id = omp_get_thread_num();
            #pragma omp for schedule(dynamic, 4)
            for (int i=0; i < end_loop; ++i) {
                int x = start_x + i;
                const int u = x;
                int64_t beg = x == 0 ? 0 : indptr[x - 1];
                int64_t end = indptr[x];
                int64_t data_size = end - beg;
                if (data_size == 0) {
                    TRACE("No data exists for {}", u);
                    continue;
                }

                const FactorType& p = P.row(u);
                const FactorType& block_p = p.block(0, block_beg, 1, block_size);
                FactorType b = p * gramian + reg * block_p;

                if (block_beg == 0 && compute_loss_on_training && axis == 1) {
                    loss_nume[worker_id] += P.row(u).dot(P.row(u) * FF_);
                    loss_deno[worker_id] += Q.rows();
                }

                for (int64_t idx=0, it = beg; it < end; ++it, ++idx) {
                    const int col = keys[it - shifted];
                    const float val = vals[it - shifted];
                    float residual = Yui[it - shifted] - 1.0;
                    const FactorType& v = block_q.row(col);
                    b.noalias() += residual * val * alpha * v;
                    if ((block_beg == 0) && compute_loss_on_training && axis == 1) {
                        float dot = P.row(u).dot(Q.row(col));
                        loss_nume[worker_id] -= dot * dot;
                        loss_nume[worker_id] += (dot - 1) * (dot - 1) * (1.0 + val * alpha);
                        loss_deno[worker_id] += val * alpha;
                    }
                }
                float ada_reg = adaptive_reg ? (float)data_size : 1.0;
                // compute loss on regularizatio term (both user and item side)
                if ((block_beg == 0) && compute_loss_on_training) {
                    loss_nume[worker_id] += ada_reg * reg * P.row(u).dot(P.row(u));
                }
                // CG Update
                b.transposeInPlace();
                {
                    Eigen::VectorXf x = Eigen::VectorXf::Zero(block_size);
                    Eigen::VectorXf r = b;
                    Eigen::VectorXf p = r;
                    Eigen::VectorXf Ap(block_size);
                    double rsold = r.dot(r);
                    if (rsold > cg_tolerance_) {
                        for (int cg_step = 0; cg_step < 3; ++cg_step) {
                            Ap = A * p;
                            for (int64_t idx=0, it = beg; it < end; ++it, ++idx) {
                                const int col = keys[it - shifted];
                                const float val = vals[it - shifted];
                                Ap.noalias() += val * alpha * (block_q.row(col) * p).value() * block_q.row(col);
                            }
                            float step_size = rsold / p.dot(Ap);
                            x.noalias() += step_size * p;
                            r.noalias() -= step_size * Ap;
                            double rsnew = r.dot(r);
                            if (rsnew < cg_tolerance_) break;
                            p.noalias() = r + (rsnew / rsold) * p;
                            rsold = rsnew;
                        }
                    }
                    P.row(u).block(0, block_beg, 1, block_size).noalias() -= x.transpose();
                    for (int64_t idx=0, it = beg; it < end; ++it, ++idx) {
                        const int col = keys[it - shifted];
                        Yui[it - shifted] -= (block_q.row(col) * x).value();
                    }
                }
            }
        }
    }

    return make_pair(accumulate(loss_nume.begin(), loss_nume.end(), 0.0),
                     accumulate(loss_deno.begin(), loss_deno.end(), 0.0));
}
} // namespace als
