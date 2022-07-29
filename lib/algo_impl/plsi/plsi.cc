#include <vector>
#include <string>
#include <numeric>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/algo_impl/plsi/plsi.hpp"


namespace plsi {


CPLSI::CPLSI():
    P_(nullptr, 0, 0), Q_(nullptr, 0, 0) {}

CPLSI::~CPLSI() {
    new (&P_) Map<MatrixType> (nullptr, 0, 0);
    new (&Q_) Map<MatrixType> (nullptr, 0, 0);
}

bool CPLSI::init(string opt_path) {
    bool ok = true;
    ok = ok & parse_option(opt_path);
    if (ok) {
        d_ = opt_["d"].int_value();
        seed_ = opt_["random_seed"].int_value();
        num_workers_ = opt_["num_workers"].int_value();
    }
    return ok;
}

bool CPLSI::parse_option(string opt_path) {
    return Algorithm::parse_option(opt_path, opt_);
}

void CPLSI::release() {
    P.resize(0, 0); Q.resize(0, 0);
}

void CPLSI::reset() {
    P.setZero(); Q.setZero();
}

void CPLSI::initialize_model(float* init_P, int P_rows,
        float* init_Q, int Q_rows) {
    new (&P_) Map<MatrixType> (init_P, P_rows, d_);
    new (&Q_) Map<MatrixType> (init_Q, Q_rows, d_);
    P.resize(P_rows, d_);
    Q.resize(Q_rows, d_);

    mt19937 RNG(seed_);
    normal_distribution<float> dist(0 , 1.0/d_);

    #pragma omp parallel for
    for (int u = 0; u < P_rows; ++u) {
        _mm_prefetch((char*)P_.row(u).data(), _MM_HINT_T0);
        for (int d = 0; d < d_; ++d)
            P_(u, d) = fabs(dist(RNG));
        P_.row(u) /= P_.row(u).sum();
    }

    #pragma omp parallel for
    for (int k = 0; k < d_; ++k) {
        _mm_prefetch((char*)Q_.col(k).data(), _MM_HINT_T0);
        for (int i = 0; i < Q_rows; ++i)
            Q_(i, k) = fabs(dist(RNG));
        Q_.col(k) /= Q_.col(k).sum();
    }

    DEBUG("Set P({} x {}) Q({} x {})",
        P.rows(), P.cols(), Q.rows(), Q.cols());
}

float CPLSI::partial_update(int start_x, int next_x, int64_t* indptr,
        int32_t* keys, float* vals) {
    omp_set_num_threads(num_workers_);

    vector<float> losses(num_workers_, 0.0);
    size_t job_size = next_x - start_x;
    const int64_t shifted = start_x == 0 ? 0 : indptr[start_x - 1];

    #pragma omp parallel for schedule(dynamic, 4)
    for (size_t i = 0; i < job_size; ++i) {
        const int x = start_x + i;
        const int64_t& beg = x == 0 ? 0 : indptr[x - 1];
        const int64_t& end = indptr[x];
        const size_t data_size = end - beg;
        if (data_size == 0) {
            TRACE("No data exists for {}", x);
            continue;
        }

        for (int64_t j = beg; j < end; ++j) {
            const int& c = keys[j - shifted];
            const float& v = vals[j - shifted];
            VectorXf latent = P_.row(x).array() * Q_.row(c).array();
            latent.noalias() = latent.cwiseMax(1e-10);
            float norm = latent.sum();
            losses[omp_get_thread_num()] -= log(norm) * v;
            latent.array() /= norm;
            P.row(x) += latent * v;
            Q.row(c) += latent * v;
        }
    }

    float loss = accumulate(losses.begin(), losses.end(), 0.0);
    return loss;
}

void CPLSI::normalize(float alpha1, float alpha2) {
    omp_set_num_threads(num_workers_);
    size_t num_users = P.rows();
    size_t num_items = Q.rows();
    alpha1 /= static_cast<float>(d_);
    alpha2 /= static_cast<float>(num_items);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < num_users; ++i) {
        P.row(i).array() += alpha1;
        P.row(i).array() /= P.row(i).sum();
    }
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < d_; ++i) {
        Q.col(i).array() += alpha2;
        Q.col(i).array() /= Q.col(i).sum();
    }
}

void CPLSI::swap() {
    P_ = P;
    Q_ = Q;
}
}  // namespace plsi
