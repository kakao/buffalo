#include "buffalo/algo_impl/eals/eals.hpp"


namespace eals {
EALS::EALS()
    : P_ptr_(nullptr),
      Q_ptr_(nullptr),
      C_ptr_(nullptr),
      P_rows_(-1),
      Q_rows_(-1),
      is_P_cached_(false),
      is_Q_cached_(false),
      kOne(1.0),
      kZero(0.0) {}

EALS::~EALS() {}

bool EALS::init(std::string opt_path) {
    const bool ok = parse_option(opt_path);
    if (ok) {
        const int32_t num_workers = opt_["num_workers"].int_value();
        omp_set_num_threads(num_workers);
    }
    return ok;
}

bool EALS::parse_option(std::string opt_path) {
    const bool ok = Algorithm::parse_option(opt_path, opt_);
    return ok;
}

void EALS::initialize_model(double* P_ptr,
                            double* Q_ptr,
                            double* C_ptr,
                            const int32_t P_rows,
                            const int32_t Q_rows) {
    const int32_t D = opt_["d"].int_value();
    P_ptr_ = P_ptr;
    Q_ptr_ = Q_ptr;
    C_ptr_ = C_ptr;
    P_rows_ = P_rows;
    Q_rows_ = Q_rows;
    is_P_cached_ = false;
    is_Q_cached_ = false;
    DEBUG("P({} x {}) Q({} x {}) set", P_rows_, D, Q_rows_, D);
}

void EALS::precompute_cache(const int32_t nnz,
                            const int64_t* indptr,
                            const int32_t* keys,
                            const int32_t axis) {
    bool& is_cached = axis == 0 ? is_P_cached_ : is_Q_cached_;
    if (is_cached) {
        return;
    }
    const double* P = axis == 0 ? P_ptr_ : Q_ptr_;
    const double* Q = axis == 0 ? Q_ptr_ : P_ptr_;
    auto& vhat_cache = axis == 0 ? vhat_cache_u_ : vhat_cache_i_;
    if (nnz != vhat_cache.size()) {
        vhat_cache.resize(nnz);
    }
    std::vector<IdxCoord> idx_xmajor(nnz);
    const int32_t end_loop = axis == 0 ? P_rows_ : Q_rows_;
    const int32_t D = opt_["d"].int_value();
    #pragma omp parallel for schedule(dynamic, 4)
    for (int32_t xidx=0; xidx < end_loop; ++xidx) {
        const int64_t beg = xidx == 0 ? 0 : indptr[xidx - 1];
        const int64_t end = indptr[xidx];
        int64_t data_size = end - beg;
        if (data_size == 0) {
            TRACE("No data exists for {}", xidx);
            continue;
        }
        const double* p_ptr = &P[xidx * D];
        for (int64_t ind=beg; ind < end; ++ind) {
            const int32_t yidx = keys[ind];
            const double* q_ptr = &Q[yidx * D];
            vhat_cache[ind] = std::inner_product(p_ptr, p_ptr + D, q_ptr, kZero);
            idx_xmajor[ind].set(yidx, xidx, ind);
        }
    }
    auto& ind_mapper = axis == 0 ? ind_u2i_ : ind_i2u_;
    if (nnz != ind_mapper.size()) {
        ind_mapper.resize(nnz);
    }
    std::sort(idx_xmajor.begin(), idx_xmajor.end(),
        [](const IdxCoord& a, const IdxCoord& b) -> bool {
            if (a.get_row() == b.get_row()) {
                return a.get_col() < b.get_col();
            } else {
                return a.get_row() < b.get_row();
            }
        }
    );
    for (int32_t ind=0; ind < nnz; ++ind) {
        const int64_t key = idx_xmajor[ind].get_key();
        ind_mapper[key] = ind;
    }
    is_cached = true;
}

bool EALS::update(const int64_t* indptr,
                  const int32_t* keys,
                  const float* vals,
                  const int32_t axis) {
    const bool is_cached = is_P_cached_ && is_Q_cached_;
    if (is_cached) {
        if (axis == 0) {
            this->update_P_(indptr, keys, vals);
        } else {
            this->update_Q_(indptr, keys, vals);
        }
    }
    return is_cached;
}

std::pair<double, double> EALS::estimate_loss(const int32_t nnz,
                                              const int64_t* indptr,
                                              const int32_t* keys,
                                              const float* vals,
                                              const int32_t axis) {
    // Loss := sum_{(u,i) \in R} (1 + alpha * v_{u,i}) * (v_{u,i} - vHat_{u,i})^2 +
    //         sum_{u} sum_{i \notin R_u} C_{i} * vHat_{u,i}^2 +
    //         reg_u * |P|^2 + reg_i * |Q|^2

    const bool is_cached = is_P_cached_ && is_Q_cached_;
    if (!is_cached) {
        TRACE("Compute cache first(P: {}, Q: {}). Empty data is returned.", is_P_cached_, is_Q_cached_);
        return std::pair<double, double>(kZero, kZero);
    }
    const int32_t num_workers = opt_["num_workers"].int_value();
    const double alpha = opt_["alpha"].number_value();
    std::vector<double> feedbacks4tid(num_workers, kZero);
    std::vector<double> mse4tid(num_workers, kZero);
    auto& vhat_cache = axis == 0 ? vhat_cache_u_ : vhat_cache_i_;
    const int32_t end_loop = axis == 0 ? P_rows_ : Q_rows_;
    #pragma omp parallel for schedule(dynamic, 8)
    for (int32_t xidx=0; xidx < end_loop; ++xidx) {
        const int32_t tid = omp_get_thread_num();
        const int64_t beg = xidx == 0 ? 0 : indptr[xidx - 1];
        const int64_t end = indptr[xidx];
        for (int64_t ind=beg; ind < end; ++ind) {
            const int32_t yidx = keys[ind];
            const double v = static_cast<double>(vals[ind]);
            const double vhat = vhat_cache[ind];
            const double error = (v - vhat);
            feedbacks4tid[tid] += (kOne + alpha * v) * error * error;
            const int32_t iidx = axis == 0 ? yidx : xidx;
            feedbacks4tid[tid] -= C_ptr_[iidx] * vhat * vhat; // To avoid duplication in a term of negative feedbacks.
            mse4tid[tid] += error * error;
        }
    }
    const double squared_error = std::accumulate(mse4tid.begin(), mse4tid.end(), kZero);
    double feedbacks = std::accumulate(feedbacks4tid.begin(), feedbacks4tid.end(), kZero);

    // Add L2 regularization terms
    const int32_t D = opt_["d"].int_value();
    const double reg_u = opt_["reg_u"].number_value();
    const double reg_i = opt_["reg_i"].number_value();
    auto op = [](const double & init, const double & v) -> double {
        return init + v * v;
    };
    const double reg = reg_u * std::accumulate(P_ptr_, P_ptr_ + (P_rows_ * D), kZero, op) +
                       reg_i * std::accumulate(Q_ptr_, Q_ptr_ + (Q_rows_ * D), kZero, op);

    // Add negative feedbacks: sum_{u} sum_{i \in R_u} C_{i} * vHat_{u,i}^2
    std::vector<double> CQ(Q_rows_ * D, kZero);
    #pragma omp parallel for schedule(dynamic, 8)
    for (int32_t iidx=0; iidx < Q_rows_; ++iidx) {
        const int32_t ridx = iidx * D;
        for (int32_t d=0; d<D; ++d) {
            CQ[ridx + d] = std::sqrt(C_ptr_[iidx]) * Q_ptr_[ridx + d];
        }
    }
    std::vector<double> Sp(D * D, kZero);
    std::vector<double> Sq(D * D, kZero);
    blas::syrk("u", "t", D, P_rows_, kOne, P_ptr_, kZero, Sp.data());
    blas::syrk("u", "t", D, Q_rows_, kOne, CQ.data(), kZero, Sq.data());
    feedbacks += std::inner_product(Sp.data(), Sp.data() + (D * D), Sq.data(), kZero);
    const double rmse = std::sqrt(squared_error / nnz);
    const double loss = feedbacks + reg;
    return std::pair<double, double>(rmse, loss);
}

void EALS::update_P_(const int64_t* indptr,
                     const int32_t* keys,
                     const float* vals) {
    const int32_t D = opt_["d"].int_value();
    std::vector<double> CQ(Q_rows_ * D, kZero);
    for (int32_t iidx=0; iidx < Q_rows_; ++iidx) {
        const double sqrt_C = std::sqrt(C_ptr_[iidx]);
        const int32_t ridx = iidx * D;
        const double* q_ptr = &Q_ptr_[ridx];
        double* cq_ptr = &CQ[ridx];
        std::transform(q_ptr, q_ptr + D, cq_ptr,
            [sqrt_C](const double elem) -> double {
                return sqrt_C * elem;
            }
        );
    }
    std::vector<double> Sq(D * D, kZero);
    blas::syrk("u", "t", D, Q_rows_, kOne, CQ.data(), kZero, Sq.data());
    const double alpha = opt_["alpha"].number_value();
    const double reg_u = opt_["reg_u"].number_value();
    #pragma omp parallel for schedule(dynamic, 8)
    for (int32_t uidx=0; uidx < P_rows_; ++uidx) {
        double* p_ptr = &P_ptr_[uidx * D];
        const int64_t beg = uidx == 0 ? 0 : indptr[uidx - 1];
        const int64_t end = indptr[uidx];
        for (int32_t d=0; d < D; ++d) {
            double numerator = kZero;
            double denominator = kZero;
            for (int64_t ind=beg; ind < end; ++ind) {
                const int32_t iidx = keys[ind];
                const double v = static_cast<double>(vals[ind]);
                const double* q_ptr = &Q_ptr_[iidx * D];
                const double vhat = vhat_cache_u_[ind];
                const double pq = p_ptr[d] * q_ptr[d];
                const double vf = vhat - pq;
                const double w = (kOne + alpha * v);
                const double wmc = w - C_ptr_[iidx];
                numerator += (w * v - wmc * vf) * q_ptr[d];
                denominator += wmc * q_ptr[d] * q_ptr[d];
                vhat_cache_u_[ind] -= pq;
                vhat_cache_i_[ind_u2i_[ind]] -= pq;
            }
            numerator += -std::inner_product(p_ptr, p_ptr + D, &Sq[D * d], kZero) + p_ptr[d] * Sq[D * d + d];
            denominator += Sq[D * d + d] + reg_u;
            p_ptr[d] = numerator / denominator;
            for (int64_t ind=beg; ind < end; ++ind) {
                const int32_t iidx = keys[ind];
                const double* q_ptr = &Q_ptr_[iidx * D];
                const double pq = p_ptr[d] * q_ptr[d];
                vhat_cache_u_[ind] += pq;
                vhat_cache_i_[ind_u2i_[ind]] += pq;
            }
        }
    }
}

void EALS::update_Q_(const int64_t* indptr,
                     const int32_t* keys,
                     const float* vals) {
    const int32_t D = opt_["d"].int_value();
    std::vector<double> Sp(D * D, kZero);
    blas::syrk("u", "t", D, P_rows_, kOne, P_ptr_, kZero, Sp.data());
    const double alpha = opt_["alpha"].number_value();
    const double reg_i = opt_["reg_i"].number_value();
    #pragma omp parallel for schedule(dynamic, 8)
    for (int32_t iidx=0; iidx < Q_rows_; ++iidx) {
        double* q_ptr = &Q_ptr_[iidx * D];
        const int64_t beg = iidx == 0 ? 0 : indptr[iidx - 1];
        const int64_t end = indptr[iidx];
        for (int32_t d=0; d < D; ++d) {
            double numerator = kZero;
            double denominator = kZero;
            for (int64_t ind=beg; ind < end; ++ind) {
                const int32_t uidx = keys[ind];
                const double v = static_cast<double>(vals[ind]);
                const double* p_ptr = &P_ptr_[uidx * D];
                const double vhat = vhat_cache_i_[ind];
                const double pq = p_ptr[d] * q_ptr[d];
                const double vf = vhat - pq;
                const double w = (kOne + alpha * v);
                const double wmc = w - C_ptr_[iidx];
                numerator += (w * v - wmc * vf) * p_ptr[d];
                denominator += wmc * p_ptr[d] * p_ptr[d];
                vhat_cache_i_[ind] -= pq;
                vhat_cache_u_[ind_i2u_[ind]] -= pq;
            }
            numerator += -C_ptr_[iidx] * (std::inner_product(q_ptr, q_ptr + D, &Sp[D * d], kZero) - q_ptr[d] * Sp[D * d + d]);
            denominator += C_ptr_[iidx] * Sp[D * d + d] + reg_i;
            q_ptr[d] = numerator / denominator;
            for (int64_t ind=beg; ind < end; ++ind) {
                const int32_t uidx = keys[ind];
                const double* p_ptr = &P_ptr_[uidx * D];
                const double pq = p_ptr[d] * q_ptr[d];
                vhat_cache_i_[ind] += pq;
                vhat_cache_u_[ind_i2u_[ind]] += pq;
            }
        }
    }
}
} // end namespace eals
