#include "buffalo/algo_impl/eals/eals.hpp"


namespace eals {

CEALS::CEALS()
    : P_ptr_(nullptr),
      Q_ptr_(nullptr),
      C_ptr_(nullptr),
      P_rows_(-1),
      Q_rows_(-1),
      is_P_cached_(false),
      is_Q_cached_(false),
      kOne(1.0),
      kZero(0.0) {}

CEALS::~CEALS() {}

bool CEALS::init(string opt_path) {
    const bool ok = parse_option(opt_path);
    if (ok) {
        const int32_t num_workers = opt_["num_workers"].int_value();
        omp_set_num_threads(num_workers);
    }
    return ok;
}

bool CEALS::parse_option(string opt_path) {
    const bool ok = Algorithm::parse_option(opt_path, opt_);
    return ok;
}

void CEALS::initialize_model(float* P_ptr,
                             float* Q_ptr,
                             float* C_ptr,
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

void CEALS::precompute_cache(const int32_t nnz,
                             const int64_t* indptr,
                             const int32_t* keys,
                             const int32_t axis) {
    bool& is_cached = axis == 0 ? is_P_cached_ : is_Q_cached_;
    if (is_cached) {
        return;
    }
    const float* P = axis == 0 ? P_ptr_ : Q_ptr_;
    const float* Q = axis == 0 ? Q_ptr_ : P_ptr_;
    auto& vhat_cache = axis == 0 ? vhat_cache_u_ : vhat_cache_i_;
    if (nnz != vhat_cache.size()) {
        vhat_cache.resize(nnz);
    }
    vector<IdxCoord> idx_xmajor(nnz);
    const int32_t end_loop = axis == 0 ? P_rows_ : Q_rows_;
    const int32_t D = opt_["d"].int_value();
    #pragma omp parallel for schedule(dynamic, 8)
    for (int32_t xidx=0; xidx < end_loop; ++xidx) {
        const int64_t beg = xidx == 0 ? 0 : indptr[xidx - 1];
        const int64_t end = indptr[xidx];
        int64_t data_size = end - beg;
        if (data_size == 0) {
            TRACE("No data exists for {}", xidx);
            continue;
        }
        const float* p_ptr = &P[xidx * D];
        for (int64_t ind=beg; ind < end; ++ind) {
            const int32_t yidx = keys[ind];
            const float* q_ptr = &Q[yidx * D];
            vhat_cache[ind] = inner_product(p_ptr, p_ptr + D, q_ptr, kZero);
            idx_xmajor[ind].set(yidx, xidx, ind);
        }
    }
    auto& ind_mapper = axis == 0 ? ind_u2i_ : ind_i2u_;
    if (nnz != ind_mapper.size()) {
        ind_mapper.resize(nnz);
    }
    sort(idx_xmajor.begin(), idx_xmajor.end(),
        [](const IdxCoord& a, const IdxCoord& b) -> bool {
            if (a.get_row() == b.get_row()) {
                return a.get_col() < b.get_col();
            } else {
                return a.get_row() < b.get_row();
            }
        });
    for (int32_t ind=0; ind < nnz; ++ind) {
        const int64_t key = idx_xmajor[ind].get_key();
        ind_mapper[key] = ind;
    }
    is_cached = true;
}

bool CEALS::update(const int64_t* indptr,
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

pair<float, float> CEALS::estimate_loss(const int32_t nnz,
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
        return pair<float, float>(kZero, kZero);
    }
    const int32_t num_workers = opt_["num_workers"].int_value();
    const float alpha = opt_["alpha"].number_value();
    float feedbacks = kZero;
    float squared_error = kZero;
    auto& vhat_cache = axis == 0 ? vhat_cache_u_ : vhat_cache_i_;
    const int32_t end_loop = axis == 0 ? P_rows_ : Q_rows_;
    for (int32_t xidx=0; xidx < end_loop; ++xidx) {
        const int32_t tid = omp_get_thread_num();
        const int64_t beg = xidx == 0 ? 0 : indptr[xidx - 1];
        const int64_t end = indptr[xidx];
        for (int64_t ind=beg; ind < end; ++ind) {
            const int32_t yidx = keys[ind];
            const float v = vals[ind];
            const float vhat = vhat_cache[ind];
            const float error = (v - vhat);
            feedbacks += (kOne + alpha * v) * error * error;
            const int32_t iidx = axis == 0 ? yidx : xidx;
            feedbacks -= C_ptr_[iidx] * vhat * vhat;  // To avoid duplication in a term of negative feedbacks.
            squared_error += error * error;
        }
    }

    // Add L2 regularization terms
    const int32_t D = opt_["d"].int_value();
    const float reg_u = opt_["reg_u"].number_value();
    const float reg_i = opt_["reg_i"].number_value();
    auto op = [](const float & init, const float & v) -> float {
        return init + v * v;
    };
    const float reg = reg_u * accumulate(P_ptr_, P_ptr_ + (P_rows_ * D), kZero, op) +
                      reg_i * accumulate(Q_ptr_, Q_ptr_ + (Q_rows_ * D), kZero, op);

    // Add negative feedbacks: sum_{u} sum_{i \in R_u} C_{i} * vHat_{u,i}^2
    vector<float> CQ(Q_rows_ * D, kZero);
    #pragma omp parallel for schedule(dynamic, 8)
    for (int32_t iidx=0; iidx < Q_rows_; ++iidx) {
        const int32_t ridx = iidx * D;
        for (int32_t d=0; d < D; ++d) {
            CQ[ridx + d] = sqrt(C_ptr_[iidx]) * Q_ptr_[ridx + d];
        }
    }
    vector<float> Sp(D * D, kZero);
    vector<float> Sq(D * D, kZero);
    blas::syrk("u", "t", D, P_rows_, kOne, P_ptr_, kZero, Sp.data());
    blas::syrk("u", "t", D, Q_rows_, kOne, CQ.data(), kZero, Sq.data());
    feedbacks += inner_product(Sp.data(), Sp.data() + (D * D), Sq.data(), kZero);
    const float rmse = sqrt(squared_error / nnz);
    const float loss = feedbacks + reg;
    return pair<float, float>(rmse, loss);
}

void CEALS::update_P_(const int64_t* indptr,
                      const int32_t* keys,
                      const float* vals) {
    const int32_t D = opt_["d"].int_value();
    vector<float> CQ(Q_rows_ * D, kZero);
    for (int32_t iidx=0; iidx < Q_rows_; ++iidx) {
        const float sqrt_C = sqrt(C_ptr_[iidx]);
        const int32_t ridx = iidx * D;
        const float* q_ptr = &Q_ptr_[ridx];
        float* cq_ptr = &CQ[ridx];
        transform(q_ptr, q_ptr + D, cq_ptr,
            [sqrt_C](const float elem) -> float {
                return sqrt_C * elem;
            });
    }
    vector<float> Sq(D * D, kZero);
    blas::syrk("u", "t", D, Q_rows_, kOne, CQ.data(), kZero, Sq.data());
    const float alpha = opt_["alpha"].number_value();
    const float reg_u = opt_["reg_u"].number_value();
    #pragma omp parallel for schedule(dynamic, 8)
    for (int32_t uidx=0; uidx < P_rows_; ++uidx) {
        float* p_ptr = &P_ptr_[uidx * D];
        const int64_t beg = uidx == 0 ? 0 : indptr[uidx - 1];
        const int64_t end = indptr[uidx];
        for (int32_t d=0; d < D; ++d) {
            float numerator = kZero;
            float denominator = kZero;
            for (int64_t ind=beg; ind < end; ++ind) {
                const int32_t iidx = keys[ind];
                const float v = vals[ind];
                const float* q_ptr = &Q_ptr_[iidx * D];
                const float vhat = vhat_cache_u_[ind];
                const float pq = p_ptr[d] * q_ptr[d];
                const float vf = vhat - pq;
                const float w = (kOne + alpha * v);
                const float wmc = w - C_ptr_[iidx];
                numerator += (w * v - wmc * vf) * q_ptr[d];
                denominator += wmc * q_ptr[d] * q_ptr[d];
                vhat_cache_u_[ind] -= pq;
                vhat_cache_i_[ind_u2i_[ind]] -= pq;
            }
            numerator += -inner_product(p_ptr, p_ptr + D, &Sq[D * d], kZero) + p_ptr[d] * Sq[D * d + d];
            denominator += Sq[D * d + d] + reg_u;
            p_ptr[d] = numerator / denominator;
            for (int64_t ind=beg; ind < end; ++ind) {
                const int32_t iidx = keys[ind];
                const float* q_ptr = &Q_ptr_[iidx * D];
                const float pq = p_ptr[d] * q_ptr[d];
                vhat_cache_u_[ind] += pq;
                vhat_cache_i_[ind_u2i_[ind]] += pq;
            }
        }
    }
}

void CEALS::update_Q_(const int64_t* indptr,
                      const int32_t* keys,
                      const float* vals) {
    const int32_t D = opt_["d"].int_value();
    vector<float> Sp(D * D, kZero);
    blas::syrk("u", "t", D, P_rows_, kOne, P_ptr_, kZero, Sp.data());
    const float alpha = opt_["alpha"].number_value();
    const float reg_i = opt_["reg_i"].number_value();
    #pragma omp parallel for schedule(dynamic, 8)
    for (int32_t iidx=0; iidx < Q_rows_; ++iidx) {
        float* q_ptr = &Q_ptr_[iidx * D];
        const int64_t beg = iidx == 0 ? 0 : indptr[iidx - 1];
        const int64_t end = indptr[iidx];
        for (int32_t d=0; d < D; ++d) {
            float numerator = kZero;
            float denominator = kZero;
            for (int64_t ind=beg; ind < end; ++ind) {
                const int32_t uidx = keys[ind];
                const float v = vals[ind];
                const float* p_ptr = &P_ptr_[uidx * D];
                const float vhat = vhat_cache_i_[ind];
                const float pq = p_ptr[d] * q_ptr[d];
                const float vf = vhat - pq;
                const float w = (kOne + alpha * v);
                const float wmc = w - C_ptr_[iidx];
                numerator += (w * v - wmc * vf) * p_ptr[d];
                denominator += wmc * p_ptr[d] * p_ptr[d];
                vhat_cache_i_[ind] -= pq;
                vhat_cache_u_[ind_i2u_[ind]] -= pq;
            }
            numerator += -C_ptr_[iidx] * (inner_product(q_ptr, q_ptr + D, &Sp[D * d], kZero) - q_ptr[d] * Sp[D * d + d]);
            denominator += C_ptr_[iidx] * Sp[D * d + d] + reg_i;
            q_ptr[d] = numerator / denominator;
            for (int64_t ind=beg; ind < end; ++ind) {
                const int32_t uidx = keys[ind];
                const float* p_ptr = &P_ptr_[uidx * D];
                const float pq = p_ptr[d] * q_ptr[d];
                vhat_cache_i_[ind] += pq;
                vhat_cache_u_[ind_i2u_[ind]] += pq;
            }
        }
    }
}

}
