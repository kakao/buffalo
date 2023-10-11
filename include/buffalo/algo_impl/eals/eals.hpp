#pragma once

#include <omp.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <functional>

#include "json11.hpp"
#include "buffalo/algo.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/misc/blas.hpp"


using namespace Eigen;

namespace eals {
class EALS : public Algorithm {
 public:
    EALS();
    virtual ~EALS();
    virtual bool init(std::string opt_path);
    virtual bool parse_option(std::string opt_path);
    void _leastsquare(Map<MatrixType>& X, int idx, MatrixType& A, VectorType& y) = delete;
    void initialize_model(double* P_ptr,
                          double* Q_ptr,
                          double* C_ptr,
                          const int32_t P_rows,
                          const int32_t Q_rows);
    void precompute_cache(const int32_t nnz,
                          const int64_t* indptr,
                          const int32_t* keys,
                          const int32_t axis);
    bool update(const int64_t* indptr,
                const int32_t* keys,
                const float* vals,
                const int32_t axis);
    std::pair<double, double> estimate_loss(const int32_t nnz,
                                            const int64_t* indptr,
                                            const int32_t* keys,
                                            const float* vals,
                                            const int32_t axis);

 private:
    class IdxCoord {
      public:
        void set(int32_t row, int32_t col, int64_t key) {
            row_ = row;
            col_ = col;
            key_ = key;
        }
        int32_t get_row() const { return row_; }
        int32_t get_col() const { return col_; }
        int64_t get_key() const { return key_; }
      private:
        int32_t row_, col_;
        int64_t key_;
    };

    void update_P_(const int64_t* indptr,
                   const int32_t* keys,
                   const float* vals);
    void update_Q_(const int64_t* indptr,
                   const int32_t* keys,
                   const float* vals);

    Json opt_;
    int32_t P_rows_, Q_rows_;
    bool is_P_cached_, is_Q_cached_;
    double* P_ptr_, * Q_ptr_, * C_ptr_;
    std::vector<double> vhat_cache_u_, vhat_cache_i_;
    std::vector<int64_t> ind_u2i_, ind_i2u_;
    const double kOne, kZero;
};
} // end namespace eals
