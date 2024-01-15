#pragma once

#include <string>
#include <algorithm>


extern "C" {
// blas subroutines
void ssyrk_(const char*, const char*, const int*, const int*, const float*, const float*, const int*,
            const float*, float*, const int*);

void dsyrk_(const char*, const char*, const int*, const int*, const double*, const double*, const int*,
            const double*, double*, const int*);
}

namespace blas {
namespace impl {
inline void syrk(const char uplo,
                 const char trans,
                 const int n,
                 const int k,
                 const float alpha,
                 const float* A,
                 const int lda,
                 const float beta,
                 float* C,
                 const int ldc) {
    ssyrk_(&uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
}

inline void syrk(const char uplo,
                 const char trans,
                 const int n,
                 const int k,
                 const double alpha,
                 const double* A,
                 const int lda,
                 const double beta,
                 double* C,
                 const int ldc) {
    dsyrk_(&uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
}
} // end namespace impl

namespace etc {
template <typename T> T max(const T a, const T b) { return ((a > b) ? a : b); }
template <typename T> T min(const T a, const T b) { return ((a > b) ? b : a); }
template <typename T>
void fill_left_elems(T* A, const int m, const std::string uplo) {
    if (!uplo.compare("u")) {
        for (int i=0; i < (m - 1); ++i) {
            for (int j=(i + 1); j < m; ++j) {
                A[j*m + i] = A[i*m + j];
            }
        }
    } else if (!uplo.compare("l")) {
        for (int i=1; i < m; ++i) {
            for (int j=0; j < i; ++j) {
                A[j*m + i] = A[i*m + j];
            }
        }
    }
}
} // end namespace etc

template <typename T>
void syrk(const std::string uplo,
          const std::string trans,
          const int n,
          const int k,
          const T alpha,
          const T* A,
          const T beta,
          T* C) {
    const char uplo_ = (uplo.c_str()[0] == 'u')? 'l' : 'u';
    const char trans_ = (trans.c_str()[0] == 't')? 'n' : 't';
    const int lda = (trans_ == 'n')? etc::max(1, n) : etc::max(1, k);
    const int ldc = etc::max(1, n);
    impl::syrk(uplo_, trans_, n, k, alpha, A, lda, beta, C, ldc);
    etc::fill_left_elems(C, n, uplo);
}
} // end namespace blas
