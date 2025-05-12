#pragma once

#include <iostream>
#include <string>
#include <algorithm>
#include <cctype>


extern "C" {
// blas subroutines
void ssyrk_(const char*, const char*, const int*, const int*, const float*, const float*, const int*,
            const float*, float*, const int*);

void dsyrk_(const char*, const char*, const int*, const int*, const double*, const double*, const int*,
            const double*, double*, const int*);
}

namespace blas {
namespace impl {
// LSAME function: Compares two characters case-insensitively
// In Fortran BLAS, LSAME(CA, CB) is true if CA is the same letter as CB regardless of case.
// CA and CB are CHARACTER*1.
inline bool lsame(char ca, char cb) {
    return (std::toupper(static_cast<unsigned char>(ca)) == std::toupper(static_cast<unsigned char>(cb)));
}

// XERBLA error handler (basic version)
inline void xerbla(const std::string& srname, int info) {
    std::cerr << "** On entry to " << srname
              << " parameter number " << info
              << " had an illegal value" << std::endl;
    // In a real BLAS library, this might call exit() or throw an exception.
    exit(EXIT_FAILURE);
}

template <typename T>
void csyrk(char uplo, char trans, int n, int k, T alpha,
           const T* a, int lda, T beta, T* c, int ldc) {
    // Parameters
    const T one = 1.0f;
    const T zero = 0.0f;

    // Local Scalars
    T temp;
    int i, info, j, l; // Fortran i, info, j, l (loop counters)
    int nrowa;
    bool upper;

    // Test the input parameters.
    if (lsame(trans, 'N')) {
        nrowa = n;
    } else {
        nrowa = k;
    }
    upper = lsame(uplo, 'U');

    info = 0;
    if (!upper && !lsame(uplo, 'L')) {
        info = 1;
    } else if (!lsame(trans, 'N') &&
               !lsame(trans, 'T') &&
               !lsame(trans, 'C')) {
        info = 2;
    } else if (n < 0) {
        info = 3;
    } else if (k < 0) {
        info = 4;
    } else if (lda < std::max(1, nrowa)) {
        info = 7;
    } else if (ldc < std::max(1, n)) {
        info = 10;
    }

    if (info != 0) {
        xerbla("SSYRK ", info);
    }

    // Quick return if possible.
    if (n == 0 || ((alpha == zero || k == 0) && beta == one)) {
        return;
    }

    // And when alpha.eq.zero.
    if (alpha == zero) {
        if (upper) {
            if (beta == zero) {
                for (j = 0; j < n; ++j) { // Fortran J = 1, N
                    for (i = 0; i <= j; ++i) { // Fortran I = 1, J
                        c[j * ldc + i] = zero; // c(i,j)
                    }
                }
            } else {
                for (j = 0; j < n; ++j) { // Fortran J = 1, N
                    for (i = 0; i <= j; ++i) { // Fortran I = 1, J
                        c[j * ldc + i] = beta * c[j * ldc + i]; // c(i,j)
                    }
                }
            }
        } else { // Lower triangular
            if (beta == zero) {
                for (j = 0; j < n; ++j) { // Fortran J = 1, N
                    for (i = j; i < n; ++i) { // Fortran I = J, N
                        c[j * ldc + i] = zero; // c(i,j)
                    }
                }
            } else {
                for (j = 0; j < n; ++j) { // Fortran J = 1, N
                    for (i = j; i < n; ++i) { // Fortran I = J, N
                        c[j * ldc + i] = beta * c[j * ldc + i]; // c(i,j)
                    }
                }
            }
        }
        return;
    }

    // Start the operations.
    // C++ indexing: array[col_idx * leading_dim + row_idx]
    // Fortran: C(row_idx_1based, col_idx_1based)
    //          A(row_idx_1based, col_idx_1based)
    // Here, C++ loop variables i, j, l are 0-based.
    // i typically corresponds to Fortran row index, j to Fortran column index.

    if (lsame(trans, 'N')) {
        // Form  C := alpha*A*A**T + beta*C.
        // A is N x K (nrowa = N)
        // A(j,l) in Fortran means A[ (l-1)*lda + (j-1) ]
        // A(i,l) in Fortran means A[ (l-1)*lda + (i-1) ]
        if (upper) {
            for (j = 0; j < n; ++j) { // Fortran J = 1, N (column of C)
                if (beta == zero) {
                    for (i = 0; i <= j; ++i) { // Fortran I = 1, J (row of C)
                        c[j * ldc + i] = zero;
                    }
                } else if (beta != one) {
                    for (i = 0; i <= j; ++i) {
                        c[j * ldc + i] *= beta;
                    }
                }
                for (l = 0; l < k; ++l) { // Fortran L = 1, K (column of A, row of A**T)
                    // A is N x K. Fortran A(j,l) corresponds to element at row j, col l.
                    // C++ access: a[l*lda + j] (0-based indices)
                    if (a[l * lda + j] != zero) { // Fortran A(j+1, l+1)
                        temp = alpha * a[l * lda + j];
                        for (i = 0; i <= j; ++i) { // Fortran I = 1, J (row of C)
                            c[j * ldc + i] += temp * a[l * lda + i]; // Fortran A(i+1, l+1)
                        }
                    }
                }
            }
        } else { // Lower triangular
            for (j = 0; j < n; ++j) { // Fortran J = 1, N (column of C)
                if (beta == zero) {
                    for (i = j; i < n; ++i) { // Fortran I = J, N (row of C)
                        c[j * ldc + i] = zero;
                    }
                } else if (beta != one) {
                    for (i = j; i < n; ++i) {
                        c[j * ldc + i] *= beta;
                    }
                }
                for (l = 0; l < k; ++l) { // Fortran L = 1, K
                    if (a[l * lda + j] != zero) { // Fortran A(j+1, l+1)
                        temp = alpha * a[l * lda + j];
                        for (i = j; i < n; ++i) { // Fortran I = J, N (row of C)
                            c[j * ldc + i] += temp * a[l * lda + i]; // Fortran A(i+1, l+1)
                        }
                    }
                }
            }
        }
    } else { // trans == 'T' or 'C'
        // Form  C := alpha*A**T*A + beta*C.
        // A is K x N (nrowa = K)
        // Fortran A(l,i) means A[ (i-1)*lda + (l-1) ]
        // Fortran A(l,j) means A[ (j-1)*lda + (l-1) ]
        if (upper) {
            for (j = 0; j < n; ++j) { // Fortran J = 1, N (column of C, also column of A)
                for (i = 0; i <= j; ++i) { // Fortran I = 1, J (row of C, also column of A)
                    temp = zero;
                    // A is K x N. Fortran A(l,i) corresponds to element at row l, col i.
                    // C++ access: a[i*lda + l] (0-based indices)
                    for (l = 0; l < k; ++l) { // Fortran L = 1, K (row of A)
                        temp += a[i * lda + l] * a[j * lda + l]; // Fortran A(l+1,i+1) * A(l+1,j+1)
                    }
                    if (beta == zero) {
                        c[j * ldc + i] = alpha * temp;
                    } else {
                        c[j * ldc + i] = alpha * temp + beta * c[j * ldc + i];
                    }
                }
            }
        } else { // Lower triangular
            for (j = 0; j < n; ++j) { // Fortran J = 1, N
                for (i = j; i < n; ++i) { // Fortran I = J, N
                    temp = zero;
                    for (l = 0; l < k; ++l) { // Fortran L = 1, K
                        temp += a[i * lda + l] * a[j * lda + l]; // Fortran A(l+1,i+1) * A(l+1,j+1)
                    }
                    if (beta == zero) {
                        c[j * ldc + i] = alpha * temp;
                    } else {
                        c[j * ldc + i] = alpha * temp + beta * c[j * ldc + i];
                    }
                }
            }
        }
    }
}


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
#ifdef BUFFALO_USE_BLAS
    ssyrk_(&uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
#else
    csyrk<float>(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#endif
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
#ifdef BUFFALO_USE_BLAS
    dsyrk_(&uplo, &trans, &n, &k, &alpha, A, &lda, &beta, C, &ldc);
#else
    csyrk<double>(uplo, trans, n, k, alpha, A, lda, beta, C, ldc);
#endif
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
