#pragma once
#include <omp.h>
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/IterativeLinearSolvers>
#include <unsupported/Eigen/SparseExtra>
#include <unsupported/Eigen/IterativeSolvers>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"

using namespace std;
using namespace json11;
using namespace Eigen;


typedef Matrix<float, Dynamic, Dynamic, RowMajor> MatrixType;
typedef RowVectorXf VectorType;

typedef Matrix<float, Dynamic, Dynamic, ColMajor> FactorType;
typedef Matrix<float, Dynamic, Dynamic, RowMajor> FactorTypeRowMajor;


class Algorithm  // Algorithm? Logic? Learner?
{
public:
    Algorithm();

    virtual ~Algorithm() {}
    virtual bool init(string opt_path) = 0;
    virtual bool parse_option(string opt_path) = 0;

    bool parse_option(string opt_path, Json& j);
    void _leastsquare(Map<MatrixType>& X, int idx, MatrixType& A, VectorType& y);

public:
    char optimizer_code_ = 0;
    int num_cg_max_iters_ = 3;
    float cg_tolerance_ = 1e-10;
    float eps_ = 1e-10;
    std::shared_ptr<spdlog::logger> logger_;
};

