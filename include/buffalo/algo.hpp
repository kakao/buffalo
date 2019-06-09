#pragma once
#include <string>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"

using namespace std;
using namespace json11;
using namespace Eigen;


class Algorithm  // Algorithm? Logic? Learner?
{
public:
    Algorithm();
    virtual ~Algorithm() {}

    virtual bool init(string opt_path) = 0;
    bool parse_option(string opt_path, Json& j);
    virtual bool parse_option(string opt_path) = 0;
    void decouple(Map<MatrixXf>& mat, float** data, int& rows, int& cols);

    std::shared_ptr<spdlog::logger> logger_;
};
