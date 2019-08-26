#pragma once
#include <string>
#include <utility>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>

#include "buffalo/algo.hpp"
using namespace std;
using namespace Eigen;


namespace als {


class CALS : public Algorithm {
public:
    CALS();
    ~CALS();

    void release();
    bool init(string opt_path);
    bool parse_option(string out_path);

    void initialize_model(
            float* P, int P_rows,
            float* Q, int Q_rows);
    void precompute(int axis);
    pair<double, double> partial_update(int start_x,
                                        int next_x,
                                        int64_t* indptr,
                                        int32_t* keys,
                                        float* vals,
                                        int axis);

private:
    Json opt_;
    FactorType FF_;
    Map<FactorTypeRowMajor> P_, Q_;
};

}
