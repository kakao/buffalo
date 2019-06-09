#include <string>
#include <fstream>
#include <streambuf>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/algo_impl/als/als.hpp"

namespace als {

CALS::CALS()
{

}

CALS::~CALS()
{

}

bool CALS::init(string opt_path) {
    bool ok = true;
    ok = ok & parse_option(opt_path);
    return ok;
}


bool CALS::parse_option(string opt_path) {
    bool ok = Algorithm::parse_option(opt_path, opt_);
    return ok;
}

void CALS::set_factors(Map<MatrixXf>& _P, Map<MatrixXf>& _Q) {
    decouple(_P, &P_data_, P_rows_, P_cols_);
    decouple(_Q, &Q_data_, Q_rows_, Q_cols_);

    Map<MatrixXf> P(P_data_, P_rows_, P_cols_);
    Map<MatrixXf> Q(Q_data_, Q_rows_, Q_cols_);

    DEBUG("P({} x {}) Q({} x {}) setted", P.rows(), P.cols(), Q.rows(), Q.cols());
}

}
