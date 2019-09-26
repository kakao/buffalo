#include <string>
#include <fstream>
#include <streambuf>

#include "buffalo/algo.hpp"

using namespace std;
using namespace json11;

Algorithm::Algorithm()
{
    logger_ = BuffaloLogger().get_logger();
}

bool Algorithm::parse_option(string opt_path, Json& j)
{
    ifstream in(opt_path.c_str());
    if (not in.is_open()) {
        INFO("File not exists: {}", opt_path);
        return false;
    }

    string str((std::istreambuf_iterator<char>(in)),
               std::istreambuf_iterator<char>());
    string err_cmt;
    auto _j = Json::parse(str, err_cmt);
    if (not err_cmt.empty()) {
        INFO("Failed to parse: {}", err_cmt);
        return false;
    }
    j = _j;
    return true;
}

void Algorithm::_leastsquare(Map<MatrixType>& X, int idx, MatrixType& A, VectorType& y)
{
    VectorType r, p;
    float rs_old, rs_new, alpha, beta;
    ConjugateGradient<MatrixType, Lower|Upper> cg;
    
    BiCGSTAB<MatrixType> bicg;
    GMRES<MatrixType> gmres;
    DGMRES<MatrixType> dgmres;
    MINRES<MatrixType> minres;
    // use switch statement instead of if statement just for the clarity of the code
    // no performance improvement
    switch (optimizer_code_){
        case 0: // llt
            X.row(idx).noalias() = A.llt().solve(y.transpose());
            break;
        case 1: // ldlt
            X.row(idx).noalias() = A.ldlt().solve(y.transpose());
            break;
        case 2: 
            // manual implementation of conjugate gradient descent
            // no preconditioning
            // thus faster in case of small number of iterations than eigen implementation
            r = y - X.row(idx) * A;
            // in case that current vector is nearer to solution than zero vector, no zero initialization
            if (y.dot(y) < r.dot(r)){
                X.row(idx).setZero(); r = y;
            }
            p = r;
            rs_old = r.dot(r);
            for (int it=0; it<num_cg_max_iters_; ++it){
                alpha = rs_old / ((p * A).dot(p) + eps_);
                X.row(idx).noalias() += alpha * p;
                r.noalias() -= alpha * (p * A);
                rs_new = r.dot(r);
                // stop iteration if rs_new is sufficiently small
                if (rs_new < cg_tolerance_)
                    break;
                beta = rs_new / (rs_old + eps_);
                p.noalias() = r + beta * p;
                rs_old = rs_new;
            }
            break;
        case 3: // eigen implementation of conjugate gradient descent
            cg.setMaxIterations(num_cg_max_iters_).setTolerance(cg_tolerance_).compute(A);
            r = y - X.row(idx) * A;
            // in case that current vector is nearer to solution than zero vector, no zero initialization
            if (y.dot(y) < r.dot(r))
                X.row(idx).noalias() = cg.solve(y.transpose());
            else
                X.row(idx).noalias() = cg.solveWithGuess(y.transpose(), X.row(idx).transpose());
            break;
        case 4: // eigen implementation of BiCGSTAB 
            bicg.setMaxIterations(num_cg_max_iters_).setTolerance(cg_tolerance_).compute(A);
            r = y - X.row(idx) * A;
            // in case that current vector is nearer to solution than zero vector, no zero initialization
            if (y.dot(y) < r.dot(r))
                X.row(idx).noalias() = bicg.solve(y.transpose());
            else
                X.row(idx).noalias() = bicg.solveWithGuess(y.transpose(), X.row(idx).transpose());
            break;
        case 5: // eigen implementation of GMRES
            gmres.setMaxIterations(num_cg_max_iters_).setTolerance(cg_tolerance_).compute(A);
            r = y - X.row(idx) * A;
            // in case that current vector is nearer to solution than zero vector, no zero initialization
            if (y.dot(y) < r.dot(r))
                X.row(idx).noalias() = gmres.solve(y.transpose());
            else
                X.row(idx).noalias() = gmres.solveWithGuess(y.transpose(), X.row(idx).transpose());
            break;
        case 6: // eigen implementation of DGMRES
            dgmres.setMaxIterations(num_cg_max_iters_).setTolerance(cg_tolerance_).compute(A);
            r = y - X.row(idx) * A;
            // in case that current vector is nearer to solution than zero vector, no zero initialization
            if (y.dot(y) < r.dot(r))
                X.row(idx).noalias() = dgmres.solve(y.transpose());
            else
                X.row(idx).noalias() = dgmres.solveWithGuess(y.transpose(), X.row(idx).transpose());
            break;
        case 7: // eigen implementation of MINRES
            minres.setMaxIterations(num_cg_max_iters_).setTolerance(cg_tolerance_).compute(A);
            r = y - X.row(idx) * A;
            // in case that current vector is nearer to solution than zero vector, no zero initialization
            if (y.dot(y) < r.dot(r))
                X.row(idx).noalias() = minres.solve(y.transpose());
            else
                X.row(idx).noalias() = minres.solveWithGuess(y.transpose(), X.row(idx).transpose());
            break;
        default:
            break;
    }
}
