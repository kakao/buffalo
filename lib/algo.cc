#include <string>
#include <fstream>
#include <streambuf>
#include <sys/time.h>

#include "buffalo/algo.hpp"

using namespace std;
using namespace json11;
static const float FEPS = 1e-10;


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

SGDAlgorithm::SGDAlgorithm() :
    P_(nullptr, 0, 0),
    Q_(nullptr, 0, 0),
    Qb_(nullptr, 0, 0)
{

}

SGDAlgorithm::~SGDAlgorithm()
{
    release();
}

void SGDAlgorithm::initialize_model(
        float* P, int32_t P_rows,
        float* Q, int32_t Q_rows,
        float* Qb,
        int64_t num_total_samples)
{
    int one = 1;
    int D = opt_["d"].int_value();

    new (&P_) Map<MatrixType>(P, P_rows, D);
    new (&Q_) Map<MatrixType>(Q, Q_rows, D);
    new (&Qb_) Map<MatrixType>(Qb, Q_rows, one);

    DEBUG("P({} x {}) Q({} x {}) Qb({} x {}) setted.",
            P_.rows(), P_.cols(),
            Q_.rows(), Q_.cols(), Qb_.rows(), Qb_.cols());

    if (optimizer_ != "sgd") {
        initialize_adam_optimizer();
    }
    else {
        initialize_sgd_optimizer();
    }
    DEBUG("Optimizer({}).", optimizer_);

    iters_ = 0;

    int num_iters = opt_["num_iters"].int_value();
    total_processed_ = (double)num_total_samples * num_iters;
}

void SGDAlgorithm::release()
{

    gradP_.resize(0, 0);
    gradQ_.resize(0, 0);
    gradQb_.resize(0, 0);
    momentumP_.resize(0, 0);
    momentumQ_.resize(0, 0);
    momentumQb_.resize(0, 0);
    velocityP_.resize(0, 0);
    velocityQ_.resize(0, 0);
    velocityQb_.resize(0, 0);
    P_samples_per_coordinates_.clear();
    P_samples_per_coordinates_.assign(1, 0);
    Q_samples_per_coordinates_.clear();
    Q_samples_per_coordinates_.assign(1, 0);

    new (&P_) Map<MatrixType>(nullptr, 0, 0);
    new (&Q_) Map<MatrixType>(nullptr, 0, 0);
    new (&Qb_) Map<MatrixType>(nullptr, 0, 0);
}

bool SGDAlgorithm::init(string opt_path) {
    bool ok = true;
    ok = ok & parse_option(opt_path);
    if (ok) {
        int num_workers = opt_["num_workers"].int_value();
        omp_set_num_threads(num_workers);
		optimizer_ = opt_["optimizer"].string_value();
    }
    return ok;
}


void SGDAlgorithm::launch_workers()
{
    int num_workers = opt_["num_workers"].int_value();
    workers_.clear();
    for (int i=0; i < num_workers; ++i) {
        workers_.emplace_back(thread(&SGDAlgorithm::worker, this, i));
    }
    progress_manager_ = new thread(&SGDAlgorithm::progress_manager, this);
}

void SGDAlgorithm::initialize_adam_optimizer()
{
    int D = opt_["d"].int_value();
    bool use_bias = opt_["use_bias"].bool_value();

    gradP_.resize(P_.rows(), D);
    gradQ_.resize(Q_.rows(), D);

    // currently only adam optimizer can be used
    momentumP_.resize(P_.rows(), D);
    momentumP_.setZero();
    momentumQ_.resize(Q_.rows(), D);
    momentumQ_.setZero();

    velocityP_.resize(P_.rows(), D);
    velocityP_.setZero();
    velocityQ_.resize(Q_.rows(), D);
    velocityQ_.setZero();

    if (use_bias) {
        gradQb_.resize(Q_.rows(), 1);
        momentumQb_.resize(Q_.rows(), 1);
        momentumQb_.setZero();
        velocityQb_.resize(Q_.rows(), 1);
        velocityQb_.setZero();
    }
    gradP_.setZero();
    gradQ_.setZero();
    gradQb_.setZero();
    if (opt_["per_coordinate_normalize"].bool_value()) {
        P_samples_per_coordinates_.assign(P_.rows(), 0);
        Q_samples_per_coordinates_.assign(Q_.rows(), 0);
    }
}

void SGDAlgorithm::initialize_sgd_optimizer()
{
    lr_ = opt_["lr"].number_value();
}

void SGDAlgorithm::progress_manager()
{
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    double alpha = opt_["lr"].number_value();
    double min_alpha = opt_["min_lr"].number_value();
    lr_ = alpha;
    double loss = 0.0;

    long long total_processed_samples = 0;
    long long processed_samples = 0;
    const double every_secs = 5.0;
    while(true)
    {
        progress_t p = progress_queue_.pop();
        if(p.num_sents == -1 && p.num_processed_samples == -1)
            break;

        total_processed_samples += p.num_total_samples;
        processed_samples += p.num_processed_samples;
        loss += p.loss;

        double progress = total_processed_samples / total_processed_;
        double new_alpha = alpha - (alpha - min_alpha) * progress;
        new_alpha = max(new_alpha, min_alpha);
        lr_ = new_alpha;

        gettimeofday(&end_time, NULL);
        double elapsed = ((end_time.tv_sec  - start_time.tv_sec) * 1000000u + end_time.tv_usec - start_time.tv_usec) / 1.e6;
        if (elapsed >= every_secs) {
            loss /= processed_samples;
            int sps = processed_samples / every_secs;
            if (optimizer_ == "adam") {
                INFO("Progress({:0.2f}{}): TrainingLoss({}) {} samples/s",
                    progress * 100.0, "%", loss, sps);
            }
            else {
                INFO("Progress({:0.2f}{}): TrainingLoss({}) Decayed learning rate {:0.6f}, {} samples/s",
                    progress * 100.0, "%", loss, lr_, sps);
            }
            loss = 0.0;
            processed_samples = 0;
            gettimeofday(&start_time, NULL);
        }
    }
}

void SGDAlgorithm::add_jobs(
        int start_x,
        int next_x,
        int64_t* indptr,
        int32_t* positives)
{
    if( (next_x - start_x) == 0) {
        WARN0("No data to process");
        return;
    }

    int batch_size = opt_["batch_size"].int_value();
    if (batch_size < 0)
        batch_size = 10000;

    job_t job;
    int job_size = 0;

    int end_loop = next_x - start_x;
    const int64_t shifted = start_x == 0 ? 0 : indptr[start_x - 1];
    vector<int> S;
    for (int i=0; i < end_loop; ++i)
    {
        int x = start_x + i;
        const int u = x;
        int64_t beg = x == 0 ? 0 : indptr[x - 1];
        int64_t end = indptr[x];
        int64_t data_size = end - beg;
        if (data_size == 0) {
            TRACE("No data exists for {}", u);
            continue;
        }

        S.push_back(u);
        for (int64_t idx=0, it = beg; it < end; ++it, ++idx)
            S.push_back(positives[it - shifted]);

        if (data_size + job_size <= batch_size) {
            job.add(S);
            job_size += (int)S.size();
        } else {
            job.alpha = lr_;
            job_queue_.push(job);
            job = job_t();
            job.add(S);
            job_size = (int)S.size();
        }
        S.clear();
	}

    if (job.size) {
        job.alpha = lr_;
        job_queue_.push(job);
    }
}


void SGDAlgorithm::update_adam(
        FactorTypeRowMajor& grad,
        FactorTypeRowMajor& momentum,
        FactorTypeRowMajor& velocity, int i, double beta1, double beta2)
{
    momentum.row(i) = beta1 * momentum.row(i) + (1.0 - beta1) * grad.row(i);
    velocity.row(i).array() = beta2 * velocity.row(i).array() + (1.0 - beta2) * grad.row(i).array().pow(2.0);
    FactorTypeRowMajor m_hat = momentum.row(i) / (1.0 - pow(beta1, iters_ + 1));
    FactorTypeRowMajor v_hat = velocity.row(i) / (1.0 - pow(beta2, iters_ + 1));
    grad.row(i).array() = m_hat.array() / (v_hat.array().sqrt() + FEPS);
}

void SGDAlgorithm::update_adagrad(FactorTypeRowMajor& grad,
        FactorTypeRowMajor& velocity, int i){
    velocity.row(i).array() = velocity.row(i).array() + grad.row(i).array().pow(2.0);
    grad.row(i).array() = grad.row(i).array() / (velocity.row(i).array().sqrt() + FEPS);
}

void SGDAlgorithm::update_parameters()
{
    int num_workers = opt_["num_workers"].int_value();
    omp_set_num_threads(num_workers);

    bool use_bias = opt_["use_bias"].bool_value();
    double reg_u = opt_["reg_u"].number_value();
    double reg_i = opt_["reg_i"].number_value();
    double reg_b = opt_["reg_b"].number_value();

    bool per_coordinate_normalize = (opt_["per_coordinate_normalize"].bool_value());
    if (optimizer_ == "adam") {
        double lr = opt_["lr"].number_value();
        double beta1 = opt_["beta1"].number_value();
        double beta2 = opt_["beta1"].number_value();

        #pragma omp parallel for schedule(static)
        for(int u=0; u < P_.rows(); ++u){
            if (per_coordinate_normalize && P_samples_per_coordinates_[u]) {
                gradP_.row(u) /= P_samples_per_coordinates_[u];
            }
            gradP_.row(u) -= (P_.row(u) * (2 * reg_u));
            update_adam(gradP_, momentumP_, velocityP_, u, beta1, beta2);
            P_.row(u) += (lr * gradP_.row(u));
        }

        #pragma omp parallel for schedule(static)
        for (int i=0; i < Q_.rows(); ++i) {
            if (per_coordinate_normalize && Q_samples_per_coordinates_[i]) {
                gradQ_.row(i) /= Q_samples_per_coordinates_[i];
                gradQb_.row(i) /= Q_samples_per_coordinates_[i];
            }
            gradQ_.row(i) -= (Q_.row(i) * (2 * reg_i));
            update_adam(gradQ_, momentumQ_, velocityQ_, i, beta1, beta2);
            Q_.row(i) += (lr * gradQ_.row(i));

            if (use_bias) {
                gradQb_(i, 0) -= (Qb_(i, 0) * (2 * reg_b));
                update_adam(gradQb_, momentumQb_, velocityQb_, i, beta1, beta2);
                Qb_(i, 0) += (lr * gradQb_(i, 0));
            }
        }
        if (per_coordinate_normalize) {
            P_samples_per_coordinates_.assign(P_.rows(), 0);
            Q_samples_per_coordinates_.assign(Q_.rows(), 0);
        }
    } else if(optimizer_ == "adagrad"){
        double lr = opt_["lr"].number_value();
        #pragma omp parallel for schedule(static)
        for(int u=0; u < P_.rows(); ++u){
            if (per_coordinate_normalize && P_samples_per_coordinates_[u]) {
                gradP_.row(u) /= P_samples_per_coordinates_[u];
            }
            gradP_.row(u) -= (P_.row(u) * (2 * reg_u));
            update_adagrad(gradP_, velocityP_, u);
            P_.row(u) += (lr * gradP_.row(u));
        }

        #pragma omp parallel for schedule(static)
        for (int i=0; i < Q_.rows(); ++i) {
            if (per_coordinate_normalize && Q_samples_per_coordinates_[i]) {
                gradQ_.row(i) /= Q_samples_per_coordinates_[i];
                gradQb_.row(i) /= Q_samples_per_coordinates_[i];
            }
            gradQ_.row(i) -= (Q_.row(i) * (2 * reg_i));
            update_adagrad(gradQ_, velocityQ_, i);
            Q_.row(i) += (lr * gradQ_.row(i));

            if (use_bias) {
                gradQb_(i, 0) -= (Qb_(i, 0) * (2 * reg_b));
                update_adagrad(gradQb_, velocityQb_, i);
                Qb_(i, 0) += (lr * gradQb_(i, 0));
            }
        }
        if (per_coordinate_normalize) {
            P_samples_per_coordinates_.assign(P_.rows(), 0);
            Q_samples_per_coordinates_.assign(Q_.rows(), 0);
        }

    } else if (optimizer_ == "SGD"){ // sgd
    }

    iters_ += 1;
}

void SGDAlgorithm::wait_until_done()
{
    while (job_queue_.get_size() > 0) {
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

double SGDAlgorithm::join()
{
    int num_workers = opt_["num_workers"].int_value();
    for (int i=0; i < num_workers; ++i) {
        job_t job;
        job.size = -1;
        job_queue_.push(job);
    }

    for (auto& t : workers_) {
        t.join();
    }
    progress_queue_.push(progress_t(-1, -1, -1));
    progress_manager_->join();
    delete progress_manager_;
    progress_manager_ = nullptr;
    workers_.clear();
    return 0.0;
}
