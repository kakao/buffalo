#include <string>
#include <numeric>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <sys/time.h>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/algo_impl/bpr/bpr.hpp"


static const float FEPS = 1e-10;
static const int MAX_EXP = 6;

namespace bpr {

struct job_t
{
    int size;
    double alpha;
    vector<vector<int>> samples;

    job_t() : size(0), alpha(0.0)
    {
    }

    void add(const vector<int>& sample)
    {
        samples.push_back(sample);
        size += (int)sample.size();
    }

    int total_samples()
    {
        return size;
    }
};


struct progress_t
{
    int num_sents;
    int num_processed_samples;
    int num_total_samples;
    double loss;
    progress_t(int a, int b, int c) :
        num_sents(a),
        num_processed_samples(b),
        num_total_samples(c) {
    }

    progress_t(int a, int b, int c, double l) :
        num_sents(a),
        num_processed_samples(b),
        num_total_samples(c),
        loss(l) {
    }
};


CBPRMF::CBPRMF() :
    P_(nullptr, 0, 0),
    Q_(nullptr, 0, 0),
    Qb_(nullptr, 0, 0)
{
    build_exp_table();
}

CBPRMF::~CBPRMF()
{
    release();
}

void CBPRMF::release()
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


bool CBPRMF::init(string opt_path) {
    bool ok = true;
    ok = ok & parse_option(opt_path);
    if (ok) {
        int num_workers = opt_["num_workers"].int_value();
        omp_set_num_threads(num_workers);
		optimizer_ = opt_["optimizer"].string_value();
    }
    return ok;
}


bool CBPRMF::parse_option(string opt_path) {
    bool ok = Algorithm::parse_option(opt_path, opt_);
    return ok;
}

void CBPRMF::build_exp_table()
{
    for (int i=0; i < EXP_TABLE_SIZE; ++i) {
        exp_table_[i] = (float)exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
        exp_table_[i] = 1.0 / (exp_table_[i] + 1);
    }
}

void CBPRMF::launch_workers()
{
    int num_workers = opt_["num_workers"].int_value();
    workers_.clear();
    for (int i=0; i < num_workers; ++i) {
        workers_.emplace_back(thread(&CBPRMF::worker, this, i));
    }
    progress_manager_ = new thread(&CBPRMF::progress_manager, this);
}

void CBPRMF::initialize_model(
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

    if (optimizer_ == "adam") {
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

void CBPRMF::initialize_adam_optimizer()
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

void CBPRMF::initialize_sgd_optimizer()
{
    lr_ = opt_["lr"].number_value();
}


void CBPRMF::set_cumulative_table(int64_t* cum_table, int size)
{
    cum_table_ = cum_table;
    cum_table_size_ = size;
}


void CBPRMF::progress_manager()
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

void CBPRMF::add_jobs(
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


void CBPRMF::worker(int worker_id)
{
    bool use_bias = opt_["use_bias"].bool_value();
    bool update_i = opt_["update_i"].bool_value();
    bool update_j = opt_["update_j"].bool_value();

    double reg_u = opt_["reg_u"].number_value();
    double reg_i = opt_["reg_i"].number_value();
    double reg_j = opt_["reg_j"].number_value();
    double reg_b = opt_["reg_b"].number_value();

    mt19937 RNG(opt_["random_seed"].int_value() + worker_id);
    int num_negative_samples = opt_["num_negative_samples"].int_value();
    uniform_int_distribution<int64_t> rng1(0, cum_table_[cum_table_size_ - 1] - 1);
    uniform_int_distribution<int64_t> rng2(0, Q_.rows() - 1);
    double sample_power = opt_["sampling_power"].number_value();
    bool verify_neg = opt_["verify_neg"].bool_value();
    int uniform_sampling = sample_power == 0.0 ? 1 : 0;

    bool per_coordinate_normalize = opt_["per_coordinate_normalize"].bool_value();
    while(true)
    {
        job_t job = job_queue_.pop();
        int processed_samples = 0, total_samples = job.size;
        if(job.size == -1)
            break;

        double alpha = job.alpha;
        for (const auto& _seen : job.samples)
        {
            const int u = _seen[0];
            unordered_set<int> seen(_seen.begin() + 1, _seen.end());
            for(const auto pos : seen)
            {
                for(int i=0; i < num_negative_samples; ++i)
                {
                    int neg = 0;
                    while(true) {
                        if(uniform_sampling) {
                            neg = rng2(RNG);
                        }
                        else {
                            int64_t r = rng1(RNG);
                            neg = (int)(lower_bound(cum_table_, cum_table_ + cum_table_size_, r) - cum_table_);
                        }

                        if (!verify_neg || (seen.find(neg) == seen.end()))
                            break;
                    }

                    float x_uij = (P_.row(u) * (Q_.row(pos) - Q_.row(neg)).transpose())(0, 0);
                    if (use_bias)
                        x_uij += (Qb_(pos, 0) - Qb_(neg, 0));

                    float logit = 0.0;
                    if (MAX_EXP < x_uij) {
                        logit = 0.0;
                    }
                    else if (x_uij < -MAX_EXP) {
                        logit = 1.0;
                    }
                    else {
                        // get 1.0 - logit(f)
                        logit = exp_table_[(int)((x_uij + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
                    }

                    FactorTypeRowMajor item_deriv;
                    if (update_i or update_j)
                        item_deriv = logit * P_.row(u);

                    // TODO: change to enum class
                    if (optimizer_ == "adam") {
                        if (per_coordinate_normalize)  // critical section
                        {
                            #pragma omp atomic
                            Q_samples_per_coordinates_[neg] += 1;
                        }
                        gradP_.row(u) += logit * (Q_.row(pos) - Q_.row(neg));

                        if (update_i) {
                            gradQ_.row(pos) += item_deriv;
                            if (use_bias)
                                gradQb_(pos, 0) += logit;
                        }

                        if (update_j) {
                            gradQ_.row(neg) -= item_deriv;
                            if (use_bias)
                                gradQb_(neg, 0) -= logit;
                        }
                    } else { // sgd
                        auto g = logit * (Q_.row(pos) - Q_.row(neg)) - reg_u * P_.row(u);
                        if (update_i) {
                            Q_.row(pos) += alpha * (item_deriv - reg_i * Q_.row(pos));
                            if (use_bias)
                                Qb_(pos, 0) += alpha * (logit - reg_b * Qb_(pos, 0));
                        }

                        if (update_j) {
                            Q_.row(neg) += alpha * (-item_deriv - reg_j * Q_.row(neg));
                            if (use_bias)
                                Qb_(neg, 0) += alpha * (-logit - reg_b * Qb_(neg, 0));
                        }

                        P_.row(u) += alpha * g;
                    }
                }

                if (optimizer_ == "adam" and per_coordinate_normalize) {
                    P_samples_per_coordinates_[u] += 1;
                    {
                        #pragma omp atomic
                        Q_samples_per_coordinates_[pos] += 1;
                    }
                }
            }
            processed_samples += (int)_seen.size() - 1;
        }
        progress_queue_.push(
                progress_t((int)job.samples.size(), processed_samples, total_samples, 0.0));
	}
}

double CBPRMF::distance(size_t p, size_t q)
{
    bool use_bias = opt_["use_bias"].bool_value();

    float ret = (P_.row(p) * Q_.row(q).transpose())(0, 0);
    if (use_bias)
        ret += Qb_(q, 0);
    return ret;
}

double CBPRMF::compute_loss(int32_t num_loss_samples,
                            int32_t* users,
                            int32_t* positives,
                            int32_t* negatives)
{
    int num_workers = opt_["num_workers"].int_value();
    omp_set_num_threads(num_workers);

    vector<double> loss(num_workers, 0.0);
    #pragma omp parallel for schedule(static)
    for (int idx=0; idx < num_loss_samples; ++idx) {
        int u = users[idx], i = positives[idx], j = negatives[idx];
        double x_uij = distance(u, i) - distance(u, j);
        loss[omp_get_thread_num()] += log(1.0 + exp(-x_uij));
    }
    double l = accumulate(loss.begin(), loss.end(), 0.0);
    return l / (double) num_loss_samples;
}

void CBPRMF::update_adam(
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

void CBPRMF::update_parameters()
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
    } else { // sgd
    }

    iters_ += 1;
}

double CBPRMF::join()
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

void CBPRMF::wait_until_done()
{
    while (job_queue_.get_size() > 0) {
        this_thread::sleep_for(chrono::milliseconds(100));
    }
}

}
