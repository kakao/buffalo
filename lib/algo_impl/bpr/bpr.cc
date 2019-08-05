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


CBPRMF::CBPRMF()
{
    build_exp_table();
}

CBPRMF::~CBPRMF()
{
    P_data_ = Q_data_ = Qb_data_ = nullptr;
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
}


bool CBPRMF::init(string opt_path) {
    bool ok = true;
    ok = ok & parse_option(opt_path);
    if (ok) {
        int num_workers = opt_["num_workers"].int_value();
        omp_set_num_threads(num_workers);
		optimizer_ = opt_["optimizer"].string_value();

        rng_.seed(opt_["random_seed"].int_value());
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
        Map<MatrixXf>& _P,
        Map<MatrixXf>& _Q,
        Map<MatrixXf>& _Qb,
        int64_t num_total_samples) 
{
    int one = 1;
    decouple(_P, &P_data_, P_rows_, P_cols_);
    decouple(_Q, &Q_data_, Q_rows_, Q_cols_);
    decouple(_Qb, &Qb_data_, Q_rows_, one);

    Map<MatrixXf> P(P_data_, P_rows_, P_cols_);
    Map<MatrixXf> Q(Q_data_, Q_rows_, Q_cols_);
    Map<MatrixXf> Qb(Qb_data_, Q_rows_, one);

    DEBUG("P({} x {}) Q({} x {}) Qb({} x {}) setted.",
            P.rows(), P.cols(),
            Q.rows(), Q.cols(), Qb.rows(), Qb.cols());

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

    gradP_.resize(P_rows_, D);
    gradQ_.resize(Q_rows_, D);

    // currently only adam optimizer can be used
    momentumP_.resize(P_rows_, D);
    momentumP_.setZero();
    momentumQ_.resize(Q_rows_, D);
    momentumQ_.setZero();

    velocityP_.resize(P_rows_, D);
    velocityP_.setZero();
    velocityQ_.resize(Q_rows_, D);
    velocityQ_.setZero();

    if (use_bias) {
        gradQb_.resize(Q_rows_, 1);
        momentumQb_.resize(Q_rows_, 1);
        momentumQb_.setZero();
        velocityQb_.resize(Q_rows_, 1);
        velocityQb_.setZero();
    }
    gradP_.setZero();
    gradQ_.setZero();
    gradQb_.setZero();
    if (opt_["per_coordinate_normalize"].bool_value()) {
        P_samples_per_coordinates_.assign(P_rows_, 0);
        Q_samples_per_coordinates_.assign(Q_rows_, 0);
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

int CBPRMF::get_negative_sample(unordered_set<int>& seen)
{
    uniform_int_distribution<int64_t> rng(0, cum_table_[cum_table_size_ - 1] - 1);
    while (true) {
        int64_t r = rng(rng_);
        int neg = (int)(lower_bound(cum_table_, cum_table_ + cum_table_size_, r) - cum_table_);
        if (seen.find(neg) == seen.end())
            return neg;
    }
}

void CBPRMF::add_jobs(
        int start_x,
        int next_x,
        int64_t* indptr,
        Map<VectorXi>& positives)
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
    Map<MatrixXf> P(P_data_, P_rows_, P_cols_),
                  Q(Q_data_, Q_rows_, Q_cols_),
                  Qb(Qb_data_, Q_rows_, 1);

    bool use_bias = opt_["use_bias"].bool_value();
    bool update_i = opt_["update_i"].bool_value();
    bool update_j = opt_["update_j"].bool_value();

    double reg_u = opt_["reg_u"].number_value();
    double reg_i = opt_["reg_i"].number_value();
    double reg_j = opt_["reg_j"].number_value();
    double reg_b = opt_["reg_b"].number_value();

    int num_negative_samples = opt_["num_negative_samples"].int_value();

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
                    int neg = get_negative_sample(seen);

                    float x_uij = (P.row(u) * (Q.row(pos) - Q.row(neg)).transpose())(0, 0);
                    if (use_bias)
                        x_uij += (Qb(pos, 0) - Qb(neg, 0)); 

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

                    FactorType item_deriv; 
                    if (update_i or update_j)
                        item_deriv = logit * P.row(u);
                    
                    // TODO: change to enum class
                    if (optimizer_ == "adam") {
                        if (per_coordinate_normalize)  // critical section
                        {
                            #pragma omp atomic
                            Q_samples_per_coordinates_[neg] += 1;
                        }
                        gradP_.row(u) += logit * (Q.row(pos) - Q.row(neg));
                        
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
                        auto g = logit * (Q.row(pos) - Q.row(neg)) - reg_u * P.row(u);
                        if (update_i) {
                            Q.row(pos) += alpha * (item_deriv - reg_i * Q.row(pos));
                            if (use_bias)
                                Qb(pos, 0) += (logit - reg_b * Qb(pos, 0));
                        }

                        if (update_j) {
                            Q.row(neg) -= alpha * (item_deriv - reg_j * Q.row(neg));
                            if (use_bias)
                                Qb(pos, 0) -= (logit - reg_b * Qb(neg, 0));
                        }

                        P.row(u) += alpha * g;
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
    Map<MatrixXf> P(P_data_, P_rows_, P_cols_),
                  Q(Q_data_, Q_rows_, Q_cols_),
                  Qb(Qb_data_, Q_rows_, 1);
    bool use_bias = opt_["use_bias"].bool_value();

    float ret = (P.row(p) * Q.row(q).transpose())(0, 0);
    if (use_bias)
        ret += Qb(q, 0);
    return ret;
}

double CBPRMF::compute_loss(Map<VectorXi>& users,
                            Map<VectorXi>& positives,
                            Map<VectorXi>& negatives)
{
    int num_workers = opt_["num_workers"].int_value();

    vector<double> loss(num_workers, 0.0);
    int num_loss_samples = (int)users.rows();
    #pragma omp parallel for schedule(static)
    for (int idx=0; idx < num_loss_samples; ++idx) {
        int u = users[idx], i = positives[idx], j = negatives[idx];
        double x_uij = distance(u, i) - distance(u, j);
        loss[omp_get_thread_num()] += log(1.0 + exp(-x_uij));
    }
    double l = accumulate(loss.begin(), loss.end(), 0.0);
    l /= (double)loss.size();
    return l;
}

void CBPRMF::update_adam(FactorType& grad, FactorType& momentum, FactorType& velocity, int i, double beta1, double beta2)
{
    momentum.row(i) = beta1 * momentum.row(i) + (1.0 - beta1) * grad.row(i);
    velocity.row(i).array() = beta2 * velocity.row(i).array() + (1.0 - beta2) * grad.row(i).array().pow(2.0);
    FactorType m_hat = momentum.row(i) / (1.0 - pow(beta1, iters_ + 1));
    FactorType v_hat = velocity.row(i) / (1.0 - pow(beta2, iters_ + 1));
    grad.row(i).array() = m_hat.array() / (v_hat.array().sqrt() + FEPS);
}

void CBPRMF::update_parameters()
{
    int num_workers = opt_["num_workers"].int_value();

    bool use_bias = opt_["use_bias"].bool_value();
    double reg_u = opt_["reg_u"].number_value();
    double reg_i = opt_["reg_i"].number_value();
    double reg_b = opt_["reg_b"].number_value();

    Map<MatrixXf> P(P_data_, P_rows_, P_cols_),
                  Q(Q_data_, Q_rows_, Q_cols_),
                  Qb(Qb_data_, Q_rows_, 1);

    bool per_coordinate_normalize = (opt_["per_coordinate_normalize"].bool_value());
    if (optimizer_ == "adam") {
        double lr = opt_["lr"].number_value();
        double beta1 = opt_["beta1"].number_value();
        double beta2 = opt_["beta1"].number_value();

        #pragma omp parallel for schedule(static)
        for(int u=0; u < P_rows_; ++u){
            if (per_coordinate_normalize && P_samples_per_coordinates_[u]) {
                gradP_.row(u) /= P_samples_per_coordinates_[u];
            }
            gradP_.row(u) -= (P.row(u) * (2 * reg_u));
            update_adam(gradP_, momentumP_, velocityP_, u, beta1, beta2);
            P.row(u) += (lr * gradP_.row(u));
        }

        #pragma omp parallel for schedule(static)
        for (int i=0; i < Q_rows_; ++i) {
            if (per_coordinate_normalize && Q_samples_per_coordinates_[i]) {
                gradQ_.row(i) /= Q_samples_per_coordinates_[i];
                gradQb_.row(i) /= Q_samples_per_coordinates_[i];
            }
            gradQ_.row(i) -= (Q.row(i) * (2 * reg_u));
            update_adam(gradQ_, momentumQ_, velocityQ_, i, beta1, beta2);
            Q.row(i) += (lr * gradQ_.row(i));

            if (use_bias) {
                gradQb_(i, 0) -= (Qb(i, 0) * (2 * reg_b));
                update_adam(gradQb_, momentumQb_, velocityQb_, i, beta1, beta2);
                Qb(i, 0) += (lr * gradQb_(i, 0));
            }
        }
        if (per_coordinate_normalize) {
            P_samples_per_coordinates_.assign(P_rows_, 0);
            Q_samples_per_coordinates_.assign(Q_rows_, 0);
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


}
