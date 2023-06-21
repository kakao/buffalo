#include <exception>
#include <iterator>
#include <sys/time.h>

#include <string>
#include <numeric>
#include <fstream>
#include <iostream>
#include <streambuf>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/algo_impl/warp/warp.hpp"


static const float FEPS = 1e-10;
static const int MAX_EXP = 6;

namespace warp {

inline float dot_score(const FactorTypeRowMajor u, const FactorTypeRowMajor i) {
    return (u * i.transpose())(0, 0);
}

inline float l2_score(const FactorTypeRowMajor& u, const FactorTypeRowMajor& i) {
    FactorTypeRowMajor diff = (u - i);
    return -(diff * diff.transpose())(0, 0);
}

inline void dot_deriv(const float Phi,
               const FactorTypeRowMajor& u,
               const FactorTypeRowMajor& i,
               const FactorTypeRowMajor& j,
               FactorTypeRowMajor& u_deriv,
               FactorTypeRowMajor& i_deriv,
               FactorTypeRowMajor& j_deriv) {
    u_deriv = Phi * (i - j);
    i_deriv = Phi * u;
    j_deriv = -i_deriv;
}

inline void l2_deriv(const float Phi,
                     const FactorTypeRowMajor& u,
                     const FactorTypeRowMajor& i,
                     const FactorTypeRowMajor& j,
                     FactorTypeRowMajor& u_deriv,
                     FactorTypeRowMajor& i_deriv,
                     FactorTypeRowMajor& j_deriv) {
    u_deriv = Phi * 2 * (i - j);
    i_deriv = Phi * (u - i);
    j_deriv = -Phi * (u -j);
}


CWARP::CWARP() { }

CWARP::~CWARP() { }

bool CWARP::parse_option(string opt_path) {
    bool ok = Algorithm::parse_option(opt_path, opt_);
    return ok;
}



void CWARP::release(){
    SGDAlgorithm::release();
}

bool CWARP::init(string opt_path) {
    bool ok = parse_option(opt_path);
    if (ok) {
        int num_workers = opt_["num_workers"].int_value();
        omp_set_num_threads(num_workers);
        optimizer_ = opt_["optimizer"].string_value();
        auto score_func = opt_["score_func"].string_value();
        if (score_func == "l2") {
            score_f_ = l2_score;
            get_deriv_ = l2_deriv;
        } else {
            score_f_ = dot_score;
            get_deriv_ = dot_deriv;
        }
    }
    return ok;
}

void CWARP::initialize_model(
        float* P, int32_t P_rows,
        float* Q, int32_t Q_rows,
        float* Qb,
        int64_t num_total_samples)
{
    SGDAlgorithm::initialize_model(P, P_rows, Q, Q_rows, Qb, num_total_samples);
}

void CWARP::set_cumulative_table(int64_t* cum_table, int size)
{
    cum_table_ = cum_table;
    cum_table_size_ = size;
}

void CWARP::worker(int worker_id)
{
    int max_trial = opt_["max_trials"].int_value();
    double threshold = opt_["threshold"].number_value();
    double reg_u = opt_["reg_u"].number_value();
    double reg_i = opt_["reg_i"].number_value();
    double reg_j = opt_["reg_j"].number_value();
    mt19937 RNG(opt_["random_seed"].int_value() + worker_id);
    int Q_rows = Q_.rows();

    uniform_int_distribution<int64_t> rng(0, Q_rows - 1);

    bool per_coordinate_normalize = opt_["per_coordinate_normalize"].bool_value();
    while (true)
    {
        job_t job = job_queue_.pop();
        int processed_samples = 0, total_samples = job.size;
        if (job.size == -1)
            break;

        // double alpha = job.alpha;
        FactorTypeRowMajor user_deriv, pos_item_deriv, neg_item_deriv;  // it's costly if initialized for every sample
        double partial_loss = 0.0;
        for (const auto& _seen : job.samples) {
            const int u = _seen[0];
            unordered_set<int> seen(_seen.begin() + 1, _seen.end());


            for (const auto pos : seen) {
                float ui = score_f_(P_.row(u), Q_.row(pos));
                float uj = 0;
                int neg = 0;
                int trial = 1;

                while (trial <= max_trial)
                {
                    neg = rng(RNG);
                    if (seen.find(neg) != seen.end())
                        continue; // False negative.
                    trial += 1;
                    uj = score_f_(P_.row(u), Q_.row(neg));

                    if ((ui - uj) < threshold) //Found a violating pair
                        break;
                    trial += 1;
                }
                if (trial >= max_trial)
                    continue;

                float Phi = log(max(1, int((Q_rows - seen.size() - 1) / trial)));

                get_deriv_(Phi, P_.row(u), Q_.row(pos), Q_.row(neg),
                           user_deriv, pos_item_deriv, neg_item_deriv);
                gradP_.row(u) += user_deriv - reg_u * P_.row(u);
                gradQ_.row(pos) += pos_item_deriv - reg_i * Q_.row(pos);
                gradQ_.row(neg) += neg_item_deriv - reg_j * Q_.row(neg);
                if (per_coordinate_normalize)  // critical section
                {
                    #pragma omp atomic
                    P_samples_per_coordinates_[u] += 1;
                    Q_samples_per_coordinates_[pos] += 1;
                    Q_samples_per_coordinates_[neg] += 1;
                }
                partial_loss += (uj - ui + threshold);
            }
            processed_samples += (int)_seen.size() - 1;
        }
        progress_queue_.push(
                progress_t((int)job.samples.size(), processed_samples, total_samples, partial_loss));
    }
}

void CWARP::add_jobs(
            int start_x,
            int next_x,
            int64_t* indptr,
            int32_t* positives){
    SGDAlgorithm::add_jobs(start_x, next_x, indptr, positives);
}

void CWARP::wait_until_done(){
    SGDAlgorithm::wait_until_done();
}


void CWARP::launch_workers(){
    SGDAlgorithm::launch_workers();
}

void CWARP::update_parameters(){
    SGDAlgorithm::update_parameters();
    #pragma omp parallel for schedule(static)
    for (int i=0; i < Q_.rows(); ++i)
        Q_.row(i) /= max(1.0f, sqrt((Q_.row(i) * Q_.row(i).transpose())(0, 0)));

    #pragma omp parallel for schedule(static)
    for (int u=0; u < P_.rows(); ++u)
        P_.row(u) /= max(1.0f, sqrt((P_.row(u) * P_.row(u).transpose())(0, 0)));
}
double CWARP::join(){
    return SGDAlgorithm::join();
}
double CWARP::compute_loss(int32_t num_loss_samples,
                            int32_t* users,
                            int32_t* positives,
                            int32_t* negatives)
{
    int num_workers = opt_["num_workers"].int_value();
    omp_set_num_threads(num_workers);
    double threshold = opt_["threshold"].number_value();
    vector<int> loss(num_workers, 0);

    #pragma omp parallel for schedule(static)
    for (int idx=0; idx < num_loss_samples; ++idx) {
        int u = users[idx], i = positives[idx], j = negatives[idx];
        double x_ui = score_f_(P_.row(u), Q_.row(i));
        double x_uj = score_f_(P_.row(u), Q_.row(j));
        loss[omp_get_thread_num()] += int((x_ui - x_uj) < threshold);
    }

    double l = double(accumulate(loss.begin(), loss.end(), 0));
    return l / (double) num_loss_samples;
    return 0.0;
}
}
