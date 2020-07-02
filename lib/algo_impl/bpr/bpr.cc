#include <string>
#include <numeric>
#include <fstream>
#include <iostream>
#include <streambuf>
#include <sys/time.h>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/algo.hpp"
#include "buffalo/algo_impl/bpr/bpr.hpp"



static const int MAX_EXP = 6;

namespace bpr {

CBPRMF::CBPRMF()
{
}

CBPRMF::~CBPRMF()
{
}

void CBPRMF::release(){
    SGDAlgorithm::release();
}


bool CBPRMF::parse_option(string opt_path) {
    bool ok = Algorithm::parse_option(opt_path, opt_);
    return ok;
}

bool CBPRMF::init(string opt_path){
    bool ok = parse_option(opt_path);
    if(ok) {
        int num_workers = opt_["num_workers"].int_value();
        omp_set_num_threads(num_workers);
        optimizer_ = opt_["optimizer"].string_value();
    }
    return ok;
}
void CBPRMF::initialize_model(
        float* P, int32_t P_rows,
        float* Q, int32_t Q_rows,
        float* Qb,
        int64_t num_total_samples)
{
    SGDAlgorithm::initialize_model(P, P_rows, Q, Q_rows, Qb, num_total_samples);
    build_exp_table();
}

void CBPRMF::build_exp_table()
{
    for (int i=0; i < EXP_TABLE_SIZE; ++i) {
        exp_table_[i] = (float)exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
        exp_table_[i] = 1.0 / (exp_table_[i] + 1);
    }
}


void CBPRMF::set_cumulative_table(int64_t* cum_table, int size)
{
    cum_table_ = cum_table;
    cum_table_size_ = size;
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
                    if (optimizer_ != "sgd") {
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

                if (optimizer_ != "sgd" and per_coordinate_normalize) {
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



void CBPRMF::add_jobs(
            int start_x,
            int next_x,
            int64_t* indptr,
            int32_t* positives){
    SGDAlgorithm::add_jobs(start_x, next_x, indptr, positives);
}

void CBPRMF::wait_until_done(){
    SGDAlgorithm::wait_until_done();
}

void CBPRMF::launch_workers(){
    SGDAlgorithm::launch_workers();
}

void CBPRMF::update_parameters(){
    SGDAlgorithm::update_parameters();
}

double CBPRMF::join(){
    return SGDAlgorithm::join();
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

}
