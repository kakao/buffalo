#include <set>
#include <map>
#include <thread>
#include <vector>
#include <random>
#include <iterator>
#include <sys/time.h>
#include <unordered_map>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/algo_impl/w2v/w2v.hpp"

static const float EPS = 1e-8;
static const int MAX_EXP = 6;

namespace w2v
{


struct job_t
{
    int size;
    double alpha;
    vector<vector<int>> sents;

    job_t() : size(0), alpha(0.0)
    {
    }

    void add(const vector<int>& sent)
    {
        sents.push_back(sent);
        size += (int)sent.size();
    }

    int total_words()
    {
        return size;
    }
};


struct progress_t
{
    int num_sents;
    int num_processed_words;
    int num_total_words;
    double loss;
    progress_t(int a, int b, int c) :
        num_sents(a),
        num_processed_words(b),
        num_total_words(c) {
    }

    progress_t(int a, int b, int c, double l) :
        num_sents(a),
        num_processed_words(b),
        num_total_words(c),
        loss(l) {
    }
};


CW2V::CW2V() :
    L0_(nullptr, 0, 0)
{
    build_exp_table();
    progress_manager_ = nullptr;
}

CW2V::~CW2V()
{
    release();
}


void CW2V::release()
{
    new (&L0_) Map<MatrixType>(nullptr, 0, 0);
    L1_.resize(0, 0);
}

bool CW2V::init(string opt_path) {
    bool ok = true;
    ok = ok & parse_option(opt_path);
    if (ok) {
        int num_workers = opt_["num_workers"].int_value();
        omp_set_num_threads(num_workers);
        processed_.assign(num_workers, 0.0);
        job_queue_.set_max_size(3 * num_workers);
    }
    return ok;
}


bool CW2V::parse_option(string opt_path) {
    bool ok = Algorithm::parse_option(opt_path, opt_);
    return ok;
}


void CW2V::initialize_model(
        float* L0, int32_t L0_rows,
        int32_t* index, uint32_t* scale, int32_t* dist,
        int64_t total_word_count)
{
    int D = opt_["d"].int_value();
    new (&L0_) Map<FactorTypeRowMajor>(L0, L0_rows, D);
    L1_.resize(L0_.rows(), L0_.cols());
    L1_.setZero();
    index_ = index;
    scale_ = scale;
    dist_ = dist;

    int num_iters = opt_["num_iters"].int_value();
    total_processed_ = (double)total_word_count * num_iters;

    INFO("TotalWords({}) x Iteration({})", total_word_count, num_iters);
    INFO("L0({} x {}) L1({} x {})", L0_.rows(), L0_.cols(), L1_.rows(), L1_.cols());
}

void CW2V::build_exp_table()
{
    for (int i=0; i < EXP_TABLE_SIZE; ++i) {
        exp_table_[i] = (float)exp((i / (float)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP);
        exp_table_[i] = exp_table_[i] / (exp_table_[i] + 1);
    }
}

void CW2V::launch_workers()
{
    int num_workers = opt_["num_workers"].int_value();
    workers_.clear();
    for (int i=0; i < num_workers; ++i) {
        workers_.emplace_back(thread(&CW2V::worker, this, i));
    }
    progress_manager_ = new thread(&CW2V::progress_manager, this);
}


void CW2V::add_jobs(
        int start_x,
        int next_x,
        int64_t* indptr,
        int32_t* sequences)
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
    const int64_t shifted = start_x == 0 ? 0 : indptr[start_x - 1];
    int end_loop = next_x - start_x;
    vector<int> W;
    for (int loop=0; loop < end_loop; ++loop)
    {
        int x = start_x + loop;
        int64_t beg = x == 0 ? 0 : indptr[x - 1];
        int64_t end = indptr[x];
        int64_t data_size = end - beg;

        if (data_size == 0) {
            TRACE0("No data exists");
            continue;
        }

        for (int64_t it = beg; it < end; ++ it)
            W.push_back(sequences[it - shifted]);

        if (data_size + job_size <= batch_size) {
            job.add(W);
            job_size += (int)W.size();
        } else {
            job.alpha = alpha_;
            job_queue_.push(job);
            job = job_t();
            job.add(W);
            job_size = (int)W.size();
        }
        W.clear();
    }
    if (job.size) {
        job.alpha = alpha_;
        job_queue_.push(job);
    }
}


void CW2V::worker(int worker_id)
{
    int window_size = opt_["window"].int_value();
    int num_negatives = opt_["num_negative_samples"].int_value();
    int random_seed = opt_["random_seed"].int_value();
    bool compute_loss = opt_["compute_loss_on_training"].bool_value();
    int vocab_size = L0_.rows();

    vector<int> N;
    vector<int> W;
    Array<float, 1, Dynamic, RowMajor> work;
    work.resize(L0_.cols()); work.setZero();

    mt19937 RNG(random_seed + worker_id);
    uniform_int_distribution<unsigned int> rng1(0, 0xFFFFFFFF);
    uniform_int_distribution<unsigned int> rng2(0, window_size - 1);
    uniform_int_distribution<int> rng3(0, dist_[vocab_size - 1] - 1);

    while(true)
    {
        job_t job = job_queue_.pop();
        int processed_words = 0, total_words = job.size;
        if(job.size == -1)
            break;

        double loss = 0.0;
        for (const auto& sent : job.sents)
        {
            W.clear();
            for (const int& w: sent)
            {
                if (!index_[w])
                    continue;
                int word_index = index_[w] - 1;
                if (scale_[word_index] <= rng1(RNG))
                    continue;
                W.push_back(word_index);
            }

            int n = W.size();
            for (int i=0; i <  n; ++i) {
                int reduced_window = rng2(RNG);
                int start = max(0, i - window_size + reduced_window);
                int end = min(n, i + window_size + 1 - reduced_window);
                int target_word_idx = W[i];
                for (int j=start; j < end; ++j) {
                    if (i == j)
                        continue;
                    N.clear();
                    N.push_back(target_word_idx);
                    while ((int)N.size() < num_negatives + 1)
                    {
                        int neg_word_idx = (int)(lower_bound(
                                    dist_,
                                    dist_ + L0_.rows(),
                                    rng3(RNG)) - dist_);
                        if (neg_word_idx != target_word_idx)
                            N.push_back(neg_word_idx);
                    }
                    loss += update_parameter(
                            work,
                            job.alpha,
                            W[j],
                            N,
                            compute_loss);
                }
            }
            processed_words += n;
        }
        // DEBUG("processed_words {} total_words {} loss {}", processed_words, total_words, loss);
        progress_queue_.push(
                progress_t((int)job.sents.size(), processed_words, total_words, loss));
    }
}


double CW2V::update_parameter(
        Array<float, 1, Dynamic, RowMajor>& work,
        double alpha,
        int input_word_idx,
        vector<int>& negatives,
        bool compute_loss)
{
    VectorXf l0 = L0_.row(input_word_idx);
    int num_negatives = (int)negatives.size();
    work.setZero();

    double loss = 0.0;

    float label = 1.0; // first index always target word index
    for(int i=0; i < num_negatives; ++i){
        const int neg_word_idx = negatives[i];
        const auto& row = L1_.row(neg_word_idx);
        float f = row * l0;
        float g = 0.0;

        if (MAX_EXP < f) {
            g = (label - 1.0);
        }
        else if (f < -MAX_EXP) {
            g = label;
        }
        else {
            // get 1.0 - logit(f)
            f = exp_table_[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
            g = (label - f);
        }

        if ( compute_loss ) {
            if ( i == 0 )
                loss -= log(g + EPS);
            else
                loss -= log(1.0 - g + EPS);
        }

        g *= alpha;

        work += g * row.array();
        L1_.row(neg_word_idx).array() += g * l0.array();

        label = 0.0;
    }
    L0_.row(input_word_idx).array() += work;
    return loss;
}

void CW2V::progress_manager()
{
    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    double alpha = opt_["lr"].number_value();
    double min_alpha = opt_["min_lr"].number_value();
    alpha_ = alpha;
    double loss = 0.0;

    long long total_processed_words = 0;
    long long processed_words = 0;
    while(true)
    {
        progress_t p = progress_queue_.pop();
        if(p.num_sents == -1 && p.num_processed_words == -1)
            break;

        total_processed_words += p.num_total_words;
        processed_words += p.num_processed_words;
        loss += p.loss;

        double progress = total_processed_words / total_processed_;
        double new_alpha = alpha - (alpha - min_alpha) * progress;
        new_alpha = max(new_alpha, min_alpha);
        alpha_ = new_alpha;

        gettimeofday(&end_time, NULL);
        double elapsed = ((end_time.tv_sec  - start_time.tv_sec) * 1000000u + end_time.tv_usec - start_time.tv_usec) / 1.e6;
        if (elapsed >= 5.0) {
            loss /= processed_words;
            int wps = processed_words / 5.0;
            DEBUG("Progress({:0.2f}{}): TrainingLoss({}) Decayed learning rate {:0.6f}, {} words/s",
                  progress * 100.0, "%", loss, alpha_, wps);
            loss = 0.0;
            processed_words = 0;
            gettimeofday(&start_time, NULL);
        }
    }
}


double CW2V::join()
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
