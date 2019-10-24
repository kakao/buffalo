#pragma once
#include <fstream>
#include <utility>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/random.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"


namespace cuda_bpr{

using namespace json11;
using namespace thrust;

class CuBPR{
public:
    CuBPR();
    ~CuBPR();

    bool init(std::string opt_path); 
    bool parse_option(std::string opt_path, Json& j);
    
    void initialize_model(
            float* P, int P_rows,
            float* Q, float* Qb, int Q_rows, 
            int64_t num_nnz, bool set_gpu);
    void set_placeholder(int64_t* indptr, size_t batch_size);
    void set_cumulative_table(int64_t* sampling_table);
    int get_vdim();
    std::pair<double, double> partial_update(int start_x, 
            int next_x,
            int64_t* indptr,
            int* keys);
    double compute_loss(int num_loss_samples,
            int* users, int* positives, int* negatives);
    void synchronize(bool device_to_host);

public:
    Json opt_;
   
    // feed data place holder
    device_vector<int64_t> indptr_;
    device_vector<int> rows_, keys_; 

    float *hostP_, *hostQ_, *hostQb_;
    device_vector<float> devP_, devQ_, devQb_;
    host_vector<float> hostLoss_;
    device_vector<float> devLoss_;
    device_vector<int> user_, pos_, neg_;
    device_vector<float> dist_;
    int dim_, vdim_, P_rows_, Q_rows_, num_iters_;
    float reg_u_, reg_i_, reg_j_, reg_b_, lr_, min_lr_;
    int64_t num_processed_, num_total_process_;
    bool update_i_, update_j_, use_bias_, compute_loss_;
    int devId_, mp_cnt_, block_cnt_, cores_;
    int64_t rand_seed_;
    bool random_positive_;
    device_vector<default_random_engine> rngs_;

    // sampling option
    bool uniform_dist_, verify_neg_;
    int num_neg_samples_;

    std::shared_ptr<spdlog::logger> logger_;

};

} // namespace cuda_bpr
