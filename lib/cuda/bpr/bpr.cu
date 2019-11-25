#include <unistd.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "buffalo/cuda/utils.cuh"
#include "buffalo/cuda/bpr/bpr.hpp"


namespace cuda_bpr{

using std::invalid_argument;
using namespace cuda_buffalo;
using namespace thrust;

static const float MAX_EXP = 6;

__global__ void init_rngs_kernel(default_random_engine* rngs, int rand_seed){
    rngs[blockIdx.x].seed(blockIdx.x + rand_seed);
}

__global__ void fill_rows_kernel(const int start_x, const int next_x,
        const int64_t* indptr, int* rows){
    int64_t shift = start_x == 0? 0: indptr[start_x - 1];
    for(int user=start_x+blockIdx.x; user<next_x; user+=gridDim.x){
        size_t beg = user == 0? 0: indptr[user - 1] - shift;
        size_t end = indptr[user] - shift;
        for(size_t idx=beg; idx<end; ++idx)
            rows[idx] = user;
        // TODO make faster by sorting keys and using binary searching when verifying negative sample
        // sort(thrust::device, keys+beg, keys+end);
    }
}

__global__ void generate_samples_kernel(const int start_x, const int next_x, 
        int* user, int* pos, int* neg,
        const int64_t* indptr, const float* dist, const int* rows, const int* keys, 
        const int num_items, const size_t sample_size, const int num_neg_samples, const bool uniform_dist, 
        const bool verify_neg, default_random_engine* rngs, const bool random_positive){
    // prepare sampling
    int64_t beg = start_x == 0? 0: indptr[start_x - 1];
    int64_t end = indptr[next_x - 1];
    float Z = dist[num_items - 1];

    // set random generator
    default_random_engine& rng = rngs[blockIdx.x];
    uniform_int_distribution<int> item_dist1(0, num_items-1); // item distribution in case of uniform sampling
    uniform_real_distribution<float> item_dist2(0.0, Z); // item distribution in case of multinomial sampling
    uniform_int_distribution<int64_t> pos_dist(0, end-beg-1); // positive sampler
    
    for(int64_t s=blockIdx.x; s<(sample_size*num_neg_samples); s+=gridDim.x){
        int64_t idx;
        if (random_positive) idx = pos_dist(rng); // draw positive index
        else idx = s % sample_size; // straight-forward positive index
        int _user = rows[idx]; // find user
        int _pos = keys[idx]; // get postive key
        // indexes for finding positive keys of target user
        size_t _beg = _user == 0? 0: indptr[_user - 1] - beg; 
        size_t _end = indptr[_user] - beg;
        // negative sampling
        int _neg;
        while (true){
            if (uniform_dist){
                _neg = item_dist1(rng);
            } else{
                float r = item_dist2(rng);
                _neg = upper_bound(thrust::device, dist, dist+num_items-1, r) - dist;
            }
            if (not verify_neg) break;
            bool exit = true;
            for(size_t _idx=_beg; _idx<_end; ++_idx){
                if (_neg == keys[_idx]){
                    exit = false;
                    break;
                }
            }
            if (exit) break;
            /*
            TODO make faster by sorting keys and using binary searching when verifying negative sample
            if (not binary_search(thrust::device, keys+_beg, keys+_end, _neg))
                break;
            */
        }
        // save samples
        user[s] = _user; pos[s] = _pos; neg[s] = _neg;
    }
}

__global__ void update_bpr_kernel(const int dim, const int vdim, 
        float* P, float* Q, float* Qb, float* loss, 
        const int* user, const int* pos, const int* neg, 
        const size_t sample_size, const float lr, const bool compute_loss,
        const float reg_u, const float reg_i, const float reg_j, const float reg_b,
        const bool update_i, const bool update_j, const bool use_bias){
    
    for (size_t s=blockIdx.x; s<sample_size; s+=gridDim.x){
        // take a sample
        const int _user = user[s], _pos = pos[s], _neg = neg[s];

        // target parameter vectors
        float* _P = P + vdim * _user;
        float* _Qp = Q + vdim * _pos;
        float* _Qn = Q + vdim * _neg;

        // compute scores
        float pos_score = dot(_P, _Qp);
        float neg_score = dot(_P, _Qn);
        if (use_bias){
            pos_score += Qb[_pos];
            neg_score += Qb[_neg];
        }
        // prepare computing gradient
        float diff = neg_score - pos_score;
        diff = max(min(diff, MAX_EXP), -MAX_EXP);
        float e = expf(diff);
        float logit = e / (1 + e);
        
        // compute loss
        if (compute_loss and threadIdx.x == 0)
            loss[blockIdx.x] += logf(1 + e);

        if (threadIdx.x < dim){
            // save parameter temporarily before update
            float tmp = _P[threadIdx.x];

            //update user
            atomicAdd(_P + threadIdx.x, lr * (logit * (_Qp[threadIdx.x] - _Qn[threadIdx.x]) - reg_u * tmp));

            //update item
            if (update_i)
                atomicAdd(_Qp + threadIdx.x, lr * (logit * tmp - reg_i * _Qp[threadIdx.x]));
            if (update_j)
                atomicAdd(_Qn + threadIdx.x, lr * (-logit * tmp - reg_j * _Qn[threadIdx.x]));
        }

        //update item bias
        if (threadIdx.x == 0 and use_bias){
            if (update_i)
                atomicAdd(Qb + _pos, lr * (logit - reg_b * Qb[_pos]));
            if (update_j)
                atomicAdd(Qb + _neg, lr * (-logit - reg_b * Qb[_neg]));
        }

        __syncthreads();
    }
}

__global__ void compute_bpr_sample_loss_kernel(const int dim, const int vdim, 
        float* P, float* Q, float* Qb, float* loss, 
        const int* user, const int* pos, const int* neg, 
        const size_t sample_size, const bool use_bias){
    
    for (size_t s=blockIdx.x; s<sample_size; s+=gridDim.x){
        // take a sample
        const int _user = user[s], _pos = pos[s], _neg = neg[s];

        // target parameter vectors
        float* _P = P + vdim * _user;
        float* _Qp = Q + vdim * _pos;
        float* _Qn = Q + vdim * _neg;

        // compute scores
        float pos_score = dot(_P, _Qp);
        float neg_score = dot(_P, _Qn);
        
        if (use_bias){
            pos_score += Qb[_pos];
            neg_score += Qb[_neg];
        }
        float diff = neg_score - pos_score;
        diff = max(min(diff, MAX_EXP), -MAX_EXP);
        float e = expf(diff);
        
        // compute loss
        if (threadIdx.x == 0)
            loss[blockIdx.x] += logf(1 + e);

        __syncthreads();
    }
}

CuBPR::CuBPR(){
    logger_ = BuffaloLogger().get_logger();
    
    num_processed_ = 0;
    CHECK_CUDA(cudaGetDevice(&devId_));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, devId_));
    mp_cnt_ = prop.multiProcessorCount;
    int major = prop.major;
    int minor = prop.minor;
    cores_ = -1;

    switch (major){
        case 2: // Fermi
            if (minor == 1) cores_ = mp_cnt_ * 48;
            else cores_ = mp_cnt_ * 32;
            break;
        case 3: // Kepler
            cores_ = mp_cnt_ * 192;
            break;
        case 5: // Maxwell
            cores_ = mp_cnt_ * 128;
            break;
        case 6: // Pascal
            if (minor == 1) cores_ = mp_cnt_ * 128;
            else if (minor == 0) cores_ = mp_cnt_ * 64;
            else INFO0("Unknown device type");
            break;
        case 7: // Volta
            if (minor == 0) cores_ = mp_cnt_ * 64;
            else INFO0("Unknown device type");
            break;
        default:
            INFO0("Unknown device type"); 
            break;
    }

    if (cores_ == -1) cores_ = mp_cnt_ * 128;
    INFO("cuda device info, major: {}, minor: {}, multi processors: {}, cores: {}",
         major, minor, mp_cnt_, cores_);
}

CuBPR::~CuBPR(){
    hostP_ = nullptr; hostQ_ = nullptr; hostQb_ = nullptr;
}

bool CuBPR::parse_option(std::string opt_path, Json& j){
    std::ifstream in(opt_path.c_str());
    if (not in.is_open()) {
        return false;
    }

    std::string str((std::istreambuf_iterator<char>(in)),
               std::istreambuf_iterator<char>());
    std::string err_cmt;
    auto _j = Json::parse(str, err_cmt);
    if (not err_cmt.empty()) {
        return false;
    }
    j = _j;
    return true;
}

bool CuBPR::init(std::string opt_path){
    // parse options
    bool ok = parse_option(opt_path, opt_);
    if (ok){
        // set options
        compute_loss_ = opt_["compute_loss_on_training"].bool_value();
        dim_ = opt_["d"].int_value();
        num_iters_ = opt_["num_iters"].int_value();
        reg_u_ = opt_["reg_u"].number_value();
        reg_i_ = opt_["reg_i"].number_value();
        reg_j_ = opt_["reg_j"].number_value();
        reg_b_ = opt_["reg_b"].number_value();
        lr_ = opt_["lr"].number_value();
        min_lr_ = opt_["min_lr"].number_value();
        update_i_ = opt_["update_i"].bool_value();
        update_j_ = opt_["update_j"].bool_value();
        use_bias_ = opt_["use_bias"].bool_value();
        rand_seed_ = opt_["rand_seed"].int_value();
        random_positive_ = opt_["random_positive"].bool_value();

        // virtual dimension
        vdim_ = (dim_ / WARP_SIZE) * WARP_SIZE;
        if (dim_ % WARP_SIZE > 0) vdim_ += WARP_SIZE;
      
        // set block count
        block_cnt_ = opt_["hyper_threads"].int_value() * (cores_ / vdim_);
        // get sampling option
        uniform_dist_ = opt_["sampling_power"].number_value() == 0.0;
        num_neg_samples_ = opt_["num_negative_samples"].int_value();
        verify_neg_ = opt_["verify_neg"].bool_value();

        if (compute_loss_){
            hostLoss_.resize(block_cnt_);
            devLoss_.resize(block_cnt_);
        }
        rngs_.resize(block_cnt_ * vdim_);
        init_rngs_kernel<<<block_cnt_*vdim_, 1>>>(raw_pointer_cast(rngs_.data()), rand_seed_);
        
    }
    return ok;
}

void CuBPR::initialize_model(
        float* P, int P_rows,
        float* Q, float* Qb, int Q_rows, 
        int64_t num_total_process, bool set_gpu)
{    
    // initialize parameters and send to gpu memory
    hostP_ = P;
    hostQ_ = Q;
    hostQb_ = Qb;
    P_rows_ = P_rows;
    Q_rows_ = Q_rows;
    num_total_process_ = num_total_process;

    if (set_gpu){
        devP_.resize(P_rows_*vdim_);
        devQ_.resize(Q_rows_*vdim_);
        devQb_.resize(Q_rows_);
        copy(hostP_, hostP_+P_rows*vdim_, devP_.begin());
        copy(hostQ_, hostQ_+Q_rows*vdim_, devQ_.begin());
        copy(hostQb_, hostQb_+Q_rows, devQb_.begin());
    }
}

void CuBPR::set_placeholder(int64_t* indptr, size_t batch_size)
{
    indptr_.resize(P_rows_);
    copy(indptr, indptr+P_rows_, indptr_.begin());
    rows_.resize(batch_size);
    keys_.resize(batch_size);
    user_.resize(batch_size*num_neg_samples_);
    pos_.resize(batch_size*num_neg_samples_);
    neg_.resize(batch_size*num_neg_samples_);
    CHECK_CUDA(cudaDeviceSynchronize());
}

void CuBPR::set_cumulative_table(int64_t* sampling_table)
{
    dist_.resize(Q_rows_);
    // CHECK_CUDA(cudaDeviceSynchronize());
    copy(sampling_table, sampling_table+Q_rows_, dist_.begin());
}


void CuBPR::synchronize(bool device_to_host)
{
    // synchronize parameters between cpu memory and gpu memory
    if (device_to_host){
        copy(devP_.begin(), devP_.end(), hostP_);
        copy(devQ_.begin(), devQ_.end(), hostQ_);
        copy(devQb_.begin(), devQb_.end(), hostQb_);
    } else{
        copy(hostP_, hostP_+P_rows_*vdim_, devP_.begin());
        copy(hostQ_, hostQ_+Q_rows_*vdim_, devQ_.begin());
        copy(hostQb_, hostQb_+Q_rows_, devQb_.begin());
    }

    CHECK_CUDA(cudaDeviceSynchronize());
}

int CuBPR::get_vdim(){
    return vdim_;
}

std::pair<double, double> CuBPR::partial_update(int start_x, 
        int next_x,
        int64_t* indptr,
        int* keys){
    
    double el;
    time_p beg_t, end_t;

    // copy data to gpu memory
    int64_t beg = start_x == 0? 0: indptr[start_x - 1];
    int64_t end = indptr[next_x - 1];
    size_t sample_size = end - beg;
    copy(keys, keys+sample_size, keys_.begin());

    // set zeros for measuring losses
    if (compute_loss_)
        fill(devLoss_.begin(), devLoss_.end(), 0.0); 
    CHECK_CUDA(cudaDeviceSynchronize());
    
    beg_t = get_now();
     
    // generate samples
    fill_rows_kernel<<<block_cnt_*vdim_, 1>>>(start_x, next_x,
            raw_pointer_cast(indptr_.data()), raw_pointer_cast(rows_.data()));
    CHECK_CUDA(cudaDeviceSynchronize());
    
    end_t = get_now();
    el = (GetTimeDiff(beg_t, end_t)) * 1000.0;
    TRACE("elapsed for filling rows: {} ms", el);

    beg_t = get_now();
    
    // generate samples
    generate_samples_kernel<<<block_cnt_*vdim_, 1>>>(start_x, next_x, 
            raw_pointer_cast(user_.data()), 
            raw_pointer_cast(pos_.data()), raw_pointer_cast(neg_.data()),
            raw_pointer_cast(indptr_.data()), raw_pointer_cast(dist_.data()), 
            raw_pointer_cast(rows_.data()), raw_pointer_cast(keys_.data()), 
            Q_rows_, sample_size, num_neg_samples_, uniform_dist_, 
            verify_neg_, raw_pointer_cast(rngs_.data()), random_positive_);
    CHECK_CUDA(cudaDeviceSynchronize());
    
    end_t = get_now();
    el = (GetTimeDiff(beg_t, end_t)) * 1000.0;
    TRACE("elapsed for sampling: {} ms", el);

    beg_t = get_now();
   
    // decay lr
    double progressed = (double) num_processed_ / ((double) num_total_process_ * (double) num_iters_);
    double alpha = lr_ + (min_lr_ - lr_) * progressed;
    alpha = fmax(min_lr_, alpha);
    
    // update bpr
    update_bpr_kernel<<<block_cnt_, vdim_>>>(dim_, vdim_, 
            raw_pointer_cast(devP_.data()), raw_pointer_cast(devQ_.data()), 
            raw_pointer_cast(devQb_.data()), raw_pointer_cast(devLoss_.data()), 
            raw_pointer_cast(user_.data()), 
            raw_pointer_cast(pos_.data()), raw_pointer_cast(neg_.data()), 
            sample_size*num_neg_samples_, alpha, compute_loss_,
            reg_u_, reg_i_, reg_j_, reg_b_,
            update_i_, update_j_, use_bias_);
    CHECK_CUDA(cudaDeviceSynchronize());
    num_processed_ += sample_size;
   

    end_t = get_now();
    el = (GetTimeDiff(beg_t, end_t)) * 1000.0;
    TRACE("elapsed for update: {} ms", el);
   
    // accumulate losses
    double loss = 0;
    if (compute_loss_){
        copy(devLoss_.begin(), devLoss_.end(), hostLoss_.begin());
        for (auto l: hostLoss_)
            loss += l;
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    return std::make_pair(loss, sample_size*num_neg_samples_);
}

double CuBPR::compute_loss(int num_loss_samples, 
        int* user, int* pos, int* neg){
    
    fill(devLoss_.begin(), devLoss_.end(), 0.0); 
    CHECK_CUDA(cudaDeviceSynchronize());

    // copy sample data to gpu memory
    copy(user, user+num_loss_samples, user_.begin());
    copy(pos, pos+num_loss_samples, pos_.begin());
    copy(neg, neg+num_loss_samples, neg_.begin());

    // update bpr
    compute_bpr_sample_loss_kernel<<<block_cnt_, vdim_>>>(dim_, vdim_, 
            raw_pointer_cast(devP_.data()), raw_pointer_cast(devQ_.data()), 
            raw_pointer_cast(devQb_.data()), raw_pointer_cast(devLoss_.data()), 
            raw_pointer_cast(user_.data()), 
            raw_pointer_cast(pos_.data()), raw_pointer_cast(neg_.data()), 
            num_loss_samples, use_bias_);
    CHECK_CUDA(cudaDeviceSynchronize());
   
    // accumulate losses
    double loss = 0;
    copy(devLoss_.begin(), devLoss_.end(), hostLoss_.begin());
    for (auto l: hostLoss_)
        loss += l;
    CHECK_CUDA(cudaDeviceSynchronize());
    return loss / (double) num_loss_samples;
}

} // namespace cuda_bpr
