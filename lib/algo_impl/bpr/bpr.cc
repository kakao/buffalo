#include <string>
#include <numeric>
#include <fstream>
#include <iostream>
#include <streambuf>

#include "json11.hpp"
#include "buffalo/misc/log.hpp"
#include "buffalo/algo_impl/bpr/bpr.hpp"


const float EXPLB = -20.0;
const float EXPUB = 20.0;
const float FEPS = 1e-10;


namespace bpr {


CBPRMF::CBPRMF()
{
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
    }
    return ok;
}


bool CBPRMF::parse_option(string opt_path) {
    bool ok = Algorithm::parse_option(opt_path, opt_);
    return ok;
}

void CBPRMF::set_factors(
        Map<MatrixXf>& _P,
        Map<MatrixXf>& _Q, Map<MatrixXf>& _Qb) 
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
    total_samples_ = 0;
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
    P_samples_per_coordinates_.assign(P_rows_, 0);
    Q_samples_per_coordinates_.assign(Q_rows_, 0);
}

void CBPRMF::initialize_sgd_optimizer()
{
    lr_ = opt_["lr"].number_value();
}

double CBPRMF::partial_update(
        int start_x,
        int next_x,
        int64_t* indptr,
        Map<VectorXi>& positives,
        Map<VectorXi>& negatives)
{
    if( (next_x - start_x) == 0) {
        // WARN0("No data to process");
        return 0.0;
    }

    total_samples_ += negatives.rows();

    Map<MatrixXf> P(P_data_, P_rows_, P_cols_),
                  Q(Q_data_, Q_rows_, Q_cols_),
                  Qb(Qb_data_, Q_rows_, 1);

    int num_workers = opt_["num_workers"].int_value();
    bool use_bias = opt_["use_bias"].bool_value();
    bool update_i = opt_["update_i"].bool_value();
    bool update_j = opt_["update_j"].bool_value();

    double reg_u = opt_["reg_u"].number_value();
    double reg_i = opt_["reg_i"].number_value();
    double reg_b = opt_["reg_b"].number_value();

    int num_negative_samples = opt_["num_negative_samples"].int_value();
    // bool evaluation_on_learning = opt_["evaluation_on_learning"].bool_value();

    vector<float> errs(num_workers, 0.0);
    int end_loop = next_x - start_x;
    const int64_t shifted = start_x == 0 ? 0 : indptr[start_x - 1];

    #pragma omp parallel
    {
        // diffent seed for each thread and each iterations
        #pragma omp for schedule(static)
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

            for (int64_t idx=0, it = beg; it < end; ++it, ++idx)
            {
                const int& pos = positives[it - shifted];

                for(int64_t it2 = it * num_negative_samples;
                    it2 < it * num_negative_samples + num_negative_samples;
                    ++it2)
                {

                    const int& neg = negatives[it2 - shifted];

                    float x_uij = (P.row(u) * (Q.row(pos) - Q.row(neg)).transpose())(0, 0);
                    if (use_bias)
                        x_uij += (Qb(pos, 0) - Qb(neg, 0)); 

                    x_uij = max(EXPLB, x_uij);
                    x_uij = min(EXPUB, x_uij);
                    float logit = 1.0 / (1.0 + exp(x_uij));

                    FactorType item_deriv; 
                    if (update_i or update_j)
                        item_deriv = logit * P.row(u);
                    
                    // TODO: change to enum class
                    if (optimizer_ == "adam") {
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
                            Q.row(pos) += lr_ * (item_deriv - reg_i * Q.row(pos));
                            if (use_bias)
                                Qb(pos, 0) += (logit - reg_b * Qb(pos, 0));
                        }

                        if (update_j) {
                            Q.row(neg) -= lr_ * (item_deriv - reg_i * Q.row(neg));
                            if (use_bias)
                                Qb(pos, 0) -= (logit - reg_b * Qb(neg, 0));
                        }

                        P.row(u) += lr_ * g;
                    }
                }
            }
        }
    }

	if (optimizer_ == "adam") {
		for (int i=0; i < end_loop; ++i)
		{
			int x = start_x + i;
			const int u = x;
			int64_t beg = x == 0 ? 0 : indptr[x - 1];
			int64_t end = indptr[x];
			int64_t data_size = end - beg;
			if (data_size == 0) {
				continue;
			}

			for (int64_t idx=0, it = beg; it < end; ++it, ++idx)
			{
				const int& pos = positives[it - shifted];

				for(int64_t it2 = it * num_negative_samples;
					it2 < it * num_negative_samples + num_negative_samples;
					++it2)
				{
					const int& neg = negatives[it2 - shifted];
					P_samples_per_coordinates_[u] += 1;
					Q_samples_per_coordinates_[pos] += 1;
					Q_samples_per_coordinates_[neg] += 1;
				}
			}
		}
	}
    return 0.0;
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
    return -l;
}

void CBPRMF::update_adam(FactorType& grad, FactorType& momentum, FactorType& velocity, int i, double beta1, double beta2)
{
    momentum.row(i) = beta1 * momentum.row(i) + (1.0 - beta1) * grad.row(i);
    velocity.row(i).array() = beta2 * velocity.row(i).array() + (1.0 - beta2) * grad.row(i).array().pow(2.0);
    FactorType m_hat = momentum.row(i) / (1.0 - pow(beta1, iters_ + 1));
    FactorType v_hat = velocity.row(i) / (1.0 - pow(beta2, iters_ + 1));
    grad.row(i).array() = m_hat.array() / (v_hat.array().sqrt() + FEPS);
}


double CBPRMF::update_parameters()
{
    int num_workers = opt_["num_workers"].int_value();
    vector<double> complexity(num_workers, 0.0);

    bool use_bias = opt_["use_bias"].bool_value();
    double reg_u = opt_["reg_u"].number_value();
    double reg_i = opt_["reg_i"].number_value();
    double reg_b = opt_["reg_b"].number_value();

    Map<MatrixXf> P(P_data_, P_rows_, P_cols_),
                  Q(Q_data_, Q_rows_, Q_cols_),
                  Qb(Qb_data_, Q_rows_, 1);

    if (optimizer_ == "adam") {
        double lr = opt_["lr"].number_value();
        double beta1 = opt_["beta1"].number_value();
        double beta2 = opt_["beta1"].number_value();

        #pragma omp parallel for schedule(static)
        for(int u=0; u < P_rows_; ++u){
            if (P_samples_per_coordinates_[u]) {
                gradP_.row(u) /= P_samples_per_coordinates_[u];
            }
            complexity[omp_get_thread_num()] += (pow(P.row(u).norm(), 2.0) * reg_u);
            gradP_.row(u) -= (P.row(u) * (2 * reg_u));
            update_adam(gradP_, momentumP_, velocityP_, u, beta1, beta2);
            P.row(u) += (lr * gradP_.row(u));
        }

        #pragma omp parallel for schedule(static)
        for (int i=0; i < Q_rows_; ++i) {
            if (Q_samples_per_coordinates_[i]) {
                gradQ_.row(i) /= Q_samples_per_coordinates_[i];
                gradQb_.row(i) /= Q_samples_per_coordinates_[i];
            }
            complexity[omp_get_thread_num()] += (pow(Q.row(i).norm(), 2.0) * reg_i);
            gradQ_.row(i) -= (Q.row(i) * (2 * reg_u));
            update_adam(gradQ_, momentumQ_, velocityQ_, i, beta1, beta2);
            Q.row(i) += (lr * gradQ_.row(i));

            if (use_bias) {
                complexity[omp_get_thread_num()] += (pow(Qb(i, 0), 2.0) * reg_b);
                gradQb_(i, 0) -= (Qb(i, 0) * (2 * reg_b));
                update_adam(gradQb_, momentumQb_, velocityQb_, i, beta1, beta2);
                Qb(i, 0) += (lr * gradQb_(i, 0));
            }
        }
        P_samples_per_coordinates_.assign(P_rows_, 0);
        Q_samples_per_coordinates_.assign(Q_rows_, 0);
    } else {
        #pragma omp parallel for schedule(static)
        for (int u=0; u < P_rows_; ++u) {
            complexity[omp_get_thread_num()] += (pow(P.row(u).norm(), 2.0) * reg_u);
        }

        #pragma omp parallel for schedule(static)
        for (int i=0; i < Q_rows_; ++i) {
            complexity[omp_get_thread_num()] += (pow(Q.row(i).norm(), 2.0) * reg_i);
            if (use_bias)  {
                complexity[omp_get_thread_num()] += (pow(Qb(i, 0), 2.0) * reg_b);
            }
        }
        double decay = opt_["lr_decay"].number_value();
        if (decay > 0.0)  {
            double lr = lr_ * decay;
            lr = max(opt_["min_lr"].number_value(), lr);
            lr = max(lr, 0.000001);
            DEBUG("Leraning rate going down from {} to {}", lr_, lr);
            lr_ = lr;
        }
    }

    double _complexity = accumulate(complexity.begin(), complexity.end(), 0.0);

    iters_ += 1;
    total_samples_ = 0;  // not good flow..

    return _complexity;
}

}
