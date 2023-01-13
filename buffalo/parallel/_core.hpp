#pragma once
#include <cmath>
#include <omp.h>
#include <vector>
#include <string>
#include <cstdio>
#include <random>
#include <limits>
#include <fstream>
#include <algorithm>
#include <unordered_set>
#include <parallel/algorithm>

#include <Eigen/Core>
#include <Eigen/Dense>


using namespace std;
using namespace Eigen;

namespace parallel
{

struct nn_t
{
    int key;
    float val;
    nn_t() : key(-1), val(numeric_limits<float>::min()) {}
    bool operator < (const float& that){
        return this->val > that;
    }

    bool operator < (const nn_t& that){
        return this->val > that.val;
    }
};

struct topn_t
{
    nn_t* nns;
    int K;
    topn_t() : K(0) {}
    void alloc(int _K){
        K = _K;
        nns = new nn_t[K];
    }
    void free() {
        if(K){
            delete[] nns;
        }
    }
    void update(int& key, float& val){
        nn_t* ptr = lower_bound(nns, nns + K, val);
        int idx = (int)(ptr - nns);
        if(idx >= K)
            return;
        if(idx + 1 == K){
            nns[idx].key = key;
            nns[idx].val = val;
        }
        else{
            memmove((void*)&nns[idx + 1], (void*)&nns[idx], sizeof(nn_t) * (K - idx - 1));
            nns[idx].key = key;
            nns[idx].val = val;
        }
    }
};

void quickselect(float* scores, int rows, int cols, int32_t* result, int k, bool sorted, int num_threads){
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>> _scores(scores, rows, cols);
    Map<Matrix<int, Dynamic, Dynamic, RowMajor>> _result(result, rows, k);
    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(dynamic, 4)
    for (int i=0; i<rows; ++i){
        vector<int> ranks(cols);
        iota(ranks.begin(), ranks.end(), 0);  // initialize ranks as {0, 1, 2, ...}
        // high score has a priority
        nth_element(ranks.begin(), ranks.begin() + k - 1, ranks.end(),
                    [&](int lhs, int rhs){return _scores(i, lhs) > _scores(i, rhs);});
        // sort ranks[: k], since it is not guaranteed to be sorted
        if (sorted)
            sort(ranks.begin(), ranks.begin() + k,
                    [&](int lhs, int rhs){return _scores(i, lhs) > _scores(i, rhs);});
        copy(ranks.begin(), ranks.begin() + k, &_result(i, 0));
    }
}

void dot_topn(
        int32_t* indexes, int num_queries,
        float* _P, int p_rows, int p_cols,
        float* _Q, int q_rows, int q_cols,
        float* _Qb, int qb_rows,
        int32_t* _out_keys, float* _out_scores,
        int32_t* _pool, int pool_size,
        int k, int num_threads)
{
    bool is_same = _P == _Q;
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>> P(_P, p_rows, p_cols);
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>> Q(_Q, q_rows, q_cols);
    Map<Matrix<float, Dynamic, Dynamic, RowMajor>> Qb(_Qb, qb_rows, 1);
    Map<Matrix<int, Dynamic, Dynamic, RowMajor>> out_keys(_out_keys, num_queries, k);
    Map<Matrix<float , Dynamic, Dynamic, RowMajor>> out_scores(_out_scores, num_queries, k);

    unordered_set<int32_t> pool;
    for (int i=0; i < pool_size; ++i)
        pool.insert(_pool[i]);

    int correct_k = min(q_rows, k);
    if (pool_size)
        correct_k = min(pool_size, correct_k);

    omp_set_num_threads(num_threads);
    #pragma omp parallel for schedule(guided)
    for (int i=0; i < num_queries; ++i){
        topn_t topn;
        topn.alloc(correct_k);
        float last_one = numeric_limits<float>::min();
        int q = indexes[i];
        for (int j=0; j < q_rows; ++j) {
            if (is_same and q == j)
                continue;
            if (pool_size and pool.find(j) == pool.end())
                continue;
            float score = P.row(q).dot(Q.row(j));
            if (qb_rows)
                score += Qb(j);
            if (score > last_one) {
                topn.update(j, score);
                last_one = topn.nns[correct_k - 1].val;
            }
        }
        for (int j=0; j < correct_k; ++j) {
            out_keys(i, j) = topn.nns[j].key;
            out_scores(i, j) = topn.nns[j].val;
        }
        for (int j=correct_k; j < k; ++j) {
            out_keys(i, j) = -1;
            out_scores(i, j) = 0.0;
        }
        topn.free();
    }
}
}
