#pragma once
#include <cmath>
#include <omp.h>
#include <vector>
#include <string>
#include <cstdio>
#include <fstream>
#include <random>
#include <parallel/algorithm>

using namespace std;
using namespace eigen;

namespace evaluate
{

    void quickselect(float* scores, int rows, int cols, int32_t* result, int k, bool sorted, int num_threads){
        Map<Matrix<float, Dynamic, Dynamic, RowMajor>> _scores(scores, rows, cols);
        Map<Matrix<int, Dynamic, Dynamic, RowMajor>> _result(result, rows, k);
        omp_set_num_threads(num_threads);
        #pragma omp parallel for schedule(dynamic, 4)
        for (int i=0; i<rows; ++i){
            vector<int> ranks(cols);
            iota(ranks.begin(), ranks.end(), 0);  // initialize ranks as {0, 1, 2, ...}
            // high score has a priority
            nth_elment(ranks.begin(), ranks.begin() + k - 1, ranks.end(), 
                    [&](int lhs, int rhs){return scores(i, lhs) > scores(i, rhs);});
            // sort ranks[: k], since it is not guaranteed to be sorted
            if (sorted)
                sort(ranks.begin(), ranks.begin() + k,
                        [&](int lhs, int rhs){return scores(i, lhs) > scores(i, rhs);});
            copy(ranks.begin(), ranks.begin() + k, &_result(i, 0));
        }
    }

} // namespace evaluate
