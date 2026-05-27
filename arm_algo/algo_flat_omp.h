#ifndef ALGO_FLAT_OMP_H
#define ALGO_FLAT_OMP_H

#define NUM_THREADS_VAL 4

#include "simd_wrapper.h"
#include "flat_simd.h"
#include <omp.h>

inline void build_flat_omp(const float* base, size_t base_number, size_t vecdim) {}

inline std::priority_queue<std::pair<float, int>> flat_omp_solve(
    const float* base, const float* query,
    size_t base_number, size_t vecdim, size_t k)
{
    int nt = NUM_THREADS_VAL;

    std::vector<std::pair<float,int>> all_cand;
    #pragma omp parallel num_threads(nt)
    {
        int tid = omp_get_thread_num(), nts = omp_get_num_threads();
        size_t chunk = base_number / nts;
        size_t start = tid * chunk;
        size_t end = (tid == nts - 1) ? base_number : start + chunk;

        std::priority_queue<std::pair<float,int>> local_pq;
        for(size_t i = start; i < end; i++){
            float dist = InnerProductSIMD(base + i * vecdim, query, vecdim);
            if(local_pq.size() < k) local_pq.push(std::make_pair(dist, (int)i));
            else if(dist < local_pq.top().first){ local_pq.pop(); local_pq.push(std::make_pair(dist, (int)i)); }
        }
        #pragma omp critical
        { while(!local_pq.empty()){ all_cand.push_back(local_pq.top()); local_pq.pop(); } }
    }

    std::priority_queue<std::pair<float,int>> result;
    for(auto& c : all_cand){
        if(result.size() < k) result.push(c);
        else if(c.first < result.top().first){ result.pop(); result.push(c); }
    }
    return result;
}

#endif
