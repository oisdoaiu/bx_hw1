#ifndef ALGO_FLAT_H
#define ALGO_FLAT_H

#include "simd_wrapper.h"
#include "flat_simd.h"

inline void build_flat(const float* base, size_t base_number, size_t vecdim) {}

inline std::priority_queue<std::pair<float, int>> flat_solve(
    const float* base, const float* query,
    size_t base_number, size_t vecdim, size_t k)
{
    return flat_simd_search(base, query, base_number, vecdim, k);
}

#endif
