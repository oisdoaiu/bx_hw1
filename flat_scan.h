#ifndef FLAT_SCAN_H
#define FLAT_SCAN_H

#include <queue>
#include <utility>
#include <cstddef>
#include <cstring>

inline float InnerProductScalar(const float* x, const float* y, size_t vecdim){
    float ip = 0.0f;
    for(size_t i = 0; i < vecdim; i++){
        ip += x[i] * y[i];
    }
    return 1.0f - ip;
}

inline std::priority_queue<std::pair<float, int>> flat_scan_search(
    const float* base,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t k)
{
    std::priority_queue<std::pair<float, int>> result;

    for(size_t i = 0; i < base_number; i++){
        float dist = InnerProductScalar(base + i * vecdim, query, vecdim);

        if(result.size() < k){
            result.push(std::make_pair(dist, static_cast<int>(i)));
        }else if(dist < result.top().first){
            result.pop();
            result.push(std::make_pair(dist, static_cast<int>(i)));
        }
    }

    return result;
}

#endif // FLAT_SCAN_H