#ifndef FLAT_SIMD_H
#define FLAT_SIMD_H

#include "simd_wrapper.h"
#include <queue>
#include <utility>
#include <cstddef>
#include <cstring>

inline float InnerProductSIMD(const float* x, const float* y, size_t vecdim){
    simd8float32 sum(0.0f);

    size_t i=0;
    for(; i+8<=vecdim; i+=8){
        simd8float32 vx(x+i), vy(y+i);
        sum += vx*vy;
    }

    float tmp[8];
    sum.storeu(tmp);
    float ip = tmp[0] + tmp[1] + tmp[2] + tmp[3]
             + tmp[4] + tmp[5] + tmp[6] + tmp[7];

    for(; i<vecdim; i++){
        ip += x[i]*y[i];
    }

    return 1.0f - ip;
}

inline std::priority_queue<std::pair<float, int>> flat_simd_search(const float* base, const float* query, size_t base_number, size_t vecdim, size_t k){
    std::priority_queue<std::pair<float, int>> result;

    for(size_t i = 0; i < base_number; i++){
        float dist = InnerProductSIMD(base + i * vecdim, query, vecdim);

        if(result.size() < k){
            result.push(std::make_pair(dist, static_cast<int>(i)));
        }

        else if(dist < result.top().first){
            result.pop();
            result.push(std::make_pair(dist, static_cast<int>(i)));
        }
    }

    return result;
}

#endif
