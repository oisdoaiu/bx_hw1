#ifndef SQ_SIMD_H
#define SQ_SIMD_H

#include "flat_simd.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <algorithm>

inline void build_sq_index(const float* base_data, uint8_t* sq_data, uint32_t* sum_qi, float& gmin, float& gscale, size_t base_number, size_t vecdim){
    gmin = base_data[0];
    float gmax = base_data[0];
    for(size_t i = 1; i < base_number*vecdim; i++){
        float v = base_data[i];
        if(v < gmin) gmin = v;
        if(v > gmax) gmax = v;
    }
    float range = gmax - gmin;
    gscale = (range > 1e-8f) ? 255.0f/range : 1.0f;

    for(size_t i = 0; i < base_number; i++){
        uint32_t sq = 0;
        for(size_t d = 0; d < vecdim; d++){
            float val = base_data[i*vecdim + d];
            int q = (int)((val - gmin) * gscale + 0.5f);
            q = std::max(0, std::min(255, q));
            uint8_t uq = (uint8_t)q;
            sq_data[i*vecdim + d] = uq;
            sq += uq;
        }
        sum_qi[i] = sq;
    }
}

inline void quantize_query(const float* query, uint8_t* query_sq, uint32_t& sum_qq, float gmin, float gscale, size_t vecdim){
    sum_qq = 0;
    for(size_t d = 0; d < vecdim; d++){
        int q = (int)((query[d] - gmin) * gscale + 0.5f);
        q = std::max(0, std::min(255, q));
        uint8_t uq = (uint8_t)q;
        query_sq[d] = uq;
        sum_qq += uq;
    }
}

#ifdef __ARM_NEON
#include <arm_neon.h>

inline uint32_t InnerProductSQ(const uint8_t* x, const uint8_t* y, size_t vecdim){
    uint32x4_t sum = vdupq_n_u32(0);

    size_t i=0;
    for(; i+16<=vecdim; i+=16){
        uint8x16_t xv = vld1q_u8(x+i);
        uint8x16_t yv = vld1q_u8(y+i);
        sum = vpadalq_u16(sum, vmull_u8(vget_low_u8(xv), vget_low_u8(yv)));
        sum = vpadalq_u16(sum, vmull_high_u8(xv, yv));
    }

    uint32_t result = vaddvq_u32(sum);
    for(; i<vecdim; i++){
        result += (uint32_t)x[i] * (uint32_t)y[i];
    }
    return result;
}

#elif defined(__AVX2__)
#include <immintrin.h>

inline uint32_t InnerProductSQ(const uint8_t* x, const uint8_t* y, size_t vecdim){
    __m256i sum = _mm256_setzero_si256();
    const __m256i zero = _mm256_setzero_si256();

    size_t i=0;
    for(; i+32<=vecdim; i+=32){
        __m256i xv = _mm256_loadu_si256((const __m256i*)(x+i));
        __m256i yv = _mm256_loadu_si256((const __m256i*)(y+i));
        __m256i x_lo = _mm256_unpacklo_epi8(xv, zero);
        __m256i x_hi = _mm256_unpackhi_epi8(xv, zero);
        __m256i y_lo = _mm256_unpacklo_epi8(yv, zero);
        __m256i y_hi = _mm256_unpackhi_epi8(yv, zero);
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(x_lo, y_lo));
        sum = _mm256_add_epi32(sum, _mm256_madd_epi16(x_hi, y_hi));
    }

    __m128i lo = _mm256_castsi256_si128(sum);
    __m128i hi = _mm256_extracti128_si256(sum, 1);
    lo = _mm_add_epi32(lo, hi);
    lo = _mm_hadd_epi32(lo, lo);
    lo = _mm_hadd_epi32(lo, lo);
    uint32_t result = _mm_cvtsi128_si32(lo);

    for(; i<vecdim; i++){
        result += (uint32_t)x[i] * (uint32_t)y[i];
    }
    return result;
}

#else
#include <emmintrin.h>
#include <tmmintrin.h>

inline uint32_t InnerProductSQ(const uint8_t* x, const uint8_t* y, size_t vecdim){
    __m128i sum = _mm_setzero_si128();
    const __m128i zero = _mm_setzero_si128();

    size_t i=0;
    for(; i+16<=vecdim; i+=16){
        __m128i xv = _mm_loadu_si128((const __m128i*)(x+i));
        __m128i yv = _mm_loadu_si128((const __m128i*)(y+i));
        __m128i x_lo = _mm_unpacklo_epi8(xv, zero);
        __m128i x_hi = _mm_unpackhi_epi8(xv, zero);
        __m128i y_lo = _mm_unpacklo_epi8(yv, zero);
        __m128i y_hi = _mm_unpackhi_epi8(yv, zero);
        sum = _mm_add_epi32(sum, _mm_madd_epi16(x_lo, y_lo));
        sum = _mm_add_epi32(sum, _mm_madd_epi16(x_hi, y_hi));
    }

    sum = _mm_hadd_epi32(sum, sum);
    sum = _mm_hadd_epi32(sum, sum);
    uint32_t result = _mm_cvtsi128_si32(sum);

    for(; i<vecdim; i++){
        result += (uint32_t)x[i] * (uint32_t)y[i];
    }
    return result;
}
#endif

inline std::priority_queue<std::pair<float, int>> sq_search(
    const float* base_float,
    const uint8_t* base_sq,
    const uint32_t* sum_qi,
    const float* query_float,
    float gmin,
    float gscale,
    size_t base_number,
    size_t vecdim,
    size_t k,
    size_t p)
{
    uint32_t sum_qq;
    std::vector<uint8_t> query_sq(vecdim);
    quantize_query(query_float, query_sq.data(), sum_qq, gmin, gscale, vecdim);

    float delta = 1.0f/gscale;
    float min_term = delta * gmin;

    std::priority_queue<std::pair<float, int>> coarse_heap;

    for(size_t i = 0; i < base_number; i++){
        uint32_t ip_u8 = InnerProductSQ(base_sq + i*vecdim, query_sq.data(), vecdim);
        float ip_approx = delta*delta*ip_u8 + min_term*(sum_qi[i] + sum_qq) + gmin*gmin*vecdim;
        float dist_approx = 1.0f - ip_approx;

        if(coarse_heap.size() < p){
            coarse_heap.push(std::make_pair(dist_approx, (int)i));
        }else if(dist_approx < coarse_heap.top().first){
            coarse_heap.pop();
            coarse_heap.push(std::make_pair(dist_approx, (int)i));
        }
    }

    std::vector<int> candidates;
    candidates.reserve(p);
    while(!coarse_heap.empty()){
        candidates.push_back(coarse_heap.top().second);
        coarse_heap.pop();
    }

    std::priority_queue<std::pair<float, int>> result;
    for(int idx : candidates){
        float dist = InnerProductSIMD(base_float + (size_t)idx*vecdim, query_float, vecdim);
        if(result.size() < k){
            result.push(std::make_pair(dist, idx));
        }else if(dist < result.top().first){
            result.pop();
            result.push(std::make_pair(dist, idx));
        }
    }

    return result;
}

inline std::vector<uint8_t> g_base_sq;
inline std::vector<uint32_t> g_sum_qi;
inline float g_gmin, g_gscale;

inline void build_index(const float* base, size_t base_number, size_t vecdim){
    g_base_sq.resize(base_number * vecdim);
    g_sum_qi.resize(base_number);
    build_sq_index(base, g_base_sq.data(), g_sum_qi.data(), g_gmin, g_gscale, base_number, vecdim);
}

inline std::priority_queue<std::pair<float, int>> flat_search(
    const float* base, const float* query,
    size_t base_number, size_t vecdim, size_t k)
{
    return sq_search(base, g_base_sq.data(), g_sum_qi.data(),
                     query, g_gmin, g_gscale, base_number, vecdim, k, 50);
}

#endif // SQ_SIMD_H
