#ifndef ALGO_PQ_H
#define ALGO_PQ_H

#define PQ_M_VAL 8
#define PQ_P_VAL 500

#include "simd_wrapper.h"
#include "flat_simd.h"
#include <queue>
#include <utility>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <cmath>
#include <cfloat>

inline void train_pq_codebook(const float* base, size_t bn, size_t vd,
    float* centroids, size_t M, size_t Ks, int max_iters)
{
    size_t ds = vd / M;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> rd(0.0f, 1.0f);
    for(size_t m = 0; m < M; m++){
        float* cm = centroids + m * Ks * ds;
        std::uniform_int_distribution<size_t> id(0, bn - 1);
        size_t f = id(rng);
        std::memcpy(cm, base + f * vd + m * ds, ds * sizeof(float));
        std::vector<float> md(bn, 1e30f);
        for(size_t k = 1; k < Ks; k++){
            float sd = 0.0f;
            for(size_t i = 0; i < bn; i++){
                const float* sv = base + i * vd + m * ds;
                float d = InnerProductSIMD(sv, cm + (k-1) * ds, ds);
                if(d < md[i]) md[i] = d;
                sd += md[i];
            }
            float r = rd(rng) * sd; float cum = 0.0f; size_t ch = 0;
            for(size_t i = 0; i < bn; i++){ cum += md[i]; if(cum >= r){ ch = i; break; } }
            std::memcpy(cm + k * ds, base + ch * vd + m * ds, ds * sizeof(float));
        }
    }
    std::vector<int> as(bn);
    for(int iter = 0; iter < max_iters; iter++){
        for(size_t m = 0; m < M; m++){
            float* cm = centroids + m * Ks * ds;
            for(size_t i = 0; i < bn; i++){
                const float* sv = base + i * vd + m * ds;
                float best = -FLT_MAX; int bk = 0;
                for(size_t k = 0; k < Ks; k++){
                    float ip = 1.0f - InnerProductSIMD(sv, cm + k * ds, ds);
                    if(ip > best){ best = ip; bk = (int)k; }
                }
                as[i] = bk;
            }
            std::vector<float> sc(Ks * ds, 0.0f); std::vector<int> cnt(Ks, 0);
            for(size_t i = 0; i < bn; i++){
                int k = as[i]; const float* sv = base + i * vd + m * ds;
                float* dst = sc.data() + k * ds;
                for(size_t d = 0; d < ds; d++) dst[d] += sv[d];
                cnt[k]++;
            }
            for(size_t k = 0; k < Ks; k++){
                if(cnt[k] > 0){
                    float inv = 1.0f / (float)cnt[k];
                    for(size_t d = 0; d < ds; d++) cm[k*ds+d] = sc[k*ds+d] * inv;
                }
            }
        }
    }
}

inline void encode_pq(const float* base, uint8_t* codes, const float* centroids,
    size_t bn, size_t vd, size_t M, size_t Ks)
{
    size_t ds = vd / M;
    for(size_t i = 0; i < bn; i++) for(size_t m = 0; m < M; m++){
        const float* sv = base + i * vd + m * ds;
        const float* cm = centroids + m * Ks * ds;
        float best = -FLT_MAX; uint8_t bk = 0;
        for(size_t k = 0; k < Ks; k++){
            float ip = 1.0f - InnerProductSIMD(sv, cm + k * ds, ds);
            if(ip > best){ best = ip; bk = (uint8_t)k; }
        }
        codes[i * M + m] = bk;
    }
}

inline void encode_pq_soa(const uint8_t* ao, uint8_t* so, size_t bn, size_t M){
    for(size_t m = 0; m < M; m++)
        for(size_t i = 0; i < bn; i++)
            so[m * bn + i] = ao[i * M + m];
}

inline void build_lut(const float* query, float* lut, const float* centroids,
    size_t vd, size_t M, size_t Ks)
{
    size_t ds = vd / M;
    for(size_t m = 0; m < M; m++){
        const float* qs = query + m * ds;
        const float* cm = centroids + m * Ks * ds;
        float* lm = lut + m * Ks;
        size_t k = 0;
#if defined(__AVX2__)
        for(; k + 8 <= Ks; k += 8){
            __m256 acc = _mm256_setzero_ps();
            for(size_t d = 0; d < ds; d++){
                __m256 qb = _mm256_set1_ps(qs[d]);
                __m256 cd2 = _mm256_setr_ps(
                    cm[(k+0)*ds+d],cm[(k+1)*ds+d],cm[(k+2)*ds+d],cm[(k+3)*ds+d],
                    cm[(k+4)*ds+d],cm[(k+5)*ds+d],cm[(k+6)*ds+d],cm[(k+7)*ds+d]);
                acc = _mm256_fmadd_ps(qb, cd2, acc);
            }
            _mm256_storeu_ps(lm + k, acc);
        }
#elif defined(__ARM_NEON)
        for(; k + 4 <= Ks; k += 4){
            float32x4_t acc = vdupq_n_f32(0.0f);
            for(size_t d = 0; d < ds; d++){
                float32x4_t qb = vdupq_n_f32(qs[d]);
                float32x4_t cd2 = vdupq_n_f32(0.0f);
                cd2 = vld1q_lane_f32(cm+(k+0)*ds+d, cd2, 0);
                cd2 = vld1q_lane_f32(cm+(k+1)*ds+d, cd2, 1);
                cd2 = vld1q_lane_f32(cm+(k+2)*ds+d, cd2, 2);
                cd2 = vld1q_lane_f32(cm+(k+3)*ds+d, cd2, 3);
                acc = vmlaq_f32(acc, qb, cd2);
            }
            vst1q_f32(lm + k, acc);
        }
#endif
        for(; k < Ks; k++)
            lm[k] = 1.0f - InnerProductSIMD(qs, cm + k * ds, ds);
    }
}

inline std::priority_queue<std::pair<float,int>> pq_search(
    const float* bf, const uint8_t* codes, const float* centroids,
    const float* query, size_t bn, size_t vd,
    size_t M, size_t Ks, size_t k, size_t p)
{
    std::vector<float> lut(M * Ks);
    build_lut(query, lut.data(), centroids, vd, M, Ks);
    std::priority_queue<std::pair<float,int>> coarse_heap;

#if defined(__AVX2__)
    float ct = FLT_MAX;
    const __m256 v1 = _mm256_set1_ps(1.0f);
    size_t i = 0;
    for(; i + 8 <= bn; i += 8){
        __m256 acc = _mm256_setzero_ps();
        for(size_t m = 0; m < M; m++){
            __m128i c8 = _mm_loadl_epi64((const __m128i*)(codes + m*bn + i));
            __m256i vi = _mm256_cvtepu8_epi32(c8);
            acc = _mm256_add_ps(acc, _mm256_i32gather_ps(lut.data()+m*Ks, vi, 4));
        }
        __m256 d = _mm256_sub_ps(v1, acc);
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(d, _mm256_set1_ps(ct), _CMP_LT_OQ));
        if(mask == 0) continue;
        float tmp[8]; _mm256_storeu_ps(tmp, d);
        for(int v = 0; v < 8; v++){
            if(mask & (1<<v)){
                coarse_heap.push(std::make_pair(tmp[v], (int)(i+v)));
                if(coarse_heap.size() > p) coarse_heap.pop();
                if(coarse_heap.size() == p) ct = coarse_heap.top().first;
            }
        }
    }
    for(; i < bn; i++){
        float is = 0.0f;
        for(size_t m = 0; m < M; m++) is += lut[m*Ks + codes[m*bn + i]];
        float da = 1.0f - is;
        if(coarse_heap.size() < p) coarse_heap.push(std::make_pair(da, (int)i));
        else if(da < coarse_heap.top().first){ coarse_heap.pop(); coarse_heap.push(std::make_pair(da, (int)i)); }
    }
#elif defined(__ARM_NEON)
    float ct = FLT_MAX;
    size_t i = 0;
    for(; i + 4 <= bn; i += 4){
        float32x4_t acc = vdupq_n_f32(0.0f);
        for(size_t m = 0; m < M; m++){
            uint32_t w; std::memcpy(&w, codes + m*bn + i, 4);
            uint8_t c0=(uint8_t)(w&0xFF),c1=(uint8_t)((w>>8)&0xFF);
            uint8_t c2=(uint8_t)((w>>16)&0xFF),c3=(uint8_t)((w>>24)&0xFF);
            float32x4_t vl = vdupq_n_f32(0.0f);
            vl=vld1q_lane_f32(lut.data()+m*Ks+c0,vl,0);
            vl=vld1q_lane_f32(lut.data()+m*Ks+c1,vl,1);
            vl=vld1q_lane_f32(lut.data()+m*Ks+c2,vl,2);
            vl=vld1q_lane_f32(lut.data()+m*Ks+c3,vl,3);
            acc = vaddq_f32(acc, vl);
        }
        float32x4_t ds2 = vsubq_f32(vdupq_n_f32(1.0f), acc);
        uint32x4_t cmp = vcltq_f32(ds2, vdupq_n_f32(ct));
        uint32_t mb[4]; vst1q_u32(mb, cmp);
        float tmp[4]; vst1q_f32(tmp, ds2);
        for(int v = 0; v < 4; v++){
            if(mb[v]){
                coarse_heap.push(std::make_pair(tmp[v], (int)(i+v)));
                if(coarse_heap.size() > p) coarse_heap.pop();
                if(coarse_heap.size() == p) ct = coarse_heap.top().first;
            }
        }
    }
    for(; i < bn; i++){
        float is = 0.0f;
        for(size_t m = 0; m < M; m++) is += lut[m*Ks + codes[m*bn + i]];
        float da = 1.0f - is;
        if(coarse_heap.size() < p) coarse_heap.push(std::make_pair(da, (int)i));
        else if(da < coarse_heap.top().first){ coarse_heap.pop(); coarse_heap.push(std::make_pair(da, (int)i)); }
    }
#else
    for(size_t i = 0; i < bn; i++){
        float is = 0.0f;
        for(size_t m = 0; m < M; m++) is += lut[m*Ks + codes[m*bn + i]];
        float da = 1.0f - is;
        if(coarse_heap.size() < p) coarse_heap.push(std::make_pair(da, (int)i));
        else if(da < coarse_heap.top().first){ coarse_heap.pop(); coarse_heap.push(std::make_pair(da, (int)i)); }
    }
#endif

    std::vector<int> cand; cand.reserve(p);
    while(!coarse_heap.empty()){ cand.push_back(coarse_heap.top().second); coarse_heap.pop(); }
    std::priority_queue<std::pair<float,int>> result;
    for(int idx : cand){
        float dist = InnerProductSIMD(bf + (size_t)idx * vd, query, vd);
        if(result.size() < k) result.push(std::make_pair(dist, idx));
        else if(dist < result.top().first){ result.pop(); result.push(std::make_pair(dist, idx)); }
    }
    return result;
}

static std::vector<uint8_t> g_pq_codes;
static std::vector<float> g_pq_centroids;
static size_t g_pq_M, g_pq_Ks;

inline void build_pq(const float* base, size_t bn, size_t vd)
{
    g_pq_M = PQ_M_VAL;
    g_pq_Ks = 256;
    size_t ds = vd / g_pq_M;
    g_pq_centroids.resize(g_pq_M * g_pq_Ks * ds);
    train_pq_codebook(base, bn, vd, g_pq_centroids.data(), g_pq_M, g_pq_Ks, 15);
    std::vector<uint8_t> ao(bn * g_pq_M);
    encode_pq(base, ao.data(), g_pq_centroids.data(), bn, vd, g_pq_M, g_pq_Ks);
    g_pq_codes.resize(bn * g_pq_M);
    encode_pq_soa(ao.data(), g_pq_codes.data(), bn, g_pq_M);
}

inline std::priority_queue<std::pair<float, int>> pq_solve(
    const float* base, const float* query, size_t bn, size_t vd, size_t k)
{
    size_t p = PQ_P_VAL;
    return pq_search(base, g_pq_codes.data(), g_pq_centroids.data(),
                     query, bn, vd, g_pq_M, g_pq_Ks, k, p);
}

#endif
