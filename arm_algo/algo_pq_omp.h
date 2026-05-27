#ifndef ALGO_PQ_OMP_H
#define ALGO_PQ_OMP_H

#define PQ_M_VAL 8
#define PQ_P_VAL 500
#define NUM_THREADS_VAL 4

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
#include <omp.h>

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

inline void build_lut_omp(const float* query, float* lut, const float* centroids,
    size_t vd, size_t M, size_t Ks, int nt)
{
    size_t ds = vd / M;
    #pragma omp parallel for num_threads(nt)
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

static std::vector<uint8_t> g_pqo_codes;
static std::vector<float> g_pqo_centroids;
static size_t g_pqo_M, g_pqo_Ks;

inline void build_pq_omp(const float* base, size_t bn, size_t vd)
{
    g_pqo_M = 8;
    g_pqo_M = PQ_M_VAL;
    g_pqo_Ks = 256;
    size_t ds = vd / g_pqo_M;
    g_pqo_centroids.resize(g_pqo_M * g_pqo_Ks * ds);
    train_pq_codebook(base, bn, vd, g_pqo_centroids.data(), g_pqo_M, g_pqo_Ks, 15);
    std::vector<uint8_t> ao(bn * g_pqo_M);
    encode_pq(base, ao.data(), g_pqo_centroids.data(), bn, vd, g_pqo_M, g_pqo_Ks);
    g_pqo_codes.resize(bn * g_pqo_M);
    encode_pq_soa(ao.data(), g_pqo_codes.data(), bn, g_pqo_M);
}

inline std::priority_queue<std::pair<float, int>> pq_omp_solve(
    const float* base, const float* query, size_t bn, size_t vd, size_t k)
{
    int nt = NUM_THREADS_VAL;
    size_t p = PQ_P_VAL;

    std::vector<float> lut(g_pqo_M * g_pqo_Ks);
    build_lut_omp(query, lut.data(), g_pqo_centroids.data(), vd, g_pqo_M, g_pqo_Ks, nt);

    std::priority_queue<std::pair<float,int>> coarse_heap;
    for(size_t i = 0; i < bn; i++){
        float is = 0.0f;
        for(size_t m = 0; m < g_pqo_M; m++) is += lut[m * g_pqo_Ks + g_pqo_codes[m * bn + i]];
        float d = 1.0f - is;
        if(coarse_heap.size() < p) coarse_heap.push(std::make_pair(d, (int)i));
        else if(d < coarse_heap.top().first){ coarse_heap.pop(); coarse_heap.push(std::make_pair(d, (int)i)); }
    }

    std::vector<int> cand; cand.reserve(p);
    while(!coarse_heap.empty()){ cand.push_back(coarse_heap.top().second); coarse_heap.pop(); }

    std::priority_queue<std::pair<float,int>> result;
    for(int idx : cand){
        float dist = InnerProductSIMD(base + (size_t)idx * vd, query, vd);
        if(result.size() < k) result.push(std::make_pair(dist, idx));
        else if(dist < result.top().first){ result.pop(); result.push(std::make_pair(dist, idx)); }
    }
    return result;
}

#endif
