#ifndef PQ_SIMD_H
#define PQ_SIMD_H

#include "flat_simd.h"
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <cmath>
#include <cfloat>

inline void train_pq_codebook(const float* base, size_t base_number, size_t vecdim,
float* centroids, size_t M, size_t Ks, int max_iters){
    size_t dsub = vecdim / M;
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> real_dist(0.0f, 1.0f);

    for(size_t m = 0; m < M; m++){
        float* cent_m = centroids + m * Ks * dsub;
        std::uniform_int_distribution<size_t> idx_dist(0, base_number - 1);
        size_t first = idx_dist(rng);
        std::memcpy(cent_m, base + first * vecdim + m * dsub, dsub * sizeof(float));

        std::vector<float> min_dist(base_number, 1e30f);
        for(size_t k = 1; k < Ks; k++){
            float sum_dist = 0.0f;
            for(size_t i = 0; i < base_number; i++){
                const float* subv = base + i * vecdim + m * dsub;
                float d = InnerProductSIMD(subv, cent_m + (k-1) * dsub, dsub);
                if(d < min_dist[i]) min_dist[i] = d;
                sum_dist += min_dist[i];
            }
            float r = real_dist(rng) * sum_dist;
            float cum = 0.0f;
            size_t chosen = 0;
            for(size_t i = 0; i < base_number; i++){
                cum += min_dist[i];
                if(cum >= r){
                    chosen = i;
                    break;
                }
            }
            std::memcpy(cent_m + k * dsub, base + chosen * vecdim + m * dsub, dsub * sizeof(float));
        }
    }

    std::vector<int> assignments(base_number);
    for(int iter = 0; iter < max_iters; iter++){
        for(size_t m = 0; m < M; m++){
            float* cent_m = centroids + m * Ks * dsub;

            for(size_t i = 0; i < base_number; i++){
                const float* subv = base + i * vecdim + m * dsub;
                float best_ip = -FLT_MAX;
                int best_k = 0;
                for(size_t k = 0; k < Ks; k++){
                    float ip = 1.0f - InnerProductSIMD(subv, cent_m + k * dsub, dsub);
                    if(ip > best_ip){
                        best_ip = ip;
                        best_k = (int)k;
                    }
                }
                assignments[i] = best_k;
            }

            std::vector<float> sum_cent(Ks * dsub, 0.0f);
            std::vector<int> counts(Ks, 0);
            for(size_t i = 0; i < base_number; i++){
                int k = assignments[i];
                const float* subv = base + i * vecdim + m * dsub;
                float* dst = sum_cent.data() + k * dsub;
                for(size_t d = 0; d < dsub; d++){
                    dst[d] += subv[d];
                }
                counts[k]++;
            }
            for(size_t k = 0; k < Ks; k++){
                if(counts[k] > 0){
                    float inv = 1.0f / (float)counts[k];
                    for(size_t d = 0; d < dsub; d++){
                        cent_m[k * dsub + d] = sum_cent[k * dsub + d] * inv;
                    }
                }
            }
        }
    }
}

inline void encode_pq(const float* base, uint8_t* codes, const float* centroids,
size_t base_number, size_t vecdim, size_t M, size_t Ks){
    size_t dsub = vecdim / M;

    for(size_t i = 0; i < base_number; i++){
        for(size_t m = 0; m < M; m++){
            const float* subv = base + i * vecdim + m * dsub;
            const float* cent_m = centroids + m * Ks * dsub;
            float best_ip = -FLT_MAX;
            uint8_t best_k = 0;
            for(size_t k = 0; k < Ks; k++){
                float ip = 1.0f - InnerProductSIMD(subv, cent_m + k * dsub, dsub);
                if(ip > best_ip){
                    best_ip = ip;
                    best_k = (uint8_t)k;
                }
            }
            codes[i * M + m] = best_k;
        }
    }
}

inline void build_lut(const float* query, float* lut, const float* centroids,
size_t vecdim, size_t M, size_t Ks){
    size_t dsub = vecdim / M;

    for(size_t m = 0; m < M; m++){
        const float* query_sub = query + m * dsub;
        const float* cent_m = centroids + m * Ks * dsub;
        float* lut_m = lut + m * Ks;
        size_t k = 0;

#ifdef __AVX2__
        for(; k + 8 <= Ks; k += 8){
            __m256 acc = _mm256_setzero_ps();
            for(size_t d = 0; d < dsub; d++){
                __m256 qb = _mm256_set1_ps(query_sub[d]);
                __m256 cd = _mm256_setr_ps(
                    cent_m[(k+0)*dsub + d], cent_m[(k+1)*dsub + d],
                    cent_m[(k+2)*dsub + d], cent_m[(k+3)*dsub + d],
                    cent_m[(k+4)*dsub + d], cent_m[(k+5)*dsub + d],
                    cent_m[(k+6)*dsub + d], cent_m[(k+7)*dsub + d]
                );
                acc = _mm256_fmadd_ps(qb, cd, acc);
            }
            _mm256_storeu_ps(lut_m + k, acc);
        }
#elif defined(__ARM_NEON)
        for(; k + 4 <= Ks; k += 4){
            float32x4_t acc = vdupq_n_f32(0.0f);
            for(size_t d = 0; d < dsub; d++){
                float32x4_t qb = vdupq_n_f32(query_sub[d]);
                float32x4_t cd = vdupq_n_f32(0.0f);
                cd = vld1q_lane_f32(cent_m + (k+0)*dsub + d, cd, 0);
                cd = vld1q_lane_f32(cent_m + (k+1)*dsub + d, cd, 1);
                cd = vld1q_lane_f32(cent_m + (k+2)*dsub + d, cd, 2);
                cd = vld1q_lane_f32(cent_m + (k+3)*dsub + d, cd, 3);
                acc = vmlaq_f32(acc, qb, cd);
            }
            vst1q_f32(lut_m + k, acc);
        }
#else
        for(; k + 4 <= Ks; k += 4){
            __m128 acc = _mm_setzero_ps();
            for(size_t d = 0; d < dsub; d++){
                __m128 qb = _mm_set1_ps(query_sub[d]);
                __m128 cd = _mm_setr_ps(
                    cent_m[(k+0)*dsub + d],
                    cent_m[(k+1)*dsub + d],
                    cent_m[(k+2)*dsub + d],
                    cent_m[(k+3)*dsub + d]
                );
                acc = _mm_add_ps(acc, _mm_mul_ps(qb, cd));
            }
            _mm_storeu_ps(lut_m + k, acc);
        }
#endif

        for(; k < Ks; k++){
            lut_m[k] = 1.0f - InnerProductSIMD(query_sub, cent_m + k * dsub, dsub);
        }
    }
}

inline std::priority_queue<std::pair<float, int>> pq_search(
    const float* base_float,
    const uint8_t* codes,
    const float* centroids,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t M,
    size_t Ks,
    size_t k,
    size_t p)
{
    std::vector<float> lut(M * Ks);
    build_lut(query, lut.data(), centroids, vecdim, M, Ks);

    std::priority_queue<std::pair<float, int>> coarse_heap;

    for(size_t i = 0; i < base_number; i++){
        const uint8_t* code = codes + i * M;
        float ip_sum = 0.0f;
        for(size_t m = 0; m < M; m++){
            ip_sum += lut[m * Ks + code[m]];
        }
        float dist_approx = 1.0f - ip_sum;

        if(coarse_heap.size() < p){
            coarse_heap.push(std::make_pair(dist_approx, (int)i));
        }
        
        else if(dist_approx < coarse_heap.top().first){
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
        float dist = InnerProductSIMD(base_float + (size_t)idx * vecdim, query, vecdim);
        if(result.size() < k){
            result.push(std::make_pair(dist, idx));
        }
        else if(dist < result.top().first){
            result.pop();
            result.push(std::make_pair(dist, idx));
        }
    }

    return result;
}

static std::vector<uint8_t> g_pq_codes;
static std::vector<float> g_pq_centroids;
static size_t g_pq_M, g_pq_Ks;

inline void build_pq(const float* base, size_t base_number, size_t vecdim){
    g_pq_M = 8;
    const char* env_m = std::getenv("PQ_M");
    if(env_m) g_pq_M = (size_t)std::atoi(env_m);
    g_pq_Ks = 256;
    size_t dsub = vecdim / g_pq_M;

    g_pq_centroids.resize(g_pq_M * g_pq_Ks * dsub);
    train_pq_codebook(base, base_number, vecdim, g_pq_centroids.data(), g_pq_M, g_pq_Ks, 15);

    g_pq_codes.resize(base_number * g_pq_M);
    encode_pq(base, g_pq_codes.data(), g_pq_centroids.data(), base_number, vecdim, g_pq_M, g_pq_Ks);
}

inline std::priority_queue<std::pair<float, int>> pq_solve(
    const float* base, 
    const float* query,
    size_t base_number, 
    size_t vecdim, 
    size_t k)
{
    return pq_search(base, g_pq_codes.data(), g_pq_centroids.data(), query, base_number, vecdim, g_pq_M, g_pq_Ks, k, 500);
}

#endif
