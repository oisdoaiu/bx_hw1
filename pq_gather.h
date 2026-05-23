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

inline void encode_pq_soa(const uint8_t* codes_aos, uint8_t* codes_soa, size_t base_number, size_t M){
    for(size_t m = 0; m < M; m++){
        for(size_t i = 0; i < base_number; i++){
            codes_soa[m * base_number + i] = codes_aos[i * M + m];
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
#if defined(__AVX2__)
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

#if defined(__AVX2__)
    float current_thresh = FLT_MAX;
    const __m256 v_one = _mm256_set1_ps(1.0f);

    size_t i = 0;
    for(; i + 8 <= base_number; i += 8){
        __m256 acc = _mm256_setzero_ps();

        for(size_t m = 0; m < M; m++){
            __m128i codes_8 = _mm_loadl_epi64((const __m128i*)(codes + m * base_number + i));
            __m256i v_idx = _mm256_cvtepu8_epi32(codes_8);
            __m256 v_lut = _mm256_i32gather_ps(lut.data() + m * Ks, v_idx, 4);
            acc = _mm256_add_ps(acc, v_lut);
        }

        __m256 dists = _mm256_sub_ps(v_one, acc);
        __m256 v_thresh = _mm256_set1_ps(current_thresh);
        __m256 cmp = _mm256_cmp_ps(dists, v_thresh, _CMP_LT_OQ);
        int mask = _mm256_movemask_ps(cmp);

        if(mask == 0) continue;

        float tmp[8];
        _mm256_storeu_ps(tmp, dists);
        for(int v = 0; v < 8; v++){
            if(mask & (1 << v)){
                coarse_heap.push(std::make_pair(tmp[v], (int)(i + v)));
                if(coarse_heap.size() > p){
                    coarse_heap.pop();
                }
                if(coarse_heap.size() == p){
                    current_thresh = coarse_heap.top().first;
                }
            }
        }
    }

    for(; i < base_number; i++){
        float ip_sum = 0.0f;
        for(size_t m = 0; m < M; m++){
            uint8_t c = codes[m * base_number + i];
            ip_sum += lut[m * Ks + c];
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
#elif defined(__ARM_NEON)
    float current_thresh = FLT_MAX;

    size_t i = 0;
    for(; i + 4 <= base_number; i += 4){
        float32x4_t acc = vdupq_n_f32(0.0f);

        for(size_t m = 0; m < M; m++){
            uint32_t word;
            std::memcpy(&word, codes + m * base_number + i, 4);
            uint8_t c0 = (uint8_t)(word & 0xFF);
            uint8_t c1 = (uint8_t)((word >> 8) & 0xFF);
            uint8_t c2 = (uint8_t)((word >> 16) & 0xFF);
            uint8_t c3 = (uint8_t)((word >> 24) & 0xFF);

            float32x4_t v_lut = vdupq_n_f32(0.0f);
            v_lut = vld1q_lane_f32(lut.data() + m * Ks + c0, v_lut, 0);
            v_lut = vld1q_lane_f32(lut.data() + m * Ks + c1, v_lut, 1);
            v_lut = vld1q_lane_f32(lut.data() + m * Ks + c2, v_lut, 2);
            v_lut = vld1q_lane_f32(lut.data() + m * Ks + c3, v_lut, 3);

            acc = vaddq_f32(acc, v_lut);
        }

        float32x4_t dists = vsubq_f32(vdupq_n_f32(1.0f), acc);
        float32x4_t v_thresh = vdupq_n_f32(current_thresh);
        uint32x4_t cmp = vcltq_f32(dists, v_thresh);

        uint32_t mask_buf[4];
        vst1q_u32(mask_buf, cmp);

        float tmp[4];
        vst1q_f32(tmp, dists);
        for(int v = 0; v < 4; v++){
            if(mask_buf[v]){
                coarse_heap.push(std::make_pair(tmp[v], (int)(i + v)));
                if(coarse_heap.size() > p){
                    coarse_heap.pop();
                }
                if(coarse_heap.size() == p){
                    current_thresh = coarse_heap.top().first;
                }
            }
        }
    }

    for(; i < base_number; i++){
        float ip_sum = 0.0f;
        for(size_t m = 0; m < M; m++){
            uint8_t c = codes[m * base_number + i];
            ip_sum += lut[m * Ks + c];
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
#else
    for(size_t i = 0; i < base_number; i++){
        float ip_sum = 0.0f;
        for(size_t m = 0; m < M; m++){
            uint8_t c = codes[m * base_number + i];
            ip_sum += lut[m * Ks + c];
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
#endif

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

    std::vector<uint8_t> codes_aos(base_number * g_pq_M);
    encode_pq(base, codes_aos.data(), g_pq_centroids.data(), base_number, vecdim, g_pq_M, g_pq_Ks);

    g_pq_codes.resize(base_number * g_pq_M);
    encode_pq_soa(codes_aos.data(), g_pq_codes.data(), base_number, g_pq_M);
}

inline std::priority_queue<std::pair<float, int>> pq_solve(
    const float* base,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t k)
{
    size_t p = 500;
    const char* env_p = std::getenv("PQ_P");
    if(env_p) p = (size_t)std::atoi(env_p);
    return pq_search(base, g_pq_codes.data(), g_pq_centroids.data(), query, base_number, vecdim, g_pq_M, g_pq_Ks, k, p);
}

#endif
