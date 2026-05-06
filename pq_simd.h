#ifndef PQ_SIMD_H
#define PQ_SIMD_H

#include "flat_simd.h"
#include <vector>
#include <cstdint>
#include <cstring>
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

#ifdef __ARM_NEON
        size_t k=0;
        for(; k+4<=Ks; k+=4){
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

#elif defined(__AVX2__)
        size_t k=0;
        for(; k+8<=Ks; k+=8){
            __m256 acc = _mm256_setzero_ps();
            for(size_t d = 0; d < dsub; d++){
                __m256 qb = _mm256_set1_ps(query_sub[d]);
                __m256 cd = _mm256_setr_ps(
                    cent_m[(k+0)*dsub+d], cent_m[(k+1)*dsub+d],
                    cent_m[(k+2)*dsub+d], cent_m[(k+3)*dsub+d],
                    cent_m[(k+4)*dsub+d], cent_m[(k+5)*dsub+d],
                    cent_m[(k+6)*dsub+d], cent_m[(k+7)*dsub+d]);
                acc = _mm256_add_ps(acc, _mm256_mul_ps(qb, cd));
            }
            _mm256_storeu_ps(lut_m + k, acc);
        }

#else
        size_t k=0;
        for(; k+4<=Ks; k+=4){
            __m128 acc = _mm_setzero_ps();
            for(size_t d = 0; d < dsub; d++){
                __m128 qb = _mm_set1_ps(query_sub[d]);
                __m128 cd = _mm_setr_ps(
                    cent_m[(k+0)*dsub+d], cent_m[(k+1)*dsub+d],
                    cent_m[(k+2)*dsub+d], cent_m[(k+3)*dsub+d]);
                acc = _mm_add_ps(acc, _mm_mul_ps(qb, cd));
            }
            _mm_storeu_ps(lut_m + k, acc);
        }
#endif

        for(; k<Ks; k++){
            float ip = 0.0f;
            for(size_t d = 0; d < dsub; d++){
                ip += query_sub[d] * cent_m[k*dsub + d];
            }
            lut_m[k] = ip;
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

#ifdef __AVX2__
    const size_t V = 8;
    __m256i zero32 = _mm256_setzero_si256();
    for(size_t i = 0; i + V <= base_number; i += V){
        __m256 acc = _mm256_setzero_ps();
        for(size_t m = 0; m < M; m++){
            const uint8_t* codes_m = codes + m * base_number;
            __m128i codes8 = _mm_loadl_epi64((const __m128i*)(codes_m + i));
            __m256i indices = _mm256_cvtepu8_epi32(codes8);
            __m256 lut_vals = _mm256_i32gather_ps(lut.data() + m * Ks, indices, 4);
            acc = _mm256_add_ps(acc, lut_vals);
        }
        __m256 dists = _mm256_sub_ps(_mm256_set1_ps(1.0f), acc);
        float tmp[8];
        _mm256_storeu_ps(tmp, dists);
        for(size_t v = 0; v < V; v++){
            float d = tmp[v];
            int idx = (int)(i + v);
            if(coarse_heap.size() < p){
                coarse_heap.push(std::make_pair(d, idx));
            }
            else if(d < coarse_heap.top().first){
                coarse_heap.pop();
                coarse_heap.push(std::make_pair(d, idx));
            }
        }
    }
    for(size_t i = (base_number / V) * V; i < base_number; i++){
        float ip_sum = 0.0f;
        for(size_t m = 0; m < M; m++){
            ip_sum += lut[m * Ks + codes[m * base_number + i]];
        }
        float d = 1.0f - ip_sum;
        if(coarse_heap.size() < p){
            coarse_heap.push(std::make_pair(d, (int)i));
        }
        else if(d < coarse_heap.top().first){
            coarse_heap.pop();
            coarse_heap.push(std::make_pair(d, (int)i));
        }
    }

#elif defined(__ARM_NEON)
    const size_t V = 4;
    for(size_t i = 0; i + V <= base_number; i += V){
        float32x4_t acc = vdupq_n_f32(0.0f);
        for(size_t m = 0; m < M; m++){
            const uint8_t* codes_m = codes + m * base_number;
            uint32_t c0 = codes_m[i+0], c1 = codes_m[i+1], c2 = codes_m[i+2], c3 = codes_m[i+3];
            const float* lut_m = lut.data() + m * Ks;
            float32x4_t lv = vdupq_n_f32(0.0f);
            lv = vld1q_lane_f32(lut_m + c0, lv, 0);
            lv = vld1q_lane_f32(lut_m + c1, lv, 1);
            lv = vld1q_lane_f32(lut_m + c2, lv, 2);
            lv = vld1q_lane_f32(lut_m + c3, lv, 3);
            acc = vaddq_f32(acc, lv);
        }
        float32x4_t ones = vdupq_n_f32(1.0f);
        float32x4_t dists = vsubq_f32(ones, acc);
        float tmp[4];
        vst1q_f32(tmp, dists);
        for(size_t v = 0; v < V; v++){
            float d = tmp[v];
            int idx = (int)(i + v);
            if(coarse_heap.size() < p){
                coarse_heap.push(std::make_pair(d, idx));
            }
            else if(d < coarse_heap.top().first){
                coarse_heap.pop();
                coarse_heap.push(std::make_pair(d, idx));
            }
        }
    }
    for(size_t i = (base_number / V) * V; i < base_number; i++){
        float ip_sum = 0.0f;
        for(size_t m = 0; m < M; m++){
            ip_sum += lut[m * Ks + codes[m * base_number + i]];
        }
        float d = 1.0f - ip_sum;
        if(coarse_heap.size() < p){
            coarse_heap.push(std::make_pair(d, (int)i));
        }
        else if(d < coarse_heap.top().first){
            coarse_heap.pop();
            coarse_heap.push(std::make_pair(d, (int)i));
        }
    }

#else
    const size_t V = 4;
    for(size_t i = 0; i + V <= base_number; i += V){
        __m128 acc = _mm_setzero_ps();
        for(size_t m = 0; m < M; m++){
            const uint8_t* codes_m = codes + m * base_number;
            uint32_t c0 = codes_m[i+0], c1 = codes_m[i+1], c2 = codes_m[i+2], c3 = codes_m[i+3];
            const float* lut_m = lut.data() + m * Ks;
            __m128 lv = _mm_setr_ps(lut_m[c0], lut_m[c1], lut_m[c2], lut_m[c3]);
            acc = _mm_add_ps(acc, lv);
        }
        __m128 dists = _mm_sub_ps(_mm_set1_ps(1.0f), acc);
        float tmp[4];
        _mm_storeu_ps(tmp, dists);
        for(size_t v = 0; v < V; v++){
            float d = tmp[v];
            int idx = (int)(i + v);
            if(coarse_heap.size() < p){
                coarse_heap.push(std::make_pair(d, idx));
            }
            else if(d < coarse_heap.top().first){
                coarse_heap.pop();
                coarse_heap.push(std::make_pair(d, idx));
            }
        }
    }
    for(size_t i = (base_number / V) * V; i < base_number; i++){
        float ip_sum = 0.0f;
        for(size_t m = 0; m < M; m++){
            ip_sum += lut[m * Ks + codes[m * base_number + i]];
        }
        float d = 1.0f - ip_sum;
        if(coarse_heap.size() < p){
            coarse_heap.push(std::make_pair(d, (int)i));
        }
        else if(d < coarse_heap.top().first){
            coarse_heap.pop();
            coarse_heap.push(std::make_pair(d, (int)i));
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

struct PQGlobal{
    std::vector<uint8_t> codes;
    std::vector<float> centroids;
    size_t M, Ks;
};

inline PQGlobal& pq_global(){
    static PQGlobal g;
    return g;
}

inline void build_pq(const float* base, size_t base_number, size_t vecdim){
    PQGlobal& g = pq_global();
    g.M = 8;
    g.Ks = 256;
    size_t dsub = vecdim / g.M;

    g.centroids.resize(g.M * g.Ks * dsub);
    train_pq_codebook(base, base_number, vecdim, g.centroids.data(), g.M, g.Ks, 15);

    std::vector<uint8_t> codes_aos(base_number * g.M);
    encode_pq(base, codes_aos.data(), g.centroids.data(), base_number, vecdim, g.M, g.Ks);
    g.codes.resize(base_number * g.M);
    encode_pq_soa(codes_aos.data(), g.codes.data(), base_number, g.M);
}

inline std::priority_queue<std::pair<float, int>> pq_solve(
    const float* base,
    const float* query,
    size_t base_number,
    size_t vecdim,
    size_t k)
{
    PQGlobal& g = pq_global();
    return pq_search(base, g.codes.data(), g.centroids.data(), query, base_number, vecdim, g.M, g.Ks, k, 100);
}

#endif
