#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <cstdlib>
#include "pq_gather.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <pthread.h>

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();
    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";
    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency;
};


struct ThreadParam
{
    int thread_id;
    int num_threads;
    const float* base;
    const float* test_query;
    const int* test_gt;
    size_t base_number;
    size_t vecdim;
    size_t test_gt_d;
    size_t test_number;
    size_t k;
    SearchResult* results;
};

void* pthread_worker(void* arg)
{
    ThreadParam* p = (ThreadParam*)arg;
    const unsigned long Converter = 1000 * 1000;

    for(size_t qi = p->thread_id; qi < p->test_number; qi += p->num_threads){
        struct timeval val;
        gettimeofday(&val, NULL);

        auto res = pq_solve(p->base, p->test_query + qi * p->vecdim,
                            p->base_number, p->vecdim, p->k);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) -
                       (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(size_t j = 0; j < p->k; ++j){
            gtset.insert((uint32_t)p->test_gt[j + qi * p->test_gt_d]);
        }

        size_t acc = 0;
        while(res.size()){
            int x = res.top().second;
            if(gtset.find((uint32_t)x) != gtset.end()) ++acc;
            res.pop();
        }
        p->results[qi] = {(float)acc / p->k, diff};
    }
    return nullptr;
}

void run_pthread(const float* base, const float* test_query, const int* test_gt,
                 size_t base_number, size_t vecdim, size_t test_gt_d,
                 size_t test_number, size_t k, SearchResult* results, int num_threads)
{
    pthread_t* threads = new pthread_t[num_threads];
    ThreadParam* params = new ThreadParam[num_threads];

    for(int t = 0; t < num_threads; t++){
        params[t].thread_id = t;
        params[t].num_threads = num_threads;
        params[t].base = base;
        params[t].test_query = test_query;
        params[t].test_gt = test_gt;
        params[t].base_number = base_number;
        params[t].vecdim = vecdim;
        params[t].test_gt_d = test_gt_d;
        params[t].test_number = test_number;
        params[t].k = k;
        params[t].results = results;
        pthread_create(&threads[t], nullptr, pthread_worker, &params[t]);
    }
    for(int t = 0; t < num_threads; t++) pthread_join(threads[t], nullptr);
    delete[] threads;
    delete[] params;
}

#ifdef _OPENMP

inline void build_lut_omp(const float* query, float* lut, const float* centroids,
                           size_t vecdim, size_t M, size_t Ks, int num_threads)
{
    size_t dsub = vecdim / M;

    #pragma omp parallel for num_threads(num_threads)
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

inline std::priority_queue<std::pair<float, int>> pq_search_omp_lut(
    const float* base_float, const uint8_t* codes, const float* centroids,
    const float* query, size_t base_number, size_t vecdim,
    size_t M, size_t Ks, size_t k, size_t p, int num_threads)
{
    std::vector<float> lut(M * Ks);
    build_lut_omp(query, lut.data(), centroids, vecdim, M, Ks, num_threads);
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
        int mask = _mm256_movemask_ps(_mm256_cmp_ps(dists, v_thresh, _CMP_LT_OQ));
        if(mask == 0) continue;
        float tmp[8]; _mm256_storeu_ps(tmp, dists);
        for(int v = 0; v < 8; v++){
            if(mask & (1 << v)){
                coarse_heap.push(std::make_pair(tmp[v], (int)(i + v)));
                if(coarse_heap.size() > p) coarse_heap.pop();
                if(coarse_heap.size() == p) current_thresh = coarse_heap.top().first;
            }
        }
    }
    for(; i < base_number; i++){
        float ip_sum = 0.0f;
        for(size_t m = 0; m < M; m++){
            ip_sum += lut[m * Ks + codes[m * base_number + i]];
        }
        float dist_approx = 1.0f - ip_sum;
        if(coarse_heap.size() < p) coarse_heap.push(std::make_pair(dist_approx, (int)i));
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
            uint8_t c0 = (uint8_t)(word & 0xFF), c1 = (uint8_t)((word >> 8) & 0xFF);
            uint8_t c2 = (uint8_t)((word >> 16) & 0xFF), c3 = (uint8_t)((word >> 24) & 0xFF);
            float32x4_t v_lut = vdupq_n_f32(0.0f);
            v_lut = vld1q_lane_f32(lut.data() + m * Ks + c0, v_lut, 0);
            v_lut = vld1q_lane_f32(lut.data() + m * Ks + c1, v_lut, 1);
            v_lut = vld1q_lane_f32(lut.data() + m * Ks + c2, v_lut, 2);
            v_lut = vld1q_lane_f32(lut.data() + m * Ks + c3, v_lut, 3);
            acc = vaddq_f32(acc, v_lut);
        }
        float32x4_t dists = vsubq_f32(vdupq_n_f32(1.0f), acc);
        uint32x4_t cmp = vcltq_f32(dists, vdupq_n_f32(current_thresh));
        uint32_t mask_buf[4]; vst1q_u32(mask_buf, cmp);
        float tmp[4]; vst1q_f32(tmp, dists);
        for(int v = 0; v < 4; v++){
            if(mask_buf[v]){
                coarse_heap.push(std::make_pair(tmp[v], (int)(i + v)));
                if(coarse_heap.size() > p) coarse_heap.pop();
                if(coarse_heap.size() == p) current_thresh = coarse_heap.top().first;
            }
        }
    }
    for(; i < base_number; i++){
        float ip_sum = 0.0f;
        for(size_t m = 0; m < M; m++){
            ip_sum += lut[m * Ks + codes[m * base_number + i]];
        }
        float dist_approx = 1.0f - ip_sum;
        if(coarse_heap.size() < p) coarse_heap.push(std::make_pair(dist_approx, (int)i));
        else if(dist_approx < coarse_heap.top().first){
            coarse_heap.pop();
            coarse_heap.push(std::make_pair(dist_approx, (int)i));
        }
    }
#else
    for(size_t i = 0; i < base_number; i++){
        float ip_sum = 0.0f;
        for(size_t m = 0; m < M; m++){
            ip_sum += lut[m * Ks + codes[m * base_number + i]];
        }
        float dist_approx = 1.0f - ip_sum;
        if(coarse_heap.size() < p) coarse_heap.push(std::make_pair(dist_approx, (int)i));
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
        if(result.size() < k) result.push(std::make_pair(dist, idx));
        else if(dist < result.top().first){
            result.pop();
            result.push(std::make_pair(dist, idx));
        }
    }
    return result;
}

void run_openmp_query(const float* base, const float* test_query, const int* test_gt,
                      size_t base_number, size_t vecdim, size_t test_gt_d,
                      size_t test_number, size_t k, SearchResult* results, int num_threads)
{
    const unsigned long Converter = 1000 * 1000;
    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for(size_t i = 0; i < test_number; ++i){
        struct timeval val;
        gettimeofday(&val, NULL);
        auto res = pq_solve(base, test_query + i * vecdim, base_number, vecdim, k);
        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) -
                       (val.tv_sec * Converter + val.tv_usec);
        std::set<uint32_t> gtset;
        for(size_t j = 0; j < k; ++j) gtset.insert((uint32_t)test_gt[j + i * test_gt_d]);
        size_t acc = 0;
        while(res.size()){
            int x = res.top().second;
            if(gtset.find((uint32_t)x) != gtset.end()) ++acc;
            res.pop();
        }
        results[i] = {(float)acc / k, diff};
    }
}

void run_pthread_subspace(const float* base, const float* test_query, const int* test_gt,
                         size_t base_number, size_t vecdim, size_t test_gt_d,
                         size_t test_number, size_t k, SearchResult* results, int num_threads)
{
    const unsigned long Converter = 1000 * 1000;
    size_t M = 8, Ks = 256;
    const char* env_m = std::getenv("PQ_M"); if(env_m) M = (size_t)std::atoi(env_m);
    size_t p = 500;
    const char* env_p = std::getenv("PQ_P"); if(env_p) p = (size_t)std::atoi(env_p);

    for(size_t i = 0; i < test_number; ++i){
        struct timeval val; gettimeofday(&val, NULL);
        const float* query = test_query + i * vecdim;

        // build_lut with Pthread per query
        std::vector<float> lut(M * Ks);
        size_t dsub = vecdim / M;
        pthread_t* threads = new pthread_t[num_threads];
        struct LutThread { int tid, nt; const float* query; float* lut; const float* centroids; size_t M, Ks, dsub; } *params = new LutThread[num_threads];

        auto lut_worker = [](void* arg) -> void* {
            auto* p = (LutThread*)arg;
            size_t chunk = p->M / p->nt;
            size_t m_start = p->tid * chunk;
            size_t m_end = (p->tid == p->nt - 1) ? p->M : m_start + chunk;
            for(size_t m = m_start; m < m_end; m++){
                const float* qs = p->query + m * p->dsub;
                const float* cm = p->centroids + m * p->Ks * p->dsub;
                float* lm = p->lut + m * p->Ks;
                for(size_t k = 0; k < p->Ks; k++)
                    lm[k] = 1.0f - InnerProductSIMD(qs, cm + k * p->dsub, p->dsub);
            }
            return nullptr;
        };

        for(int t = 0; t < num_threads; t++){
            params[t] = {t, num_threads, query, lut.data(), g_pq_centroids.data(), M, Ks, dsub};
            pthread_create(&threads[t], nullptr, lut_worker, &params[t]);
        }
        for(int t = 0; t < num_threads; t++) pthread_join(threads[t], nullptr);
        delete[] threads; delete[] params;

        // table scan (serial)
        std::priority_queue<std::pair<float,int>> coarse_heap;
        for(size_t j = 0; j < base_number; j++){
            float is = 0.0f;
            for(size_t m = 0; m < M; m++) is += lut[m*Ks + g_pq_codes[m*base_number + j]];
            float d = 1.0f - is;
            if(coarse_heap.size() < p) coarse_heap.push(std::make_pair(d, (int)j));
            else if(d < coarse_heap.top().first){ coarse_heap.pop(); coarse_heap.push(std::make_pair(d, (int)j)); }
        }
        std::vector<int> cand; cand.reserve(p);
        while(!coarse_heap.empty()){ cand.push_back(coarse_heap.top().second); coarse_heap.pop(); }
        std::priority_queue<std::pair<float,int>> result;
        for(int idx : cand){
            float dist = InnerProductSIMD(base + (size_t)idx * vecdim, query, vecdim);
            if(result.size() < k) result.push(std::make_pair(dist, idx));
            else if(dist < result.top().first){ result.pop(); result.push(std::make_pair(dist, idx)); }
        }

        struct timeval nv; gettimeofday(&nv, NULL);
        int64_t diff = (nv.tv_sec*Converter + nv.tv_usec) - (val.tv_sec*Converter + val.tv_usec);
        std::set<uint32_t> gt; for(size_t j=0;j<k;++j) gt.insert((uint32_t)test_gt[j+i*test_gt_d]);
        size_t acc=0; while(result.size()){ if(gt.find((uint32_t)result.top().second)!=gt.end())++acc; result.pop(); }
        results[i] = {(float)acc/k, diff};
    }
}

void run_openmp_subspace(const float* base, const float* test_query, const int* test_gt,
                         size_t base_number, size_t vecdim, size_t test_gt_d,
                         size_t test_number, size_t k, SearchResult* results, int num_threads)
{
    const unsigned long Converter = 1000 * 1000;
    size_t M = 8, Ks = 256;
    const char* env_m = std::getenv("PQ_M");
    if(env_m) M = (size_t)std::atoi(env_m);
    size_t p = 500;
    const char* env_p = std::getenv("PQ_P");
    if(env_p) p = (size_t)std::atoi(env_p);

    for(size_t i = 0; i < test_number; ++i){
        struct timeval val;
        gettimeofday(&val, NULL);
        auto res = pq_search_omp_lut(base, g_pq_codes.data(), g_pq_centroids.data(),
                                      test_query + i * vecdim, base_number, vecdim,
                                      M, Ks, k, p, num_threads);
        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) -
                       (val.tv_sec * Converter + val.tv_usec);
        std::set<uint32_t> gtset;
        for(size_t j = 0; j < k; ++j) gtset.insert((uint32_t)test_gt[j + i * test_gt_d]);
        size_t acc = 0;
        while(res.size()){
            int x = res.top().second;
            if(gtset.find((uint32_t)x) != gtset.end()) ++acc;
            res.pop();
        }
        results[i] = {(float)acc / k, diff};
    }
}
#endif

void run_serial(const float* base, const float* test_query, const int* test_gt,
                size_t base_number, size_t vecdim, size_t test_gt_d,
                size_t test_number, size_t k, SearchResult* results)
{
    const unsigned long Converter = 1000 * 1000;
    for(size_t i = 0; i < test_number; ++i){
        struct timeval val;
        gettimeofday(&val, NULL);
        auto res = pq_solve(base, test_query + i * vecdim, base_number, vecdim, k);
        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) -
                       (val.tv_sec * Converter + val.tv_usec);
        std::set<uint32_t> gtset;
        for(size_t j = 0; j < k; ++j) gtset.insert((uint32_t)test_gt[j + i * test_gt_d]);
        size_t acc = 0;
        while(res.size()){
            int x = res.top().second;
            if(gtset.find((uint32_t)x) != gtset.end()) ++acc;
            res.pop();
        }
        results[i] = {(float)acc / k, diff};
    }
}

int main()
{
    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    std::string data_path = "data/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    test_number = 2000;
    const size_t k = 10;

    build_pq(base, base_number, vecdim);

    const char* mode = std::getenv("PQ_MODE");
    std::string mode_str = mode ? mode : "serial";
    const char* nt = std::getenv("NUM_THREADS");
    int num_threads = nt ? std::atoi(nt) : 4;

    std::vector<SearchResult> results(test_number);

    if(mode_str == "pthread"){
        std::cerr << "PQ Pthread query-level, threads=" << num_threads << "\n";
        run_pthread(base, test_query, test_gt, base_number, vecdim,
                    test_gt_d, test_number, k, results.data(), num_threads);
    }
#ifdef _OPENMP
    else if(mode_str == "openmp"){
        std::cerr << "PQ OpenMP query-level, threads=" << num_threads << "\n";
        run_openmp_query(base, test_query, test_gt, base_number, vecdim,
                         test_gt_d, test_number, k, results.data(), num_threads);
    }
    else if(mode_str == "pthread_subspace"){
        std::cerr << "PQ Pthread subspace-level (build_lut), threads=" << num_threads << "\n";
        run_pthread_subspace(base, test_query, test_gt, base_number, vecdim,
                             test_gt_d, test_number, k, results.data(), num_threads);
    }
    else if(mode_str == "openmp_subspace"){
        std::cerr << "PQ OpenMP subspace-level (build_lut), threads=" << num_threads << "\n";
        run_openmp_subspace(base, test_query, test_gt, base_number, vecdim,
                            test_gt_d, test_number, k, results.data(), num_threads);
    }
#endif
    else {
        std::cerr << "PQ serial mode\n";
        run_serial(base, test_query, test_gt, base_number, vecdim,
                   test_gt_d, test_number, k, results.data());
    }

    float avg_recall = 0, avg_latency = 0;
    for(size_t i = 0; i < test_number; ++i){
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }
    std::cout << "mode: " << mode_str << "  num_threads: " << num_threads << "\n";
    std::cout << "average recall: " << avg_recall / test_number << "\n";
    std::cout << "average latency (us): " << avg_latency / test_number << "\n";

    delete[] base;
    delete[] test_query;
    delete[] test_gt;
    return 0;
}
