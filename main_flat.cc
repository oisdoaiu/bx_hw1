#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <cstdlib>
#include "flat_simd.h"

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

        auto res = flat_simd_search(p->base, p->test_query + qi * p->vecdim,
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
            if(gtset.find((uint32_t)x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc / p->k;

        p->results[qi] = {recall, diff};
    }
    return nullptr;
}

void run_serial(const float* base, const float* test_query, const int* test_gt,
                size_t base_number, size_t vecdim, size_t test_gt_d,
                size_t test_number, size_t k, SearchResult* results)
{
    const unsigned long Converter = 1000 * 1000;

    for(size_t i = 0; i < test_number; ++i){
        struct timeval val;
        gettimeofday(&val, NULL);

        auto res = flat_simd_search(base, test_query + i * vecdim,
                                     base_number, vecdim, k);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) -
                       (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(size_t j = 0; j < k; ++j){
            gtset.insert((uint32_t)test_gt[j + i * test_gt_d]);
        }

        size_t acc = 0;
        while(res.size()){
            int x = res.top().second;
            if(gtset.find((uint32_t)x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc / k;

        results[i] = {recall, diff};
    }
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

    for(int t = 0; t < num_threads; t++){
        pthread_join(threads[t], nullptr);
    }

    delete[] threads;
    delete[] params;
}

#ifdef _OPENMP
void run_openmp(const float* base, const float* test_query, const int* test_gt,
                size_t base_number, size_t vecdim, size_t test_gt_d,
                size_t test_number, size_t k, SearchResult* results, int num_threads)
{
    const unsigned long Converter = 1000 * 1000;

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for(size_t i = 0; i < test_number; ++i){
        struct timeval val;
        gettimeofday(&val, NULL);

        auto res = flat_simd_search(base, test_query + i * vecdim,
                                     base_number, vecdim, k);

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) -
                       (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(size_t j = 0; j < k; ++j){
            gtset.insert((uint32_t)test_gt[j + i * test_gt_d]);
        }

        size_t acc = 0;
        while(res.size()){
            int x = res.top().second;
            if(gtset.find((uint32_t)x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc / k;

        results[i] = {recall, diff};
    }
}
#endif

// ========== Base-level parallel (divide base vectors per query) ==========

struct BaseThreadParam
{
    int thread_id;
    int num_threads;
    const float* base;
    const float* query;
    size_t base_number;
    size_t vecdim;
    size_t k;
    std::vector<std::pair<float, int>> local_candidates;
};

void* base_worker(void* arg)
{
    BaseThreadParam* p = (BaseThreadParam*)arg;
    size_t chunk = p->base_number / p->num_threads;
    size_t start = p->thread_id * chunk;
    size_t end = (p->thread_id == p->num_threads - 1) ? p->base_number : start + chunk;

    std::priority_queue<std::pair<float, int>> local_pq;

    for(size_t i = start; i < end; i++){
        float dist = InnerProductSIMD(p->base + i * p->vecdim, p->query, p->vecdim);

        if(local_pq.size() < p->k){
            local_pq.push(std::make_pair(dist, static_cast<int>(i)));
        }else if(dist < local_pq.top().first){
            local_pq.pop();
            local_pq.push(std::make_pair(dist, static_cast<int>(i)));
        }
    }

    p->local_candidates.clear();
    while(!local_pq.empty()){
        p->local_candidates.push_back(local_pq.top());
        local_pq.pop();
    }

    return nullptr;
}

void run_pthread_basepar(const float* base, const float* test_query, const int* test_gt,
                         size_t base_number, size_t vecdim, size_t test_gt_d,
                         size_t test_number, size_t k, SearchResult* results, int num_threads)
{
    const unsigned long Converter = 1000 * 1000;
    std::vector<BaseThreadParam> params(num_threads);
    std::vector<pthread_t> threads(num_threads);

    for(size_t qi = 0; qi < test_number; qi++){
        struct timeval val;
        gettimeofday(&val, NULL);

        for(int t = 0; t < num_threads; t++){
            params[t].thread_id = t;
            params[t].num_threads = num_threads;
            params[t].base = base;
            params[t].query = test_query + qi * vecdim;
            params[t].base_number = base_number;
            params[t].vecdim = vecdim;
            params[t].k = k;
            params[t].local_candidates.clear();
            pthread_create(&threads[t], nullptr, base_worker, &params[t]);
        }

        for(int t = 0; t < num_threads; t++){
            pthread_join(threads[t], nullptr);
        }

        // merge local results
        std::priority_queue<std::pair<float, int>> merged;
        for(int t = 0; t < num_threads; t++){
            for(auto& cand : params[t].local_candidates){
                if(merged.size() < k){
                    merged.push(cand);
                }else if(cand.first < merged.top().first){
                    merged.pop();
                    merged.push(cand);
                }
            }
        }

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) -
                       (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(size_t j = 0; j < k; ++j){
            gtset.insert((uint32_t)test_gt[j + qi * test_gt_d]);
        }

        size_t acc = 0;
        while(merged.size()){
            int x = merged.top().second;
            if(gtset.find((uint32_t)x) != gtset.end()){
                ++acc;
            }
            merged.pop();
        }
        float recall = (float)acc / k;

        results[qi] = {recall, diff};
    }
}

#ifdef _OPENMP
void run_openmp_basepar(const float* base, const float* test_query, const int* test_gt,
                        size_t base_number, size_t vecdim, size_t test_gt_d,
                        size_t test_number, size_t k, SearchResult* results, int num_threads)
{
    const unsigned long Converter = 1000 * 1000;

    for(size_t qi = 0; qi < test_number; qi++){
        struct timeval val;
        gettimeofday(&val, NULL);

        const float* query = test_query + qi * vecdim;

        // thread-local storage for merge
        std::vector<std::pair<float, int>> all_candidates;

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();
            int nt = omp_get_num_threads();
            size_t chunk = base_number / nt;
            size_t start = tid * chunk;
            size_t end = (tid == nt - 1) ? base_number : start + chunk;

            // Manual chunk assignment (equivalent to schedule(static))
            // Each thread processes its own [start, end) range

            std::priority_queue<std::pair<float, int>> local_pq;

            for(size_t i = start; i < end; i++){
                float dist = InnerProductSIMD(base + i * vecdim, query, vecdim);

                if(local_pq.size() < k){
                    local_pq.push(std::make_pair(dist, static_cast<int>(i)));
                }else if(dist < local_pq.top().first){
                    local_pq.pop();
                    local_pq.push(std::make_pair(dist, static_cast<int>(i)));
                }
            }

            // collect local results into shared vector (critical section)
            #pragma omp critical
            {
                while(!local_pq.empty()){
                    all_candidates.push_back(local_pq.top());
                    local_pq.pop();
                }
            }
        }

        // merge
        std::priority_queue<std::pair<float, int>> merged;
        for(auto& cand : all_candidates){
            if(merged.size() < k){
                merged.push(cand);
            }else if(cand.first < merged.top().first){
                merged.pop();
                merged.push(cand);
            }
        }

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) -
                       (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(size_t j = 0; j < k; ++j){
            gtset.insert((uint32_t)test_gt[j + qi * test_gt_d]);
        }

        size_t acc = 0;
        while(merged.size()){
            int x = merged.top().second;
            if(gtset.find((uint32_t)x) != gtset.end()){
                ++acc;
            }
            merged.pop();
        }
        float recall = (float)acc / k;

        results[qi] = {recall, diff};
    }
}
#endif

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "data/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    test_number = 2000;

    const size_t k = 10;

    const char* mode = std::getenv("FLAT_MODE");
    std::string mode_str = mode ? mode : "serial";

    const char* nt = std::getenv("NUM_THREADS");
    int num_threads = nt ? std::atoi(nt) : 4;

    std::vector<SearchResult> results;
    results.resize(test_number);

    if(mode_str == "pthread"){
        std::cerr << "Running Pthread mode (query-level), threads=" << num_threads << "\n";
        run_pthread(base, test_query, test_gt, base_number, vecdim,
                    test_gt_d, test_number, k, results.data(), num_threads);
    }
    else if(mode_str == "pthread_base"){
        std::cerr << "Running Pthread base-parallel mode, threads=" << num_threads << "\n";
        run_pthread_basepar(base, test_query, test_gt, base_number, vecdim,
                           test_gt_d, test_number, k, results.data(), num_threads);
    }
#ifdef _OPENMP
    else if(mode_str == "openmp"){
        std::cerr << "Running OpenMP mode (query-level), threads=" << num_threads << "\n";
        run_openmp(base, test_query, test_gt, base_number, vecdim,
                   test_gt_d, test_number, k, results.data(), num_threads);
    }
    else if(mode_str == "openmp_base"){
        std::cerr << "Running OpenMP base-parallel mode, threads=" << num_threads << "\n";
        run_openmp_basepar(base, test_query, test_gt, base_number, vecdim,
                          test_gt_d, test_number, k, results.data(), num_threads);
    }
#endif
    else {
        std::cerr << "Running serial mode\n";
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
