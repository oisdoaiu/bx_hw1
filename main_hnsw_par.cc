#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <cstdlib>
#include <algorithm>
#include "hnswlib/hnswlib/hnswlib.h"
#include "hnswlib/hnswlib/hnswalg.h"
#include "hnsw_parallel_exp.h"

template<typename T>
T* LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n, 4); fin.read((char*)&d, 4);
    T* data = new T[n * d];
    int sz = sizeof(T);
    for (int i = 0; i < n; ++i) fin.read(((char*)data + i * d * sz), d * sz);
    fin.close();
    return data;
}

struct SearchResult { float recall; int64_t latency; };

// Serial baseline: uses hnswlib's native searchKnn
void run_hnswlib_serial(
    hnswlib::HierarchicalNSW<float>* hnsw,
    const float* query, const int* gt,
    size_t vd, size_t gtd, size_t tn, size_t k,
    SearchResult* res)
{
    for (size_t i = 0; i < tn; ++i) {
        struct timeval val; gettimeofday(&val, NULL);
        auto pq = hnsw->searchKnn((void*)(query + i * vd), k);
        struct timeval nv; gettimeofday(&nv, NULL);
        int64_t d = (nv.tv_sec * 1000000LL + nv.tv_usec) -
                    (val.tv_sec * 1000000LL + val.tv_usec);

        std::set<uint32_t> gt2;
        for (size_t j = 0; j < k; ++j) gt2.insert((uint32_t)gt[j + i * gtd]);
        size_t acc = 0;
        while (!pq.empty()) {
            if (gt2.find((uint32_t)pq.top().second) != gt2.end()) ++acc;
            pq.pop();
        }
        res[i] = {(float)acc / k, d};
    }
}

// Parallel expansion: uses our custom parallel frontier search
void run_par_exp(
    hnswlib::HierarchicalNSW<float>* hnsw,
    const float* query, const int* gt,
    size_t vd, size_t gtd, size_t tn, size_t k,
    int expansion_K, int num_threads,
    SearchResult* res)
{
    for (size_t i = 0; i < tn; ++i) {
        struct timeval val; gettimeofday(&val, NULL);
        auto pq = hnsw_par_search(
            hnsw, (void*)(query + i * vd), k, expansion_K, num_threads);
        struct timeval nv; gettimeofday(&nv, NULL);
        int64_t d = (nv.tv_sec * 1000000LL + nv.tv_usec) -
                    (val.tv_sec * 1000000LL + val.tv_usec);

        std::set<uint32_t> gt2;
        for (size_t j = 0; j < k; ++j) gt2.insert((uint32_t)gt[j + i * gtd]);
        size_t acc = 0;
        while (!pq.empty()) {
            if (gt2.find((uint32_t)pq.top().second) != gt2.end()) ++acc;
            pq.pop();
        }
        res[i] = {(float)acc / k, d};
    }
}

int main()
{
    size_t tn = 0, bn = 0, gtd = 0, vd = 0;
    std::string dp = "data/";
    auto tq   = LoadData<float>(dp + "DEEP100K.query.fbin", tn, vd);
    auto tg   = LoadData<int>(dp + "DEEP100K.gt.query.100k.top100.bin", tn, gtd);
    auto base = LoadData<float>(dp + "DEEP100K.base.100k.fbin", bn, vd);
    tn = 2000;
    const size_t k = 10;

    // --- Build hnsw index ---
    int M = 16, efc = 200;
    const char* env_ef = std::getenv("HNSW_EF");
    int ef_val = env_ef ? std::atoi(env_ef) : 50;

    hnswlib::InnerProductSpace space(vd);
    hnswlib::HierarchicalNSW<float> hnsw(&space, bn, M, efc);
    hnsw.ef_ = ef_val;

    hnsw.addPoint((void*)base, 0);
#pragma omp parallel for
    for (size_t i = 1; i < bn; i++)
        hnsw.addPoint((void*)(base + i * vd), (size_t)i);

    std::cerr << "HNSW built: n=" << bn << " dim=" << vd
              << " M=" << M << " efc=" << efc
              << " ef=" << ef_val << "\n";

    // --- Benchmark configs ---
    // Test expansion K = 1 (serial), 2, 4, 8, 16
    const int num_threads = 4; // fixed thread count for fair comparison
    std::vector<int> K_values = {1, 2, 4, 8, 16};

    // --- Serial baseline (native hnswlib) ---
    std::vector<SearchResult> serial_res(tn);
    std::cerr << "Running serial baseline (native hnswlib)...\n";
    run_hnswlib_serial(&hnsw, tq, tg, vd, gtd, tn, k, serial_res.data());
    float serial_recall = 0;
    float serial_lat = 0;
    for (size_t i = 0; i < tn; ++i) {
        serial_recall += serial_res[i].recall;
        serial_lat += serial_res[i].latency;
    }
    serial_recall /= tn;
    serial_lat /= tn;

    // --- Parallel expansion benchmarks ---
    std::cerr << "\n=== HNSW Parallel Expansion Benchmark ===\n";
    std::cerr << "ef=" << ef_val << "  num_threads=" << num_threads << "\n\n";
    std::cerr << "K\trecall\tlatency(us)\tvs_serial\n";

    for (int K : K_values) {
        std::vector<SearchResult> res(tn);
        run_par_exp(&hnsw, tq, tg, vd, gtd, tn, k, K, num_threads, res.data());

        float avg_recall = 0;
        float avg_lat = 0;
        for (size_t i = 0; i < tn; ++i) {
            avg_recall += res[i].recall;
            avg_lat += res[i].latency;
        }
        avg_recall /= tn;
        avg_lat /= tn;

        std::cerr << K << "\t" << avg_recall << "\t" << avg_lat
                  << "\t" << (serial_lat / avg_lat) << "x\n";
    }

    // Also print header for easy copy-paste
    std::cout << "\n--- SUMMARY TABLE ---\n";
    std::cout << "ef=" << ef_val << "  nt=" << num_threads << "\n";
    std::cout << "serial\t" << serial_recall << "\t" << serial_lat << "\t1.00x\n";
    for (int K : K_values) {
        std::vector<SearchResult> res(tn);
        run_par_exp(&hnsw, tq, tg, vd, gtd, tn, k, K, num_threads, res.data());
        float ar = 0, al = 0;
        for (size_t i = 0; i < tn; ++i) { ar += res[i].recall; al += res[i].latency; }
        std::cout << "par_K" << K << "\t" << ar/tn << "\t" << al/tn
                  << "\t" << (serial_lat / (al/tn)) << "x\n";
    }

    delete[] base; delete[] tq; delete[] tg;
    return 0;
}
