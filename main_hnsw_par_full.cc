#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
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

struct Result { float recall; float latency; };

void run_serial(hnswlib::HierarchicalNSW<float>* hnsw,
    const float* q, const int* gt, size_t vd, size_t gtd,
    size_t tn, size_t k, Result* res, int sample)
{
    size_t step = sample > 0 ? tn / sample : 1;
    for (size_t i = 0; i < tn; i += step) {
        struct timeval val; gettimeofday(&val, NULL);
        auto pq = hnsw->searchKnn((void*)(q + i * vd), k);
        struct timeval nv; gettimeofday(&nv, NULL);
        float d = (nv.tv_sec * 1e6 + nv.tv_usec) -
                  (val.tv_sec * 1e6 + val.tv_usec);
        std::set<uint32_t> gt2;
        for (size_t j = 0; j < k; ++j) gt2.insert((uint32_t)gt[j + i * gtd]);
        size_t acc = 0;
        while (!pq.empty()) {
            if (gt2.find((uint32_t)pq.top().second) != gt2.end()) ++acc;
            pq.pop();
        }
        res[i / step] = {(float)acc / k, d};
    }
}

void run_par(hnswlib::HierarchicalNSW<float>* hnsw,
    const float* q, const int* gt, size_t vd, size_t gtd,
    size_t tn, size_t k, int K, int nt,
    Result* res, int sample)
{
    size_t step = sample > 0 ? tn / sample : 1;
    for (size_t i = 0; i < tn; i += step) {
        struct timeval val; gettimeofday(&val, NULL);
        auto pq = hnsw_par_search(hnsw, (void*)(q + i * vd), k, K, nt);
        struct timeval nv; gettimeofday(&nv, NULL);
        float d = (nv.tv_sec * 1e6 + nv.tv_usec) -
                  (val.tv_sec * 1e6 + val.tv_usec);
        std::set<uint32_t> gt2;
        for (size_t j = 0; j < k; ++j) gt2.insert((uint32_t)gt[j + i * gtd]);
        size_t acc = 0;
        while (!pq.empty()) {
            if (gt2.find((uint32_t)pq.top().second) != gt2.end()) ++acc;
            pq.pop();
        }
        res[i / step] = {(float)acc / k, d};
    }
}

int main()
{
    size_t tn = 0, bn = 0, gtd = 0, vd = 0;
    auto tq   = LoadData<float>("data/DEEP100K.query.fbin", tn, vd);
    auto tg   = LoadData<int>("data/DEEP100K.gt.query.100k.top100.bin", tn, gtd);
    auto base = LoadData<float>("data/DEEP100K.base.100k.fbin", bn, vd);
    tn = 2000; const size_t k = 10;
    const int sample = 500; // 500 sample queries

    int ef_vals[] = {50, 100, 200};
    int nt_vals[] = {2, 4, 8};
    int K_vals[]  = {1, 2, 4, 8, 16, 32};
    const int M = 16, efc = 200;

    std::vector<Result> res(sample);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "ef\tnt\tK\trecall\tlatency(us)\tvs_serial\n";

    for (int ef_val : ef_vals) {
        // Build fresh index for each ef
        hnswlib::InnerProductSpace space(vd);
        hnswlib::HierarchicalNSW<float> hnsw(&space, bn, M, efc);
        hnsw.ef_ = ef_val;
        hnsw.addPoint((void*)base, 0);
        #pragma omp parallel for
        for (size_t i = 1; i < bn; i++)
            hnsw.addPoint((void*)(base + i * vd), (size_t)i);

        // Serial baseline
        run_serial(&hnsw, tq, tg, vd, gtd, tn, k, res.data(), sample);
        float s_recall = 0, s_lat = 0;
        for (size_t i = 0; i < res.size(); i++)
            { s_recall += res[i].recall; s_lat += res[i].latency; }
        s_recall /= res.size(); s_lat /= res.size();

        for (int nt : nt_vals) {
            for (int K : K_vals) {
                run_par(&hnsw, tq, tg, vd, gtd, tn, k, K, nt, res.data(), sample);
                float p_recall = 0, p_lat = 0;
                for (size_t i = 0; i < res.size(); i++)
                    { p_recall += res[i].recall; p_lat += res[i].latency; }
                p_recall /= res.size(); p_lat /= res.size();

                std::cout << ef_val << "\t" << nt << "\t" << K << "\t"
                          << p_recall << "\t" << p_lat << "\t"
                          << (s_lat / p_lat) << "x\n";
            }
        }
        // Print serial baseline once per ef
        std::cout << "# serial ef=" << ef_val << " recall=" << s_recall
                  << " latency=" << s_lat << "\n";
    }

    delete[] base; delete[] tq; delete[] tg;
    return 0;
}
