#include "pq_simd.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <iomanip>

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
    return data;
}

int main(){
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "data/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    test_number = 2000;
    const size_t k = 10;

    std::vector<size_t> M_vals = {4, 6, 8, 12, 16, 24};
    std::vector<size_t> p_vals = {20, 50, 100, 200, 500};
    size_t Ks = 256;

    std::ofstream out("pq_results.txt");
    out << "=== PQ-SIMD Cross-Centroid Benchmark: M x p ===\n";
    out << "vecdim="<<vecdim<<" base_number="<<base_number<<" test_number="<<test_number<<" Ks="<<Ks<<" k="<<k<<"\n\n";

    std::cout << "M\\p";
    for(size_t pi=0; pi<p_vals.size(); pi++) std::cout << "\t" << p_vals[pi];
    std::cout << "\n";

    for(size_t mi=0; mi<M_vals.size(); mi++){
        size_t M = M_vals[mi];
        std::cout << M;

        size_t dsub = vecdim / M;
        std::vector<float> centroids(M * Ks * dsub);
        std::vector<uint8_t> codes_aos(base_number * M);
        std::vector<uint8_t> codes_soa(base_number * M);

        std::cerr << "training M="<<M<<" (dsub="<<dsub<<")...\n";
        train_pq_codebook(base, base_number, vecdim, centroids.data(), M, Ks, 15);
        encode_pq(base, codes_aos.data(), centroids.data(), base_number, vecdim, M, Ks);
        encode_pq_soa(codes_aos.data(), codes_soa.data(), base_number, M);
        std::cerr << "done\n";

        for(size_t pi=0; pi<p_vals.size(); pi++){
            size_t p = p_vals[pi];
            float sum_recall = 0, sum_latency = 0;

            for(int qi = 0; qi < test_number; qi++){
                const unsigned long Converter = 1000 * 1000;
                struct timeval val;
                gettimeofday(&val, NULL);

                auto res = pq_search(base, codes_soa.data(), centroids.data(),
                                     test_query + qi*vecdim, base_number, vecdim, M, Ks, k, p);

                struct timeval newVal;
                gettimeofday(&newVal, NULL);
                int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

                std::set<uint32_t> gtset;
                for(int j = 0; j < k; j++){
                    gtset.insert((uint32_t)test_gt[j + qi*test_gt_d]);
                }

                size_t acc = 0;
                while(res.size()){
                    int x = res.top().second;
                    if(gtset.find((uint32_t)x) != gtset.end()) acc++;
                    res.pop();
                }

                sum_recall += (float)acc/k;
                sum_latency += (float)diff;
            }

            float avg_recall = sum_recall / test_number;
            float avg_latency = sum_latency / test_number;

            std::cout << "\t" << std::fixed << std::setprecision(3) << avg_recall;
            std::cout.flush();

            out << "M="<<M<<" p="<<p
                << " recall="<<avg_recall
                << " latency_us="<<avg_latency
                << " dsub="<<dsub<<"\n";
        }
        std::cout << "\n";
    }

    out.close();
    std::cerr << "\nsaved to pq_results.txt\n";
    return 0;
}
