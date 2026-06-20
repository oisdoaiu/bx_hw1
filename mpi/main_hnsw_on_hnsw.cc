#include <mpi.h>
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <algorithm>
#include <sys/time.h>
#include <cstdlib>
#include <cfloat>
#include "../hnswlib/hnswlib/hnswlib.h"
#include "../hnswlib/hnswlib/hnswalg.h"
#include "../flat_simd.h"
#include "mpi_utils.h"


template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d){

    std::ifstream fin(data_path, std::ios::in | std::ios::binary);
    if(!fin.is_open()){ std::cerr<<"ERROR: cannot open "<<data_path<<"\n"; std::exit(1); }
    fin.read((char*)&n,4); fin.read((char*)&d,4);
    T* data = new T[n*d]; int sz = sizeof(T);
    for(size_t i=0;i<n;++i) fin.read(((char*)data+i*d*sz), d*sz);
    fin.close(); return data;
}

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    const char* env_ef = std::getenv("HNSW_EF");
    int ef = env_ef ? std::atoi(env_ef) : 50;
    const char* env_coarse = std::getenv("HNSW_COARSE");
    size_t N_COARSE = env_coarse ? (size_t)std::atoi(env_coarse) : 256;
    const char* env_topm = std::getenv("HNSW_TOPM");
    size_t M_PROBE = env_topm ? (size_t)std::atoi(env_topm) : 2;
    std::string dp = "../data/";
    const size_t k = 10;
    // 加载数据
    size_t test_number=0, base_number=0, test_gt_d=0, vecdim=0;
    float *base=nullptr, *test_query=nullptr; int *test_gt=nullptr;
    if(rank == 0){
        test_query = LoadData<float>(dp+"DEEP100K.query.fbin", test_number, vecdim);
        test_gt = LoadData<int>(dp+"DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
        base = LoadData<float>(dp+"DEEP100K.base.100k.fbin", base_number, vecdim);
        test_number = 2000;
    }
    size_t meta[4];
    if(rank==0){meta[0]=base_number;meta[1]=vecdim;meta[2]=test_number;meta[3]=test_gt_d;}
    MPI_Bcast(meta,4, MPI_UNSIGNED_LONG,0, MPI_COMM_WORLD);
    base_number=meta[0]; vecdim=meta[1]; test_number=meta[2]; test_gt_d=meta[3];
    if(rank!=0){
        base=new float[base_number*vecdim];
        test_query=new float[test_number*vecdim];
        test_gt=new int[test_number*test_gt_d];
    }
    MPI_Bcast(base, base_number*vecdim, MPI_FLOAT,0, MPI_COMM_WORLD);
    MPI_Bcast(test_query, test_number*vecdim, MPI_FLOAT,0, MPI_COMM_WORLD);
    MPI_Bcast(test_gt, test_number*test_gt_d, MPI_INT,0, MPI_COMM_WORLD);
    // rank 0 构建两层索引
    // 2a. 选取代表点（均匀采样）
    std::vector<size_t> coarse_ids(N_COARSE);
    std::vector<float> coarse_vectors(N_COARSE * vecdim);
    if(rank == 0){
        size_t step = base_number / N_COARSE;
        for(size_t i = 0; i < N_COARSE; i++){
            size_t idx = i * step;
            if(idx >= base_number) idx = base_number - 1;
            coarse_ids[i] = idx;
            std::memcpy(coarse_vectors.data() + i * vecdim, base + idx * vecdim, vecdim * sizeof(float));
        }
        std::cerr << "Sampled " << N_COARSE << " coarse points\n";
    }
    // 2b. 构建上层 HNSW
    hnswlib::InnerProductSpace coarse_space(vecdim);
    hnswlib::HierarchicalNSW<float> coarse_hnsw(&coarse_space, N_COARSE, 16, 200);
    coarse_hnsw.ef_ = ef;
    if(rank == 0){
        for(size_t i = 0; i < N_COARSE; i++)
            coarse_hnsw.addPoint((void*)(coarse_vectors.data() + i * vecdim), i);
        std::cerr << "Upper HNSW built: " << N_COARSE << " nodes\n";
    }
    // 2c. 分配向量到代表点形成分区
    std::vector<int> partition_assign(base_number);
    if(rank == 0){
        for(size_t i = 0; i < base_number; i++){

            float best_d = FLT_MAX; int best_c = 0;
            for(size_t c = 0; c < N_COARSE; c++){
                float d = InnerProductSIMD(base + i * vecdim,
                    coarse_vectors.data() + c * vecdim, vecdim);
                if(d < best_d){ best_d = d; best_c = (int)c; }
            }
            partition_assign[i] = best_c;
        }
    }
    MPI_Bcast(partition_assign.data(), base_number, MPI_INT, 0, MPI_COMM_WORLD);
    // 2d. 统计每分区大小
    std::vector<int> part_sizes(N_COARSE, 0);
    for(size_t i = 0; i < base_number; i++)
        part_sizes[partition_assign[i]]++;
    // 2e. 按分区收集数据
    std::vector<std::vector<float>> part_vectors(N_COARSE);
    std::vector<std::vector<size_t>> part_labels(N_COARSE);
    for(size_t c = 0; c < N_COARSE; c++){
        part_vectors[c].reserve(part_sizes[c] * vecdim);
        part_labels[c].reserve(part_sizes[c]);
    }
    {
        std::vector<size_t> pos(N_COARSE, 0);
        for(size_t c = 0; c < N_COARSE; c++){
            part_vectors[c].resize(part_sizes[c] * vecdim);
            part_labels[c].resize(part_sizes[c]);
        }
        for(size_t i = 0; i < base_number; i++){
            int c = partition_assign[i];
            size_t p = pos[c]++;
            std::memcpy(part_vectors[c].data() + p * vecdim, base + i * vecdim, vecdim * sizeof(float));
            part_labels[c][p] = i;
        }
    }
    // 2f. 各 rank 构建下层 HNSW（只构建自己负责的分区）
    int parts_per_rank = (N_COARSE + nranks - 1) / nranks;
    int my_start = rank * parts_per_rank;
    int my_end = std::min(my_start + parts_per_rank, (int)N_COARSE);
    std::vector<hnswlib::InnerProductSpace*> lower_spaces;
    std::vector<hnswlib::HierarchicalNSW<float>*> lower_hnsws;
    std::vector<int> lower_part_ids; // 该 HNSW 对应的分区 ID
    for(int c = my_start; c < my_end; c++){
        if(part_sizes[c] == 0) continue;
        auto* sp = new hnswlib::InnerProductSpace(vecdim);
        auto* ix = new hnswlib::HierarchicalNSW<float>(sp, part_sizes[c], 16, 200);
        ix->ef_ = ef;
        for(int j = 0; j < part_sizes[c]; j++)
            ix->addPoint((void*)(part_vectors[c].data() + j * vecdim), part_labels[c][j]);
        lower_spaces.push_back(sp);
        lower_hnsws.push_back(ix);
        lower_part_ids.push_back(c);
    }
    if(rank == 0)
        std::cerr << "Lower HNSWs built: " << N_COARSE << " partitions\n";
    MPI_Barrier(MPI_COMM_WORLD);
    // 搜索
    double t_start = MPI_Wtime();
    std::vector<float> recalls(test_number, 0.0f);
    for(size_t qi = 0; qi < test_number; qi++){
        const float* q = test_query + qi * vecdim;
        // 3a. rank 0 搜索上层 HNSW
        std::vector<int> top_partitions;
        if(rank == 0){
            auto coarse_pq = coarse_hnsw.searchKnn((void*)q, M_PROBE);
            std::vector<std::pair<float, int>> tmp;
            while(!coarse_pq.empty()){
                tmp.push_back(std::make_pair(coarse_pq.top().first, (int)coarse_pq.top().second));
                coarse_pq.pop();
            }
            for(int j = (int)tmp.size()-1; j >= 0; j--)
                top_partitions.push_back(tmp[j].second);
        }
        // 广播 top partitions
        int top_count = (int)top_partitions.size();
        MPI_Bcast(&top_count, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if(rank != 0) top_partitions.resize(top_count);
        MPI_Bcast(top_partitions.data(), top_count, MPI_INT, 0, MPI_COMM_WORLD);
        // 3b. 各 rank 在负责的下层 HNSW 中搜索
        std::priority_queue<std::pair<float, int>> local_result;
        for(size_t li = 0; li < lower_hnsws.size(); li++){
            int part_id = lower_part_ids[li];
            // 检查该分区是否在 top-M 中
            bool in_top = false;
            for(int tp : top_partitions) if(tp == part_id){ in_top = true; break; }
            if(!in_top) continue;
            auto hnsw_pq = lower_hnsws[li]->searchKnn((void*)q, k);
            std::vector<std::pair<float, int>> tmp;
            while(!hnsw_pq.empty()){
                tmp.push_back(std::make_pair(hnsw_pq.top().first, (int)hnsw_pq.top().second));
                hnsw_pq.pop();
            }
            for(int j = (int)tmp.size()-1; j >= 0; j--)
                local_result.push(tmp[j]);
        }
        // 3c. MPI gather + merge
        std::vector<char> buf; serialize_pq(local_result, buf);
        int sz = (int)buf.size();
        std::vector<int> all_sizes(nranks);
        MPI_Gather(&sz,1, MPI_INT, all_sizes.data(),1, MPI_INT,0, MPI_COMM_WORLD);
        std::vector<int> displs(nranks,0); std::vector<char> all_bufs;
        if(rank==0){ int t=0; for(int r=0;r<nranks;r++){displs[r]=t;t+=all_sizes[r];} all_bufs.resize(t); }
        MPI_Gatherv(buf.data(), sz, MPI_BYTE, all_bufs.data(), all_sizes.data(), displs.data(), MPI_BYTE,0, MPI_COMM_WORLD);
        if(rank==0){
            std::vector<std::vector<char>> rbufs(nranks);
            for(int r=0;r<nranks;r++) if(all_sizes[r]>0) rbufs[r].assign(all_bufs.begin()+displs[r], all_bufs.begin()+displs[r]+all_sizes[r]);
            auto gpq = merge_topk(rbufs, k);
            recalls[qi] = compute_recall(gpq, test_gt + qi * test_gt_d, k, test_gt_d);
        }
    }
    double t_end = MPI_Wtime();
    // 汇总
    float local_ar = 0;
    for(size_t i=0;i<test_number;i++) local_ar += recalls[i];
    float global_ar = 0;
    MPI_Reduce(&local_ar,&global_ar,1, MPI_FLOAT, MPI_SUM,0, MPI_COMM_WORLD);
    if(rank == 0){
        std::cout << "=== HNSW-on-HNSW MPI Results ===\n";
        std::cout << "nranks: " << nranks << " N_coarse: " << N_COARSE
                  << " M_probe: " << M_PROBE << " ef: " << ef << "\n";
        std::cout << "average recall: " << global_ar / test_number << "\n";
        std::cout << "total_time (s): " << (t_end - t_start) << "\n";
        std::cout << "average latency (us): " << ((t_end - t_start) * 1e6 / test_number) << "\n";
    }
    for(size_t i=0;i<lower_hnsws.size();i++){ delete lower_hnsws[i]; delete lower_spaces[i]; }
    delete[] base; delete[] test_query; delete[] test_gt;
    MPI_Finalize(); return 0;
}
