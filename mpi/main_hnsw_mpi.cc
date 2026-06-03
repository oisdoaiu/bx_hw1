/*
 * main_hnsw_mpi.cc — MPI 并行化 HNSW 搜索
 *
 * 方案 B: 数据集划分 + 独立 HNSW + MPI 并行搜索 + top-k merge
 *   划分方式: random (按索引范围) / kmeans (启发式聚类)
 *
 * 环境变量:
 *   HNSW_MODE=random | kmeans (划分方式, 默认 random)
 *   HNSW_EF (默认 50)
 *   NUM_THREADS (未在纯 MPI 模式中使用)
 */

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
#include <cmath>
#include "../hnswlib/hnswlib/hnswlib.h"
#include "../hnswlib/hnswlib/hnswalg.h"
#include "../flat_simd.h"
#include "mpi_utils.h"

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if(!fin.is_open()){
        std::cerr << "ERROR: cannot open " << data_path << "\n";
        std::exit(1);
    }
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    int sz = sizeof(T);
    for(size_t i = 0; i < n; ++i){
        fin.read(((char*)data + i * d * sz), d * sz);
    }
    fin.close();
    return data;
}

/* ================================================================
 * K-Means 聚类划分（启发式划分）
 * 将 base 向量聚成 nparts 个簇，每个 rank 获得一个簇的全部向量
 *
 * 与随机划分相比: K-Means 将相似向量分到同一 partition，
 * 理论上 HNSW 在同类数据上构建的图质量更好
 * ================================================================ */
inline void kmeans_partition(
    const float* base, size_t base_number, size_t vecdim, int nparts,
    std::vector<int>& assignment)  // assignment[i] = rank for vector i
{
    assignment.resize(base_number);

    // 初始化质心：均匀采样
    std::vector<float> centroids(nparts * vecdim);
    for(int p = 0; p < nparts; p++){
        size_t idx = (p * (base_number / nparts)) % base_number;
        std::memcpy(centroids.data() + p * vecdim,
                    base + idx * vecdim, vecdim * sizeof(float));
    }

    // K-Means 迭代
    for(int iter = 0; iter < 10; iter++){
        // 分配步骤
        for(size_t i = 0; i < base_number; i++){
            const float* v = base + i * vecdim;
            float best_dist = FLT_MAX;
            int best_p = 0;
            for(int p = 0; p < nparts; p++){
                float d = InnerProductSIMD(v, centroids.data() + p * vecdim, vecdim);
                if(d < best_dist){ best_dist = d; best_p = p; }
            }
            assignment[i] = best_p;
        }

        // 更新步骤
        std::vector<float> sum(nparts * vecdim, 0.0f);
        std::vector<int> counts(nparts, 0);
        for(size_t i = 0; i < base_number; i++){
            int p = assignment[i];
            const float* v = base + i * vecdim;
            float* dst = sum.data() + p * vecdim;
            for(size_t d = 0; d < vecdim; d++) dst[d] += v[d];
            counts[p]++;
        }
        for(int p = 0; p < nparts; p++){
            if(counts[p] > 0){
                float inv = 1.0f / (float)counts[p];
                float* cent = centroids.data() + p * vecdim;
                for(size_t d = 0; d < vecdim; d++)
                    cent[d] = sum[p * vecdim + d] * inv;
            }
        }
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const char* part_mode = std::getenv("HNSW_MODE");
    std::string pm = part_mode ? part_mode : "random";

    std::string dp = "../data/";

    // === 阶段 1: rank 0 加载数据 ===
    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    float* base = nullptr;
    float* test_query = nullptr;
    int* test_gt = nullptr;

    if(rank == 0){
        test_query = LoadData<float>(dp + "DEEP100K.query.fbin", test_number, vecdim);
        test_gt    = LoadData<int>(dp + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
        base       = LoadData<float>(dp + "DEEP100K.base.100k.fbin", base_number, vecdim);
        test_number = 2000;
    }

    // 广播元数据
    size_t meta[4];
    if(rank == 0){ meta[0]=base_number; meta[1]=vecdim; meta[2]=test_number; meta[3]=test_gt_d; }
    MPI_Bcast(meta, 4, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    base_number=meta[0]; vecdim=meta[1]; test_number=meta[2]; test_gt_d=meta[3];
    const size_t k = 10;

    // 非 rank 0 分配内存
    if(rank != 0){
        base       = new float[base_number * vecdim];
        test_query = new float[test_number * vecdim];
        test_gt    = new int[test_number * test_gt_d];
    }
    MPI_Bcast(base,       base_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_query, test_number * vecdim,  MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_gt,    test_number * test_gt_d, MPI_INT, 0, MPI_COMM_WORLD);

    // === 阶段 2: 数据划分 ===
    std::vector<int> partition_assignment(base_number);

    if(pm == "kmeans" && rank == 0){
        std::cerr << "Rank 0: K-Means partitioning (nparts=" << nranks << ")...\n";
        kmeans_partition(base, base_number, vecdim, nranks, partition_assignment);
    } else if(pm == "random"){
        // 随机划分 = 按索引范围块划分（数据本身无序，等价于随机）
        for(size_t i = 0; i < base_number; i++)
            partition_assignment[i] = (int)((size_t)i * nranks / base_number);
    }

    // 广播 K-Means 划分结果
    if(pm == "kmeans")
        MPI_Bcast(partition_assignment.data(), base_number, MPI_INT, 0, MPI_COMM_WORLD);

    // 统计每个 rank 分配到的向量数
    size_t my_count = 0;
    for(size_t i = 0; i < base_number; i++)
        if(partition_assignment[i] == rank) my_count++;

    // 收集本 rank 的数据（向量 + 全局索引）
    std::vector<float> my_base(my_count * vecdim);
    std::vector<size_t> my_labels(my_count);
    {
        size_t pos = 0;
        for(size_t i = 0; i < base_number; i++){
            if(partition_assignment[i] == rank){
                std::memcpy(my_base.data() + pos * vecdim,
                            base + i * vecdim, vecdim * sizeof(float));
                my_labels[pos] = i; // 保持全局索引
                pos++;
            }
        }
    }

    if(rank == 0){
        std::cerr << "Partition mode: " << pm << "\n";
        std::cerr << "Min vectors/rank: " << (base_number / nranks)
                  << " (ideal), actual range: varies\n";
    }

    // === 阶段 3: 各 rank 独立构建 HNSW ===
    const char* env_ef = std::getenv("HNSW_EF");
    int ef = env_ef ? std::atoi(env_ef) : 50;

    hnswlib::InnerProductSpace hnsw_space(vecdim);
    hnswlib::HierarchicalNSW<float> hnsw_index(&hnsw_space, my_count, 16, 200);
    hnsw_index.ef_ = ef;

    for(size_t i = 0; i < my_count; i++){
        // 使用全局原始索引作为 label，保证 merge 时索引一致性
        hnsw_index.addPoint((void*)(my_base.data() + i * vecdim), my_labels[i]);
    }

    if(rank == 0)
        std::cerr << "HNSW built per rank: n=" << my_count
                  << " dim=" << vecdim << " ef=" << ef << "\n";

    // === 阶段 4: 并行搜索 + recall 计算 ===
    double t_start = MPI_Wtime();
    std::vector<float> recalls(test_number, 0.0f);

    for(size_t qi = 0; qi < test_number; qi++){
        const float* q = test_query + qi * vecdim;

        // 本地 HNSW 搜索
        auto local_pq = hnsw_index.searchKnn((void*)q, k);

        // hnswlib InnerProductSpace 返回 distance = 1.0 - inner_product
        // 与 IVF 统一：较小距离 = 更好匹配
        // searchKnn 返回的 priority_queue: top() = 最大距离 = 最差匹配
        std::vector<std::pair<float,int>> tmp;
        while(!local_pq.empty()){
            tmp.push_back(std::make_pair(local_pq.top().first,
                                         (int)local_pq.top().second));
            local_pq.pop();
        }
        // tmp[0]=worst, tmp[k-1]=best (最小距离)
        // 反转压入 max-heap: top()=最大距离=最差, 与 IVF 的 result 语义一致
        std::priority_queue<std::pair<float,int>> local_result;
        for(int j = (int)tmp.size()-1; j >= 0; j--)
            local_result.push(tmp[j]);

        // 序列化
        std::vector<char> local_buf;
        serialize_pq(local_result, local_buf);
        int local_sz = (int)local_buf.size();

        // Gather 所有 rank 的局部 top-k
        std::vector<int> all_sizes(nranks);
        MPI_Gather(&local_sz, 1, MPI_INT,
                   all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

        std::vector<int> displs(nranks, 0);
        std::vector<char> all_bufs;
        if(rank == 0){
            int total_sz = 0;
            for(int r = 0; r < nranks; r++){
                displs[r] = total_sz;
                total_sz += all_sizes[r];
            }
            all_bufs.resize(total_sz);
        }

        MPI_Gatherv(local_buf.data(), local_sz, MPI_BYTE,
                    all_bufs.data(), all_sizes.data(), displs.data(), MPI_BYTE,
                    0, MPI_COMM_WORLD);

        // rank 0: merge 并计算 recall
        if(rank == 0){
            std::vector<std::vector<char>> rank_bufs(nranks);
            for(int r = 0; r < nranks; r++){
                if(all_sizes[r] > 0)
                    rank_bufs[r].assign(all_bufs.begin() + displs[r],
                                        all_bufs.begin() + displs[r] + all_sizes[r]);
            }
            auto global_pq = merge_topk(rank_bufs, k);
            recalls[qi] = compute_recall(global_pq, test_gt + qi * test_gt_d, k, test_gt_d);
        }
    }

    double t_end = MPI_Wtime();

    // === 阶段 5: 汇总结果 ===
    float local_ar = 0;
    for(size_t i = 0; i < test_number; i++) local_ar += recalls[i];
    float global_ar = 0;
    MPI_Reduce(&local_ar, &global_ar, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        double total_time = t_end - t_start;
        double avg_latency_us = (total_time * 1e6) / test_number;
        std::cout << "=== MPI HNSW Results ===\n";
        std::cout << "nranks: " << nranks << "\n";
        std::cout << "partition_mode: " << pm << "\n";
        std::cout << "ef: " << ef << "\n";
        std::cout << "average recall: " << global_ar / test_number << "\n";
        std::cout << "total_time (s): " << total_time << "\n";
        std::cout << "average latency (us): " << avg_latency_us << "\n";
    }

    // 清理
    delete[] base;
    delete[] test_query;
    delete[] test_gt;

    MPI_Finalize();
    return 0;
}
