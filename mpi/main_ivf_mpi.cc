/*
 * main_ivf_mpi.cc — MPI 并行化 IVF 搜索
 *
 * 算法：rank 0 构建全局 IVF 索引 → 广播质心和 base 向量到所有 rank
 * → 按簇分配 inverted list（各 rank 只保留自己负责的簇）
 * → 广播查询 → 各 rank 本地搜索 → top-k 归并
 *
 * 环境变量:
 *   IVF_MODE=mpi_gather | mpi_reduce  (归并方式)
 *   IVF_NLIST (默认 256)
 *   IVF_NPROBE (默认 8)
 *   NUM_THREADS (未在纯 MPI 模式中使用，为后续混合并行预留)
 */

#include <mpi.h>
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <cstdlib>
#include "../ivf_simd.h"
#include "mpi_utils.h"
#include "mpi_ivf_hybrid.h"

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

struct SearchResult { float recall; int64_t latency; };

/* 序列化 vector<vector<int>> 为连续缓冲区
 * 格式: [nlist(int)][for each cluster: size(int), data...] */
inline std::vector<char> serialize_lists(const std::vector<std::vector<int>>& lists)
{
    size_t total = sizeof(int); // nlist
    for(const auto& L : lists)
        total += sizeof(int) + L.size() * sizeof(int);

    std::vector<char> buf(total);
    char* ptr = buf.data();
    int nlist = (int)lists.size();
    *(int*)ptr = nlist; ptr += sizeof(int);
    for(const auto& L : lists){
        int sz = (int)L.size();
        *(int*)ptr = sz; ptr += sizeof(int);
        if(sz > 0){
            std::memcpy(ptr, L.data(), sz * sizeof(int));
            ptr += sz * sizeof(int);
        }
    }
    return buf;
}

/* 从序列化数据恢复 vector<vector<int>>，并清空不属于本 rank 的簇 */
inline void deserialize_and_filter_lists(const char* buf, int rank, int nranks,
                                          std::vector<std::vector<int>>& lists)
{
    const char* ptr = buf;
    int nlist = *(const int*)ptr; ptr += sizeof(int);
    lists.resize(nlist);

    // 按块分配: rank r 负责簇 [r * nlist/nranks, (r+1) * nlist/nranks)
    int start = (int)((size_t)rank * nlist / nranks);
    int end   = (int)(((size_t)rank + 1) * nlist / nranks);

    for(int c = 0; c < nlist; c++){
        int sz = *(const int*)ptr; ptr += sizeof(int);
        if(c >= start && c < end){
            lists[c].resize(sz);
            if(sz > 0){
                std::memcpy(lists[c].data(), ptr, sz * sizeof(int));
            }
        } else {
            lists[c].clear(); // 不负责此簇 → 置空
        }
        ptr += sz * sizeof(int);
    }
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    // === 数据路径（相对于 mpi/ 子文件夹） ===
    std::string dp = "../data/";

    // === 阶段 1: rank 0 加载数据并构建 IVF ===
    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;

    float* base = nullptr;
    float* test_query = nullptr;
    int* test_gt = nullptr;

    if(rank == 0){
        test_query = LoadData<float>(dp + "DEEP100K.query.fbin", test_number, vecdim);
        test_gt    = LoadData<int>(dp + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
        base       = LoadData<float>(dp + "DEEP100K.base.100k.fbin", base_number, vecdim);
        test_number = 2000;

        std::cerr << "Rank 0: building IVF index...\n";
        build_ivf(base, base_number, vecdim);
        std::cerr << "IVF built: nlist=" << g_ivf_nlist << "\n";
    }

    // === 阶段 2: 广播元数据 ===
    // 先广播基本大小信息，所有 rank 分配内存
    size_t meta[4];
    if(rank == 0){
        meta[0] = base_number;
        meta[1] = vecdim;
        meta[2] = test_number;
        meta[3] = test_gt_d;
    }
    MPI_Bcast(meta, 4, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    base_number = meta[0]; vecdim = meta[1];
    test_number = meta[2]; test_gt_d = meta[3];
    const size_t k = 10;

    // 广播 nlist
    size_t nlist_buf = 0;
    if(rank == 0) nlist_buf = g_ivf_nlist;
    MPI_Bcast(&nlist_buf, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    g_ivf_nlist = nlist_buf;

    // 非 rank 0 分配 base/query/gt 内存
    if(rank != 0){
        base       = new float[base_number * vecdim];
        test_query = new float[test_number * vecdim];
        test_gt    = new int[test_number * test_gt_d];
    }

    // 广播 base, query, gt 数据
    MPI_Bcast(base,       base_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_query, test_number * vecdim,  MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_gt,    test_number * test_gt_d, MPI_INT,   0, MPI_COMM_WORLD);

    // 广播质心（所有 rank 必须都调用 MPI_Bcast）
    g_ivf_centroids.resize(g_ivf_nlist * vecdim);
    MPI_Bcast(g_ivf_centroids.data(), g_ivf_nlist * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);

    // 设置全局 base 指针（每个 rank 指向自己的本地副本）
    g_ivf_base = base;

    // === 阶段 3: 广播 inverted list 并按 rank 过滤 ===
    std::vector<char> serialized_lists;
    size_t serialized_size = 0;
    if(rank == 0){
        serialized_lists = serialize_lists(g_ivf_lists);
        serialized_size = serialized_lists.size();
    }
    MPI_Bcast(&serialized_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if(rank != 0) serialized_lists.resize(serialized_size);
    MPI_Bcast(serialized_lists.data(), serialized_size, MPI_BYTE, 0, MPI_COMM_WORLD);

    // === 读取模式 ===
    const char* mode_env = std::getenv("IVF_MODE");
    std::string ivf_mode = mode_env ? mode_env : "mpi_gather";

    const char* nt_env = std::getenv("NUM_THREADS");
    int nthreads = nt_env ? std::atoi(nt_env) : 4;

    const char* env_np = std::getenv("IVF_NPROBE");
    size_t nprobe = env_np ? (size_t)std::atoi(env_np) : 8;

    // === 阶段 3: 广播 inverted list，按模式决定是否过滤 ===
    // mpi_omp_query: 不分发查询到 rank，每个 rank 需要完整索引
    // 其他模式: 按簇分配到不同 rank
    bool filter_lists = (ivf_mode != "mpi_omp_query");

    {
        std::vector<char> serialized_lists;
        size_t serialized_size = 0;
        if(rank == 0){
            serialized_lists = serialize_lists(g_ivf_lists);
            serialized_size = serialized_lists.size();
        }
        MPI_Bcast(&serialized_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        if(rank != 0) serialized_lists.resize(serialized_size);
        MPI_Bcast(serialized_lists.data(), serialized_size, MPI_BYTE, 0, MPI_COMM_WORLD);

        if(filter_lists)
            deserialize_and_filter_lists(serialized_lists.data(), rank, nranks, g_ivf_lists);
        else
            deserialize_and_filter_lists(serialized_lists.data(), 0, 1, g_ivf_lists);
            // rank=0, nranks=1 → 不过滤，保留所有簇
    }

    // 打印负载
    {
        size_t local_total = 0;
        for(const auto& L : g_ivf_lists) local_total += L.size();
        if(rank == 0){
            std::cerr << "IVF mode: " << ivf_mode << "\n";
            std::cerr << "Vector coverage per rank: " << local_total << "\n";
        }
    }

    // === 阶段 4: 按模式分发搜索 ===
    if(ivf_mode == "mpi_omp_query"){
        // MPI 分散查询 + rank 内 OpenMP query 级并行
        run_mpi_omp_query(base, test_query, test_gt,
                          base_number, vecdim, test_gt_d,
                          test_number, k, nranks, rank, nthreads, nprobe);
    }
    else if(ivf_mode == "mpi_omp_cluster"){
        // MPI 广播查询 + rank 内 OpenMP cluster 级并行
        run_mpi_omp_cluster(base, test_query, test_gt,
                            base_number, vecdim, test_gt_d,
                            test_number, k, nranks, rank, nthreads, nprobe);
    }
    else {
        // 纯 MPI 模式: mpi_gather 或 mpi_reduce
        const char* merge_mode = std::getenv("IVF_MERGE_MODE");
        std::string mm = merge_mode ? merge_mode : "gather";

        double t_start = MPI_Wtime();

        std::vector<SearchResult> results(test_number);

        if(mm == "reduce"){
        // === 模式: 树形规约 (自定义 MPI_Op) ===
        // 对每个 query 分别处理
        for(size_t qi = 0; qi < test_number; ++qi){
            const float* q = test_query + qi * vecdim;

            // 本地搜索
            auto local_pq = ivf_search(q, base_number, vecdim, k, nprobe);

            // 序列化本地结果
            std::vector<char> local_buf;
            serialize_pq(local_pq, local_buf);
            size_t local_sz = local_buf.size();

            // 收集所有 rank 的序列化大小
            std::vector<int> all_sizes(nranks);
            int my_sz = (int)local_sz;
            MPI_Allgather(&my_sz, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, MPI_COMM_WORLD);

            // 计算位移
            std::vector<int> displs(nranks);
            int total_sz = 0;
            for(int r = 0; r < nranks; r++){
                displs[r] = total_sz;
                total_sz += all_sizes[r];
            }

            // 收集所有序列化结果
            std::vector<char> all_bufs(total_sz);
            MPI_Allgatherv(local_buf.data(), my_sz, MPI_BYTE,
                           all_bufs.data(), all_sizes.data(), displs.data(), MPI_BYTE,
                           MPI_COMM_WORLD);

            // 拆分为各 rank 的缓冲区并合并
            std::vector<std::vector<char>> rank_bufs(nranks);
            for(int r = 0; r < nranks; r++){
                if(all_sizes[r] > 0){
                    rank_bufs[r].assign(all_bufs.begin() + displs[r],
                                        all_bufs.begin() + displs[r] + all_sizes[r]);
                }
            }
            auto global_pq = merge_topk(rank_bufs, k);

            // 只在 rank 0 计算 recall（Allgather 后所有 rank 数据相同）
            if(rank == 0){
                results[qi].recall  = compute_recall(global_pq, test_gt + qi * test_gt_d, k, test_gt_d);
            }
            results[qi].latency = 0;
        }
    } else {
        // === 模式: MPI_Gather + rank 0 merge（默认） ===
        for(size_t qi = 0; qi < test_number; ++qi){
            const float* q = test_query + qi * vecdim;

            // 本地搜索
            auto local_pq = ivf_search(q, base_number, vecdim, k, nprobe);

            // 序列化
            std::vector<char> local_buf;
            serialize_pq(local_pq, local_buf);
            int local_sz = (int)local_buf.size();

            // 收集各 rank 的大小
            std::vector<int> all_sizes(nranks);
            MPI_Gather(&local_sz, 1, MPI_INT,
                       all_sizes.data(), 1, MPI_INT,
                       0, MPI_COMM_WORLD);

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

            if(rank == 0){
                std::vector<std::vector<char>> rank_bufs(nranks);
                for(int r = 0; r < nranks; r++){
                    if(all_sizes[r] > 0){
                        rank_bufs[r].assign(all_bufs.begin() + displs[r],
                                            all_bufs.begin() + displs[r] + all_sizes[r]);
                    }
                }
                auto global_pq = merge_topk(rank_bufs, k);
                results[qi].recall  = compute_recall(global_pq, test_gt + qi * test_gt_d, k, test_gt_d);
                results[qi].latency = 0;
            }
        }
    }

        double t_end = MPI_Wtime();

        // === 阶段 5: 汇总结果 ===
        float local_ar = 0;
        for(size_t i = 0; i < test_number; ++i)
            local_ar += results[i].recall;
        float global_ar = 0;
        MPI_Reduce(&local_ar, &global_ar, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

        if(rank == 0){
            double total_time = t_end - t_start;
            double avg_latency_us = (total_time * 1e6) / test_number;
            std::cout << "=== MPI IVF Results ===\n";
            std::cout << "nranks: " << nranks << "\n";
            std::cout << "merge_mode: " << mm << "\n";
            std::cout << "nlist: " << g_ivf_nlist << "  nprobe: " << nprobe << "\n";
            std::cout << "average recall: " << global_ar / test_number << "\n";
            std::cout << "total_time (s): " << total_time << "\n";
            std::cout << "average latency (us): " << avg_latency_us << "\n";
        }
    }
    // end 纯 MPI 模式

    // 清理
    delete[] base;
    delete[] test_query;
    delete[] test_gt;

    MPI_Finalize();
    return 0;
}
