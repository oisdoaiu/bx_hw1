/*
 * mpi_ivf_hybrid.h — MPI + OpenMP 混合并行 IVF 搜索
 *
 * 环境变量:
 *   NUM_THREADS (默认 4) — 每个 MPI rank 内的 OpenMP 线程数
 */

#ifndef MPI_IVF_HYBRID_H
#define MPI_IVF_HYBRID_H

#include <mpi.h>
#include <vector>
#include <queue>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <set>
#include <sys/time.h>
#include "../ivf_simd.h"
#include "mpi_utils.h"

#ifdef _OPENMP
#include <omp.h>
#endif

/* ================================================================
 * 模式: mpi_omp_query
 *
 * MPI 将查询分散到不同 rank
 * → 每个 rank 内用 OpenMP 并行处理其分配的查询
 * → MPI_Gatherv 收集结果到 rank 0
 *
 * 优点: 零 per-query 通信，查询级并行天然无同步
 * ================================================================ */
inline void run_mpi_omp_query(
    const float* base, const float* query, const int* gt,
    size_t base_number, size_t vecdim, size_t test_gt_d,
    size_t test_number, size_t k,
    int nranks, int rank, int nthreads,
    size_t nprobe)
{
    const unsigned long C = 1000 * 1000;

    // 计算每个 rank 处理的查询范围（前 remainder 个 rank 多处理一个）
    size_t base_q = test_number / nranks;
    size_t rem    = test_number % nranks;
    size_t my_start, my_count;
    if((size_t)rank < rem){
        my_start = rank * (base_q + 1);
        my_count = base_q + 1;
    } else {
        my_start = rem * (base_q + 1) + (rank - rem) * base_q;
        my_count = base_q;
    }

    // 准备 Scatterv 参数
    std::vector<int> sc(nranks), sd(nranks);
    if(rank == 0){
        for(int r = 0; r < nranks; r++){
            size_t c = ((size_t)r < rem) ? base_q + 1 : base_q;
            sc[r] = (int)(c * vecdim);
            sd[r] = (r == 0) ? 0 : sd[r-1] + sc[r-1];
        }
    }

    // 接收查询
    std::vector<float> my_queries(my_count * vecdim);
    MPI_Scatterv(query, sc.data(), sd.data(), MPI_FLOAT,
                 my_queries.data(), (int)(my_count * vecdim), MPI_FLOAT,
                 0, MPI_COMM_WORLD);

    // OpenMP 并行处理查询
    std::vector<float> my_recalls(my_count);
    std::vector<int64_t> my_latencies(my_count);

#ifdef _OPENMP
    #pragma omp parallel for num_threads(nthreads) schedule(static)
#endif
    for(size_t qi = 0; qi < my_count; qi++){
        struct timeval val; gettimeofday(&val, NULL);
        auto res = ivf_search(my_queries.data() + qi * vecdim,
                               base_number, vecdim, k, nprobe);
        struct timeval nv; gettimeofday(&nv, NULL);
        my_latencies[qi] = (nv.tv_sec*C + nv.tv_usec) - (val.tv_sec*C + val.tv_usec);

        size_t gqi = my_start + qi;
        std::set<uint32_t> gtset;
        for(size_t j = 0; j < k; ++j)
            gtset.insert((uint32_t)gt[j + gqi * test_gt_d]);
        size_t acc = 0;
        while(res.size()){
            if(gtset.find((uint32_t)res.top().second) != gtset.end()) ++acc;
            res.pop();
        }
        my_recalls[qi] = (float)acc / k;
    }

    // Gather 结果到 rank 0
    std::vector<int> rcounts(nranks), rdispls(nranks);
    int mc = (int)my_count;
    MPI_Gather(&mc, 1, MPI_INT, rcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);

    int total_q = 0;
    if(rank == 0){
        for(int r = 0; r < nranks; r++){
            rdispls[r] = total_q;
            total_q += rcounts[r];
        }
    }

    std::vector<float> all_r(total_q);
    std::vector<int64_t> all_l(total_q);
    MPI_Gatherv(my_recalls.data(), mc, MPI_FLOAT,
                all_r.data(), rcounts.data(), rdispls.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(my_latencies.data(), mc, MPI_LONG_LONG,
                all_l.data(), rcounts.data(), rdispls.data(), MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if(rank == 0){
        float ar = 0, al = 0;
        for(int i = 0; i < total_q; i++){ ar += all_r[i]; al += all_l[i]; }
        std::cout << "=== MPI+OMP Query-Level Hybrid ===\n";
        std::cout << "nranks: " << nranks << "  threads/rank: " << nthreads;
        std::cout << "  total_parallel: " << (nranks * nthreads) << "\n";
        std::cout << "nlist: " << g_ivf_nlist << "  nprobe: " << nprobe << "\n";
        std::cout << "average recall: " << ar / total_q << "\n";
        std::cout << "average latency (us): " << al / total_q << "\n";
    }
}

/* ================================================================
 * 模式: mpi_omp_cluster
 *
 * MPI 广播查询 + 按簇分配 inverted list
 * → 每个 rank 内对查询串行处理
 * → 每个查询内用 OpenMP 并行扫描分配的簇
 *
 * 优点: 适合簇数多、单查询延迟敏感的场景
 * 注意: MPI 通信在 OpenMP 区域外，避免冲突
 * ================================================================ */
inline void run_mpi_omp_cluster(
    const float* base, const float* query, const int* gt,
    size_t base_number, size_t vecdim, size_t test_gt_d,
    size_t test_number, size_t k,
    int nranks, int rank, int nthreads,
    size_t nprobe)
{
    const unsigned long C = 1000 * 1000;
    size_t nlist = g_ivf_nlist;

    // 本 rank 负责的簇范围
    int my_start = (int)((size_t)rank * nlist / nranks);
    int my_end   = (int)(((size_t)rank + 1) * nlist / nranks);

    // 收集本 rank 负责的簇 ID（用于 OpenMP 并行扫描）
    std::vector<int> my_clusters;
    for(int c = my_start; c < my_end; c++)
        if(!g_ivf_lists[c].empty())
            my_clusters.push_back(c);

    std::vector<float> my_recalls(test_number);
    std::vector<int64_t> my_latencies(test_number);

    // 串行遍历查询（MPI 与 OpenMP 不混用）
    for(size_t qi = 0; qi < test_number; qi++){
        struct timeval val; gettimeofday(&val, NULL);
        const float* q = query + qi * vecdim;
        size_t nlist_local = nlist;

        // 质心扫描（串行，数据量小）
        std::priority_queue<std::pair<float,int>> cluster_heap;
        for(size_t c = 0; c < nlist_local; c++){
            float d = InnerProductSIMD(q, g_ivf_centroids.data() + c * vecdim, vecdim);
            cluster_heap.push(std::make_pair(-d, (int)c));
        }

        std::vector<int> probes;
        probes.reserve(nprobe);
        for(size_t i = 0; i < nprobe && !cluster_heap.empty(); i++){
            probes.push_back(cluster_heap.top().second);
            cluster_heap.pop();
        }

        // 用 OpenMP 并行扫描本 rank 的簇
        // 每个线程扫描部分簇，维护线程局部 top-k，最后合并
        std::vector<std::pair<float,int>> all_local;
#ifdef _OPENMP
        omp_set_num_threads(nthreads);
        #pragma omp parallel
        {
            std::priority_queue<std::pair<float,int>> thread_pq;
            #pragma omp for schedule(dynamic, 1)
            for(size_t pi = 0; pi < probes.size(); pi++){
                int probe_c = probes[pi];
                // 只扫描本 rank 负责的簇
                if(probe_c < my_start || probe_c >= my_end) continue;
                for(int idx : g_ivf_lists[probe_c]){
                    float dist = InnerProductSIMD(
                        g_ivf_base + (size_t)idx * vecdim, q, vecdim);
                    if(thread_pq.size() < k)
                        thread_pq.push(std::make_pair(dist, idx));
                    else if(dist < thread_pq.top().first){
                        thread_pq.pop();
                        thread_pq.push(std::make_pair(dist, idx));
                    }
                }
            }
            #pragma omp critical
            {
                while(!thread_pq.empty()){
                    all_local.push_back(thread_pq.top());
                    thread_pq.pop();
                }
            }
        }
#else
        // 无 OpenMP 时的串行退路
        {
            std::priority_queue<std::pair<float,int>> thread_pq;
            for(size_t pi = 0; pi < probes.size(); pi++){
                int probe_c = probes[pi];
                if(probe_c < my_start || probe_c >= my_end) continue;
                for(int idx : g_ivf_lists[probe_c]){
                    float dist = InnerProductSIMD(
                        g_ivf_base + (size_t)idx * vecdim, q, vecdim);
                    if(thread_pq.size() < k)
                        thread_pq.push(std::make_pair(dist, idx));
                    else if(dist < thread_pq.top().first){
                        thread_pq.pop();
                        thread_pq.push(std::make_pair(dist, idx));
                    }
                }
            }
            while(!thread_pq.empty()){
                all_local.push_back(thread_pq.top());
                thread_pq.pop();
            }
        }
#endif

        // 合并所有线程的局部结果
        std::priority_queue<std::pair<float,int>> merged;
        for(auto& cand : all_local){
            if(merged.size() < k)
                merged.push(cand);
            else if(cand.first < merged.top().first){
                merged.pop();
                merged.push(cand);
            }
        }

        struct timeval nv; gettimeofday(&nv, NULL);
        int64_t diff = (nv.tv_sec*C + nv.tv_usec) - (val.tv_sec*C + val.tv_usec);
        my_latencies[qi] = diff;

        // 序列化本 rank 的局部 top-k
        std::vector<char> local_buf;
        serialize_pq(merged, local_buf);
        int local_sz = (int)local_buf.size();

        // MPI Gather（在 OpenMP 区域外）
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

        if(rank == 0){
            std::vector<std::vector<char>> rbufs(nranks);
            for(int r = 0; r < nranks; r++){
                if(all_sizes[r] > 0)
                    rbufs[r].assign(all_bufs.begin() + displs[r],
                                    all_bufs.begin() + displs[r] + all_sizes[r]);
            }
            auto gpq = merge_topk(rbufs, k);
            my_recalls[qi] = compute_recall(gpq, gt + qi * test_gt_d, k, test_gt_d);
        }
    }

    // 汇总 recall 和 latency（MPI_Reduce 是集体操作，所有 rank 必须调用）
    float local_ar = 0;
    for(size_t i = 0; i < test_number; i++) local_ar += my_recalls[i];
    float global_ar = 0;
    MPI_Reduce(&local_ar, &global_ar, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    int64_t local_max_lat = 0;
    for(size_t i = 0; i < test_number; i++)
        if(my_latencies[i] > local_max_lat) local_max_lat = my_latencies[i];
    int64_t global_max_lat = 0;
    MPI_Reduce(&local_max_lat, &global_max_lat, 1, MPI_LONG_LONG, MPI_MAX, 0, MPI_COMM_WORLD);

    if(rank == 0){
        std::cout << "=== MPI+OMP Cluster-Level Hybrid ===\n";
        std::cout << "nranks: " << nranks << "  threads/rank: " << nthreads;
        std::cout << "  total_parallel: " << (nranks * nthreads) << "\n";
        std::cout << "nlist: " << g_ivf_nlist << "  nprobe: " << nprobe << "\n";
        std::cout << "clusters/rank: ~" << (g_ivf_nlist / nranks) << "\n";
        std::cout << "average recall: " << global_ar / test_number << "\n";
        std::cout << "query avg latency (us): " << global_max_lat << "\n";
    }
}

#endif // MPI_IVF_HYBRID_H
