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
T *LoadData(std::string data_path, size_t& n, size_t& d){

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

struct SearchResult{ float recall; int64_t latency; };

inline std::vector<char> serialize_lists(const std::vector<std::vector<int>>& lists){

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

inline void deserialize_and_filter_lists(const char* buf, int rank, int nranks,
                                          std::vector<std::vector<int>>& lists){

    const char* ptr = buf;
    int nlist = *(const int*)ptr; ptr += sizeof(int);
    lists.resize(nlist);
    int start = (int)((size_t)rank * nlist / nranks);
    int end = (int)(((size_t)rank + 1) * nlist / nranks);
    for(int c = 0; c < nlist; c++){
        int sz = *(const int*)ptr; ptr += sizeof(int);
        if(c >= start && c < end){
            lists[c].resize(sz);
            if(sz > 0)
                std::memcpy(lists[c].data(), ptr, sz * sizeof(int));
        }
        else{
            lists[c].clear();
        }
        ptr += sz * sizeof(int);
    }
}

inline std::vector<std::vector<int>> balanced_cluster_assignment(
    const std::vector<std::vector<int>>& lists, int nranks){

    int nlist = (int)lists.size();
    std::vector<std::pair<int, int>> cluster_info; 
    for(int c = 0; c < nlist; c++)
        cluster_info.push_back(std::make_pair((int)lists[c].size(), c));
    std::sort(cluster_info.begin(), cluster_info.end(), std::greater<std::pair<int, int>>());
    std::vector<size_t> rank_loads(nranks, 0);
    std::vector<std::vector<int>> rank_clusters(nranks);
    for(auto& ci : cluster_info){
        // 找负载最小的 rank
        size_t min_load = (size_t)-1;
        int best_r = 0;
        for(int r = 0; r < nranks; r++){
            if(rank_loads[r] < min_load){
                min_load = rank_loads[r];
                best_r = r;
            }
        }
        rank_loads[best_r] += ci.first;
        rank_clusters[best_r].push_back(ci.second);
    }
    return rank_clusters;
}

inline void deserialize_and_balance_lists(const char* buf, int rank, int nranks,
                                           std::vector<std::vector<int>>& lists){

    const char* ptr = buf;
    int nlist = *(const int*)ptr; ptr += sizeof(int);
    std::vector<std::vector<int>> full_lists(nlist);
    for(int c = 0; c < nlist; c++){
        int sz = *(const int*)ptr; ptr += sizeof(int);
        full_lists[c].resize(sz);
        if(sz > 0)
            std::memcpy(full_lists[c].data(), ptr, sz * sizeof(int));
        ptr += sz * sizeof(int);
    }
    // 计算贪心分配
    auto rank_clusters = balanced_cluster_assignment(full_lists, nranks);
    // 只保留本 rank 的簇
    lists.resize(nlist);
    for(int c = 0; c < nlist; c++) lists[c].clear();
    for(int c : rank_clusters[rank])
        lists[c] = std::move(full_lists[c]);
}

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    // 数据路径
    std::string dp = "../data/";
    // rank 0 加载数据并构建 IVF
    size_t test_number = 0, base_number = 0, test_gt_d = 0, vecdim = 0;
    float* base = nullptr;
    float* test_query = nullptr;
    int* test_gt = nullptr;
    if(rank == 0){
        test_query = LoadData<float>(dp + "DEEP100K.query.fbin", test_number, vecdim);
        test_gt = LoadData<int>(dp + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
        base = LoadData<float>(dp + "DEEP100K.base.100k.fbin", base_number, vecdim);
        test_number = 2000;
        std::cerr << "Rank 0: building IVF index...\n";
        build_ivf(base, base_number, vecdim);
        std::cerr << "IVF built: nlist=" << g_ivf_nlist << "\n";
    }
    // 广播元数据
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
        base = new float[base_number * vecdim];
        test_query = new float[test_number * vecdim];
        test_gt = new int[test_number * test_gt_d];
    }
    // 广播 base, query, gt 数据
    MPI_Bcast(base, base_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_query, test_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_gt, test_number * test_gt_d, MPI_INT, 0, MPI_COMM_WORLD);
    // 广播质心（所有 rank 必须都调用 MPI_Bcast）
    g_ivf_centroids.resize(g_ivf_nlist * vecdim);
    MPI_Bcast(g_ivf_centroids.data(), g_ivf_nlist * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // 设置全局 base 指针（每个 rank 指向自己的本地副本）
    g_ivf_base = base;
    // 广播 inverted list 并按 rank 过滤
    std::vector<char> serialized_lists;
    size_t serialized_size = 0;
    if(rank == 0){
        serialized_lists = serialize_lists(g_ivf_lists);
        serialized_size = serialized_lists.size();
    }
    MPI_Bcast(&serialized_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if(rank != 0) serialized_lists.resize(serialized_size);
    MPI_Bcast(serialized_lists.data(), serialized_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    // 读取模式
    const char* mode_env = std::getenv("IVF_MODE");
    std::string ivf_mode = mode_env ? mode_env : "mpi_gather";
    const char* nt_env = std::getenv("NUM_THREADS");
    int nthreads = nt_env ? std::atoi(nt_env) : 4;
    const char* env_np = std::getenv("IVF_NPROBE");
    size_t nprobe = env_np ? (size_t)std::atoi(env_np) : 8;
    // 广播 inverted list，按模式决定分配策略
    const char* part_env = std::getenv("IVF_PARTITION");
    std::string part_mode = part_env ? part_env : "block"; // "block" 或 "balanced"
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
        if(!filter_lists)
            deserialize_and_filter_lists(serialized_lists.data(), 0, 1, g_ivf_lists);
        else if(part_mode == "balanced")
            deserialize_and_balance_lists(serialized_lists.data(), rank, nranks, g_ivf_lists);
        else
            deserialize_and_filter_lists(serialized_lists.data(), rank, nranks, g_ivf_lists);
    }
    // 打印负载分布
    {
        size_t local_total = 0;
        for(const auto& L : g_ivf_lists) local_total += L.size();
        std::vector<size_t> all_loads(nranks);
        MPI_Gather(&local_total, 1, MPI_UNSIGNED_LONG, all_loads.data(), 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
        if(rank == 0){
            size_t max_l = 0, min_l = (size_t)-1;
            for(int r = 0; r < nranks; r++){
                if(all_loads[r] > max_l) max_l = all_loads[r];
                if(all_loads[r] < min_l) min_l = all_loads[r];
            }
            float imbalance = (max_l - min_l) / (float)(base_number / nranks) * 100;
            std::cerr << "IVF mode: " << ivf_mode << " partition: " << part_mode << "\n";
            std::cerr << "Load per rank: " << min_l << "~" << max_l
                      << " (imbalance: " << imbalance << "%)\n";
        }
    }
    // 按模式分发搜索
    if(ivf_mode == "mpi_omp_query"){
        // MPI 分散查询 + rank 内 OpenMP query 级并行
        run_mpi_omp_query(base, test_query, test_gt, base_number, vecdim, test_gt_d,
                          test_number, k, nranks, rank, nthreads, nprobe);
    }
    else if(ivf_mode == "mpi_omp_cluster"){
        // MPI 广播查询 + rank 内 OpenMP cluster 级并行
        run_mpi_omp_cluster(base, test_query, test_gt, base_number, vecdim, test_gt_d,
                            test_number, k, nranks, rank, nthreads, nprobe);
    }
    else if(ivf_mode == "mpi_dynamic"){
        // 模式: 主从动态任务池
        // rank 0 = master，维护查询队列
        // worker 空闲时主动请求下一个查询（TAG=0: request, TAG=1: query_idx, TAG=2: result）
        double t_start = MPI_Wtime();
        std::vector<float> recalls(test_number, 0.0f);
        if(rank == 0){
            int next_query = 0;
            int active_workers = nranks - 1;
            std::vector<char> recv_buf;
            MPI_Status st;
            while(active_workers > 0){
                // 接收任何 worker 的请求
                int dummy;
                MPI_Recv(&dummy, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &st);
                int worker = st.MPI_SOURCE;
                if(next_query < (int)test_number){
                    // 分配查询
                    int qi = next_query++;
                    MPI_Send(&qi, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
                    // 接收结果
                    int sz;
                    MPI_Recv(&sz, 1, MPI_INT, worker, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    recv_buf.resize(sz);
                    MPI_Recv(recv_buf.data(), sz, MPI_BYTE, worker, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    // 计算 recall（worker 返回的是局部结果，直接就是全局的，因为 worker 有完整索引）
                    std::priority_queue<std::pair<float, int>> w_pq;
                    deserialize_pq(recv_buf.data(), w_pq);
                    recalls[qi] = compute_recall(w_pq, test_gt + qi * test_gt_d, k, test_gt_d);
                }
                else {
                    // 没有更多查询
                    int done = -1;
                    MPI_Send(&done, 1, MPI_INT, worker, 1, MPI_COMM_WORLD);
                    active_workers--;
                }
            }
        }
        else {
            // Worker: 反复请求任务
            while(true){
                int dummy = 0;
                MPI_Send(&dummy, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
                int qi;
                MPI_Recv(&qi, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                if(qi < 0) break; // 没有更多任务
                const float* q = test_query + qi * vecdim;
                auto local_pq = ivf_search(q, base_number, vecdim, k, nprobe);
                std::vector<char> buf;
                serialize_pq(local_pq, buf);
                int sz = (int)buf.size();
                MPI_Send(&sz, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
                MPI_Send(buf.data(), sz, MPI_BYTE, 0, 2, MPI_COMM_WORLD);
            }
        }
        double t_end = MPI_Wtime();
        float local_ar = 0;
        for(size_t i = 0; i < test_number; i++) local_ar += recalls[i];
        float global_ar = 0;
        MPI_Reduce(&local_ar, &global_ar, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if(rank == 0){
            std::cout << "=== MPI Dynamic Master-Worker ===\n";
            std::cout << "nranks: " << nranks << " partition: " << part_mode << "\n";
            std::cout << "nlist: " << g_ivf_nlist << " nprobe: " << nprobe << "\n";
            std::cout << "average recall: " << global_ar / test_number << "\n";
            std::cout << "total_time (s): " << (t_end - t_start) << "\n";
            std::cout << "average latency (us): " << ((t_end - t_start) * 1e6 / test_number) << "\n";
        }
    }
    else if(ivf_mode == "mpi_async"){
        // 模式: 非阻塞通信 pipeline（通信与计算重叠）
        // 用 MPI_Isend/MPI_Irecv 替代 MPI_Gather
        // Worker: 计算 query[i] → Isend 结果 → 立即计算 query[i+1]（后台传输）
        // Master: 接收 worker 结果的同时处理自己的查询
        double t_start = MPI_Wtime();
        std::vector<float> recalls(test_number, 0.0f);
        const int TAG_SZ = 20, TAG_DAT = 21;
        const int BATCH = 32;
        if(rank == 0){
            // Master: 串行处理自己的查询 + 非阻塞接收 worker 结果
            int next_qi = 0;
            std::vector<MPI_Request> recv_reqs;
            std::vector<std::vector<char>> worker_bufs(nranks);
            // 为每个 worker 预发 Irecv（探测模式：先收大小，再收数据）
            // 简化：使用 MPI_Probe + MPI_Recv（本质仍是阻塞，但可展示概念）
            // 真正的 overlap 需要双缓冲，这里实现的是 "查询级流水线"
            for(size_t qi = 0; qi < test_number; qi++){
                // 处理自己的查询
                auto local_pq = ivf_search(test_query + qi * vecdim, base_number, vecdim, k, nprobe);
                // 接收来自每个 worker 的 query[qi] 结果（如果 qi < test_number）
                std::vector<std::vector<char>> rbufs(nranks);
                // 自己的结果
                std::vector<char> own_buf; serialize_pq(local_pq, own_buf);
                rbufs[0] = own_buf;
                // 接收 worker 结果
                for(int src = 1; src < nranks; src++){
                    int sz;
                    MPI_Recv(&sz, 1, MPI_INT, src, TAG_SZ, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                    rbufs[src].resize(sz);
                    MPI_Recv(rbufs[src].data(), sz, MPI_BYTE, src, TAG_DAT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                }
                auto gpq = merge_topk(rbufs, k);
                recalls[qi] = compute_recall(gpq, test_gt + qi * test_gt_d, k, test_gt_d);
            }
        }
        else {
            // Worker: 流水线, Isend query[i] 的结果，同时计算 query[i+1]
            MPI_Request prev_sz_req, prev_dat_req;
            bool has_pending = false;
            std::vector<char> pending_buf;
            for(size_t qi = 0; qi < test_number; qi++){
                auto local_pq = ivf_search(test_query + qi * vecdim, base_number, vecdim, k, nprobe);
                std::vector<char> buf; serialize_pq(local_pq, buf);
                // 等上一轮发送完成
                if(has_pending){
                    MPI_Wait(&prev_sz_req, MPI_STATUS_IGNORE);
                    MPI_Wait(&prev_dat_req, MPI_STATUS_IGNORE);
                }
                // 启动本轮非阻塞发送
                int sz = (int)buf.size();
                pending_buf = std::move(buf);
                MPI_Isend(&sz, 1, MPI_INT, 0, TAG_SZ, MPI_COMM_WORLD, &prev_sz_req);
                MPI_Isend(pending_buf.data(), sz, MPI_BYTE, 0, TAG_DAT, MPI_COMM_WORLD, &prev_dat_req);
                has_pending = true;
                // 注意：pending_buf 必须在 Isend 完成前保持有效！
                // 简化处理：每次 send 完再复用缓冲区，但这破坏了 overlap
                // 真正 overlap 需要双缓冲：一个在发送，一个在填充
                // 由于 k=10 消息极小（~88B），发送几乎瞬时完成，overlap 窗口极小
            }
            if(has_pending){
                MPI_Wait(&prev_sz_req, MPI_STATUS_IGNORE);
                MPI_Wait(&prev_dat_req, MPI_STATUS_IGNORE);
            }
        }
        double t_end = MPI_Wtime();
        float local_ar = 0;
        for(size_t i = 0; i < test_number; i++) local_ar += recalls[i];
        float global_ar = 0;
        MPI_Reduce(&local_ar, &global_ar, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);
        if(rank == 0){
            std::cout << "=== MPI Async Pipeline ===\n";
            std::cout << "nranks: " << nranks << " partition: " << part_mode << "\n";
            std::cout << "nlist: " << g_ivf_nlist << " nprobe: " << nprobe << "\n";
            std::cout << "average recall: " << global_ar / test_number << "\n";
            std::cout << "total_time (s): " << (t_end - t_start) << "\n";
            std::cout << "average latency (us): " << ((t_end - t_start) * 1e6 / test_number) << "\n";
        }
    }
    else {
        // 纯 MPI 模式: mpi_gather 或 mpi_reduce
        const char* merge_mode = std::getenv("IVF_MERGE_MODE");
        std::string mm = merge_mode ? merge_mode : "gather";
        double t_start = MPI_Wtime();
        std::vector<SearchResult> results(test_number);
        if(mm == "reduce"){
        // 模式: 树形规约 (自定义 MPI_Op)
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
            MPI_Allgatherv(local_buf.data(), my_sz, MPI_BYTE, all_bufs.data(), all_sizes.data(), displs.data(), MPI_BYTE, MPI_COMM_WORLD);
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
                results[qi].recall = compute_recall(global_pq, test_gt + qi * test_gt_d, k, test_gt_d);
            }
            results[qi].latency = 0;
        }
    }
    else {
        // 模式: MPI_Gather + rank 0 merge（默认）
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
            MPI_Gather(&local_sz, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
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
            MPI_Gatherv(local_buf.data(), local_sz, MPI_BYTE, all_bufs.data(), all_sizes.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
            if(rank == 0){
                std::vector<std::vector<char>> rank_bufs(nranks);
                for(int r = 0; r < nranks; r++){
                    if(all_sizes[r] > 0){
                        rank_bufs[r].assign(all_bufs.begin() + displs[r],
                                            all_bufs.begin() + displs[r] + all_sizes[r]);
                    }
                }
                auto global_pq = merge_topk(rank_bufs, k);
                results[qi].recall = compute_recall(global_pq, test_gt + qi * test_gt_d, k, test_gt_d);
                results[qi].latency = 0;
            }
        }
    }
        double t_end = MPI_Wtime();
        // 汇总结果
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
            std::cout << "nlist: " << g_ivf_nlist << " nprobe: " << nprobe << "\n";
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
