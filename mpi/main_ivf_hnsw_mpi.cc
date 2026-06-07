/*
 * main_ivf_hnsw_mpi.cc — MPI IVF+HNSW 方案 A
 *
 * 先构建全局 IVF 索引 → 每个簇内构建小型 HNSW → 按簇分配 HNSW 到各 rank
 * → 广播查询 → 质心扫描 + HNSW 搜索 → top-k merge
 *
 * 环境变量:
 *   HNSW_EF (默认 50) — 每个簇内 HNSW 的 ef 参数
 *   IVF_NLIST (默认 256)
 *   IVF_NPROBE (默认 8)
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
#include "../ivf_simd.h"
#include "../hnswlib/hnswlib/hnswlib.h"
#include "../hnswlib/hnswlib/hnswalg.h"
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
    for(size_t i = 0; i < n; ++i)
        fin.read(((char*)data + i * d * sz), d * sz);
    fin.close();
    return data;
}

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

    const char* env_ef = std::getenv("HNSW_EF");
    int ef = env_ef ? std::atoi(env_ef) : 50;

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

    // 广播元数据
    size_t meta[4];
    if(rank == 0){ meta[0]=base_number; meta[1]=vecdim; meta[2]=test_number; meta[3]=test_gt_d; }
    MPI_Bcast(meta, 4, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    base_number=meta[0]; vecdim=meta[1]; test_number=meta[2]; test_gt_d=meta[3];
    const size_t k = 10;

    // 广播 nlist
    size_t nlist = 0;
    if(rank == 0) nlist = g_ivf_nlist;
    MPI_Bcast(&nlist, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);

    // 分配内存
    if(rank != 0){
        base       = new float[base_number * vecdim];
        test_query = new float[test_number * vecdim];
        test_gt    = new int[test_number * test_gt_d];
    }
    MPI_Bcast(base,       base_number * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_query, test_number * vecdim,  MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Bcast(test_gt,    test_number * test_gt_d, MPI_INT, 0, MPI_COMM_WORLD);

    // 广播质心
    g_ivf_nlist = nlist;
    g_ivf_centroids.resize(nlist * vecdim);
    MPI_Bcast(g_ivf_centroids.data(), nlist * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    g_ivf_base = base;

    // === 阶段 2: 按簇收集数据并分发到各 rank ===
    // rank 0: 对每个簇，提取向量数据 + 原始索引
    // 然后发送给负责该簇的 rank

    // 计算每个 rank 负责的簇范围
    int my_start = (int)(rank * nlist / nranks);
    int my_end   = (int)((rank + 1) * nlist / nranks);

    // 先统计簇大小（所有 rank 都需要知道，便于接收）
    std::vector<int> cluster_sizes(nlist, 0);
    if(rank == 0){
        for(size_t c = 0; c < nlist; c++)
            cluster_sizes[c] = (int)g_ivf_lists[c].size();
    }
    MPI_Bcast(cluster_sizes.data(), nlist, MPI_INT, 0, MPI_COMM_WORLD);

    // 收集本 rank 负责的簇的数据
    size_t my_total_vectors = 0;
    for(int c = my_start; c < my_end; c++)
        my_total_vectors += cluster_sizes[c];

    std::vector<float> my_base(my_total_vectors * vecdim);
    std::vector<size_t> my_labels(my_total_vectors);

    if(rank == 0){
        // rank 0 从本地内存直接复制自己负责的簇
        size_t pos = 0;
        for(int c = my_start; c < my_end; c++){
            for(int idx : g_ivf_lists[c]){
                std::memcpy(my_base.data() + pos * vecdim,
                            base + (size_t)idx * vecdim, vecdim * sizeof(float));
                my_labels[pos] = (size_t)idx;
                pos++;
            }
        }
        // 发送其他 rank 的簇数据
        for(int r = 1; r < nranks; r++){
            int r_start = (int)(r * nlist / nranks);
            int r_end   = (int)((r + 1) * nlist / nranks);
            size_t r_count = 0;
            for(int c = r_start; c < r_end; c++)
                r_count += cluster_sizes[c];
            // 发送向量数据
            std::vector<float> r_base(r_count * vecdim);
            std::vector<size_t> r_labels(r_count);
            size_t rp = 0;
            for(int c = r_start; c < r_end; c++){
                for(int idx : g_ivf_lists[c]){
                    std::memcpy(r_base.data() + rp * vecdim,
                                base + (size_t)idx * vecdim, vecdim * sizeof(float));
                    r_labels[rp] = (size_t)idx;
                    rp++;
                }
            }
            MPI_Send(r_base.data(),  r_count * vecdim, MPI_FLOAT, r, 0, MPI_COMM_WORLD);
            MPI_Send(r_labels.data(), r_count,         MPI_UNSIGNED_LONG, r, 1, MPI_COMM_WORLD);
        }
    } else {
        // 非 rank 0: 接收数据
        if(my_total_vectors > 0){
            MPI_Recv(my_base.data(),  my_total_vectors * vecdim, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(my_labels.data(), my_total_vectors, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if(rank == 0){
        size_t max_v = 0, min_v = (size_t)-1;
        for(int r = 0; r < nranks; r++){
            int rs = (int)(r * nlist / nranks);
            int re = (int)((r + 1) * nlist / nranks);
            size_t rc = 0;
            for(int c = rs; c < re; c++) rc += cluster_sizes[c];
            if(rc > max_v) max_v = rc;
            if(rc < min_v) min_v = rc;
        }
        std::cerr << "Cluster data distributed: clusters/rank=" << (nlist/nranks)
                  << " vectors/rank: " << min_v << "~" << max_v << "\n";
    }

    MPI_Barrier(MPI_COMM_WORLD);

    // === 阶段 3: 各 rank 为每个簇构建 HNSW ===
    // 为每个簇构建独立的 HNSW 索引
    std::vector<hnswlib::InnerProductSpace*> cluster_spaces;
    std::vector<hnswlib::HierarchicalNSW<float>*> cluster_hnsws;
    std::vector<size_t> cluster_offsets; // 每个簇在 my_base 中的起始位置

    {
        size_t offset = 0;
        for(int c = my_start; c < my_end; c++){
            int csize = cluster_sizes[c];
            cluster_offsets.push_back(offset);
            if(csize > 0){
                auto* space = new hnswlib::InnerProductSpace(vecdim);
                auto* hnsw = new hnswlib::HierarchicalNSW<float>(space, csize, 16, 200);
                hnsw->ef_ = ef;
                for(int j = 0; j < csize; j++){
                    hnsw->addPoint((void*)(my_base.data() + (offset + j) * vecdim),
                                   my_labels[offset + j]);
                }
                cluster_spaces.push_back(space);
                cluster_hnsws.push_back(hnsw);
            } else {
                cluster_spaces.push_back(nullptr);
                cluster_hnsws.push_back(nullptr);
            }
            offset += csize;
        }
    }

    if(rank == 0)
        std::cerr << "HNSW built per cluster, ef=" << ef << "\n";

    MPI_Barrier(MPI_COMM_WORLD);

    // === 阶段 4: 搜索 ===
    const char* env_np = std::getenv("IVF_NPROBE");
    size_t nprobe = env_np ? (size_t)std::atoi(env_np) : 8;

    double t_start = MPI_Wtime();
    std::vector<float> recalls(test_number, 0.0f);

    for(size_t qi = 0; qi < test_number; qi++){
        const float* q = test_query + qi * vecdim;

        // 质心扫描（所有 rank 相同）
        std::priority_queue<std::pair<float,int>> ch;
        for(size_t c = 0; c < nlist; c++){
            float d = InnerProductSIMD(q, g_ivf_centroids.data() + c * vecdim, vecdim);
            ch.push(std::make_pair(-d, (int)c));
        }

        // 选出 top nprobe
        std::vector<int> probes;
        for(size_t i = 0; i < nprobe && !ch.empty(); i++){
            probes.push_back(ch.top().second);
            ch.pop();
        }

        // 在本 rank 负责的簇中，用 HNSW 搜索
        std::priority_queue<std::pair<float,int>> local_result;
        for(int probe_c : probes){
            if(probe_c < my_start || probe_c >= my_end) continue;
            int local_c = probe_c - my_start;
            if(cluster_hnsws[local_c] == nullptr) continue;

            auto hnsw_pq = cluster_hnsws[local_c]->searchKnn((void*)q, k);
            // 转换为统一格式 (distance = 1.0 - inner_product, 越小越好)
            std::vector<std::pair<float,int>> tmp;
            while(!hnsw_pq.empty()){
                tmp.push_back(std::make_pair(hnsw_pq.top().first,
                                             (int)hnsw_pq.top().second));
                hnsw_pq.pop();
            }
            for(int j = (int)tmp.size()-1; j >= 0; j--){
                if(local_result.size() < k)
                    local_result.push(tmp[j]);
                else if(tmp[j].first < local_result.top().first){
                    local_result.pop();
                    local_result.push(tmp[j]);
                }
            }
        }

        // 序列化 + MPI_Gather
        std::vector<char> local_buf;
        serialize_pq(local_result, local_buf);
        int local_sz = (int)local_buf.size();

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
            recalls[qi] = compute_recall(gpq, test_gt + qi * test_gt_d, k, test_gt_d);
        }
    }

    double t_end = MPI_Wtime();

    // === 汇总 ===
    float local_ar = 0;
    for(size_t i = 0; i < test_number; i++) local_ar += recalls[i];
    float global_ar = 0;
    MPI_Reduce(&local_ar, &global_ar, 1, MPI_FLOAT, MPI_SUM, 0, MPI_COMM_WORLD);

    if(rank == 0){
        double total_time = t_end - t_start;
        double avg_latency_us = (total_time * 1e6) / test_number;
        std::cout << "=== IVF+HNSW MPI Results ===\n";
        std::cout << "nranks: " << nranks << "\n";
        std::cout << "nlist: " << nlist << "  nprobe: " << nprobe << "  ef: " << ef << "\n";
        std::cout << "average recall: " << global_ar / test_number << "\n";
        std::cout << "total_time (s): " << total_time << "\n";
        std::cout << "average latency (us): " << avg_latency_us << "\n";
    }

    // 清理 HNSW 对象
    for(size_t i = 0; i < cluster_hnsws.size(); i++){
        if(cluster_hnsws[i]) delete cluster_hnsws[i];
        if(cluster_spaces[i]) delete cluster_spaces[i];
    }
    delete[] base;
    delete[] test_query;
    delete[] test_gt;

    MPI_Finalize();
    return 0;
}
