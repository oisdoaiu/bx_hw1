#ifndef ARM_IVF_H
#define ARM_IVF_H
#define ARM_IVF_NPROBE 8
#define ARM_IVF_PARTITION 0
#include <mpi.h>
#include "flat_simd.h"
#include <vector>
#include <queue>
#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <algorithm>
#include <set>
#include <iostream>


/*MPI 初始化*/

struct _IvfGuard{
    _IvfGuard(){
        int flag;
        MPI_Initialized(&flag);
        if(!flag) MPI_Init(nullptr, nullptr);
    }
    ~_IvfGuard() { MPI_Finalize(); }
};
static _IvfGuard _ivf_guard;

/*MPI 工具函数*/
inline void _ivf_serialize_pq(const std::priority_queue<std::pair<float, int>>& pq, std::vector<char>& buf){

    size_t n = pq.size();
    buf.resize(sizeof(size_t) + n * (sizeof(float) + sizeof(int)));
    char* ptr = buf.data();
    *(size_t*)ptr = n;
    ptr += sizeof(size_t);
    auto copy = pq;
    while(!copy.empty()){
        auto& top = copy.top();
        *(float*)ptr = top.first;
        ptr += sizeof(float);
        *(int*)ptr = top.second;
        ptr += sizeof(int);
        copy.pop();
    }
}

inline void _ivf_deserialize_pq(const char* buf, std::priority_queue<std::pair<float, int>>& pq){

    size_t n = *(const size_t*)buf;
    const char* ptr = buf + sizeof(size_t);
    for(size_t i = 0; i < n; i++){
        float f = *(const float*)ptr;
        ptr += sizeof(float);
        int id = *(const int*)ptr;
        ptr += sizeof(int);
        pq.push(std::make_pair(f, id));
    }
}

inline std::priority_queue<std::pair<float, int>>
_ivf_merge_topk(const std::vector<std::vector<char>>& bufs, size_t k){

    std::priority_queue<std::pair<float, int>> result;
    for(const auto& buf : bufs){
        if(buf.empty()) continue;
        size_t n = *(const size_t*)buf.data();
        const char* ptr = buf.data() + sizeof(size_t);
        for(size_t i = 0; i < n; i++){
            float f = *(const float*)ptr;
            ptr += sizeof(float);
            int id = *(const int*)ptr;
            ptr += sizeof(int);
            if(result.size() < k){
                result.push(std::make_pair(f, id));
            }
            else{
                result.pop();
                result.push(std::make_pair(f, id));
            }
        }
    }
    return result;
}

inline float _ivf_compute_recall(
    const std::priority_queue<std::pair<float, int>>& result,
    const int* gt, size_t k){

    std::set<uint32_t> gt_set;
    for(size_t j = 0; j < k; j++)
        gt_set.insert((uint32_t)gt[j]);
    size_t acc = 0;
    auto copy = result;
    while(copy.size()){
        if(gt_set.find((uint32_t)copy.top().second) != gt_set.end())
            acc++;
        copy.pop();
    }
    return (float)acc / k;
}

/*IVF 算法*/
static std::vector<float> _ivf_centroids;
static std::vector<std::vector<int>> _ivf_lists;
static size_t _ivf_nlist;
static const float* _ivf_base;
inline void _ivf_kmeans(const float* base, size_t bn, size_t vd,
                         float* centroids, size_t nlist, int max_iters){

    for(size_t c = 0; c < nlist; c++){

        size_t idx = (c * (bn / nlist)) % bn;
        std::memcpy(centroids + c * vd, base + idx * vd, vd * sizeof(float));
    }
    std::vector<int> assignments(bn);
    for(int iter = 0; iter < max_iters; iter++){
        /* 分配步骤 */
        for(size_t i = 0; i < bn; i++){
            const float* v = base + i * vd;
            float best_d = FLT_MAX;
            int best_c = 0;
            for(size_t c = 0; c < nlist; c++){
                float d = InnerProductSIMD(v, centroids + c * vd, vd);
                if(d < best_d){
                    best_d = d;
                    best_c = (int)c;
                }
            }
            assignments[i] = best_c;
        }

        /* 更新步骤 */
        std::vector<float> sum(nlist * vd, 0.0f);
        std::vector<int> counts(nlist, 0);
        for(size_t i = 0; i < bn; i++){
            int c = assignments[i];
            const float* v = base + i * vd;
            float* dst = sum.data() + c * vd;
            for(size_t d = 0; d < vd; d++)
                dst[d] += v[d];
            counts[c]++;
        }
        for(size_t c = 0; c < nlist; c++){
            if(counts[c] > 0){
                float inv = 1.0f / (float)counts[c];
                float* cent = centroids + c * vd;
                for(size_t d = 0; d < vd; d++)
                    cent[d] = sum[c * vd + d] * inv;
            }
        }
    }
}

inline void _ivf_build_index(const float* base, size_t bn, size_t vd){

    _ivf_nlist = 256;
    _ivf_base = base;
    _ivf_centroids.resize(_ivf_nlist * vd);
    _ivf_kmeans(base, bn, vd, _ivf_centroids.data(), _ivf_nlist, 15);
    _ivf_lists.resize(_ivf_nlist);
    for(size_t c = 0; c < _ivf_nlist; c++)
        _ivf_lists[c].clear();
    for(size_t i = 0; i < bn; i++){
        const float* v = base + i * vd;
        float best_d = FLT_MAX;
        int best_c = 0;
        for(size_t c = 0; c < _ivf_nlist; c++){
            float d = InnerProductSIMD(v, _ivf_centroids.data() + c * vd, vd);
            if(d < best_d){
                best_d = d;
                best_c = (int)c;
            }
        }
        _ivf_lists[best_c].push_back((int)i);
    }
}

inline std::priority_queue<std::pair<float, int>>
_ivf_search_query(const float* query, size_t bn, size_t vd, size_t k, size_t nprobe){

    (void)bn;

    /* 粗排：选 nprobe 个最近簇 */
    std::priority_queue<std::pair<float, int>> cluster_heap;
    for(size_t c = 0; c < _ivf_nlist; c++){
        float d = InnerProductSIMD(query, _ivf_centroids.data() + c * vd, vd);
        cluster_heap.push(std::make_pair(-d, (int)c));
    }
    std::vector<int> probes;
    probes.reserve(nprobe);
    for(size_t i = 0; i < nprobe && !cluster_heap.empty(); i++){
        probes.push_back(cluster_heap.top().second);
        cluster_heap.pop();
    }

    /* 精排：扫描选中簇 */
    std::priority_queue<std::pair<float, int>> result;
    for(int c : probes){
        for(int idx : _ivf_lists[c]){

            float dist = InnerProductSIMD(_ivf_base + (size_t)idx * vd, query, vd);
            if(result.size() < k){
                result.push(std::make_pair(dist, idx));
            }
            else{
                result.pop();
                result.push(std::make_pair(dist, idx));
            }
        }
    }
    return result;
}

/*inverted list 序列化/分配*/
inline std::vector<char> _ivf_serialize_lists(
    const std::vector<std::vector<int>>& lists){

    size_t total = sizeof(int);
    for(const auto& L : lists)
        total += sizeof(int) + L.size() * sizeof(int);
    std::vector<char> buf(total);
    char* ptr = buf.data();
    int nlist = (int)lists.size();
    *(int*)ptr = nlist;
    ptr += sizeof(int);
    for(const auto& L : lists){
        int sz = (int)L.size();
        *(int*)ptr = sz;
        ptr += sizeof(int);
        if(sz > 0){
            std::memcpy(ptr, L.data(), sz * sizeof(int));
            ptr += sz * sizeof(int);
        }
    }
    return buf;
}

inline void _ivf_deser_block(const char* buf, int rank, int nranks, std::vector<std::vector<int>>& lists){

    const char* ptr = buf;
    int nlist = *(const int*)ptr;
    ptr += sizeof(int);
    lists.resize(nlist);
    int start = (int)((size_t)rank * nlist / nranks);
    int end = (int)(((size_t)rank + 1) * nlist / nranks);
    for(int c = 0; c < nlist; c++){
        int sz = *(const int*)ptr;
        ptr += sizeof(int);
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

/* 贪心均匀分配 */
inline std::vector<std::vector<int>> _ivf_balanced_assign(
    const std::vector<std::vector<int>>& lists, int nranks){

    int nlist = (int)lists.size();
    std::vector<std::pair<int, int>> cluster_info;
    for(int c = 0; c < nlist; c++)
        cluster_info.push_back(std::make_pair((int)lists[c].size(), c));
    std::sort(cluster_info.begin(), cluster_info.end(), std::greater<std::pair<int, int>>());
    std::vector<size_t> rank_loads(nranks, 0);
    std::vector<std::vector<int>> rank_clusters(nranks);
    for(auto& ci : cluster_info){
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

inline void _ivf_deser_balance(const char* buf, int rank, int nranks, std::vector<std::vector<int>>& lists){

    const char* ptr = buf;
    int nlist = *(const int*)ptr;
    ptr += sizeof(int);
    std::vector<std::vector<int>> full_lists(nlist);
    for(int c = 0; c < nlist; c++){
        int sz = *(const int*)ptr;
        ptr += sizeof(int);
        full_lists[c].resize(sz);
        if(sz > 0)
            std::memcpy(full_lists[c].data(), ptr, sz * sizeof(int));
        ptr += sz * sizeof(int);
    }
    auto rank_clusters = _ivf_balanced_assign(full_lists, nranks);
    lists.resize(nlist);
    for(int c = 0; c < nlist; c++)
        lists[c].clear();
    for(int c : rank_clusters[rank])
        lists[c] = std::move(full_lists[c]);
}

/*MPI 状态*/
static int _ivf_rank = 0, _ivf_nranks = 1;
static size_t _ivf_bn = 0, _ivf_vd = 0;

/*对外接口*/
inline void ivf_mpi_build(const float* base, size_t base_number, size_t vecdim){

    MPI_Comm_rank(MPI_COMM_WORLD, &_ivf_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_ivf_nranks);
    _ivf_bn = base_number;
    _ivf_vd = vecdim;

    /* 构建 IVF 索引 */
    if(_ivf_rank == 0)
        _ivf_build_index(base, base_number, vecdim);
    size_t nlist = 0;
    if(_ivf_rank == 0) nlist = _ivf_nlist;
    MPI_Bcast(&nlist, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    _ivf_nlist = nlist;
    _ivf_centroids.resize(nlist * vecdim);
    MPI_Bcast(_ivf_centroids.data(), nlist * vecdim, MPI_FLOAT, 0, MPI_COMM_WORLD);
    _ivf_base = base;

    /* 广播倒排列表 */
    std::vector<char> serialized;
    size_t serialized_size = 0;
    if(_ivf_rank == 0){
        serialized = _ivf_serialize_lists(_ivf_lists);
        serialized_size = serialized.size();
    }
    MPI_Bcast(&serialized_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if(_ivf_rank != 0)
        serialized.resize(serialized_size);
    MPI_Bcast(serialized.data(), serialized_size, MPI_BYTE, 0, MPI_COMM_WORLD);

    /* 分配簇 */
    if(ARM_IVF_PARTITION == 1)
        _ivf_deser_balance(serialized.data(), _ivf_rank, _ivf_nranks, _ivf_lists);
    else
        _ivf_deser_block(serialized.data(), _ivf_rank, _ivf_nranks, _ivf_lists);
}

inline std::priority_queue<std::pair<float, int>>
ivf_mpi_solve(const float* base, const float* query, size_t base_number, size_t vecdim, size_t k){

    (void)base_number;
    auto local_result = _ivf_search_query(query, _ivf_bn, vecdim, k, ARM_IVF_NPROBE);
    std::vector<char> buf;
    _ivf_serialize_pq(local_result, buf);
    int local_sz = (int)buf.size();
    std::vector<int> all_sizes(_ivf_nranks);
    MPI_Gather(&local_sz, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> displs(_ivf_nranks, 0);
    std::vector<char> all_bufs;
    if(_ivf_rank == 0){
        int total = 0;
        for(int r = 0; r < _ivf_nranks; r++){
            displs[r] = total;
            total += all_sizes[r];
        }
        all_bufs.resize(total);
    }
    MPI_Gatherv(buf.data(), local_sz, MPI_BYTE, all_bufs.data(), all_sizes.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if(_ivf_rank == 0){
        std::vector<std::vector<char>> rank_bufs(_ivf_nranks);
        for(int r = 0; r < _ivf_nranks; r++){
            if(all_sizes[r] > 0){
                rank_bufs[r].assign(all_bufs.begin() + displs[r],
                                    all_bufs.begin() + displs[r] + all_sizes[r]);
            }
        }
        return _ivf_merge_topk(rank_bufs, k);
    }
    return std::priority_queue<std::pair<float, int>>();
}
#endif
