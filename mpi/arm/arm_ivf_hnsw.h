#ifndef ARM_IVF_HNSW_H
#define ARM_IVF_HNSW_H
#define ARM_IVF_HNSW_NPROBE 8
#define ARM_IVF_HNSW_EF 50
#include <mpi.h>
#include "flat_simd.h"
#include "hnswlib/hnswlib/hnswlib.h"
#include "hnswlib/hnswlib/hnswalg.h"
#include <vector>
#include <queue>
#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <set>


struct _IvhGuard{
    _IvhGuard(){
        int flag;
        MPI_Initialized(&flag);
        if(!flag) MPI_Init(nullptr, nullptr);
    }
    ~_IvhGuard() { MPI_Finalize(); }
};
static _IvhGuard _ivh_guard;

inline void _ivh_serialize_pq(const std::priority_queue<std::pair<float, int>>& pq, std::vector<char>& buf){

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

inline std::priority_queue<std::pair<float, int>>
_ivh_merge_topk(const std::vector<std::vector<char>>& bufs, size_t k){
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


/*IVF 算法*/
static std::vector<float> _ivh_centroids;
static std::vector<std::vector<int>> _ivh_lists;
static size_t _ivh_nlist;
static const float* _ivh_base;
inline void _ivh_kmeans(const float* base, size_t bn, size_t vd,
                         float* centroids, size_t nlist, int max_iters){

    for(size_t c = 0; c < nlist; c++){
        size_t idx = (c * (bn / nlist)) % bn;
        std::memcpy(centroids + c * vd, base + idx * vd, vd * sizeof(float));
    }
    std::vector<int> assignments(bn);
    for(int iter = 0; iter < max_iters; iter++){
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

inline void _ivh_build_ivf(const float* base, size_t bn, size_t vd){
    _ivh_nlist = 256;
    _ivh_base = base;
    _ivh_centroids.resize(_ivh_nlist * vd);
    _ivh_kmeans(base, bn, vd, _ivh_centroids.data(), _ivh_nlist, 15);
    _ivh_lists.resize(_ivh_nlist);
    for(size_t c = 0; c < _ivh_nlist; c++)
        _ivh_lists[c].clear();
    for(size_t i = 0; i < bn; i++){
        const float* v = base + i * vd;
        float best_d = FLT_MAX;
        int best_c = 0;
        for(size_t c = 0; c < _ivh_nlist; c++){
            float d = InnerProductSIMD(v, _ivh_centroids.data() + c * vd, vd);
            if(d < best_d){
                best_d = d;
                best_c = (int)c;
            }
        }
        _ivh_lists[best_c].push_back((int)i);
    }
}


/*MPI 状态*/
static int _ivh_rank = 0, _ivh_nranks = 1;
static int _ivh_my_start = 0, _ivh_my_end = 0;
static std::vector<hnswlib::InnerProductSpace*> _ivh_spaces;
static std::vector<hnswlib::HierarchicalNSW<float>*> _ivh_indexes;

inline void ivf_hnsw_mpi_build(const float* base, size_t base_number, size_t vecdim){

    MPI_Comm_rank(MPI_COMM_WORLD, &_ivh_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_ivh_nranks);
    int ef = ARM_IVF_HNSW_EF;
    size_t bn = base_number;
    size_t vd = vecdim;

    /* 构建全局 IVF */
    if(_ivh_rank == 0)
        _ivh_build_ivf(base, bn, vd);
    size_t nlist = 0;
    if(_ivh_rank == 0) nlist = _ivh_nlist;
    MPI_Bcast(&nlist, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    _ivh_nlist = nlist;
    _ivh_centroids.resize(nlist * vd);
    MPI_Bcast(_ivh_centroids.data(), nlist * vd, MPI_FLOAT, 0, MPI_COMM_WORLD);
    _ivh_base = base;

    /* 广播各簇大小 */
    std::vector<int> cluster_sizes(nlist, 0);
    if(_ivh_rank == 0){
        for(size_t c = 0; c < nlist; c++)
            cluster_sizes[c] = (int)_ivh_lists[c].size();
    }
    MPI_Bcast(cluster_sizes.data(), nlist, MPI_INT, 0, MPI_COMM_WORLD);
    _ivh_my_start = (int)(_ivh_rank * nlist / _ivh_nranks);
    _ivh_my_end = (int)((_ivh_rank + 1) * nlist / _ivh_nranks);

    /* 收集本 rank 簇的向量数据 */
    size_t my_total = 0;
    for(int c = _ivh_my_start; c < _ivh_my_end; c++)
        my_total += cluster_sizes[c];
    std::vector<float> my_base(my_total * vd);
    std::vector<size_t> my_labels(my_total);
    if(_ivh_rank == 0){
        size_t pos = 0;
        for(int c = _ivh_my_start; c < _ivh_my_end; c++){
            for(int idx : _ivh_lists[c]){
                std::memcpy(my_base.data() + pos * vd, base + idx * vd, vd * sizeof(float));
                my_labels[pos] = (size_t)idx;
                pos++;
            }
        }

        /* 发送给其他 rank */
        for(int r = 1; r < _ivh_nranks; r++){
            int rs = (int)(r * nlist / _ivh_nranks);
            int re = (int)((r + 1) * nlist / _ivh_nranks);
            size_t rc = 0;
            for(int c = rs; c < re; c++)
                rc += cluster_sizes[c];
            std::vector<float> r_base(rc * vd);
            std::vector<size_t> r_labels(rc);
            size_t rp = 0;
            for(int c = rs; c < re; c++){
                for(int idx : _ivh_lists[c]){
                    std::memcpy(r_base.data() + rp * vd, base + idx * vd, vd * sizeof(float));
                    r_labels[rp] = (size_t)idx;
                    rp++;
                }
            }
            MPI_Send(r_base.data(), rc * vd, MPI_FLOAT, r, 0, MPI_COMM_WORLD);
            MPI_Send(r_labels.data(), rc, MPI_UNSIGNED_LONG, r, 1, MPI_COMM_WORLD);
        }
    }
    else{
        MPI_Recv(my_base.data(), my_total * vd, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(my_labels.data(), my_total, MPI_UNSIGNED_LONG, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);

    /* 为每个簇构建 HNSW */
    size_t offset = 0;
    for(int c = _ivh_my_start; c < _ivh_my_end; c++){
        int nv = cluster_sizes[c];
        if(nv > 0){
            auto* space = new hnswlib::InnerProductSpace(vd);
            auto* index = new hnswlib::HierarchicalNSW<float>(space, nv, 16, 200);
            index->ef_ = ef;
            for(int j = 0; j < nv; j++){
                index->addPoint((void*)(my_base.data() + (offset + j) * vd), my_labels[offset + j]);
            }
            _ivh_spaces.push_back(space);
            _ivh_indexes.push_back(index);
        }
        else{
            _ivh_spaces.push_back(nullptr);
            _ivh_indexes.push_back(nullptr);
        }
        offset += nv;
    }
}

inline std::priority_queue<std::pair<float, int>>
ivf_hnsw_mpi_solve(const float* base, const float* query, size_t base_number, size_t vecdim, size_t k){
    (void)base;
    (void)base_number;
    size_t nprobe = ARM_IVF_HNSW_NPROBE;
    size_t nlist = _ivh_nlist;
    size_t vd = vecdim;

    /* 质心扫描 */
    std::priority_queue<std::pair<float, int>> cluster_heap;
    for(size_t c = 0; c < nlist; c++){
        float d = InnerProductSIMD(query,_ivh_centroids.data() + c * vd, vd);
        cluster_heap.push(std::make_pair(-d, (int)c));
    }
    std::vector<int> probes;
    for(size_t i = 0; i < nprobe && !cluster_heap.empty(); i++){
        probes.push_back(cluster_heap.top().second);
        cluster_heap.pop();
    }

    /* 在负责的簇内 HNSW 搜索 */
    std::priority_queue<std::pair<float, int>> local_result;
    for(int probe_c : probes){
        if(probe_c < _ivh_my_start || probe_c >= _ivh_my_end)
            continue;
        int local_c = probe_c - _ivh_my_start;
        if(_ivh_indexes[local_c] == nullptr)
            continue;
        auto hnsw_pq = _ivh_indexes[local_c]->searchKnn((void*)query, k);
        std::vector<std::pair<float, int>> tmp;
        while(!hnsw_pq.empty()){
            tmp.push_back(std::make_pair(hnsw_pq.top().first,(int)hnsw_pq.top().second));
            hnsw_pq.pop();
        }
        for(int j = (int)tmp.size() - 1; j >= 0; j--)
            local_result.push(tmp[j]);
    }

    /* MPI gather + merge */
    std::vector<char> buf;
    _ivh_serialize_pq(local_result, buf);
    int local_sz = (int)buf.size();
    std::vector<int> all_sizes(_ivh_nranks);
    MPI_Gather(&local_sz, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> displs(_ivh_nranks, 0);
    std::vector<char> all_bufs;
    if(_ivh_rank == 0){
        int total = 0;
        for(int r = 0; r < _ivh_nranks; r++){
            displs[r] = total;
            total += all_sizes[r];
        }
        all_bufs.resize(total);
    }
    MPI_Gatherv(buf.data(), local_sz, MPI_BYTE, all_bufs.data(), all_sizes.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if(_ivh_rank == 0){
        std::vector<std::vector<char>> rank_bufs(_ivh_nranks);
        for(int r = 0; r < _ivh_nranks; r++){
            if(all_sizes[r] > 0){
                rank_bufs[r].assign(all_bufs.begin() + displs[r], all_bufs.begin() + displs[r] + all_sizes[r]);
            }
        }
        return _ivh_merge_topk(rank_bufs, k);
    }
    return std::priority_queue<std::pair<float, int>>();
}
#endif
