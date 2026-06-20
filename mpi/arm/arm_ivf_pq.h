#ifndef ARM_IVF_PQ_H
#define ARM_IVF_PQ_H
#define ARM_IVF_PQ_NPROBE 16
#define ARM_IVF_PQ_RERANK 200
#include <mpi.h>
#include "flat_simd.h"
#include <vector>
#include <queue>
#include <cstring>
#include <cstdlib>
#include <cfloat>
#include <cstdint>
#include <set>


struct _IvpGuard{
    _IvpGuard(){
        int flag;
        MPI_Initialized(&flag);
        if(!flag) MPI_Init(nullptr, nullptr);
    }
    ~_IvpGuard() { MPI_Finalize(); }
};
static _IvpGuard _ivp_guard;

/*MPI 工具函数*/
inline void _ivp_serialize_pq(const std::priority_queue<std::pair<float, int>>& pq, std::vector<char>& buf){

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
_ivp_merge_topk(const std::vector<std::vector<char>>& bufs, size_t k){

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

/*K-Means 聚类*/
inline void _ivp_kmeans(const float* base, size_t bn, size_t vd,
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

/*IVF 索引*/
static std::vector<float> _ivp_ivf_cents;
static std::vector<std::vector<int>> _ivp_ivf_lists;
static size_t _ivp_ivf_nlist;
static const float* _ivp_ivf_base;
inline void _ivp_build_ivf(const float* base, size_t bn, size_t vd){

    _ivp_ivf_nlist = 256;
    _ivp_ivf_base = base;
    _ivp_ivf_cents.resize(_ivp_ivf_nlist * vd);
    _ivp_kmeans(base, bn, vd, _ivp_ivf_cents.data(), _ivp_ivf_nlist, 15);
    _ivp_ivf_lists.resize(_ivp_ivf_nlist);
    for(size_t c = 0; c < _ivp_ivf_nlist; c++)
        _ivp_ivf_lists[c].clear();
    for(size_t i = 0; i < bn; i++){
        const float* v = base + i * vd;
        float best_d = FLT_MAX;
        int best_c = 0;
        for(size_t c = 0; c < _ivp_ivf_nlist; c++){
            float d = InnerProductSIMD(v, _ivp_ivf_cents.data() + c * vd, vd);
            if(d < best_d){
                best_d = d;
                best_c = (int)c;
            }
        }
        _ivp_ivf_lists[best_c].push_back((int)i);
    }
}

/*PQ 编码*/
static std::vector<float> _ivp_pq_cents;
static std::vector<uint8_t> _ivp_pq_codes;
static size_t _ivp_M = 8;
static size_t _ivp_Ks = 256;
inline void _ivp_build_pq(const float* base, size_t bn, size_t vd){

    size_t M = _ivp_M;
    size_t Ks = _ivp_Ks;
    size_t ds = vd / M;
    _ivp_pq_cents.resize(M * Ks * ds);

    /* 对每个子空间做 K-Means */
    for(size_t m = 0; m < M; m++){
        std::vector<float> sub_vectors(bn * ds);
        for(size_t i = 0; i < bn; i++)
            std::memcpy(sub_vectors.data() + i * ds, base + i * vd + m * ds, ds * sizeof(float));
        _ivp_kmeans(sub_vectors.data(), bn, ds, _ivp_pq_cents.data() + m * Ks * ds, Ks, 10);
    }

    /* 编码 */
    _ivp_pq_codes.resize(M * bn);
    for(size_t i = 0; i < bn; i++){
        for(size_t m = 0; m < M; m++){

            const float* v = base + i * vd + m * ds;
            const float* cents = _ivp_pq_cents.data() + m * Ks * ds;
            float best_d = FLT_MAX;
            uint8_t best_k = 0;
            for(size_t k = 0; k < Ks; k++){
                float d = InnerProductSIMD(v, cents + k * ds, ds);
                if(d < best_d){
                    best_d = d;
                    best_k = (uint8_t)k;
                }
            }
            _ivp_pq_codes[m * bn + i] = best_k;
        }
    }
}

inline void _ivp_build_lut(const float* query, float* lut, const float* pq_cents, size_t vd,
                            size_t M, size_t Ks){

    size_t ds = vd / M;
    for(size_t m = 0; m < M; m++){
        const float* qsub = query + m * ds;
        const float* cents_m = pq_cents + m * Ks * ds;
        float* lut_m = lut + m * Ks;
        for(size_t k = 0; k < Ks; k++)
            lut_m[k] = 1.0f - InnerProductSIMD(qsub, cents_m + k * ds, ds);
    }
}

/*inverted list 序列化*/
inline std::vector<char> _ivp_serialize_lists(
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

inline void _ivp_deser_filter(const char* buf, int rank, int nranks, std::vector<std::vector<int>>& lists){

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

/*MPI 状态*/
static int _ivp_rank = 0, _ivp_nranks = 1;

/*对外接口*/
inline void ivf_pq_mpi_build(const float* base, size_t base_number, size_t vecdim){

    MPI_Comm_rank(MPI_COMM_WORLD, &_ivp_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_ivp_nranks);
    size_t bn = base_number;
    size_t vd = vecdim;

    /* 构建 IVF + PQ */
    if(_ivp_rank == 0){
        _ivp_build_ivf(base, bn, vd);
        _ivp_build_pq(base, bn, vd);
    }

    /* 广播 IVF */
    size_t nlist = 0;
    if(_ivp_rank == 0) nlist = _ivp_ivf_nlist;
    MPI_Bcast(&nlist, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    _ivp_ivf_nlist = nlist;
    _ivp_ivf_cents.resize(nlist * vd);
    MPI_Bcast(_ivp_ivf_cents.data(), nlist * vd, MPI_FLOAT, 0, MPI_COMM_WORLD);
    _ivp_ivf_base = base;

    /* 广播 PQ */
    size_t M = _ivp_M;
    size_t Ks = _ivp_Ks;
    size_t ds = vd / M;
    _ivp_pq_cents.resize(M * Ks * ds);
    MPI_Bcast(_ivp_pq_cents.data(), M * Ks * ds, MPI_FLOAT, 0, MPI_COMM_WORLD);
    _ivp_pq_codes.resize(M * bn);
    MPI_Bcast(_ivp_pq_codes.data(), M * bn, MPI_BYTE, 0, MPI_COMM_WORLD);

    /* 广播倒排列表 */
    std::vector<char> serialized;
    size_t serialized_size = 0;
    if(_ivp_rank == 0){
        serialized = _ivp_serialize_lists(_ivp_ivf_lists);
        serialized_size = serialized.size();
    }
    MPI_Bcast(&serialized_size, 1, MPI_UNSIGNED_LONG, 0, MPI_COMM_WORLD);
    if(_ivp_rank != 0)
        serialized.resize(serialized_size);
    MPI_Bcast(serialized.data(), serialized_size, MPI_BYTE, 0, MPI_COMM_WORLD);
    _ivp_deser_filter(serialized.data(), _ivp_rank, _ivp_nranks, _ivp_ivf_lists);
}

inline std::priority_queue<std::pair<float, int>>
ivf_pq_mpi_solve(const float* base, const float* query, size_t base_number, size_t vecdim, size_t k){

    size_t nprobe = ARM_IVF_PQ_NPROBE;
    size_t rerank_p = ARM_IVF_PQ_RERANK;
    size_t M = _ivp_M;
    size_t Ks = _ivp_Ks;
    size_t nlist = _ivp_ivf_nlist;
    size_t bn = base_number;
    size_t vd = vecdim;

    /* 构建 LUT */
    std::vector<float> lut(M * Ks);
    _ivp_build_lut(query, lut.data(), _ivp_pq_cents.data(), vd, M, Ks);

    /* 质心扫描 */
    std::priority_queue<std::pair<float, int>> cluster_heap;
    for(size_t c = 0; c < nlist; c++){
        float d = InnerProductSIMD(query, _ivp_ivf_cents.data() + c * vd, vd);
        cluster_heap.push(std::make_pair(-d, (int)c));
    }
    std::vector<int> probes;
    for(size_t i = 0; i < nprobe && !cluster_heap.empty(); i++){
        probes.push_back(cluster_heap.top().second);
        cluster_heap.pop();
    }
    std::priority_queue<std::pair<float, int>> local_result;
    if(rerank_p == 0){
        /* pq_only: 直接 PQ 距离取 top-k */
        for(int pc : probes){
            if((size_t)pc >= _ivp_ivf_lists.size() || _ivp_ivf_lists[pc].empty())
                continue;
            for(int idx : _ivp_ivf_lists[pc]){
                float ip_sum = 0.0f;
                for(size_t m = 0; m < M; m++)
                    ip_sum += lut[m * Ks + _ivp_pq_codes[m * bn + idx]];
                float dist = 1.0f - ip_sum;
                if(local_result.size() < k){
                    local_result.push(std::make_pair(dist, idx));
                }
                else{
                    local_result.pop();
                    local_result.push(std::make_pair(dist, idx));
                }
            }
        }
    }
    else{

        /* PQ 粗排 → 精确 rerank */
        std::priority_queue<std::pair<float, int>> pq_heap;
        for(int pc : probes){
            if((size_t)pc >= _ivp_ivf_lists.size() || _ivp_ivf_lists[pc].empty())
                continue;
            for(int idx : _ivp_ivf_lists[pc]){
                float ip_sum = 0.0f;
                for(size_t m = 0; m < M; m++)
                    ip_sum += lut[m * Ks + _ivp_pq_codes[m * bn + idx]];
                float dist = 1.0f - ip_sum;
                if(pq_heap.size() < rerank_p){
                    pq_heap.push(std::make_pair(dist, idx));
                }
                else{
                    pq_heap.pop();
                    pq_heap.push(std::make_pair(dist, idx));
                }
            }
        }

        /* 精确重排 */
        while(!pq_heap.empty()){
            int idx = pq_heap.top().second;
            pq_heap.pop();
            float dist = InnerProductSIMD(base + (size_t)idx * vd, query, vd);
            if(local_result.size() < k){
                local_result.push(std::make_pair(dist, idx));
            }
            else{
                local_result.pop();
                local_result.push(std::make_pair(dist, idx));
            }
        }
    }

    /* MPI gather + merge */
    std::vector<char> buf;
    _ivp_serialize_pq(local_result, buf);
    int local_sz = (int)buf.size();
    std::vector<int> all_sizes(_ivp_nranks);
    MPI_Gather(&local_sz, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> displs(_ivp_nranks, 0);
    std::vector<char> all_bufs;
    if(_ivp_rank == 0){
        int total = 0;
        for(int r = 0; r < _ivp_nranks; r++){
            displs[r] = total;
            total += all_sizes[r];
        }
        all_bufs.resize(total);
    }
    MPI_Gatherv(buf.data(), local_sz, MPI_BYTE, all_bufs.data(), all_sizes.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if(_ivp_rank == 0){
        std::vector<std::vector<char>> rank_bufs(_ivp_nranks);
        for(int r = 0; r < _ivp_nranks; r++){
            if(all_sizes[r] > 0){
                rank_bufs[r].assign(all_bufs.begin() + displs[r], all_bufs.begin() + displs[r] + all_sizes[r]);
            }
        }
        return _ivp_merge_topk(rank_bufs, k);
    }
    return std::priority_queue<std::pair<float, int>>();
}
#endif
