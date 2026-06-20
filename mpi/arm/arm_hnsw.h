#ifndef ARM_HNSW_H
#define ARM_HNSW_H
#define ARM_HNSW_EF 50
#define ARM_HNSW_MODE 0
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


struct _HnswGuard{
    _HnswGuard(){
        int flag;
        MPI_Initialized(&flag);
        if(!flag) MPI_Init(nullptr, nullptr);
    }
    ~_HnswGuard() { MPI_Finalize(); }
};
static _HnswGuard _hnsw_guard;

inline void _hnsw_serialize_pq(const std::priority_queue<std::pair<float, int>>& pq, std::vector<char>& buf){

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
_hnsw_merge_topk(const std::vector<std::vector<char>>& bufs, size_t k){

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


inline void _hnsw_kmeans_partition(const float* base, size_t bn, size_t vd,
                                    int nparts, std::vector<int>& assign){

    assign.resize(bn);
    std::vector<float> centroids(nparts * vd);
    for(int p = 0; p < nparts; p++){
        size_t idx = (p * (bn / nparts)) % bn;
        std::memcpy(centroids.data() + p * vd, base + idx * vd, vd * sizeof(float));
    }
    for(int iter = 0; iter < 10; iter++){
        /* 分配步骤 */
        for(size_t i = 0; i < bn; i++){
            const float* v = base + i * vd;
            float best_d = FLT_MAX;
            int best_p = 0;
            for(int p = 0; p < nparts; p++){
                float d = InnerProductSIMD(v, centroids.data() + p * vd, vd);
                if(d < best_d){
                    best_d = d;
                    best_p = p;
                }
            }
            assign[i] = best_p;
        }

        /* 更新步骤 */
        std::vector<float> sum(nparts * vd, 0.0f);
        std::vector<int> counts(nparts, 0);
        for(size_t i = 0; i < bn; i++){
            int p = assign[i];
            const float* v = base + i * vd;
            float* dst = sum.data() + p * vd;
            for(size_t d = 0; d < vd; d++)
                dst[d] += v[d];
            counts[p]++;
        }
        for(int p = 0; p < nparts; p++){
            if(counts[p] > 0){
                float inv = 1.0f / (float)counts[p];
                float* cent = centroids.data() + p * vd;
                for(size_t d = 0; d < vd; d++)
                    cent[d] = sum[p * vd + d] * inv;
            }
        }
    }
}


static int _hnsw_rank = 0, _hnsw_nranks = 1;
static hnswlib::HierarchicalNSW<float>* _hnsw_index = nullptr;
static hnswlib::InnerProductSpace* _hnsw_space = nullptr;

inline void hnsw_mpi_build(const float* base, size_t base_number, size_t vecdim){
    MPI_Comm_rank(MPI_COMM_WORLD, &_hnsw_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &_hnsw_nranks);
    int ef = ARM_HNSW_EF;
    size_t bn = base_number;
    size_t vd = vecdim;

    /* 数据划分 */
    std::vector<int> assign(bn);
    if(_hnsw_rank == 0){
        if(ARM_HNSW_MODE == 1){
            _hnsw_kmeans_partition(base, bn, vd, _hnsw_nranks, assign);
        }
        else{
            for(size_t i = 0; i < bn; i++)
                assign[i] = (int)(i * _hnsw_nranks / bn);
        }
    }
    MPI_Bcast(assign.data(), bn, MPI_INT, 0, MPI_COMM_WORLD);

    /* 收集本 rank 的数据 */
    size_t my_count = 0;
    for(size_t i = 0; i < bn; i++)
        if(assign[i] == _hnsw_rank) my_count++;
    std::vector<float> my_base(my_count * vd);
    std::vector<size_t> my_labels(my_count);
    {
        size_t pos = 0;
        for(size_t i = 0; i < bn; i++){
            if(assign[i] == _hnsw_rank){
                std::memcpy(my_base.data() + pos * vd, base + i * vd, vd * sizeof(float));
                my_labels[pos] = i;
                pos++;
            }
        }
    }

    /* 构建本地 HNSW */
    _hnsw_space = new hnswlib::InnerProductSpace(vd);
    _hnsw_index = new hnswlib::HierarchicalNSW<float>(
        _hnsw_space, my_count, 16, 200);
    _hnsw_index->ef_ = ef;
    for(size_t i = 0; i < my_count; i++){
        _hnsw_index->addPoint((void*)(my_base.data() + i * vd), my_labels[i]);
    }
}

inline std::priority_queue<std::pair<float, int>>
hnsw_mpi_solve(const float* base, const float* query, size_t base_number, size_t vecdim, size_t k){
    (void)base;
    (void)base_number;
    auto local_pq = _hnsw_index->searchKnn((void*)query, k);

    /* 转换为统一格式 */
    std::vector<std::pair<float, int>> tmp;
    while(!local_pq.empty()){
        tmp.push_back(std::make_pair(local_pq.top().first,(int)local_pq.top().second));
        local_pq.pop();
    }
    std::priority_queue<std::pair<float, int>> local_result;
    for(int j = (int)tmp.size() - 1; j >= 0; j--)
        local_result.push(tmp[j]);
    std::vector<char> buf;
    _hnsw_serialize_pq(local_result, buf);
    int local_sz = (int)buf.size();
    std::vector<int> all_sizes(_hnsw_nranks);
    MPI_Gather(&local_sz, 1, MPI_INT, all_sizes.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
    std::vector<int> displs(_hnsw_nranks, 0);
    std::vector<char> all_bufs;
    if(_hnsw_rank == 0){
        int total = 0;
        for(int r = 0; r < _hnsw_nranks; r++){
            displs[r] = total;
            total += all_sizes[r];
        }
        all_bufs.resize(total);
    }
    MPI_Gatherv(buf.data(), local_sz, MPI_BYTE, all_bufs.data(), all_sizes.data(), displs.data(), MPI_BYTE, 0, MPI_COMM_WORLD);
    if(_hnsw_rank == 0){
        std::vector<std::vector<char>> rank_bufs(_hnsw_nranks);
        for(int r = 0; r < _hnsw_nranks; r++){
            if(all_sizes[r] > 0){
                rank_bufs[r].assign(all_bufs.begin() + displs[r], all_bufs.begin() + displs[r] + all_sizes[r]);
            }
        }
        return _hnsw_merge_topk(rank_bufs, k);
    }
    return std::priority_queue<std::pair<float, int>>();
}
#endif
