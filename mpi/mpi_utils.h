/*
 * mpi_utils.h — MPI 并行化通用工具函数
 * 用于 main_ivf_mpi.cc 和 main_hnsw_mpi.cc
 */

#ifndef MPI_UTILS_H
#define MPI_UTILS_H

#include <mpi.h>
#include <vector>
#include <queue>
#include <cstring>
#include <cstdlib>
#include <sys/time.h>
#include <iostream>

/* 将 priority_queue 序列化为连续缓冲区，用于 MPI 通信
 * 每对 (float, int) 占 sizeof(float)+sizeof(int) 字节
 * 注意：priority_queue 是最大堆，top() 返回最大距离 */
inline void serialize_pq(const std::priority_queue<std::pair<float,int>>& pq,
                          std::vector<char>& buf)
{
    size_t n = pq.size();
    buf.resize(sizeof(size_t) + n * (sizeof(float) + sizeof(int)));
    char* ptr = buf.data();
    *(size_t*)ptr = n; ptr += sizeof(size_t);

    auto copy = pq;  // 拷贝一份用于遍历
    while(!copy.empty()){
        auto& p = copy.top();
        *(float*)ptr = p.first;  ptr += sizeof(float);
        *(int*)ptr  = p.second;  ptr += sizeof(int);
        copy.pop();
    }
}

/* 从缓冲区反序列化为 priority_queue */
inline void deserialize_pq(const char* buf, std::priority_queue<std::pair<float,int>>& pq)
{
    size_t n = *(const size_t*)buf;
    const char* ptr = buf + sizeof(size_t);
    for(size_t i=0; i<n; ++i){
        float f = *(const float*)ptr; ptr += sizeof(float);
        int   id = *(const int*)ptr;   ptr += sizeof(int);
        pq.push(std::make_pair(f, id));
    }
}

/* 合并多个 local top-k 为全局 top-k
 * all_queues: 每个 rank 的 top-k 序列化数据
 * k: 需要的 top-k 数量
 * 返回: 全局 top-k (priority_queue, 最大堆)
 *
 * 算法：收集所有候选（最多 rank_count * k 个），选 top-k */
inline std::priority_queue<std::pair<float,int>>
merge_topk(const std::vector<std::vector<char>>& all_queues, size_t k)
{
    std::priority_queue<std::pair<float,int>> result;
    for(const auto& buf : all_queues){
        if(buf.empty()) continue;
        size_t n = *(const size_t*)buf.data();
        const char* ptr = buf.data() + sizeof(size_t);
        for(size_t i=0; i<n; ++i){
            float f = *(const float*)ptr; ptr += sizeof(float);
            int   id = *(const int*)ptr;   ptr += sizeof(int);
            if(result.size() < k)
                result.push(std::make_pair(f, id));
            else if(f < result.top().first){
                result.pop();
                result.push(std::make_pair(f, id));
            }
        }
    }
    return result;
}

/* 计算召回率 */
inline float compute_recall(const std::priority_queue<std::pair<float,int>>& result,
                             const int* gt, size_t k, size_t gtd)
{
    std::set<uint32_t> gtset;
    for(size_t j=0; j<k; ++j)
        gtset.insert((uint32_t)gt[j]);
    size_t acc = 0;
    auto copy = result;
    while(copy.size()){
        if(gtset.find((uint32_t)copy.top().second) != gtset.end())
            ++acc;
        copy.pop();
    }
    return (float)acc / k;
}

#endif // MPI_UTILS_H
