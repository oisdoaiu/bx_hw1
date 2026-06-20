#ifndef MPI_UTILS_H
#define MPI_UTILS_H
#include <mpi.h>
#include <vector>
#include <queue>
#include <cstring>
#include <cstdlib>
#include <set>
#include <sys/time.h>
#include <iostream>


inline void serialize_pq(const std::priority_queue<std::pair<float, int>>& pq,
                          std::vector<char>& buf){
    size_t n = pq.size();
    buf.resize(sizeof(size_t) + n * (sizeof(float) + sizeof(int)));
    char* ptr = buf.data();
    *(size_t*)ptr = n; ptr += sizeof(size_t);
    auto copy = pq; // 拷贝一份用于遍历
    while(!copy.empty()){
        auto& p = copy.top();
        *(float*)ptr = p.first; ptr += sizeof(float);
        *(int*)ptr = p.second; ptr += sizeof(int);
        copy.pop();
    }
}

inline void deserialize_pq(const char* buf, std::priority_queue<std::pair<float, int>>& pq){
    size_t n = *(const size_t*)buf;
    const char* ptr = buf + sizeof(size_t);
    for(size_t i=0; i<n; ++i){
        float f = *(const float*)ptr; ptr += sizeof(float);
        int id = *(const int*)ptr; ptr += sizeof(int);
        pq.push(std::make_pair(f, id));
    }
}

inline std::priority_queue<std::pair<float, int>>
merge_topk(const std::vector<std::vector<char>>& all_queues, size_t k){
    std::priority_queue<std::pair<float, int>> result;
    for(const auto& buf : all_queues){
        if(buf.empty()) continue;
        size_t n = *(const size_t*)buf.data();
        const char* ptr = buf.data() + sizeof(size_t);
        for(size_t i=0; i<n; ++i){
            float f = *(const float*)ptr; ptr += sizeof(float);
            int id = *(const int*)ptr; ptr += sizeof(int);
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
inline float compute_recall(const std::priority_queue<std::pair<float, int>>& result,
                             const int* gt, size_t k, size_t gtd){
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
