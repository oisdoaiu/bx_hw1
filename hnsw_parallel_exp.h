#ifndef HNSW_PARALLEL_EXP_H
#define HNSW_PARALLEL_EXP_H

#include <queue>
#include <vector>
#include <utility>
#include <algorithm>
#include <omp.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "hnswlib/hnswlib/hnswalg.h"

// Parallel expansion search for HNSW base layer.
// At each iteration, pops K candidates from the candidate_set (instead of 1),
// processes their neighbors in parallel via OpenMP, then merges results.
// K=1 is equivalent to serial search; larger K trades extra distance
// computations for parallelism.
template<typename dist_t>
std::priority_queue<std::pair<dist_t, hnswlib::tableint>,
                    std::vector<std::pair<dist_t, hnswlib::tableint>>,
                    typename hnswlib::HierarchicalNSW<dist_t>::CompareByFirst>
hnsw_searchBaseLayer_par(
    const hnswlib::HierarchicalNSW<dist_t>* hnsw,
    hnswlib::tableint ep_id,
    const void* data_point,
    size_t ef,
    int K,
    int num_threads)
{
    using hnswlib::tableint;
    using hnswlib::linklistsizeint;
    using hnswlib::vl_type;
    using Cmp = typename hnswlib::HierarchicalNSW<dist_t>::CompareByFirst;

    hnswlib::VisitedList* vl = hnsw->visited_list_pool_->getFreeVisitedList();
    vl_type* visited_array = vl->mass;
    vl_type visited_array_tag = vl->curV;

    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>, Cmp> top_candidates;
    std::priority_queue<std::pair<dist_t, tableint>,
                        std::vector<std::pair<dist_t, tableint>>, Cmp> candidate_set;

    dist_t lowerBound;
    dist_t dist = hnsw->fstdistfunc_(data_point,
                                     hnsw->getDataByInternalId(ep_id),
                                     hnsw->dist_func_param_);
    top_candidates.emplace(dist, ep_id);
    lowerBound = dist;
    candidate_set.emplace(-dist, ep_id);
    visited_array[ep_id] = visited_array_tag;

    // Per-thread result buffers (allocated once, reused each iteration)
    struct ThreadBuf {
        std::vector<std::pair<dist_t, tableint>> candidates;
    };
    std::vector<ThreadBuf> thread_bufs(num_threads);

    while (!candidate_set.empty()) {
        // --- Extract up to K candidates as the parallel frontier ---
        std::vector<std::pair<dist_t, tableint>> frontier;
        {
            while ((int)frontier.size() < K && !candidate_set.empty()) {
                auto p = candidate_set.top();
                candidate_set.pop();
                dist_t d = -p.first;
                tableint id = p.second;

                // Standard HNSW stopping condition: if the closest remaining
                // candidate is already worse than lowerBound and we have
                // enough results, stop the entire search.
                if (top_candidates.size() == ef && d > lowerBound) {
                    // Put it back and break — same as serial HNSW.
                    candidate_set.emplace(-d, id);
                    goto search_done;
                }
                frontier.push_back({d, id});
            }
        }

        if (frontier.empty()) break;

        // Clear per-thread buffers
        for (int t = 0; t < num_threads; t++)
            thread_bufs[t].candidates.clear();

        // --- Process frontier nodes' neighbors ---
        // Fast path: single-threaded or single-frontier → skip atomic CAS + OpenMP
        if (num_threads == 1 || frontier.size() == 1) {
            auto& local = thread_bufs[0];
            for (size_t fi = 0; fi < frontier.size(); fi++) {
                tableint node_id = frontier[fi].second;
                int* data = (int*)hnsw->get_linklist0(node_id);
                size_t size = hnsw->getListCount((linklistsizeint*)data);

                for (size_t j = 1; j <= size; j++) {
                    tableint candidate_id = *(data + j);
                    if (visited_array[candidate_id] == visited_array_tag) continue;
                    visited_array[candidate_id] = visited_array_tag;

                    dist_t d = hnsw->fstdistfunc_(
                        data_point,
                        hnsw->getDataByInternalId(candidate_id),
                        hnsw->dist_func_param_);

                    local.candidates.push_back({d, candidate_id});
                }
            }
        } else {
            // Parallel path with atomic CAS for thread safety
#pragma omp parallel num_threads(num_threads)
            {
                int tid = omp_get_thread_num();
                auto& local = thread_bufs[tid];

#pragma omp for schedule(dynamic, 1) nowait
                for (size_t fi = 0; fi < frontier.size(); fi++) {
                    tableint node_id = frontier[fi].second;
                    int* data = (int*)hnsw->get_linklist0(node_id);
                    size_t size = hnsw->getListCount((linklistsizeint*)data);

                    for (size_t j = 1; j <= size; j++) {
                        tableint candidate_id = *(data + j);

                        // Atomically check-and-set visited flag
                        vl_type old_val = __atomic_load_n(
                            &visited_array[candidate_id], __ATOMIC_RELAXED);
                        if (old_val == visited_array_tag) continue;

                        if (!__atomic_compare_exchange_n(
                                &visited_array[candidate_id], &old_val,
                                visited_array_tag, false,
                                __ATOMIC_RELAXED, __ATOMIC_RELAXED))
                            continue;

                        dist_t d = hnsw->fstdistfunc_(
                            data_point,
                            hnsw->getDataByInternalId(candidate_id),
                            hnsw->dist_func_param_);

                        local.candidates.push_back({d, candidate_id});
                    }
                }
            }  // end parallel region
        }

        // --- Merge thread-local results ---
        for (int t = 0; t < num_threads; t++) {
            for (auto& cand : thread_bufs[t].candidates) {
                if (top_candidates.size() < ef || lowerBound > cand.first) {
                    candidate_set.emplace(-cand.first, cand.second);
                    if (!hnsw->isMarkedDeleted(cand.second))
                        top_candidates.emplace(cand.first, cand.second);
                    if (top_candidates.size() > ef)
                        top_candidates.pop();
                    if (!top_candidates.empty())
                        lowerBound = top_candidates.top().first;
                }
            }
        }
    }

search_done:
    hnsw->visited_list_pool_->releaseVisitedList(vl);
    return top_candidates;
}


// Full parallel-expansion HNSW search (replaces searchKnn).
// Multi-layer descent is sequential; only the base layer (layer 0)
// uses parallel expansion.
inline std::priority_queue<std::pair<float, hnswlib::labeltype>>
hnsw_par_search(
    const hnswlib::HierarchicalNSW<float>* hnsw,
    const void* query_data,
    size_t k,
    int expansion_K,
    int num_threads)
{
    typedef float dist_t;
    std::priority_queue<std::pair<dist_t, hnswlib::labeltype>> result;
    if (hnsw->cur_element_count == 0) return result;

    // --- Multi-layer descent (sequential, same as original) ---
    hnswlib::tableint currObj = hnsw->enterpoint_node_;
    dist_t curdist = hnsw->fstdistfunc_(query_data,
        hnsw->getDataByInternalId(hnsw->enterpoint_node_),
        hnsw->dist_func_param_);

    for (int level = hnsw->maxlevel_; level > 0; level--) {
        bool changed = true;
        while (changed) {
            changed = false;
            unsigned int* data = (unsigned int*)hnsw->get_linklist(currObj, level);
            int size = hnsw->getListCount(data);
            hnswlib::tableint* datal = (hnswlib::tableint*)(data + 1);
            for (int i = 0; i < size; i++) {
                hnswlib::tableint cand = datal[i];
                dist_t d = hnsw->fstdistfunc_(query_data,
                    hnsw->getDataByInternalId(cand),
                    hnsw->dist_func_param_);
                if (d < curdist) {
                    curdist = d;
                    currObj = cand;
                    changed = true;
                }
            }
        }
    }

    // --- Base layer: parallel expansion search ---
    size_t ef = std::max(hnsw->ef_, k);
    auto top_candidates = hnsw_searchBaseLayer_par<dist_t>(
        hnsw, currObj, query_data, ef, expansion_K, num_threads);

    while (top_candidates.size() > k)
        top_candidates.pop();

    while (!top_candidates.empty()) {
        auto rez = top_candidates.top();
        result.push(std::make_pair(rez.first,
            hnsw->getExternalLabel(rez.second)));
        top_candidates.pop();
    }
    return result;
}

#endif
