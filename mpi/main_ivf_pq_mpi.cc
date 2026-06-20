#include <mpi.h>
#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <cstdlib>
#include "../ivf_pq_simd.h"
#include "mpi_utils.h"
#ifdef _OPENMP
#include <omp.h>


#endif

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d){

    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if(!fin.is_open()){ std::cerr<<"ERROR: cannot open "<<data_path<<"\n"; std::exit(1); }
    fin.read((char*)&n,4); fin.read((char*)&d,4);
    T* data = new T[n*d]; int sz = sizeof(T);
    for(size_t i=0;i<n;++i) fin.read(((char*)data+i*d*sz), d*sz);
    fin.close(); return data;
}

/* 序列化 vector<vector<int>> */
inline std::vector<char> serialize_lists(const std::vector<std::vector<int>>& lists){

    size_t total = sizeof(int);
    for(const auto& L : lists) total += sizeof(int) + L.size()*sizeof(int);
    std::vector<char> buf(total); char* ptr = buf.data();
    int nlist = (int)lists.size(); *(int*)ptr = nlist; ptr += sizeof(int);
    for(const auto& L : lists){
        int sz = (int)L.size(); *(int*)ptr = sz; ptr += sizeof(int);
        if(sz>0){ std::memcpy(ptr, L.data(), sz*sizeof(int)); ptr += sz*sizeof(int); }
    }
    return buf;
}

inline void deserialize_and_filter(const char* buf, int rank, int nranks,
                                    std::vector<std::vector<int>>& lists){

    const char* ptr = buf;
    int nlist = *(const int*)ptr; ptr += sizeof(int);
    lists.resize(nlist);
    int start = (int)((size_t)rank*nlist/nranks);
    int end = (int)(((size_t)rank+1)*nlist/nranks);
    for(int c=0;c<nlist;c++){
        int sz = *(const int*)ptr; ptr += sizeof(int);
        if(c>=start && c<end){ lists[c].resize(sz); if(sz>0) std::memcpy(lists[c].data(), ptr, sz*sizeof(int)); }
else lists[c].clear();
        ptr += sz*sizeof(int);
    }
}

int main(int argc, char* argv[]){

    MPI_Init(&argc, &argv);
    int rank, nranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);
    std::string dp = "../data/";
    const char* mode_env = std::getenv("IVF_MODE");
    std::string ivf_mode = mode_env ? mode_env : "rerank_serial";
    const char* nt_env = std::getenv("NUM_THREADS");
    int nthreads = nt_env ? std::atoi(nt_env) : 4;
    const char* env_np = std::getenv("IVF_NPROBE");
    size_t nprobe = env_np ? (size_t)std::atoi(env_np) : 16;
    const char* env_rp = std::getenv("IVF_RERANK");
    size_t rerank_p = env_rp ? (size_t)std::atoi(env_rp) : 200;
    // rank 0 加载数据，构建 IVF+PQ
    size_t test_number=0, base_number=0, test_gt_d=0, vecdim=0;
    float *base=nullptr, *test_query=nullptr; int *test_gt=nullptr;
    if(rank == 0){
        test_query = LoadData<float>(dp+"DEEP100K.query.fbin", test_number, vecdim);
        test_gt = LoadData<int>(dp+"DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
        base = LoadData<float>(dp+"DEEP100K.base.100k.fbin", base_number, vecdim);
        test_number = 2000;
        std::cerr << "Rank 0: building IVF+PQ...\n";
        build_ivf(base, base_number, vecdim); // IVF 索引
        build_pq(base, base_number, vecdim); // PQ 编码
    }
    size_t meta[4];
    if(rank==0){meta[0]=base_number;meta[1]=vecdim;meta[2]=test_number;meta[3]=test_gt_d;}
    MPI_Bcast(meta,4, MPI_UNSIGNED_LONG,0, MPI_COMM_WORLD);
    base_number=meta[0]; vecdim=meta[1]; test_number=meta[2]; test_gt_d=meta[3];
    const size_t k = 10;
    size_t nlist=0, M=0, Ks=0;
    if(rank==0){ nlist=g_ivf_nlist; M=g_pq_M; Ks=g_pq_Ks; }
    MPI_Bcast(&nlist,1, MPI_UNSIGNED_LONG,0, MPI_COMM_WORLD);
    MPI_Bcast(&M,1, MPI_UNSIGNED_LONG,0, MPI_COMM_WORLD);
    MPI_Bcast(&Ks,1, MPI_UNSIGNED_LONG,0, MPI_COMM_WORLD);
    size_t dsub = vecdim / M;
    if(rank!=0){
        base=new float[base_number*vecdim];
        test_query=new float[test_number*vecdim];
        test_gt=new int[test_number*test_gt_d];
    }
    MPI_Bcast(base, base_number*vecdim, MPI_FLOAT,0, MPI_COMM_WORLD);
    MPI_Bcast(test_query, test_number*vecdim, MPI_FLOAT,0, MPI_COMM_WORLD);
    MPI_Bcast(test_gt, test_number*test_gt_d, MPI_INT,0, MPI_COMM_WORLD);
    // 广播质心
    g_ivf_nlist=nlist; g_ivf_centroids.resize(nlist*vecdim);
    MPI_Bcast(g_ivf_centroids.data(), nlist*vecdim, MPI_FLOAT,0, MPI_COMM_WORLD);
    g_ivf_base=base;
    // 广播 PQ 质心和编码
    g_pq_M=M; g_pq_Ks=Ks;
    g_pq_centroids.resize(M*Ks*dsub);
    MPI_Bcast(g_pq_centroids.data(), M*Ks*dsub, MPI_FLOAT,0, MPI_COMM_WORLD);
    g_pq_codes.resize(M*base_number);
    MPI_Bcast(g_pq_codes.data(), M*base_number, MPI_BYTE,0, MPI_COMM_WORLD);
    // 广播 inverted list 并过滤
    std::vector<char> ser;
    size_t ser_sz=0;
    if(rank==0){ ser=serialize_lists(g_ivf_lists); ser_sz=ser.size(); }
    MPI_Bcast(&ser_sz,1, MPI_UNSIGNED_LONG,0, MPI_COMM_WORLD);
    if(rank!=0) ser.resize(ser_sz);
    MPI_Bcast(ser.data(), ser_sz, MPI_BYTE,0, MPI_COMM_WORLD);
    deserialize_and_filter(ser.data(), rank, nranks, g_ivf_lists);
    if(rank==0) std::cerr<<"IVF-PQ ready, mode="<<ivf_mode<<" nprobe="<<nprobe<<" rerank_p="<<rerank_p<<"\n";
    // 搜索
    double t_start = MPI_Wtime();
    std::vector<float> recalls(test_number,0.0f);
    if(ivf_mode == "pq_only"){
        // PQ 扫描，无 rerank
        for(size_t qi=0;qi<test_number;qi++){
            auto pq = ivf_pq_search(test_query+qi*vecdim, base_number, vecdim, k, nprobe);
            std::vector<char> buf; serialize_pq(pq, buf);
            int sz=(int)buf.size();
            std::vector<int> all_sizes(nranks);
            MPI_Gather(&sz,1, MPI_INT, all_sizes.data(),1, MPI_INT,0, MPI_COMM_WORLD);
            std::vector<int> disp(nranks,0); std::vector<char> all_bufs;
            if(rank==0){ int t=0; for(int r=0;r<nranks;r++){disp[r]=t;t+=all_sizes[r];} all_bufs.resize(t); }
            MPI_Gatherv(buf.data(), sz, MPI_BYTE, all_bufs.data(), all_sizes.data(), disp.data(), MPI_BYTE,0, MPI_COMM_WORLD);
            if(rank==0){
                std::vector<std::vector<char>> rbufs(nranks);
                for(int r=0;r<nranks;r++) if(all_sizes[r]>0) rbufs[r].assign(all_bufs.begin()+disp[r], all_bufs.begin()+disp[r]+all_sizes[r]);
                recalls[qi]=compute_recall(merge_topk(rbufs, k), test_gt+qi*test_gt_d, k, test_gt_d);
            }
        }
    }
    else if(ivf_mode == "rerank_serial"){
        // PQ 扫描 + 串行 rerank（每个 rank 独立做 PQ 扫描取 top-rerank_p，再精确重排）
        for(size_t qi=0;qi<test_number;qi++){
            const float* q = test_query + qi*vecdim;
            // 构建 LUT
            std::vector<float> lut(M*Ks);
            build_lut(q, lut.data(), g_pq_centroids.data(), vecdim, M, Ks);
            // 质心扫描选 nprobe 簇
            std::priority_queue<std::pair<float, int>> ch;
            for(size_t c=0;c<nlist;c++){
                float d=InnerProductSIMD(q, g_ivf_centroids.data()+c*vecdim, vecdim);
                ch.push(std::make_pair(-d,(int)c));
            }
            std::vector<int> probes;
            for(size_t i=0;i<nprobe&&!ch.empty();i++){probes.push_back(ch.top().second);ch.pop();}
            // PQ 扫描：本 rank 负责的簇，收集 top-rerank_p
            std::priority_queue<std::pair<float, int>> pq_heap;
            for(int c : probes){
                if((size_t)c >= g_ivf_lists.size() || g_ivf_lists[c].empty()) continue;
                for(int idx : g_ivf_lists[c]){
                    float ip_sum=0.0f;
                    for(size_t m=0;m<M;m++) ip_sum+=lut[m*Ks+g_pq_codes[m*base_number+(size_t)idx]];
                    float dist=1.0f-ip_sum;
                    if(pq_heap.size()<rerank_p) pq_heap.push(std::make_pair(dist, idx));
                    else if(dist<pq_heap.top().first){pq_heap.pop();pq_heap.push(std::make_pair(dist, idx));}
                }
            }
            // 串行 rerank
            std::priority_queue<std::pair<float, int>> local_result;
            while(!pq_heap.empty()){
                int idx=pq_heap.top().second; pq_heap.pop();
                float dist=InnerProductSIMD(base+(size_t)idx*vecdim, q, vecdim);
                if(local_result.size()<k) local_result.push(std::make_pair(dist, idx));
                else if(dist<local_result.top().first){local_result.pop();local_result.push(std::make_pair(dist, idx));}
            }
            // MPI gather + merge
            std::vector<char> buf; serialize_pq(local_result, buf);
            int sz=(int)buf.size();
            std::vector<int> all_sizes(nranks);
            MPI_Gather(&sz,1, MPI_INT, all_sizes.data(),1, MPI_INT,0, MPI_COMM_WORLD);
            std::vector<int> disp(nranks,0); std::vector<char> all_bufs;
            if(rank==0){ int t=0;for(int r=0;r<nranks;r++){disp[r]=t;t+=all_sizes[r];}all_bufs.resize(t);}
            MPI_Gatherv(buf.data(), sz, MPI_BYTE, all_bufs.data(), all_sizes.data(), disp.data(), MPI_BYTE,0, MPI_COMM_WORLD);
            if(rank==0){
                std::vector<std::vector<char>> rbufs(nranks);
                for(int r=0;r<nranks;r++) if(all_sizes[r]>0) rbufs[r].assign(all_bufs.begin()+disp[r], all_bufs.begin()+disp[r]+all_sizes[r]);
                recalls[qi]=compute_recall(merge_topk(rbufs, k), test_gt+qi*test_gt_d, k, test_gt_d);
            }
        }
    }
    else if(ivf_mode == "rerank_omp"){
        // PQ 扫描 + OpenMP 并行 rerank
#ifdef _OPENMP
        for(size_t qi=0;qi<test_number;qi++){
            const float* q = test_query + qi*vecdim;
            std::vector<float> lut(M*Ks);
            build_lut(q, lut.data(), g_pq_centroids.data(), vecdim, M, Ks);
            std::priority_queue<std::pair<float, int>> ch;
            for(size_t c=0;c<nlist;c++){
                float d=InnerProductSIMD(q, g_ivf_centroids.data()+c*vecdim, vecdim);
                ch.push(std::make_pair(-d,(int)c));
            }
            std::vector<int> probes;
            for(size_t i=0;i<nprobe&&!ch.empty();i++){probes.push_back(ch.top().second);ch.pop();}
            // PQ 扫描
            std::priority_queue<std::pair<float, int>> pq_heap;
            for(int c : probes){
                if((size_t)c >= g_ivf_lists.size() || g_ivf_lists[c].empty()) continue;
                for(int idx : g_ivf_lists[c]){
                    float ip_sum=0.0f;
                    for(size_t m=0;m<M;m++) ip_sum+=lut[m*Ks+g_pq_codes[m*base_number+(size_t)idx]];
                    float dist=1.0f-ip_sum;
                    if(pq_heap.size()<rerank_p) pq_heap.push(std::make_pair(dist, idx));
                    else if(dist<pq_heap.top().first){pq_heap.pop();pq_heap.push(std::make_pair(dist, idx));}
                }
            }
            // Rerank: OpenMP 并行
            std::vector<std::pair<float, int>> candidates;
            while(!pq_heap.empty()){candidates.push_back(pq_heap.top());pq_heap.pop();}
            // candidates 现在是 desc (worst first)，rerank_p 个候选
            std::vector<std::pair<float, int>> reranked(candidates.size());
            #pragma omp parallel for num_threads(nthreads) schedule(static)
            for(size_t ci=0;ci<candidates.size();ci++){
                int idx=candidates[ci].second;
                float dist=InnerProductSIMD(base+(size_t)idx*vecdim, q, vecdim);
                reranked[ci]=std::make_pair(dist, idx);
            }
            // 取 top-k
            std::priority_queue<std::pair<float, int>> local_result;
            for(auto& r : reranked){
                if(local_result.size()<k) local_result.push(r);
                else if(r.first<local_result.top().first){local_result.pop();local_result.push(r);}
            }
            // MPI gather + merge
            std::vector<char> buf; serialize_pq(local_result, buf);
            int sz=(int)buf.size();
            std::vector<int> all_sizes(nranks);
            MPI_Gather(&sz,1, MPI_INT, all_sizes.data(),1, MPI_INT,0, MPI_COMM_WORLD);
            std::vector<int> disp(nranks,0); std::vector<char> all_bufs;
            if(rank==0){ int t=0;for(int r=0;r<nranks;r++){disp[r]=t;t+=all_sizes[r];}all_bufs.resize(t);}
            MPI_Gatherv(buf.data(), sz, MPI_BYTE, all_bufs.data(), all_sizes.data(), disp.data(), MPI_BYTE,0, MPI_COMM_WORLD);
            if(rank==0){
                std::vector<std::vector<char>> rbufs(nranks);
                for(int r=0;r<nranks;r++) if(all_sizes[r]>0) rbufs[r].assign(all_bufs.begin()+disp[r], all_bufs.begin()+disp[r]+all_sizes[r]);
                recalls[qi]=compute_recall(merge_topk(rbufs, k), test_gt+qi*test_gt_d, k, test_gt_d);
            }
        }
#else
        if(rank==0) std::cerr<<"OpenMP not available, use rerank_serial\n";
#endif
    }
    double t_end = MPI_Wtime();
    float local_ar=0;
    for(size_t i=0;i<test_number;i++) local_ar+=recalls[i];
    float global_ar=0;
    MPI_Reduce(&local_ar,&global_ar,1, MPI_FLOAT, MPI_SUM,0, MPI_COMM_WORLD);
    if(rank==0){
        std::cout<<"=== IVF-PQ MPI ===\n";
        std::cout<<"mode: "<<ivf_mode<<" nranks: "<<nranks<<" nthreads: "<<nthreads<<"\n";
        std::cout<<"nlist: "<<nlist<<" nprobe: "<<nprobe<<" rerank_p: "<<rerank_p<<"\n";
        std::cout<<"average recall: "<<global_ar/test_number<<"\n";
        std::cout<<"total_time (s): "<<(t_end-t_start)<<"\n";
        std::cout<<"average latency (us): "<<((t_end-t_start)*1e6/test_number)<<"\n";
    }
    delete[] base; delete[] test_query; delete[] test_gt;
    MPI_Finalize(); return 0;
}
