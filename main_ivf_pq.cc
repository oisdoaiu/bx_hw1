#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <cstdlib>
#include "ivf_pq_simd.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <pthread.h>

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4); fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i) fin.read(((char*)data + i*d*sz), d*sz);
    fin.close();
    std::cerr<<"load "<<data_path<<"\n";
    return data;
}

struct SearchResult { float recall; int64_t latency; };

// ===== serial =====
void run_serial(const float* base, const float* query, const int* gt,
                size_t bn, size_t vd, size_t gtd, size_t tn, size_t k, SearchResult* res)
{
    const unsigned long C=1000*1000;
    for(size_t i=0;i<tn;++i){
        struct timeval val; gettimeofday(&val,NULL);
        auto r=ivf_pq_solve(base, query+i*vd, bn, vd, k);
        struct timeval nv; gettimeofday(&nv,NULL);
        int64_t d=(nv.tv_sec*C+nv.tv_usec)-(val.tv_sec*C+val.tv_usec);
        std::set<uint32_t> gt2;
        for(size_t j=0;j<k;++j) gt2.insert((uint32_t)gt[j+i*gtd]);
        size_t acc=0;
        while(r.size()){ if(gt2.find((uint32_t)r.top().second)!=gt2.end())++acc; r.pop(); }
        res[i]={(float)acc/k,d};
    }
}

// ===== Pthread query-level =====
struct QParam{ int tid,nt; const float *base,*query; const int* gt;
    size_t bn,vd,gtd,tn,k; SearchResult* res; };
void* qw(void* arg){
    QParam* p=(QParam*)arg; const unsigned long C=1000*1000;
    for(size_t i=p->tid;i<p->tn;i+=p->nt){
        struct timeval val; gettimeofday(&val,NULL);
        auto r=ivf_pq_solve(p->base,p->query+i*p->vd,p->bn,p->vd,p->k);
        struct timeval nv; gettimeofday(&nv,NULL);
        int64_t d=(nv.tv_sec*C+nv.tv_usec)-(val.tv_sec*C+val.tv_usec);
        std::set<uint32_t> gt2;
        for(size_t j=0;j<p->k;++j) gt2.insert((uint32_t)p->gt[j+i*p->gtd]);
        size_t acc=0;
        while(r.size()){ if(gt2.find((uint32_t)r.top().second)!=gt2.end())++acc; r.pop(); }
        p->res[i]={(float)acc/p->k,d};
    }
    return nullptr;
}
void run_pthread(const float* base, const float* query, const int* gt,
                 size_t bn, size_t vd, size_t gtd, size_t tn, size_t k, SearchResult* res, int nt){
    pthread_t* th=new pthread_t[nt]; QParam* pr=new QParam[nt];
    for(int t=0;t<nt;t++){
        pr[t]={t,nt,base,query,gt,bn,vd,gtd,tn,k,res};
        pthread_create(&th[t],nullptr,qw,&pr[t]);
    }
    for(int t=0;t<nt;t++) pthread_join(th[t],nullptr);
    delete[] th; delete[] pr;
}

#ifdef _OPENMP
// OpenMP query-level
void run_omp_query(const float* base, const float* query, const int* gt,
                   size_t bn, size_t vd, size_t gtd, size_t tn, size_t k, SearchResult* res, int nt){
    const unsigned long C=1000*1000;
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(size_t i=0;i<tn;++i){
        struct timeval val; gettimeofday(&val,NULL);
        auto r=ivf_pq_solve(base, query+i*vd, bn, vd, k);
        struct timeval nv; gettimeofday(&nv,NULL);
        int64_t d=(nv.tv_sec*C+nv.tv_usec)-(val.tv_sec*C+val.tv_usec);
        std::set<uint32_t> gt2;
        for(size_t j=0;j<k;++j) gt2.insert((uint32_t)gt[j+i*gtd]);
        size_t acc=0;
        while(r.size()){ if(gt2.find((uint32_t)r.top().second)!=gt2.end())++acc; r.pop(); }
        res[i]={(float)acc/k,d};
    }
}

// OpenMP cluster-level: divide nprobe clusters among threads, PQ scan + rerank + merge
void run_omp_cluster(const float* base, const float* query, const int* gt,
                     size_t bn, size_t vd, size_t gtd, size_t tn, size_t k, SearchResult* res, int nt){
    const unsigned long C=1000*1000;
    size_t nprobe=16; const char* env=std::getenv("IVF_NPROBE");
    if(env) nprobe=(size_t)std::atoi(env);
    size_t rerank_p=500; const char* env_rr=std::getenv("IVF_RERANK");
    if(env_rr) rerank_p=(size_t)std::atoi(env_rr);

    for(size_t qi=0;qi<tn;++qi){
        struct timeval val; gettimeofday(&val,NULL);
        const float* q=query+qi*vd;

        // LUT
        std::vector<float> lut(g_pq_M*g_pq_Ks);
        build_lut(q,lut.data(),g_pq_centroids.data(),vd,g_pq_M,g_pq_Ks);

        // coarse
        std::priority_queue<std::pair<float,int>> ch;
        for(size_t c=0;c<g_ivf_nlist;c++){
            float d=InnerProductSIMD(q,g_ivf_centroids.data()+c*vd,vd);
            ch.push(std::make_pair(-d,(int)c));
        }
        std::vector<int> probes; probes.reserve(nprobe);
        for(size_t p=0;p<nprobe&&!ch.empty();p++){ probes.push_back(ch.top().second); ch.pop(); }

        // fine: PQ scan (cluster-level parallel) → get top rerank_p candidates
        std::priority_queue<std::pair<float,int>> pq_heap;
        #pragma omp parallel num_threads(nt)
        {
            int tid=omp_get_thread_num(); int nts=omp_get_num_threads();
            size_t chunk=(probes.size()+nts-1)/nts;
            size_t cs=tid*chunk, ce=std::min(cs+chunk,probes.size());
            std::priority_queue<std::pair<float,int>> local_pq;
            for(size_t pi=cs;pi<ce;pi++){
                int c=probes[pi];
                for(int idx:g_ivf_lists[c]){
                    float ip_sum=0.0f;
                    for(size_t m=0;m<g_pq_M;m++)
                        ip_sum+=lut[m*g_pq_Ks+g_pq_codes[m*bn+(size_t)idx]];
                    float dist=1.0f-ip_sum;
                    if(local_pq.size()<rerank_p) local_pq.push(std::make_pair(dist,idx));
                    else if(dist<local_pq.top().first){ local_pq.pop(); local_pq.push(std::make_pair(dist,idx)); }
                }
            }
            #pragma omp critical
            { while(!local_pq.empty()){ pq_heap.push(local_pq.top()); local_pq.pop(); } }
        }

        // merge PQ results: keep top rerank_p
        std::vector<std::pair<float,int>> pq_cand;
        while(!pq_heap.empty()){ pq_cand.push_back(pq_heap.top()); pq_heap.pop(); }
        // sort ascending by dist
        std::sort(pq_cand.begin(), pq_cand.end());
        if(pq_cand.size()>rerank_p) pq_cand.resize(rerank_p);

        // rerank: exact float distance on top rerank_p candidates
        std::priority_queue<std::pair<float,int>> merged;
        for(auto& cand:pq_cand){
            float dist=InnerProductSIMD(base+(size_t)cand.second*vd,q,vd);
            if(merged.size()<k) merged.push(std::make_pair(dist,cand.second));
            else if(dist<merged.top().first){ merged.pop(); merged.push(std::make_pair(dist,cand.second)); }
        }

        struct timeval nv; gettimeofday(&nv,NULL);
        int64_t d=(nv.tv_sec*C+nv.tv_usec)-(val.tv_sec*C+val.tv_usec);
        std::set<uint32_t> gt2;
        for(size_t j=0;j<k;++j) gt2.insert((uint32_t)gt[j+qi*gtd]);
        size_t acc=0;
        while(merged.size()){ if(gt2.find((uint32_t)merged.top().second)!=gt2.end())++acc; merged.pop(); }
        res[qi]={(float)acc/k,d};
    }
}
#endif

int main()
{
    size_t tn=0,bn=0,gtd=0,vd=0;
    std::string dp="data/";
    auto tq=LoadData<float>(dp+"DEEP100K.query.fbin",tn,vd);
    auto tg=LoadData<int>(dp+"DEEP100K.gt.query.100k.top100.bin",tn,gtd);
    auto base=LoadData<float>(dp+"DEEP100K.base.100k.fbin",bn,vd);
    tn=2000; const size_t k=10;

    build_pq(base,bn,vd);
    build_ivf(base,bn,vd);

    const char* mode=std::getenv("IVF_MODE");
    std::string ms=mode?mode:"serial";
    const char* env_nt=std::getenv("NUM_THREADS");
    int nt=env_nt?std::atoi(env_nt):4;

    std::vector<SearchResult> results(tn);

    if(ms=="pthread"){
        std::cerr<<"IVF-PQ Pthread query-level, threads="<<nt<<"\n";
        run_pthread(base,tq,tg,bn,vd,gtd,tn,k,results.data(),nt);
    }
#ifdef _OPENMP
    else if(ms=="openmp"){
        std::cerr<<"IVF-PQ OpenMP query-level, threads="<<nt<<"\n";
        run_omp_query(base,tq,tg,bn,vd,gtd,tn,k,results.data(),nt);
    }
    else if(ms=="openmp_cluster"){
        std::cerr<<"IVF-PQ OpenMP cluster-level, threads="<<nt<<"\n";
        run_omp_cluster(base,tq,tg,bn,vd,gtd,tn,k,results.data(),nt);
    }
#endif
    else {
        std::cerr<<"IVF-PQ serial mode\n";
        run_serial(base,tq,tg,bn,vd,gtd,tn,k,results.data());
    }

    float ar=0,al=0;
    for(size_t i=0;i<tn;++i){ ar+=results[i].recall; al+=results[i].latency; }
    std::cout<<"mode: "<<ms<<"  num_threads: "<<nt<<"\n";
    std::cout<<"average recall: "<<ar/tn<<"\n";
    std::cout<<"average latency (us): "<<al/tn<<"\n";

    delete[] base; delete[] tq; delete[] tg;
    return 0;
}
