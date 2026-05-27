#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <cstdlib>
#include "ivf_simd.h"

#ifdef _OPENMP
#include <omp.h>
#endif
#include <pthread.h>

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();
    std::cerr<<"load "<<data_path<<"\n";
    return data;
}

struct SearchResult { float recall; int64_t latency; };

void run_serial(const float* base, const float* test_query, const int* test_gt,
                size_t base_number, size_t vecdim, size_t test_gt_d,
                size_t test_number, size_t k, SearchResult* results)
{
    const unsigned long C = 1000 * 1000;
    for(size_t i = 0; i < test_number; ++i){
        struct timeval val; gettimeofday(&val, NULL);
        auto res = ivf_solve(base, test_query + i * vecdim, base_number, vecdim, k);
        struct timeval nv; gettimeofday(&nv, NULL);
        int64_t diff = (nv.tv_sec*C + nv.tv_usec) - (val.tv_sec*C + val.tv_usec);
        std::set<uint32_t> gt;
        for(size_t j = 0; j < k; ++j) gt.insert((uint32_t)test_gt[j + i * test_gt_d]);
        size_t acc = 0;
        while(res.size()){ if(gt.find((uint32_t)res.top().second)!=gt.end()) ++acc; res.pop(); }
        results[i] = {(float)acc/k, diff};
    }
}

struct QThreadParam {
    int tid, nt; const float *base, *query; const int* gt;
    size_t bn, vd, gtd, tn, k; SearchResult* res;
};
void* qworker(void* arg){
    QThreadParam* p=(QThreadParam*)arg; const unsigned long C=1000*1000;
    for(size_t i=p->tid; i<p->tn; i+=p->nt){
        struct timeval val; gettimeofday(&val, NULL);
        auto res=ivf_solve(p->base, p->query+i*p->vd, p->bn, p->vd, p->k);
        struct timeval nv; gettimeofday(&nv, NULL);
        int64_t d=(nv.tv_sec*C+nv.tv_usec)-(val.tv_sec*C+val.tv_usec);
        std::set<uint32_t> gt;
        for(size_t j=0;j<p->k;++j) gt.insert((uint32_t)p->gt[j+i*p->gtd]);
        size_t acc=0;
        while(res.size()){ if(gt.find((uint32_t)res.top().second)!=gt.end())++acc; res.pop(); }
        p->res[i]={(float)acc/p->k, d};
    }
    return nullptr;
}
void run_pthread(const float* base, const float* query, const int* gt,
                 size_t bn, size_t vd, size_t gtd, size_t tn, size_t k,
                 SearchResult* res, int nt){
    pthread_t* th=new pthread_t[nt]; QThreadParam* pr=new QThreadParam[nt];
    for(int t=0;t<nt;t++){
        pr[t]={t,nt,base,query,gt,bn,vd,gtd,tn,k,res};
        pthread_create(&th[t],nullptr,qworker,&pr[t]);
    }
    for(int t=0;t<nt;t++) pthread_join(th[t],nullptr);
    delete[] th; delete[] pr;
}

#ifdef _OPENMP
void run_omp_query(const float* base, const float* query, const int* gt,
                    size_t bn, size_t vd, size_t gtd, size_t tn, size_t k,
                    SearchResult* res, int nt){
    const unsigned long C=1000*1000;
    const int* gtp = gt; size_t kg = k; size_t gtd2 = gtd;
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(size_t i=0;i<tn;++i){
        struct timeval val; gettimeofday(&val, NULL);
        auto r=ivf_solve(base, query+i*vd, bn, vd, kg);
        struct timeval nv; gettimeofday(&nv, NULL);
        int64_t d=(nv.tv_sec*C+nv.tv_usec)-(val.tv_sec*C+val.tv_usec);
        std::set<uint32_t> gtset;
        for(size_t j=0;j<kg;++j) gtset.insert((uint32_t)gtp[j+i*gtd2]);
        size_t acc=0;
        while(r.size()){ if(gtset.find((uint32_t)r.top().second)!=gtset.end())++acc; r.pop(); }
        res[i]={(float)acc/kg, d};
    }
}

void run_omp_cluster(const float* base, const float* query, const int* gt,
                     size_t bn, size_t vd, size_t gtd, size_t tn, size_t k,
                     SearchResult* res, int nt){
    const unsigned long C=1000*1000;
    size_t nprobe = 8;
    const char* env_np = std::getenv("IVF_NPROBE");
    if(env_np) nprobe = (size_t)std::atoi(env_np);

    for(size_t qi=0; qi<tn; ++qi){
        struct timeval val; gettimeofday(&val, NULL);
        const float* q = query + qi*vd;

        std::priority_queue<std::pair<float,int>> ch;
        for(size_t c=0; c<g_ivf_nlist; c++){
            float d=InnerProductSIMD(q, g_ivf_centroids.data()+c*vd, vd);
            ch.push(std::make_pair(-d,(int)c));
        }
        std::vector<int> probes; probes.reserve(nprobe);
        for(size_t p=0; p<nprobe && !ch.empty(); p++){
            probes.push_back(ch.top().second); ch.pop();
        }

        std::vector<std::pair<float,int>> all_cand;
        #pragma omp parallel num_threads(nt)
        {
            int tid=omp_get_thread_num();
            int nts=omp_get_num_threads();
            size_t chunk=(probes.size()+nts-1)/nts;
            size_t cstart=tid*chunk;
            size_t cend=std::min(cstart+chunk, probes.size());

            std::priority_queue<std::pair<float,int>> local_pq;
            for(size_t pi=cstart; pi<cend; pi++){
                int c=probes[pi];
                for(int idx : g_ivf_lists[c]){
                    float dist=InnerProductSIMD(base+(size_t)idx*vd, q, vd);
                    if(local_pq.size()<k) local_pq.push(std::make_pair(dist,idx));
                    else if(dist<local_pq.top().first){
                        local_pq.pop(); local_pq.push(std::make_pair(dist,idx));
                    }
                }
            }
            #pragma omp critical
            { while(!local_pq.empty()){ all_cand.push_back(local_pq.top()); local_pq.pop(); } }
        }

        std::priority_queue<std::pair<float,int>> merged;
        for(auto& cand : all_cand){
            if(merged.size()<k) merged.push(cand);
            else if(cand.first<merged.top().first){ merged.pop(); merged.push(cand); }
        }

        struct timeval nv; gettimeofday(&nv, NULL);
        int64_t d=(nv.tv_sec*C+nv.tv_usec)-(val.tv_sec*C+val.tv_usec);
        std::set<uint32_t> gtset;
        for(size_t j=0;j<k;++j) gtset.insert((uint32_t)gt[j+qi*gtd]);
        size_t acc=0;
        while(merged.size()){ if(gtset.find((uint32_t)merged.top().second)!=gtset.end())++acc; merged.pop(); }
        res[qi]={(float)acc/k, d};
    }
}
#endif

int main()
{
    size_t test_number=0, base_number=0, test_gt_d=0, vecdim=0;
    std::string dp="data/";
    auto test_query=LoadData<float>(dp+"DEEP100K.query.fbin",test_number,vecdim);
    auto test_gt=LoadData<int>(dp+"DEEP100K.gt.query.100k.top100.bin",test_number,test_gt_d);
    auto base=LoadData<float>(dp+"DEEP100K.base.100k.fbin",base_number,vecdim);
    test_number=2000; const size_t k=10;

    build_ivf(base, base_number, vecdim);

    const char* mode=std::getenv("IVF_MODE");
    std::string ms=mode?mode:"serial";
    const char* nt=std::getenv("NUM_THREADS");
    int nthreads=nt?std::atoi(nt):4;

    std::vector<SearchResult> results(test_number);

    if(ms=="pthread"){
        std::cerr<<"IVF Pthread query-level, threads="<<nthreads<<"\n";
        run_pthread(base,test_query,test_gt,base_number,vecdim,test_gt_d,test_number,k,results.data(),nthreads);
    }
#ifdef _OPENMP
    else if(ms=="openmp"){
        std::cerr<<"IVF OpenMP query-level, threads="<<nthreads<<"\n";
        run_omp_query(base,test_query,test_gt,base_number,vecdim,test_gt_d,test_number,k,results.data(),nthreads);
    }
    else if(ms=="openmp_cluster"){
        std::cerr<<"IVF OpenMP cluster-level, threads="<<nthreads<<"\n";
        run_omp_cluster(base,test_query,test_gt,base_number,vecdim,test_gt_d,test_number,k,results.data(),nthreads);
    }
#endif
    else {
        std::cerr<<"IVF serial mode\n";
        run_serial(base,test_query,test_gt,base_number,vecdim,test_gt_d,test_number,k,results.data());
    }

    float ar=0, al=0;
    for(size_t i=0;i<test_number;++i){ ar+=results[i].recall; al+=results[i].latency; }
    std::cout<<"mode: "<<ms<<"  num_threads: "<<nthreads<<"\n";
    std::cout<<"average recall: "<<ar/test_number<<"\n";
    std::cout<<"average latency (us): "<<al/test_number<<"\n";

    delete[] base; delete[] test_query; delete[] test_gt;
    return 0;
}
