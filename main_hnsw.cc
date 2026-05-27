#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <cstdlib>
#include "hnsw_wrapper.h"

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
    T* data = new T[n*d]; int sz = sizeof(T);
    for(int i = 0; i < n; ++i) fin.read(((char*)data + i*d*sz), d*sz);
    fin.close();
    std::cerr<<"load "<<data_path<<"\n";
    return data;
}

struct SearchResult { float recall; int64_t latency; };
const unsigned long C = 1000*1000;

void run_serial(const float* base, const float* query, const int* gt,
                size_t bn, size_t vd, size_t gtd, size_t tn, size_t k, SearchResult* res){
    for(size_t i=0;i<tn;++i){
        struct timeval val; gettimeofday(&val,NULL);
        auto r=hnsw_solve(base, query+i*vd, bn, vd, k);
        struct timeval nv; gettimeofday(&nv,NULL);
        int64_t d=(nv.tv_sec*C+nv.tv_usec)-(val.tv_sec*C+val.tv_usec);
        std::set<uint32_t> gt2;
        for(size_t j=0;j<k;++j) gt2.insert((uint32_t)gt[j+i*gtd]);
        size_t acc=0;
        while(r.size()){ if(gt2.find((uint32_t)r.top().second)!=gt2.end())++acc; r.pop(); }
        res[i]={(float)acc/k,d};
    }
}

struct QP{ int tid,nt; const float *base,*query; const int* gt;
    size_t bn,vd,gtd,tn,k; SearchResult* res; };
void* qw(void* arg){
    QP* p=(QP*)arg;
    for(size_t i=p->tid;i<p->tn;i+=p->nt){
        struct timeval val; gettimeofday(&val,NULL);
        auto r=hnsw_solve(p->base,p->query+i*p->vd,p->bn,p->vd,p->k);
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
    pthread_t* th=new pthread_t[nt]; QP* pr=new QP[nt];
    for(int t=0;t<nt;t++){
        pr[t]={t,nt,base,query,gt,bn,vd,gtd,tn,k,res};
        pthread_create(&th[t],nullptr,qw,&pr[t]);
    }
    for(int t=0;t<nt;t++) pthread_join(th[t],nullptr);
    delete[] th; delete[] pr;
}

#ifdef _OPENMP
void run_openmp(const float* base, const float* query, const int* gt,
                size_t bn, size_t vd, size_t gtd, size_t tn, size_t k, SearchResult* res, int nt){
    #pragma omp parallel for num_threads(nt) schedule(static)
    for(size_t i=0;i<tn;++i){
        struct timeval val; gettimeofday(&val,NULL);
        auto r=hnsw_solve(base, query+i*vd, bn, vd, k);
        struct timeval nv; gettimeofday(&nv,NULL);
        int64_t d=(nv.tv_sec*C+nv.tv_usec)-(val.tv_sec*C+val.tv_usec);
        std::set<uint32_t> gt2;
        for(size_t j=0;j<k;++j) gt2.insert((uint32_t)gt[j+i*gtd]);
        size_t acc=0;
        while(r.size()){ if(gt2.find((uint32_t)r.top().second)!=gt2.end())++acc; r.pop(); }
        res[i]={(float)acc/k,d};
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

    build_hnsw(base,bn,vd);

    const char* mode=std::getenv("HNSW_MODE");
    std::string ms=mode?mode:"serial";
    const char* env_nt=std::getenv("NUM_THREADS");
    int nt=env_nt?std::atoi(env_nt):4;

    std::vector<SearchResult> results(tn);

    if(ms=="pthread"){
        std::cerr<<"HNSW Pthread query-level, threads="<<nt<<"\n";
        run_pthread(base,tq,tg,bn,vd,gtd,tn,k,results.data(),nt);
    }
#ifdef _OPENMP
    else if(ms=="openmp"){
        std::cerr<<"HNSW OpenMP query-level, threads="<<nt<<"\n";
        run_openmp(base,tq,tg,bn,vd,gtd,tn,k,results.data(),nt);
    }
#endif
    else {
        std::cerr<<"HNSW serial mode\n";
        run_serial(base,tq,tg,bn,vd,gtd,tn,k,results.data());
    }

    float ar=0,al=0;
    for(size_t i=0;i<tn;++i){ ar+=results[i].recall; al+=results[i].latency; }
    std::cout<<"mode: "<<ms<<"  num_threads: "<<nt<<"\n";
    std::cout<<"average recall: "<<ar/tn<<"\n";
    std::cout<<"average latency (us): "<<al/tn<<"\n";

    delete g_hnsw; delete g_hnsw_space;
    delete[] base; delete[] tq; delete[] tg;
    return 0;
}
