#ifndef ALGO_IVF_PQ_H
#define ALGO_IVF_PQ_H

#define PQ_M_VAL 8
#define IVF_NLIST_VAL 256
#define IVF_NPROBE_VAL 32
#define IVF_RERANK_VAL 500

#include "simd_wrapper.h"
#include "flat_simd.h"
#include <queue>
#include <utility>
#include <vector>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <algorithm>
#include <random>
#include <cmath>
#include <cfloat>
#include <iostream>

static std::vector<uint8_t> g_cd;
static std::vector<float> g_ct, g_ic;
static size_t g_M,g_Ks,g_nl;
static std::vector<std::vector<int>> g_il;
static const float* g_bb;

inline void trpq(const float* b, size_t n, size_t d, float* ct, size_t M, size_t Ks, int it)
{
    size_t ds=d/M; std::mt19937 rng(42); std::uniform_real_distribution<float> rd(0,1);
    for(size_t m=0;m<M;m++){
        float* cm=ct+m*Ks*ds; std::uniform_int_distribution<size_t> id(0,n-1);
        size_t f=id(rng); std::memcpy(cm,b+f*d+m*ds,ds*sizeof(float));
        std::vector<float> md(n,1e30f);
        for(size_t k=1;k<Ks;k++){
            float sd=0;
            for(size_t i=0;i<n;i++){
                const float* sv=b+i*d+m*ds;
                float d2=InnerProductSIMD(sv,cm+(k-1)*ds,ds);
                if(d2<md[i]) md[i]=d2; sd+=md[i];
            }
            float r=rd(rng)*sd; float cum=0; size_t ch=0;
            for(size_t i=0;i<n;i++){cum+=md[i];if(cum>=r){ch=i;break;}}
            std::memcpy(cm+k*ds,b+ch*d+m*ds,ds*sizeof(float));
        }
    }
    std::vector<int> as(n);
    for(int iter=0;iter<it;iter++){
        for(size_t m=0;m<M;m++){
            float* cm=ct+m*Ks*ds;
            for(size_t i=0;i<n;i++){
                const float* sv=b+i*d+m*ds;
                float best=-FLT_MAX; int bk=0;
                for(size_t k=0;k<Ks;k++){
                    float ip=1.0f-InnerProductSIMD(sv,cm+k*ds,ds);
                    if(ip>best){best=ip;bk=(int)k;}
                }
                as[i]=bk;
            }
            std::vector<float> sc(Ks*ds,0); std::vector<int> cnt(Ks,0);
            for(size_t i=0;i<n;i++){
                int k=as[i]; const float* sv=b+i*d+m*ds;
                float* dst=sc.data()+k*ds;
                for(size_t dd=0;dd<ds;dd++) dst[dd]+=sv[dd];
                cnt[k]++;
            }
            for(size_t k=0;k<Ks;k++)
                if(cnt[k]>0){float inv=1.0f/(float)cnt[k];
                    for(size_t dd=0;dd<ds;dd++) cm[k*ds+dd]=sc[k*ds+dd]*inv;}
        }
    }
}

inline void enc2(const float* b, uint8_t* cd, const float* ct, size_t n, size_t d, size_t M, size_t Ks)
{
    size_t ds=d/M;
    for(size_t i=0;i<n;i++) for(size_t m=0;m<M;m++){
        const float* sv=b+i*d+m*ds; const float* cm=ct+m*Ks*ds;
        float best=-FLT_MAX; uint8_t bk=0;
        for(size_t k=0;k<Ks;k++){
            float ip=1.0f-InnerProductSIMD(sv,cm+k*ds,ds);
            if(ip>best){best=ip;bk=(uint8_t)k;}
        }
        cd[i*M+m]=bk;
    }
}

inline void soa(const uint8_t* ao, uint8_t* so, size_t n, size_t M){
    for(size_t m=0;m<M;m++) for(size_t i=0;i<n;i++) so[m*n+i]=ao[i*M+m];
}

inline void blut(const float* q, float* lut, const float* ct, size_t d, size_t M, size_t Ks)
{
    size_t ds=d/M;
    for(size_t m=0;m<M;m++){
        const float* qs=q+m*ds; const float* cm=ct+m*Ks*ds; float* lm=lut+m*Ks; size_t k=0;
#if defined(__AVX2__)
        for(;k+8<=Ks;k+=8){
            __m256 acc=_mm256_setzero_ps();
            for(size_t dd=0;dd<ds;dd++){
                __m256 qb=_mm256_set1_ps(qs[dd]);
                __m256 cd2=_mm256_setr_ps(
                    cm[(k+0)*ds+dd],cm[(k+1)*ds+dd],cm[(k+2)*ds+dd],cm[(k+3)*ds+dd],
                    cm[(k+4)*ds+dd],cm[(k+5)*ds+dd],cm[(k+6)*ds+dd],cm[(k+7)*ds+dd]);
                acc=_mm256_fmadd_ps(qb,cd2,acc);
            }
            _mm256_storeu_ps(lm+k,acc);
        }
#elif defined(__ARM_NEON)
        for(;k+4<=Ks;k+=4){
            float32x4_t acc=vdupq_n_f32(0);
            for(size_t dd=0;dd<ds;dd++){
                float32x4_t qb=vdupq_n_f32(qs[dd]); float32x4_t cd2=vdupq_n_f32(0);
                cd2=vld1q_lane_f32(cm+(k+0)*ds+dd,cd2,0); cd2=vld1q_lane_f32(cm+(k+1)*ds+dd,cd2,1);
                cd2=vld1q_lane_f32(cm+(k+2)*ds+dd,cd2,2); cd2=vld1q_lane_f32(cm+(k+3)*ds+dd,cd2,3);
                acc=vmlaq_f32(acc,qb,cd2);
            }
            vst1q_f32(lm+k,acc);
        }
#endif
        for(;k<Ks;k++) lm[k]=1.0f-InnerProductSIMD(qs,cm+k*ds,ds);
    }
}

inline void kmeans(const float* b, size_t n, size_t d, float* ct, size_t nl, int it)
{
    for(size_t c=0;c<nl;c++){size_t idx=(c*(n/nl))%n; std::memcpy(ct+c*d,b+idx*d,d*sizeof(float));}
    std::vector<int> as(n);
    for(int iter=0;iter<it;iter++){
        for(size_t i=0;i<n;i++){
            const float* v=b+i*d; float bd=FLT_MAX; int bc=0;
            for(size_t c=0;c<nl;c++){
                float dd=InnerProductSIMD(v,ct+c*d,d);
                if(dd<bd){bd=dd;bc=(int)c;}
            }
            as[i]=bc;
        }
        std::vector<float> sum(nl*d,0); std::vector<int> cnt(nl,0);
        for(size_t i=0;i<n;i++){
            int c=as[i]; const float* v=b+i*d;
            float* dst=sum.data()+c*d;
            for(size_t dd=0;dd<d;dd++) dst[dd]+=v[dd];
            cnt[c]++;
        }
        for(size_t c=0;c<nl;c++)
            if(cnt[c]>0){float inv=1.0f/(float)cnt[c];
                for(size_t dd=0;dd<d;dd++) ct[c*d+dd]=sum[c*d+dd]*inv;}
    }
}

inline void build_ivf_pq_algo(const float* b, size_t n, size_t d)
{
    g_M=PQ_M_VAL;
    g_Ks=256; size_t ds=d/g_M;
    g_ct.resize(g_M*g_Ks*ds); trpq(b,n,d,g_ct.data(),g_M,g_Ks,15);
    std::vector<uint8_t> ao(n*g_M); enc2(b,ao.data(),g_ct.data(),n,d,g_M,g_Ks);
    g_cd.resize(n*g_M); soa(ao.data(),g_cd.data(),n,g_M);
    g_nl=IVF_NLIST_VAL;
    g_bb=b; g_ic.resize(g_nl*d); kmeans(b,n,d,g_ic.data(),g_nl,15);
    g_il.resize(g_nl); for(size_t c=0;c<g_nl;c++) g_il[c].clear();
    for(size_t i=0;i<n;i++){
        const float* v=b+i*d; float bd=FLT_MAX; int bc=0;
        for(size_t c=0;c<g_nl;c++){
            float dd=InnerProductSIMD(v,g_ic.data()+c*d,d);
            if(dd<bd){bd=dd;bc=(int)c;}
        }
        g_il[bc].push_back((int)i);
    }
}

inline std::priority_queue<std::pair<float,int>> ivf_pq_algo_solve(
    const float* b, const float* q, size_t n, size_t d, size_t k)
{
    size_t np=IVF_NPROBE_VAL;
    size_t rp=IVF_RERANK_VAL;
    std::vector<float> lut(g_M*g_Ks); blut(q,lut.data(),g_ct.data(),d,g_M,g_Ks);

    std::priority_queue<std::pair<float,int>> ch;
    for(size_t c=0;c<g_nl;c++){
        float dd=InnerProductSIMD(q,g_ic.data()+c*d,d);
        ch.push(std::make_pair(-dd,(int)c));
    }
    std::vector<int> pr; pr.reserve(np);
    for(size_t p=0;p<np&&!ch.empty();p++){pr.push_back(ch.top().second);ch.pop();}

    std::priority_queue<std::pair<float,int>> pqh;
    for(int c:pr) for(int idx:g_il[c]){
        float is=0;
        for(size_t m=0;m<g_M;m++) is+=lut[m*g_Ks+g_cd[m*n+(size_t)idx]];
        float dist=1.0f-is;
        if(pqh.size()<rp) pqh.push(std::make_pair(dist,idx));
        else if(dist<pqh.top().first){pqh.pop();pqh.push(std::make_pair(dist,idx));}
    }

    std::vector<std::pair<float,int>> pc;
    while(!pqh.empty()){pc.push_back(pqh.top());pqh.pop();}
    std::sort(pc.begin(),pc.end()); if(pc.size()>rp) pc.resize(rp);

    std::priority_queue<std::pair<float,int>> result;
    for(auto& cc:pc){
        float dist=InnerProductSIMD(b+(size_t)cc.second*d,q,d);
        if(result.size()<k) result.push(std::make_pair(dist,cc.second));
        else if(dist<result.top().first){result.pop();result.push(std::make_pair(dist,cc.second));}
    }
    return result;
}

#endif
