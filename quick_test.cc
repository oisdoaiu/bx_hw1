#include "pq_simd.h"
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d){
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4); fin.read((char*)&d,4);
    T* data = new T[n*d]; int sz = sizeof(T);
    for(int i = 0; i < n; ++i) fin.read(((char*)data + i*d*sz), d*sz);
    fin.close(); return data;
}

int main(){
    size_t tn=0, bn=0, gtd=0, vd=0;
    auto tq = LoadData<float>("data/DEEP100K.query.fbin", tn, vd);
    auto gt = LoadData<int>("data/DEEP100K.gt.query.100k.top100.bin", tn, gtd);
    auto ba = LoadData<float>("data/DEEP100K.base.100k.fbin", bn, vd);
    tn=2000; const size_t k=10, M=24, Ks=256, p=50;
    size_t dsub = vd/M;
    std::vector<float> cents(M*Ks*dsub);
    std::vector<uint8_t> codes(bn*M);
    std::cerr << "training...\n";
    train_pq_codebook(ba, bn, vd, cents.data(), M, Ks, 15);
    encode_pq(ba, codes.data(), cents.data(), bn, vd, M, Ks);
    std::cerr << "done\n";
    float sr=0, sl=0;
    for(int qi=0; qi<tn; qi++){
        struct timeval val; gettimeofday(&val, NULL);
        auto res = pq_search(ba, codes.data(), cents.data(), tq+qi*vd, bn, vd, M, Ks, k, p);
        struct timeval nv; gettimeofday(&nv, NULL);
        sl += (nv.tv_sec*1000000+nv.tv_usec)-(val.tv_sec*1000000+val.tv_usec);
        std::set<uint32_t> gs;
        for(int j=0; j<k; j++) gs.insert((uint32_t)gt[j+qi*gtd]);
        size_t ac=0; while(res.size()){ if(gs.find((uint32_t)res.top().second)!=gs.end()) ac++; res.pop(); }
        sr += (float)ac/k;
    }
    std::cout << "recall="<<sr/tn<<" latency="<<sl/tn<<"us\n";
    return 0;
}
