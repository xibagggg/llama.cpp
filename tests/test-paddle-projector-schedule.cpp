#include "../ggml/src/ggml-spacemit/paddle-f16-kblock-plan.h"
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void require(bool condition,const char *message) {
    if(!condition){std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1);}
}

int main() {
    using namespace paddle_f16_ime2;
    int capacity_cases=0,schedules=0;
    for(int requested:{0,1,1024,1440})
    for(std::size_t available:std::vector<std::size_t>{0,344319,344320,344321,370944,381183,381184,381185,393216}) {
        const auto plan=make_kblock_plan(requested,available);
        const int expected=requested==1440&&available>=381184?1440:
                           (requested==1024||requested==1440)&&available>=344320?1024:0;
        require(plan.block_k==expected,"capacity/fallback selection");
        require(plan.panels==(expected==1440?3:expected==1024?4:0),"panel count");
        if(expected)require(kblock_tcm_bytes(plan.block_k,plan.panels)<=available,"TCM overflow");
        ++capacity_cases;
    }
    for(int k:{4304,4608})for(int n:{1024,1152,4608})
    for(int block:{1024,1440})for(int workers:{1,2,3,4,5,6,7,12})
    for(int m:{1,7,8,9,17,33,212,216,375,848,1121}) {
        const auto plan=make_kblock_plan(block,393216);
        const int groups=(m+7)/8,ntiles=(n+31)/32;
        std::vector<int> owners(ntiles,0);
        for(int worker=0;worker<workers;++worker)
        for(int first=worker;first<ntiles;first+=plan.panels*workers) {
            const int active=std::min(plan.panels,1+(ntiles-1-first)/workers);
            for(int p=0;p<active;++p)++owners[first+p*workers];
            int expected_k8=0;
            for(int k0=0;k0<k;k0+=plan.block_k) {
                const int kb=std::min(plan.block_k,k-k0);
                require(kb>0&&kb%8==0,"invalid final K chunk");
                require(kblock_tcm_bytes(kb,active)<=393216,"active chunk TCM overflow");
                for(int k8=k0;k8<k0+kb;k8+=8) {
                    require(k8==expected_k8,"changed K8 reduction order");expected_k8+=8;
                }
                for(int p=0;p<active;++p)for(int o=0;o<4;++o) {
                    const auto weight_end=std::size_t((first+p*workers)*32+o*8)*k+
                                          std::size_t(k0)*8+std::size_t(8)*kb;
                    require(weight_end<=std::size_t((n+31)/32)*32*k,"W source overflow");
                    const auto staged_end=std::size_t(p*32+o*8)*kb+std::size_t(8)*kb;
                    require(staged_end<=std::size_t(plan.panels)*32*plan.block_k,"W TCM overflow");
                }
                for(int g=0;g<groups;g+=2) {
                    const int count=std::min(2,groups-g);
                    for(int j=0;j<count;++j) {
                        const auto input_end=std::size_t(g+j)*8*k+std::size_t(k0)*8+std::size_t(8)*kb;
                        require(input_end<=std::size_t(groups)*8*k,"A source overflow");
                        require(std::size_t(j+1)*8*kb<=std::size_t(16)*plan.block_k,"A staging overflow");
                    }
                    for(int p=0;p<active;++p) {
                        const std::size_t end=(std::size_t(p)*groups+g+count)*256;
                        require(end<=std::size_t(4)*groups*256,"partial workspace high/low overlap");
                    }
                    // Even for an odd A8 tail, high and low staging banks
                    // remain disjoint in the ordinary K-block scheduler.
                    if(count==1) {
                        const std::size_t high_end=std::size_t(8)*kb;
                        const std::size_t low_begin=std::size_t(16)*plan.block_k;
                        require(high_end<=low_begin,"high/low staging overlaps");
                    }
                }
            }
            require(expected_k8==k,"missing K terms");
        }
        for(int count:owners)require(count==1,"missing/duplicate output ownership");
        ++schedules;
    }
    std::printf("PASS: %d capacity choices, %d projector/vision ownership/K8/workspace schedules\n",capacity_cases,schedules);
}
