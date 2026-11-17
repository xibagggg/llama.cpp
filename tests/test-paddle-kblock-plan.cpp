#include "../ggml/src/ggml-spacemit/paddle-f16-kblock-plan.h"
#include <cstdio>
#include <algorithm>
#include <cstdlib>
#include <vector>

static void require(bool value,const char *message) {
    if(!value){std::fprintf(stderr,"FAIL: %s\n",message);std::exit(1);}
}
int main() {
    using namespace paddle_f16_ime2;
    int choices=0,schedules=0;
    for(int requested:{0,1,1024,1440})
    for(std::size_t available:std::vector<std::size_t>{0,344319,344320,344321,381183,381184,381185,393216}) {
        const auto plan=make_kblock_plan(requested,available);
        const int expected=requested==1440&&available>=381184?1440:
                           (requested==1024||requested==1440)&&available>=344320?1024:0;
        require(plan.block_k==expected,"capacity/fallback selection");
        require(plan.panels==(expected==1440?3:expected==1024?4:0),"panel count");
        if(expected)require(kblock_tcm_bytes(plan.block_k,plan.panels)<=available,"TCM overflow");
        ++choices;
    }
    for(int block:{1024,1440})for(int workers:{1,2,3,4,5,6,7,12})
    for(int m:{1,7,8,9,17,33,216,848,1121}) {
        const auto plan=make_kblock_plan(block,393216);
        const int groups=(m+7)/8;
        std::vector<int> owners(36,0);
        for(int worker=0;worker<workers;++worker)
        for(int first=worker;first<36;first+=plan.panels*workers) {
            const int active=std::min(plan.panels,1+(35-first)/workers);
            for(int p=0;p<active;++p)++owners[first+p*workers];
            int expected_k8=0;
            for(int k0=0;k0<4304;k0+=plan.block_k) {
                const int kb=std::min(plan.block_k,4304-k0);
                require(kb>0&&kb%8==0,"invalid final K chunk");
                for(int k8=k0;k8<k0+kb;k8+=8) {
                    require(k8==expected_k8,"changed K8 reduction order");expected_k8+=8;
                }
                for(int g=0;g<groups;g+=2) {
                    const int count=std::min(2,groups-g);
                    for(int p=0;p<active;++p) {
                        const std::size_t end=(std::size_t(p)*groups+g+count)*256;
                        require(end<=std::size_t(4)*groups*256,"partial workspace high/low overlap");
                    }
                }
            }
            require(expected_k8==4304,"missing K terms");
        }
        for(int count:owners)require(count==1,"missing/duplicate output ownership");
        ++schedules;
    }
    std::printf("PASS: %d capacity choices, %d ownership/K8/workspace schedules\n",choices,schedules);
}
