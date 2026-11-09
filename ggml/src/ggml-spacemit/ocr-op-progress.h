#pragma once
// Optional mmap telemetry, initialized on the host. AI workers only store
// bounded counters: no file I/O, locks, extra barriers or signal handlers.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <fcntl.h>
#include <new>
#include <sys/mman.h>
#include <unistd.h>

namespace ocr_op_progress {
struct alignas(128) lane {
    std::atomic<uint64_t> sequence{0},index{0},phase{0},op{0},tensor{0};
};
struct storage { lane lanes[8]; };
inline storage * get() {
    static storage * p=[]() -> storage * {
        const char * path=std::getenv("GGML_SPACEMIT_OP_PROGRESS_PATH");
        if(!path||!*path)return nullptr;
        int fd=::open(path,O_RDWR|O_CREAT|O_EXCL,0600);
        if(fd<0)return nullptr;
        if(::ftruncate(fd,4096)!=0){::close(fd);return nullptr;}
        void * base=::mmap(nullptr,4096,PROT_READ|PROT_WRITE,MAP_SHARED,fd,0);
        ::close(fd);if(base==MAP_FAILED)return nullptr;
        return new(base) storage;
    }();
    return p;
}
inline void record(storage * p,unsigned worker,uint64_t sequence,unsigned index,
                   uint64_t phase,const ggml_tensor * op) {
    if(!p||worker>=8)return;
    auto & l=p->lanes[worker];
    l.sequence.store(sequence,std::memory_order_relaxed);
    l.index.store(index,std::memory_order_relaxed);
    l.op.store(uint64_t(op->op),std::memory_order_relaxed);
    l.tensor.store(reinterpret_cast<uintptr_t>(op),std::memory_order_relaxed);
    l.phase.store(phase,std::memory_order_release);
}
}
