#pragma once
// Optional mmap telemetry, initialized and locked on the host. AI workers only
// store bounded counters: no file I/O, locks, extra barriers or signal handlers.
#include <atomic>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <sys/mman.h>
#include <unistd.h>

namespace ocr_op_progress {
struct alignas(128) lane {
    std::atomic<uint64_t> sequence{0},index{0},phase{0},op{0},tensor{0};
    std::atomic<uint64_t> version{0}, subphase{0};
};
struct storage { lane lanes[8]; };
static_assert(std::atomic<uint64_t>::is_always_lock_free, "worker telemetry must be lock free");
static_assert(sizeof(storage) <= 4096, "telemetry must fit in one mapped page");

inline const char * path() { return std::getenv("GGML_SPACEMIT_OP_PROGRESS_PATH"); }

class mapping {
  public:
    mapping() {
        const char * filename = path();
        if (!filename || !*filename) return;
        const int fd = ::open(filename, O_RDWR | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
        if (fd < 0) throw std::system_error(errno, std::generic_category(), "progress open");
        if (::ftruncate(fd, 4096) != 0) {
            const int error = errno;
            ::close(fd);
            throw std::system_error(error, std::generic_category(), "progress ftruncate");
        }
        void * base = ::mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
        const int error = errno;
        ::close(fd);
        if (base == MAP_FAILED) throw std::system_error(error, std::generic_category(), "progress mmap");
        std::memset(base, 0, 4096);
        if (::mlock(base, 4096) != 0) {
            const int lock_error = errno;
            ::munmap(base, 4096);
            throw std::system_error(lock_error, std::generic_category(), "progress mlock");
        }
        value = new(base) storage;
    }
    ~mapping() { if (value) ::munmap(value, 4096); }
    mapping(const mapping &) = delete;
    mapping & operator=(const mapping &) = delete;
    storage * value = nullptr;
};

inline storage * get() {
    static mapping owner;
    return owner.value;
}

inline uint64_t next_sequence() {
    static std::atomic<uint64_t> next{0};
    return next.fetch_add(1, std::memory_order_relaxed) + 1;
}

// Persist the graph-to-index mapping before dispatch; workers never format it.
template<class Nodes>
inline void catalog(storage * p, uint64_t sequence, uint64_t uid, const Nodes & nodes) {
    if (!p) return;
    const std::string filename = std::string(path()) + ".catalog";
    FILE * file = std::fopen(filename.c_str(), "ab");
    if (!file) throw std::system_error(errno, std::generic_category(), "progress catalog open");
    bool good = true;
    size_t index = 0;
    for (const auto & node : nodes) {
        const auto * tensor = node.node;
        good &= std::fprintf(file, "%llu\t%llu\t%zu\t%d\t%d\t%s\t%lld,%lld,%lld,%lld\t%s\n",
            (unsigned long long) sequence, (unsigned long long) uid, index++,
            int(node.opcode), int(tensor->op), ggml_type_name(tensor->type),
            (long long) tensor->ne[0], (long long) tensor->ne[1],
            (long long) tensor->ne[2], (long long) tensor->ne[3], tensor->name) >= 0;
    }
    good &= std::fflush(file) == 0;
    good &= ::fsync(::fileno(file)) == 0;
    good &= std::fclose(file) == 0;
    if (!good) throw std::runtime_error("progress catalog could not be persisted");
}
inline void record(storage * p,unsigned worker,uint64_t sequence,unsigned index,
                   uint64_t phase,const ggml_tensor * op) {
    if(!p||worker>=8)return;
    auto & l=p->lanes[worker];
    l.version.fetch_add(1,std::memory_order_acq_rel);
    l.subphase.store(0,std::memory_order_relaxed);
    l.sequence.store(sequence,std::memory_order_relaxed);
    l.index.store(index,std::memory_order_relaxed);
    l.op.store(op?uint64_t(op->op):0,std::memory_order_relaxed);
    l.tensor.store(reinterpret_cast<uintptr_t>(op),std::memory_order_relaxed);
    l.phase.store(phase,std::memory_order_release);
    l.version.fetch_add(1,std::memory_order_release);
}
}
