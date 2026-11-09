#include "spacemit-resident-scratch.h"

#include <atomic>
#include <cstdio>
#include <fstream>
#include <limits>
#include <string>
#include <sys/resource.h>
#include <sys/wait.h>

using ggml::spacemit::resident_scratch;
using ggml::spacemit::tcm_lease;

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while (0)

static size_t locked_kib() {
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line))
        if (line.compare(0, 6, "VmLck:") == 0) return std::strtoull(line.c_str() + 6, nullptr, 10);
    return size_t(-1);
}

static bool faults_on_write(unsigned char * address) {
    const pid_t child = fork();
    if (child < 0) return false;
    if (!child) {
        rlimit limit{0, 0};
        setrlimit(RLIMIT_CORE, &limit);
        *static_cast<volatile unsigned char *>(address) = 1;
        _exit(0);
    }
    int status = 0;
    if (waitpid(child, &status, 0) != child) return false;
    return WIFSIGNALED(status) && WTERMSIG(status) == SIGSEGV;
}

int main(int argc, char **) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // Allocation tests run before any runtime threads exist (fork is safe).
    const size_t before = locked_kib(), page = size_t(sysconf(_SC_PAGESIZE));
    resident_scratch scratch;
    scratch.prepare(2, page);
    CHECK(locked_kib() == before + 2 * page / 1024);
    auto a = scratch.lane(0), b = scratch.lane(1);
    CHECK(a && b && a.data != b.data && a.size == page && !scratch.lane(2));
    std::memset(a.data, 0xa5, a.size);
    std::memset(b.data, 0x5a, b.size);
    scratch.prepare(2, page);
    CHECK(scratch.lane(0).data == a.data);
    for (size_t i = 0; i < page; ++i) {
        CHECK(static_cast<unsigned char *>(a.data)[i] == 0xa5);
        CHECK(static_cast<unsigned char *>(b.data)[i] == 0x5a);
    }
    CHECK(faults_on_write(static_cast<unsigned char *>(a.data) - 1));
    CHECK(faults_on_write(static_cast<unsigned char *>(a.data) + page));
    bool overflow = false;
    try { scratch.prepare(1, std::numeric_limits<size_t>::max()); }
    catch (const std::overflow_error &) { overflow = true; }
    CHECK(overflow && scratch.lane(0).data == a.data);
    // Fail after one successful lane lock; the previous allocation survives.
    rlimit saved{};
    CHECK(getrlimit(RLIMIT_MEMLOCK, &saved) == 0);
    rlimit limited = saved;
    limited.rlim_cur = (before * 1024) + 3 * page;
    CHECK(setrlimit(RLIMIT_MEMLOCK, &limited) == 0);
    bool failed = false;
    try { scratch.prepare(3, page); }
    catch (const std::system_error &) { failed = true; }
    CHECK(setrlimit(RLIMIT_MEMLOCK, &saved) == 0);
    CHECK(failed && scratch.lane(0).data == a.data);
    CHECK(locked_kib() == before + 2 * page / 1024);
    scratch.clear();
    CHECK(locked_kib() == before && !scratch.lane(0));
    std::puts("PASS DDR guards, isolation, reuse, overflow, partial-lock rollback, cleanup");
    if (argc == 1) return 0;

    // Real runtime: full-pool reservations retain both pointer and contents
    // across barriers and are reusable by a subsequent graph launch.
    spert::Stream stream{2};
    CHECK(stream.valid());
    const size_t bytes = spert::backend_info().shared_mem_size;
    CHECK(bytes > 0);
    std::atomic<unsigned> errors{0}, completed{0};
    for (unsigned repeat = 0; repeat < 2; ++repeat) {
        auto future = stream.launch(spert::Grid{2}, [&](spert::Context * ctx) {
            tcm_lease lease(*ctx, bytes, true);
            const auto view = ctx->shared_buffer();
            const auto pattern = static_cast<unsigned char>(ctx->program_id() + 17);
            std::memset(view.data, pattern, bytes);
            for (unsigned round = 0; round < 3; ++round) {
                if (ctx->sync() != spert::Status::Ok) ++errors;
                const auto current = ctx->shared_buffer();
                if (current.data != view.data || current.size != view.size) ++errors;
                for (size_t i = 0; i < bytes; i += 128)
                    if (static_cast<unsigned char *>(view.data)[i] != pattern) ++errors;
            }
            ++completed;
        });
        CHECK(future.valid() && future.sync(3000) == spert::Status::Ok);
    }
    CHECK(errors == 0 && completed == 4);
    std::puts("PASS TCM full-layout reservation, barrier contents, release and relaunch");
}
