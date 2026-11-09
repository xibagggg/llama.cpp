#include "ggml.h"
#include "ocr-op-progress.h"

#include <fstream>
#include <sys/resource.h>

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); return 1; } } while (0)

static size_t locked_kib() {
    std::ifstream file("/proc/self/status");
    std::string line;
    while (std::getline(file, line))
        if (line.compare(0, 6, "VmLck:") == 0) return std::strtoull(line.c_str() + 6, nullptr, 10);
    return size_t(-1);
}

int main() {
    const size_t before = locked_kib();
    char directory[] = "/tmp/ocr-progress-test-XXXXXX";
    CHECK(mkdtemp(directory));
    const std::string filename = std::string(directory) + "/lanes";
    CHECK(setenv("GGML_SPACEMIT_OP_PROGRESS_PATH", filename.c_str(), 1) == 0);
    {
        ocr_op_progress::mapping owner;
        CHECK(owner.value && locked_kib() == before + 4);
        ggml_tensor tensor{};
        tensor.op = GGML_OP_MUL_MAT;
        ocr_op_progress::record(owner.value, 2, 19, 23, 4, &tensor);
        CHECK(owner.value->lanes[2].phase.load(std::memory_order_acquire) == 4);
        CHECK(owner.value->lanes[2].sequence == 19 && owner.value->lanes[2].index == 23);
        for (unsigned lane = 0; lane < 8; ++lane)
            if (lane != 2) CHECK(owner.value->lanes[lane].sequence == 0);
        ocr_op_progress::record(owner.value, 8, 99, 99, 99, &tensor);
        bool exclusive = false;
        try { ocr_op_progress::mapping second; }
        catch (const std::system_error &) { exclusive = true; }
        CHECK(exclusive);
    }
    CHECK(locked_kib() == before);
    CHECK(unlink(filename.c_str()) == 0);
    rlimit saved{};
    CHECK(getrlimit(RLIMIT_MEMLOCK, &saved) == 0);
    rlimit limited = saved;
    limited.rlim_cur = 0;
    CHECK(setrlimit(RLIMIT_MEMLOCK, &limited) == 0);
    bool rejected = false;
    try { ocr_op_progress::mapping failed; }
    catch (const std::system_error &) { rejected = true; }
    CHECK(setrlimit(RLIMIT_MEMLOCK, &saved) == 0);
    CHECK(rejected && locked_kib() == before);
    CHECK(unlink(filename.c_str()) == 0 && rmdir(directory) == 0);
    CHECK(ocr_op_progress::next_sequence() == 1 && ocr_op_progress::next_sequence() == 2);
    std::puts("PASS progress lock, lane isolation, exclusive create, failure rejection, sequence, cleanup");
}
