#include <atomic>
#include <cstdlib>
#include <new>

static std::atomic<size_t> allocations{0};
void * operator new(std::size_t size) {
    allocations.fetch_add(1, std::memory_order_relaxed);
    if (void * p = std::malloc(size ? size : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }

#include "ocr-float-graph.h"
#include <limits>

int main() {
    // The full mask output has this many elements. Measure the actual common
    // validation helper, with values read from memory and no timed inference.
    std::vector<float> values(12000000, 0.25f);
    const auto before = allocations.load();
    const auto started = std::chrono::steady_clock::now();
    for (float x : values) require(std::isfinite(x), "nonfinite layout output");
    const auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
    const auto count = allocations.load()-before;
    int rejected = 0;
    for (float x : {std::numeric_limits<float>::infinity(),
                    -std::numeric_limits<float>::infinity(),
                    std::numeric_limits<float>::quiet_NaN()}) {
        try { require(std::isfinite(x), "nonfinite layout output"); }
        catch (const std::runtime_error & e) {
            if (std::strcmp(e.what(), "nonfinite layout output")) return 2;
            ++rejected;
        }
    }
    const std::string dynamic_message = std::string("invalid weight: ")+"test";
    try { require(false, dynamic_message); }
    catch (const std::runtime_error & e) {
        if (e.what() != dynamic_message) return 3;
        ++rejected;
    }
    std::printf("%s\n", json{{"passed",rejected==4},{"elements",values.size()},
                {"success_heap_allocations",count},{"validation_s",elapsed},
                {"rejected_with_original_message",rejected}}.dump().c_str());
    return rejected==4 ? 0 : 4;
}
