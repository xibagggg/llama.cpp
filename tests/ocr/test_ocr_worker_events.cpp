#include "ocr-worker-events.h"

#include <array>
#include <cstdio>
#include <cstdlib>
#include <new>
#include <thread>
#include <vector>

namespace events = ocr_worker_events;

static thread_local bool in_worker = false;
static std::atomic<unsigned> worker_allocations{0};

void * operator new(std::size_t n) {
    if (in_worker) worker_allocations.fetch_add(1, std::memory_order_relaxed);
    if (void * p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete(void * p) noexcept { std::free(p); }
void operator delete(void * p, std::size_t) noexcept { std::free(p); }

static events::values payload(uint64_t marker) {
    return {marker, marker ^ 0x1beef5a1234ULL, marker*17, ~marker, marker+42, marker*marker};
}

int main() {
    // Prepare process storage on the host, as the real graph dispatcher does.
    auto & store = events::get();
    constexpr unsigned count = unsigned(events::kind::count), threads = 32;
    std::array<unsigned, count> seen{};
    bool valid = true;
    auto consume = [&](events::kind event, const events::values & actual) {
        const unsigned i = unsigned(event);
        const auto expected = payload(actual.a);
        valid = valid && actual.a > 0 && actual.a <= threads &&
            actual.b == expected.b && actual.c == expected.c && actual.d == expected.d &&
            actual.e == expected.e && actual.f == expected.f;
        ++seen[i];
    };
    store.drain(consume);
    std::atomic<unsigned> finished{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> workers;
    workers.reserve(threads);
    for(unsigned i = 0; i < threads; ++i) workers.emplace_back([&, i] {
        while(!start.load(std::memory_order_acquire)) std::this_thread::yield();
        in_worker = true;
        for(unsigned repeat = 0; repeat < 1000; ++repeat) {
            for(unsigned j = 0; j < count; ++j) {
                events::publish(events::kind((j+i)%count), payload(i+1));
            }
        }
        in_worker = false;
        finished.fetch_add(1, std::memory_order_release);
    });
    start.store(true, std::memory_order_release);
    // The consumer deliberately overlaps publication to exercise release /
    // acquire visibility, including seeing a claimed but incomplete slot.
    while(finished.load(std::memory_order_acquire) != threads) store.drain(consume);
    for(auto & worker : workers) worker.join();
    store.drain(consume);
    store.drain(consume);
    for(unsigned n : seen) valid = valid && n == 1;
    if(!valid || worker_allocations.load() != 0) {
        std::fprintf(stderr, "FAIL: publication/once/allocation invariant\n"); return 1;
    }
    std::printf("PASS: %u contending workers; %u publish attempts; %u complete once-only events; 0 worker allocations\n",
        threads, threads*1000*count, count);
}
