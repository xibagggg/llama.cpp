#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>

// Kernels publish bounded, process-once diagnostic facts. Formatting and the
// user's logging callback run on the host after the graph future completes.
// No allocation, file I/O, or extra grid barrier on the worker. First-writer
// arbitration uses a lock-free CAS; later calls only perform a relaxed load.
namespace ocr_worker_events {

enum class kind : unsigned { pack, full_kv, direct, lazy, x8, group, pv, count };

struct values {
    uint64_t a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;
};

class storage {
    static_assert(std::atomic<unsigned>::is_always_lock_free,
                  "AI-worker diagnostics require lock-free state publication");
    enum : unsigned { empty, writing, ready, consumed };
    struct alignas(128) slot {
        std::atomic<unsigned> state{empty};
        values data{};
    };
    slot slots_[unsigned(kind::count)];

public:
    void publish(kind event, values data = {}) noexcept {
        auto & target = slots_[unsigned(event)];
        // After the first publisher, avoid repeatedly taking exclusive cache
        // line ownership merely to discover that this message was announced.
        if (target.state.load(std::memory_order_relaxed) != empty) return;
        unsigned expected = empty;
        if (!target.state.compare_exchange_strong(expected, writing, std::memory_order_relaxed)) return;
        target.data = data;
        target.state.store(ready, std::memory_order_release);
    }

    template<class Consumer> void drain(Consumer && consume) {
        for (unsigned i = 0; i < unsigned(kind::count); ++i) {
            auto & source = slots_[i];
            unsigned expected = ready;
            if (source.state.compare_exchange_strong(expected, consumed, std::memory_order_acquire)) {
                consume(kind(i), source.data);
            }
        }
    }
};

inline storage & get() {
    static storage instance;
    return instance;
}

inline void publish(kind event, values data = {}) noexcept { get().publish(event, data); }

} // namespace ocr_worker_events
