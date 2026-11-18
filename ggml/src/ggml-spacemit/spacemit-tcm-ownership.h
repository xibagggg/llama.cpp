#pragma once

// No SPERT, Linux or driver dependency: this policy is exercised by host mocks.
#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

namespace ggml::spacemit::tcm_ownership {

inline constexpr size_t max_lanes = 8;

enum class error {
    ok, invalid_api, invalid_lane, unavailable, fake_tcm, invalid_layout, invalid_block,
    duplicate_mapping, unknown_pointer, insufficient_size, cpu_mismatch,
    busy, returned_pointer_mismatch, metadata_failed, owner_mismatch, scratch_unavailable,
    thread_changed, release_failed, release_still_owned, barrier_failed
};

inline const char * name(error value) noexcept {
    switch (value) {
#define TCM_ERROR_NAME(x) case error::x: return #x
        TCM_ERROR_NAME(ok); TCM_ERROR_NAME(invalid_api); TCM_ERROR_NAME(invalid_lane); TCM_ERROR_NAME(unavailable);
        TCM_ERROR_NAME(fake_tcm); TCM_ERROR_NAME(invalid_layout); TCM_ERROR_NAME(invalid_block);
        TCM_ERROR_NAME(duplicate_mapping); TCM_ERROR_NAME(unknown_pointer);
        TCM_ERROR_NAME(insufficient_size); TCM_ERROR_NAME(cpu_mismatch); TCM_ERROR_NAME(busy);
        TCM_ERROR_NAME(returned_pointer_mismatch); TCM_ERROR_NAME(metadata_failed);
        TCM_ERROR_NAME(owner_mismatch); TCM_ERROR_NAME(thread_changed);
        TCM_ERROR_NAME(scratch_unavailable);
        TCM_ERROR_NAME(release_failed); TCM_ERROR_NAME(release_still_owned);
        TCM_ERROR_NAME(barrier_failed);
#undef TCM_ERROR_NAME
    }
    return "unknown";
}

struct block {
    void * data = nullptr;
    size_t size = 0;
    uint64_t cpu_mask = 0;
    uint64_t physical = 0;
    int owner_tid = 0;
    int acquired = 0;
};

struct driver {
    void * opaque = nullptr;
    int (*available)(void *) = nullptr;
    int (*layout)(void *, size_t *, size_t *, int *) = nullptr;
    int (*info)(void *, int, block *) = nullptr;
    void * (*try_wait)(void *, int, size_t) = nullptr;
    int (*release)(void *, int) = nullptr;
    int (*tid)(void *) = nullptr;
    int (*cpu)(void *) = nullptr;
};

// Prepared on the host after SPERT initializes its backing buffers. In v2,
// info() exposes each existing mapping without mem_get()/refcount ownership.
struct catalog {
    driver api{};
    std::array<block, max_lanes> blocks{};
    size_t count = 0;
    size_t bytes = 0;

    error prepare(driver input, size_t expected_bytes) noexcept {
        api = input;
        count = bytes = 0;
        if (!api.available || !api.layout || !api.info || !api.try_wait ||
            !api.release || !api.tid || !api.cpu) return error::invalid_api;
        if (api.available(api.opaque) != 1) return error::unavailable;
        size_t block_bytes = 0, block_count = 0;
        int fake = -1;
        if (api.layout(api.opaque, &block_bytes, &block_count, &fake) != 0)
            return error::invalid_layout;
        if (fake != 0) return error::fake_tcm;
        if (!expected_bytes || block_bytes < expected_bytes || !block_count || block_count > max_lanes)
            return error::invalid_layout;
        for (size_t i = 0; i < block_count; ++i) {
            block value{};
            if (api.info(api.opaque, int(i), &value) != 0 || !value.data ||
                value.size < expected_bytes || !value.cpu_mask) return error::invalid_block;
            for (size_t j = 0; j < i; ++j)
                if (blocks[j].data == value.data) return error::duplicate_mapping;
            blocks[i] = value;
        }
        count = block_count;
        bytes = expected_bytes;
        return error::ok;
    }
};

struct record {
    void * data = nullptr;
    size_t size = 0;
    int id = -1, cpu = -1, tid = -1;
    bool claimed = false, released = false;
    error acquire_error = error::ok;
    error release_error = error::ok;
};

struct launch_state {
    std::atomic<int> first_error{0};
    std::array<record, max_lanes> lanes{};

    void fail(error value) noexcept {
        if (value == error::ok) return;
        int expected = 0;
        first_error.compare_exchange_strong(expected, int(value), std::memory_order_acq_rel);
    }
    bool failed() const noexcept { return first_error.load(std::memory_order_acquire) != 0; }
};

// Construct after the full SPERT pool lease so that destruction releases the
// v2 lock before returning the worker scratch to SPERT. The v2 release ABI is
// TID-specific: acquiring on the host for a worker is not valid.
class lease {
  public:
    lease(const catalog * values, launch_state * state, size_t lane) noexcept :
        values_(values), state_(state),
        record_(state && lane < max_lanes ? &state->lanes[lane] : nullptr) {}
    lease(const lease &) = delete;
    lease & operator=(const lease &) = delete;
    ~lease() { release(); }

    error acquire(void * data, size_t size, error preparation = error::ok) noexcept {
        if (!values_) return error::ok;
        if (!record_) return fail_acquire(error::invalid_lane);
        if (preparation != error::ok) return fail_acquire(preparation);
        if (!values_->count || values_->count > max_lanes || !values_->bytes)
            return fail_acquire(error::invalid_layout);
        auto & out = *record_;
        const auto & api = values_->api;
        out.data = data; out.size = size;
        out.cpu = api.cpu(api.opaque); out.tid = api.tid(api.opaque);
        for (size_t i = 0; i < values_->count; ++i)
            if (data == values_->blocks[i].data) { out.id = int(i); break; }
        if (out.id < 0) return fail_acquire(error::unknown_pointer);
        const auto & known = values_->blocks[size_t(out.id)];
        if (size < values_->bytes || size > known.size) return fail_acquire(error::insufficient_size);
        if (out.cpu < 0 || out.cpu >= 64 || !(known.cpu_mask & (uint64_t(1) << out.cpu)))
            return fail_acquire(error::cpu_mismatch);
        if (out.tid <= 0) return fail_acquire(error::owner_mismatch);
        // No wait while holding a SPERT core grant: contention fails the graph.
        void * acquired = api.try_wait(api.opaque, out.id, 0);
        if (!acquired) return fail_acquire(error::busy);
        held_ = out.claimed = true;
        if (acquired != data) return fail_acquire(error::returned_pointer_mismatch);
        block current{};
        if (api.info(api.opaque, out.id, &current) != 0) return fail_acquire(error::metadata_failed);
        if (current.data != data || current.size < size || current.acquired != 1 ||
            current.owner_tid != out.tid) return fail_acquire(error::owner_mismatch);
        return error::ok;
    }

    error release() noexcept {
        if (!held_) return error::ok;
        const auto & api = values_->api;
        auto & out = *record_;
        // A failed release is recorded once; never retry it from a different
        // thread or force-release a block that may have changed owners.
        held_ = false;
        if (api.tid(api.opaque) != out.tid) return fail_release(error::thread_changed);
        if (api.release(api.opaque, out.id) != 0) return fail_release(error::release_failed);
        out.released = true;
        block current{};
        if (api.info(api.opaque, out.id, &current) != 0) return fail_release(error::metadata_failed);
        // Another process may already have claimed the block; only our own
        // still-present TID is evidence that this release did not finish.
        if (current.acquired == 1 && current.owner_tid == out.tid)
            return fail_release(error::release_still_owned);
        return error::ok;
    }

  private:
    error fail_acquire(error value) noexcept {
        if (record_) record_->acquire_error = value;
        if (state_) state_->fail(value);
        return value;
    }
    error fail_release(error value) noexcept {
        record_->release_error = value; state_->fail(value); return value;
    }
    const catalog * values_ = nullptr;
    launch_state * state_ = nullptr;
    record * record_ = nullptr;
    bool held_ = false;
};

} // namespace ggml::spacemit::tcm_ownership
