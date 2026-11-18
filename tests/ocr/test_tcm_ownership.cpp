#include "../../ggml/src/ggml-spacemit/spacemit-tcm-ownership.h"

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace tcm = ggml::spacemit::tcm_ownership;

namespace {
thread_local int current_lane = 0;
thread_local int tid_override = 0;
constexpr size_t bytes = 393216;
int checks = 0;

void check(bool condition, const char * message) {
    ++checks;
    if (!condition) { std::cerr << "FAIL: " << message << '\n'; std::exit(1); }
}

struct mock {
    std::mutex mutex;
    std::array<tcm::block, tcm::max_lanes> blocks{};
    std::array<char, tcm::max_lanes> addresses{};
    int available = 1, fake = 0, layout_status = 0;
    size_t block_count = tcm::max_lanes, block_bytes = bytes;
    int info_failure_id = -1, mismatch_id = -1, wrong_owner_id = -1;
    int release_failure_id = -1, leave_owned_id = -1, handoff_id = -1;
    int cpu_offset = 0;
    std::atomic<int> claims{0}, releases{0}, infos{0}, waits_nonzero{0};

    mock() {
        for (size_t i = 0; i < blocks.size(); ++i) {
            const int cpu = 8 + int(i);
            // Real K3 has paired CPU masks; pointer equality selects the ID.
            const int pair = cpu & ~1;
            blocks[i] = {&addresses[i], bytes, uint64_t(3) << pair,
                         0x100000 + i * bytes, 0, 0};
        }
    }
    static mock & get(void * value) { return *static_cast<mock *>(value); }
    tcm::driver api() {
        tcm::driver out{};
        out.opaque = this;
        out.available = [](void * p) { return get(p).available; };
        out.layout = [](void * p, size_t * size, size_t * count, int * fake) {
            auto & m = get(p); *size = m.block_bytes; *count = m.block_count; *fake = m.fake;
            return m.layout_status;
        };
        out.info = [](void * p, int id, tcm::block * block) {
            auto & m = get(p); ++m.infos;
            std::lock_guard<std::mutex> lock(m.mutex);
            if (id == m.info_failure_id) return -1;
            *block = m.blocks[size_t(id)];
            return 0;
        };
        out.try_wait = [](void * p, int id, size_t timeout) -> void * {
            auto & m = get(p); ++m.claims;
            if (timeout) ++m.waits_nonzero;
            std::lock_guard<std::mutex> lock(m.mutex);
            auto & block = m.blocks[size_t(id)];
            if (block.acquired) return nullptr;
            block.acquired = 1;
            block.owner_tid = id == m.wrong_owner_id ? 999 : 100 + current_lane;
            return id == m.mismatch_id ? &m : block.data;
        };
        out.release = [](void * p, int id) {
            auto & m = get(p); ++m.releases;
            std::lock_guard<std::mutex> lock(m.mutex);
            auto & block = m.blocks[size_t(id)];
            if (id == m.release_failure_id) return -1;
            if (block.owner_tid != 100 + current_lane) return -1;
            if (id != m.leave_owned_id) { block.acquired = 0; block.owner_tid = 0; }
            if (id == m.handoff_id) { block.acquired = 1; block.owner_tid = 777; }
            return 0;
        };
        out.tid = [](void *) { return tid_override ? tid_override : 100 + current_lane; };
        out.cpu = [](void * p) { return 8 + current_lane + get(p).cpu_offset; };
        return out;
    }
};

struct barrier {
    std::mutex mutex;
    std::condition_variable cv;
    int count = 0;
    void arrive() {
        std::unique_lock<std::mutex> lock(mutex);
        if (++count == 6) cv.notify_all();
        else cv.wait(lock, [this] { return count == 6; });
    }
};

tcm::catalog prepare(mock & m) {
    tcm::catalog catalog;
    check(catalog.prepare(m.api(), bytes) == tcm::error::ok, "valid host catalog");
    return catalog;
}

void host_validation() {
    mock m;
    tcm::catalog c;
    auto api = m.api(); api.release = nullptr;
    check(c.prepare(api, bytes) == tcm::error::invalid_api, "missing callback rejected");
    m.available = 0;
    check(c.prepare(m.api(), bytes) == tcm::error::unavailable, "unavailable rejected");
    m.available = 1; m.fake = 1;
    check(c.prepare(m.api(), bytes) == tcm::error::fake_tcm, "fake rejected");
    m.fake = -1;
    check(c.prepare(m.api(), bytes) == tcm::error::fake_tcm, "unknown fake status rejected");
    m.fake = 0; m.block_count = tcm::max_lanes + 1;
    check(c.prepare(m.api(), bytes) == tcm::error::invalid_layout, "too many blocks rejected");
    m.block_count = tcm::max_lanes; m.block_bytes = bytes - 1;
    check(c.prepare(m.api(), bytes) == tcm::error::invalid_layout, "small layout rejected");
    m.block_bytes = bytes; m.blocks[7].data = nullptr;
    check(c.prepare(m.api(), bytes) == tcm::error::invalid_block, "unmapped block rejected");
    m.blocks[7].data = m.blocks[0].data;
    check(c.prepare(m.api(), bytes) == tcm::error::duplicate_mapping, "duplicate mapping rejected");
    check(m.claims == 0 && m.releases == 0, "host validation does not claim or release");
}

void success_and_disabled() {
    mock m; auto c = prepare(m); tcm::launch_state state;
    {
        tcm::lease ownership(&c, &state, 0);
        check(ownership.acquire(m.blocks[0].data, bytes) == tcm::error::ok, "claim exact TCM pointer");
        check(state.lanes[0].id == 0 && state.lanes[0].tid == 100, "records block and worker TID");
        check(m.blocks[0].acquired == 1, "claimed during graph");
        check(ownership.release() == tcm::error::ok, "same worker release succeeds");
        check(ownership.release() == tcm::error::ok, "release idempotent");
    }
    check(m.claims == 1 && m.releases == 1 && m.waits_nonzero == 0, "one fail-fast claim and release");
    check(!state.failed() && state.lanes[0].released, "success state clear");
    {
        tcm::lease disabled(nullptr, nullptr, 9999);
        check(disabled.acquire(nullptr, 0) == tcm::error::ok, "disabled does nothing");
    }
    check(m.claims == 1 && m.releases == 1, "disabled has zero driver calls");
    // IDs 0 and 1 share affinity mask: CPU alone cannot distinguish them.
    {
        tcm::launch_state other;
        tcm::lease paired(&c, &other, 0);
        check(paired.acquire(m.blocks[1].data, bytes) == tcm::error::ok, "paired CPU resolves by pointer");
        check(other.lanes[0].id == 1, "exact pointer identifies paired block");
    }
}

void preclaim_failures() {
    mock m; auto c = prepare(m);
    {
        tcm::launch_state state; tcm::lease ownership(&c, &state, 0);
        int ddr = 0;
        check(ownership.acquire(&ddr, bytes) == tcm::error::unknown_pointer, "DDR fallback rejected");
        check(state.failed(), "unknown pointer marks graph failure");
    }
    {
        tcm::launch_state state; tcm::lease ownership(&c, &state, 0);
        check(ownership.acquire(m.blocks[0].data, bytes - 1) == tcm::error::insufficient_size, "small view rejected");
    }
    m.cpu_offset = 2;
    {
        tcm::launch_state state; tcm::lease ownership(&c, &state, 0);
        check(ownership.acquire(m.blocks[0].data, bytes) == tcm::error::cpu_mismatch, "wrong CPU pair rejected");
    }
    m.cpu_offset = 0;
    {
        tcm::launch_state state; tcm::lease ownership(&c, &state, tcm::max_lanes);
        check(ownership.acquire(m.blocks[0].data, bytes) == tcm::error::invalid_lane, "lane out of bounds rejected");
        check(state.failed(), "invalid lane marks launch failure without array access");
    }
    {
        tcm::lease ownership(&c, nullptr, 0);
        check(ownership.acquire(m.blocks[0].data, bytes) == tcm::error::invalid_lane, "missing state rejected safely");
    }
    {
        tcm::launch_state state; tcm::lease ownership(&c, &state, 0);
        check(ownership.acquire(nullptr, 0, tcm::error::scratch_unavailable) == tcm::error::scratch_unavailable,
              "scratch preparation failure can join collective without attempting claim");
        check(state.failed(), "scratch failure marks launch error");
    }
    check(m.claims == 0 && m.releases == 0, "preclaim failure touches no ownership lock");
    m.blocks[0].acquired = 1; m.blocks[0].owner_tid = 999;
    {
        tcm::launch_state state; tcm::lease ownership(&c, &state, 0);
        check(ownership.acquire(m.blocks[0].data, bytes) == tcm::error::busy, "foreign claim fails fast");
    }
    check(m.blocks[0].owner_tid == 999 && m.releases == 0, "foreign owner never released");
}

void postclaim_failures() {
    for (int failure = 0; failure < 3; ++failure) {
        mock m; auto c = prepare(m); tcm::launch_state state;
        if (failure == 0) m.mismatch_id = 0;
        if (failure == 1) m.info_failure_id = 0;
        if (failure == 2) m.wrong_owner_id = 0;
        {
            tcm::lease ownership(&c, &state, 0);
            const auto want = failure == 0 ? tcm::error::returned_pointer_mismatch :
                failure == 1 ? tcm::error::metadata_failed : tcm::error::owner_mismatch;
            check(ownership.acquire(m.blocks[0].data, bytes) == want, "postclaim validation catches corruption");
        }
        check(m.releases == 1 && (failure == 2 ? m.blocks[0].owner_tid == 999 : !m.blocks[0].acquired),
              "postclaim failure cleans own claim and preserves a changed foreign owner");
        check(state.failed(), "postclaim corruption fails graph");
    }
}

void release_failures() {
    for (int failure = 0; failure < 5; ++failure) {
        mock m; auto c = prepare(m); tcm::launch_state state;
        {
            tcm::lease ownership(&c, &state, 0);
            check(ownership.acquire(m.blocks[0].data, bytes) == tcm::error::ok, "claim before release test");
            if (failure == 0) tid_override = 999;
            if (failure == 1) m.release_failure_id = 0;
            if (failure == 2) m.leave_owned_id = 0;
            if (failure == 3) m.info_failure_id = 0;
            if (failure == 4) m.handoff_id = 0;
            const auto want = failure == 0 ? tcm::error::thread_changed :
                failure == 1 ? tcm::error::release_failed :
                failure == 2 ? tcm::error::release_still_owned :
                failure == 3 ? tcm::error::metadata_failed : tcm::error::ok;
            check(ownership.release() == want, "release outcome matches metadata");
        }
        tid_override = 0;
        check(m.releases == (failure == 0 ? 0 : 1), "failed release never retried by destructor");
        check(state.failed() == (failure != 4), "new foreign claim after release is allowed");
    }
}

void collective_failure(bool scratch_failure) {
    mock m; auto c = prepare(m); tcm::launch_state state;
    if (!scratch_failure) { m.blocks[3].acquired = 1; m.blocks[3].owner_tid = 999; }
    barrier sync;
    std::atomic<int> operators{0};
    std::vector<std::thread> workers;
    for (int lane = 0; lane < 6; ++lane) workers.emplace_back([&, lane] {
        current_lane = lane;
        tcm::lease ownership(&c, &state, size_t(lane));
        ownership.acquire(m.blocks[size_t(lane)].data, bytes, scratch_failure && lane == 3 ?
                          tcm::error::scratch_unavailable : tcm::error::ok);
        sync.arrive();
        if (state.failed()) { ownership.release(); return; }
        ++operators;
    });
    for (auto & worker : workers) worker.join();
    check(state.failed() && operators == 0, "one failed lane makes all six skip operators");
    check(m.claims == (scratch_failure ? 5 : 6) && m.releases == 5 && m.waits_nonzero == 0,
          "all successful lanes release after collective abort");
    for (int lane = 0; lane < 6; ++lane)
        check(m.blocks[size_t(lane)].owner_tid == (lane == 3 && !scratch_failure ? 999 : 0),
              "collective abort preserves foreign lock only");
}
} // namespace

int main() {
    host_validation(); success_and_disabled(); preclaim_failures();
    postclaim_failures(); release_failures(); collective_failure(false); collective_failure(true);
    std::cout << "TCM ownership mock: " << checks << " checks passed\n";
}
