#pragma once

// Candidate-only harness. These production headers are copied verbatim and
// their hashes recorded by the launcher; no approximation of the ownership API.
#include "spacemit-resident-scratch.h"
#include "spacemit-tcm-ownership-linux.h"
#include <optional>
#include <stdexcept>
#include <string>

namespace standalone_real_tcm {
namespace policy = ggml::spacemit::tcm_ownership;

struct host {
    policy::catalog values;
    size_t bytes = 0;

    // Construct after backend / Stream initialization, on a host CPU.
    host() : bytes(spert::backend_info().shared_mem_size) {
        const auto status = policy::prepare_platform_catalog(values, bytes);
        if (status != policy::error::ok)
            throw std::runtime_error(std::string("real TCM catalog: ") + policy::name(status));
    }
};

class worker {
public:
    worker(spert::Context & runtime, const host & platform,
           policy::launch_state & state, unsigned lane) noexcept :
        runtime_(runtime), state_(state), ownership_(&platform.values, &state, lane) {
        policy::error preparation = policy::error::ok;
        try {
            pool_.emplace(runtime, platform.bytes, true);
            view_ = runtime.shared_buffer();
            if (!view_ || view_.size < platform.bytes) preparation = policy::error::scratch_unavailable;
        } catch (...) {
            preparation = policy::error::scratch_unavailable;
        }
        ownership_.acquire(view_.data, view_.size, preparation);
    }
    worker(const worker &) = delete;
    worker & operator=(const worker &) = delete;

    // Every lane must enter once, even when its preparation failed. No kernel
    // may write scratch until the collective claim succeeded for all lanes.
    bool ready() noexcept {
        if (runtime_.sync() != spert::Status::Ok) state_.fail(policy::error::barrier_failed);
        return !state_.failed();
    }
    spert::SharedBufferView full() const noexcept { return view_; }
    void release() noexcept { ownership_.release(); }

private:
    spert::Context & runtime_;
    policy::launch_state & state_;
    // Reverse destruction order releases v2 ownership before the SPERT pool.
    std::optional<ggml::spacemit::tcm_lease> pool_;
    policy::lease ownership_;
    spert::SharedBufferView view_{};
};

inline unsigned errors(const policy::launch_state & state, unsigned lanes) noexcept {
    unsigned result = state.failed();
    for (unsigned lane = 0; lane < lanes; ++lane) {
        const auto & record = state.lanes[lane];
        result += !record.claimed || !record.released;
        result += record.acquire_error != policy::error::ok || record.release_error != policy::error::ok;
        result += record.cpu < 8 || record.cpu > 13 || record.tid <= 0 || record.id < 0;
    }
    return result;
}
} // namespace standalone_real_tcm
