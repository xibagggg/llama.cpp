#pragma once

#include "spacemit-tcm-ownership.h"

#if defined(__linux__) && __has_include(<spine_tcm.h>)
#include <spine_tcm.h>
#include <sched.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace ggml::spacemit::tcm_ownership {

inline error prepare_platform_catalog(catalog & out, size_t bytes) noexcept {
    // This must run on the host: the header loader can dlopen/allocate and the
    // library constructor discovers/maps TCM. It must never first run on an AI
    // worker. Reuse the same global instance already loaded by SPERT.
    if (spine_tcm_open_handle(nullptr) != 0) return error::unavailable;
    driver api{};
    api.available = [](void *) { return spine_tcm_is_available(); };
    api.layout = [](void *, size_t * size, size_t * count, int * fake) {
        spine_tcm_mem_info_t info{};
        const int status = spine_tcm_mem_info(&info);
        *size = info.blk_size; *count = info.blk_num; *fake = info.is_fake_tcm;
        return status;
    };
    api.info = [](void *, int id, block * out) {
        spine_tcm_block_info_t info{};
        const int status = spine_tcm_block_info(id, &info);
        if (status == 0) {
            *out = {info.va, info.size, info.cpu_affinity_mask, info.phys_addr,
                    info.owner_tid, info.is_acquired};
        }
        return status;
    };
    api.try_wait = [](void *, int id, size_t timeout) { return spine_tcm_mem_try_wait(id, timeout); };
    api.release = [](void *, int id) { return spine_tcm_mem_release(id); };
    api.tid = [](void *) { return int(::syscall(SYS_gettid)); };
    api.cpu = [](void *) { return ::sched_getcpu(); };
    return out.prepare(api, bytes);
}

inline const char * platform_version() noexcept { return spine_tcm_version(); }

} // namespace ggml::spacemit::tcm_ownership
#else
namespace ggml::spacemit::tcm_ownership {
inline error prepare_platform_catalog(catalog &, size_t) noexcept { return error::unavailable; }
inline const char * platform_version() noexcept { return "unavailable"; }
} // namespace ggml::spacemit::tcm_ownership
#endif
