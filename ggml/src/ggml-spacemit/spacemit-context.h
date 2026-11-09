#pragma once

#include "ggml.h"

#include <spert.hpp>

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <stdexcept>

struct spacemit_opnode;  // forward declaration (defined in spacemit-opnode.h)

namespace ggml::spacemit {

inline constexpr size_t cache_line_size_f32 = 64 / sizeof(float);

struct context {
    spert::Context *        runtime       = nullptr;
    uint32_t                ith           = 0;
    uint32_t                nth           = 0;
    void *                  workspace     = nullptr;
    size_t                  workspace_size = 0;
    spert::SharedBufferView shared        = {};
    bool                    check_tcm_view = false;
    std::atomic<int64_t> * work_queue     = nullptr;

    // Valid only across adjacent PC8 nodes inside one graph launch. The graph
    // executor clears this for every other op; reset prevents cross-call reuse.
    const ggml_tensor * pc8_packed_source = nullptr;
    // Produced by the immediately preceding cast-pair node only. Never persists
    // across a graph launch or an intervening workspace-writing operation.
    const ggml_tensor * ocr_prepared_attention = nullptr;

    void reset(spert::Context & runtime_in, uint32_t ith_in, uint32_t nth_in,
               void * workspace_in, size_t workspace_size_in, spert::SharedBufferView shared_in,
               std::atomic<int64_t> * work_queue_in = nullptr, bool check_tcm_view_in = false) {
        pc8_packed_source = nullptr;
        ocr_prepared_attention = nullptr;
        runtime        = &runtime_in;
        ith            = ith_in;
        nth            = nth_in;
        workspace      = workspace_in;
        workspace_size = workspace_size_in;
        shared         = shared_in;
        check_tcm_view = check_tcm_view_in;
        work_queue     = work_queue_in;
    }

    void clear() {
        pc8_packed_source = nullptr;
        ocr_prepared_attention = nullptr;
        runtime        = nullptr;
        ith            = 0;
        nth            = 0;
        workspace      = nullptr;
        workspace_size = 0;
        shared         = {};
        check_tcm_view = false;
        work_queue     = nullptr;
    }

    spert::Status sync() {
        auto check = [&] {
            if (!check_tcm_view) return;
            const auto current = runtime->shared_buffer();
            if (current.data != shared.data || current.size < shared.size)
                throw std::runtime_error("TCM view changed across a cooperative barrier");
        };
        check();
        const auto status = runtime->sync();
        check();
        return status;
    }
};

class tensor_traits_base {
  public:
    virtual ~tensor_traits_base() = default;

    virtual bool work_size(int n_threads, const ggml_tensor * op, size_t & size) const = 0;
    virtual bool compute_forward(context & ctx, ggml_tensor * op) const                 = 0;
    // Fused multi-op dispatch, keyed on spacemit_op_code. Default is "not supported".
    virtual bool compute_forward_fused(context & ctx, const spacemit_opnode & node) const {
        GGML_UNUSED(ctx);
        GGML_UNUSED(node);
        return false;
    }
    virtual int  repack(ggml_tensor * tensor, const void * data, size_t size) const     = 0;
};

}  // namespace ggml::spacemit
