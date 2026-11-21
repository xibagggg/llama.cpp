#pragma once

#include "ggml.h"

#include <spert.hpp>

#include <cstddef>
#include <cstdint>
#include <atomic>
#include <stdexcept>

struct spacemit_opnode;  // forward declaration (defined in spacemit-opnode.h)

namespace paddle_qkv_content_cache { struct view; }
namespace ocr_d2t_scatter { struct launch_catalog; }

namespace ggml::spacemit {

inline constexpr size_t cache_line_size_f32 = 64 / sizeof(float);

struct context {
    const ocr_d2t_scatter::launch_catalog * d2t_scatter = nullptr;
    spert::Context *        runtime       = nullptr;
    uint32_t                ith           = 0;
    uint32_t                nth           = 0;
    void *                  workspace     = nullptr;
    size_t                  workspace_size = 0;
    spert::SharedBufferView shared        = {};
    bool                    check_tcm_view = false;
    std::atomic<uint64_t> * diagnostic_subphase = nullptr;
    std::atomic<int64_t> * work_queue     = nullptr;

    // Valid only across adjacent PC8 nodes inside one graph launch. The graph
    // executor clears this for every other op; reset prevents cross-call reuse.
    const ggml_tensor * pc8_packed_source = nullptr;
    // Produced by the immediately preceding cast-pair node only. Never persists
    // across a graph launch or an intervening workspace-writing operation.
    const ggml_tensor * ocr_prepared_attention = nullptr;
    // The conv state GET_ROWS of the recurrent layer whose conv state is updated in place (see
    // conv_inplace_match), set when that node is reached and compared by the later opnodes of the
    // layer. The row index is only guaranteed to be readable at that position: its last reader
    // comes before the SSM_CONV, and the allocator may reuse its buffer after that.
    const ggml_tensor * conv_state_inplace = nullptr;
    const float * prepared_vision_rope = nullptr;
    paddle_qkv_content_cache::view * qkv_content_cache = nullptr;
    // GGML_SPACEMIT_GEMV_TILE as the host parsed it before this graph launch, 0 when unset. The
    // direct GEMV path runs once per lane on every decode matrix product, so it must not scan the
    // environment itself.
    int64_t gemv_tile_cols = 0;

    void reset(spert::Context & runtime_in, uint32_t ith_in, uint32_t nth_in,
               void * workspace_in, size_t workspace_size_in, spert::SharedBufferView shared_in,
               std::atomic<int64_t> * work_queue_in = nullptr, bool check_tcm_view_in = false) {
        pc8_packed_source = nullptr;
        ocr_prepared_attention = nullptr;
        conv_state_inplace = nullptr;
        prepared_vision_rope = nullptr;
        qkv_content_cache = nullptr;
        gemv_tile_cols = 0;
        d2t_scatter = nullptr;
        runtime        = &runtime_in;
        ith            = ith_in;
        nth            = nth_in;
        workspace      = workspace_in;
        workspace_size = workspace_size_in;
        shared         = shared_in;
        check_tcm_view = check_tcm_view_in;
        diagnostic_subphase = nullptr;
        work_queue     = work_queue_in;
    }

    void clear() {
        pc8_packed_source = nullptr;
        ocr_prepared_attention = nullptr;
        conv_state_inplace = nullptr;
        prepared_vision_rope = nullptr;
        qkv_content_cache = nullptr;
        gemv_tile_cols = 0;
        d2t_scatter = nullptr;
        runtime        = nullptr;
        ith            = 0;
        nth            = 0;
        workspace      = nullptr;
        workspace_size = 0;
        shared         = {};
        check_tcm_view = false;
        diagnostic_subphase = nullptr;
        work_queue     = nullptr;
    }

    void mark_subphase(uint64_t value) {
        if (diagnostic_subphase) diagnostic_subphase->store(value, std::memory_order_release);
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
