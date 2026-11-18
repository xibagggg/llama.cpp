#pragma once

#include "ggml-impl.h"
#include "ocr-mapped-argmax-rvv.h"
#include <cstdlib>
#include <atomic>
#include <cstdint>
#include <cstring>

// Private, versioned contract for dflash.cpp's reviewed compact argmax callback.
// All registration and admission happen on the host. This is NOT a generic
// custom-op whitelist, name match, inference offload bypass or callback runner.
namespace ocr_dspark_custom_contract {
inline std::atomic<ggml_custom_op_t> registered_callback{nullptr};
inline std::atomic<bool> vector_enabled{false};

// Called only by trusted model-building code via backend get_proc_address.
// The supplied function must implement the reviewed v1 callback: it reads
// src[0]/src[1], treats userdata solely as an integer n_vocab, and writes exactly
// one I32 to dst->data on ith0. Callback code must remain loaded throughout the
// backend's lifetime. No model, graph, tensor, data or userdata pointer is kept.
// First identity is pinned until backend unload. Repeat identity is idempotent;
// null or a different identity fails closed. Registration never invokes it.
inline bool register_callback(ggml_custom_op_t callback) {
    if (!callback) return false;
    ggml_custom_op_t expected = nullptr;
    const bool accepted = registered_callback.compare_exchange_strong(expected, callback,
               std::memory_order_release, std::memory_order_relaxed) || expected == callback;
    if (accepted) {
        const char * option = std::getenv("GGML_SPACEMIT_DSPARK_MAPPED_ARGMAX_RVV");
        vector_enabled.store(option && !std::strcmp(option, "1"), std::memory_order_release);
    }
    return accepted;
}

inline bool disjoint(const void * a, size_t an, const void * b, size_t bn) {
    const uintptr_t av = reinterpret_cast<uintptr_t>(a), bv = reinterpret_cast<uintptr_t>(b);
    return an <= UINTPTR_MAX - av && bn <= UINTPTR_MAX - bv &&
           (av + an <= bv || bv + bn <= av);
}

// Host metadata-only proof of the exact current compact callback's write-set.
// The enclosing D2T catalog still checks every ordinary graph output and scans
// all signed I64 IDs for range/uniqueness on every graph invocation.
inline bool writes_only_output(const ggml_tensor * op, const ggml_tensor * ids_root) {
    if (!op || op->op != GGML_OP_CUSTOM || !ids_root) return false;
    const ggml_custom_op_t trusted = registered_callback.load(std::memory_order_acquire);
    if (!trusted) return false;
    ggml_custom_op_params params;
    static_assert(sizeof(params) <= sizeof(op->op_params), "custom params fit GGML storage");
    std::memcpy(&params, op->op_params, sizeof(params));
    if (params.fun != trusted || params.n_tasks != 1 ||
        reinterpret_cast<uintptr_t>(params.userdata) != uintptr_t(248100)) return false;
    const auto * values = op->src[0];
    const auto * ids = op->src[1];
    if (!values || !ids || !op->data || !values->data || !ids->data || !ids_root->data ||
        op->type != GGML_TYPE_I32 || values->type != GGML_TYPE_F32 || ids->type != GGML_TYPE_I64 ||
        op->view_src || !ggml_is_contiguous(op) || !ggml_is_contiguous(values) || !ggml_is_contiguous(ids) ||
        op->ne[0] != 1 || op->ne[1] != 1 || op->ne[2] != 1 || op->ne[3] != 1 ||
        values->ne[0] != 114688 || values->ne[1] != 1 || values->ne[2] != 1 || values->ne[3] != 1 ||
        ids->ne[0] != 114688 || ids->ne[1] != 1 || ids->ne[2] != 1 || ids->ne[3] != 1 ||
        reinterpret_cast<uintptr_t>(op->data) % alignof(int32_t) ||
        reinterpret_cast<uintptr_t>(values->data) % alignof(float) ||
        reinterpret_cast<uintptr_t>(ids->data) % alignof(int64_t) ||
        ids->data != ids_root->data || ggml_nbytes(ids_root) != size_t(114688) * sizeof(int64_t)) return false;
    for (int i = 2; i < GGML_MAX_SRC; ++i) if (op->src[i]) return false;
    // These also prove nonwrapping address ranges before the callback's reads.
    return disjoint(op->data, sizeof(int32_t), ids_root->data, ggml_nbytes(ids_root)) &&
           disjoint(op->data, sizeof(int32_t), values->data, ggml_nbytes(values));
}
// The trusted callback's non-NaN scan only compares narrowed IDs; it never
// indexes memory through them. The vector helper is identical even for duplicate
// or out-of-range IDs. The existing host D2T catalog's stronger unique/in-range
// proof remains intact. Recheck metadata here; keep all NaN work in the callback.
inline bool vector_forward(ggml_tensor * op) {
    if (!vector_enabled.load(std::memory_order_acquire) || !op ||
        !writes_only_output(op, op->src[1])) return false;
    int32_t result;
    if (!ocr_mapped_argmax_rvv::compute(static_cast<const float *>(op->src[0]->data),
            static_cast<const int64_t *>(op->src[1]->data), op->src[0]->ne[0], 248100, result)) return false;
    *static_cast<int32_t *>(op->data) = result;
    return true;
}
} // namespace ocr_dspark_custom_contract
