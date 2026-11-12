#ifndef SPACEMIT_OPNODE_H
#define SPACEMIT_OPNODE_H

#define GGML_COMMON_IMPL_CPP
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-impl.h"

#include <algorithm>
#include <string>
#include <vector>
#include <stdio.h>

enum spacemit_op_code {
    SPACEMIT_OP_INVALID = 0,
    SPACEMIT_OP_MUL_MAT,
    SPACEMIT_OP_MUL_MAT_ID,
    SPACEMIT_OP_MUL_MAT_ADD,
    SPACEMIT_OP_MUL_MAT_QKV,
    SPACEMIT_OP_MUL_MAT_FFN,
    SPACEMIT_OP_RMS_NORM,
    SPACEMIT_OP_RMS_NORM_MUL,
    SPACEMIT_OP_CONCAT_STATE,    // concat writing only the trailing columns (fused with a later SSM_CONV)
    SPACEMIT_OP_SSM_CONV_CONCAT, // ssm_conv reading the concat inputs directly (the concat was partial)
    SPACEMIT_OP_ADD,
    SPACEMIT_OP_UNARY_SILU,
    SPACEMIT_OP_UNARY_GELU,
    SPACEMIT_OP_ROPE,
    SPACEMIT_OP_SOFTMAX,
    SPACEMIT_OP_RESHAPE,
    SPACEMIT_OP_VIEW,
    SPACEMIT_OP_PERMUTE,
    SPACEMIT_OP_TRANSPOSE,
    SPACEMIT_OP_NONE,
    SPACEMIT_OP_ADD_GELU,
    SPACEMIT_OP_IM2COL_MUL_MAT,
    SPACEMIT_OP_NORM_AFFINE,
    SPACEMIT_OP_RESIDUAL_NORM_AFFINE,
    SPACEMIT_OP_PC8_PAD_MUL_MAT,
    SPACEMIT_OP_PC8_MUL_MAT_BIAS,
    SPACEMIT_OP_OCR_CAST_KV_PACK,
    SPACEMIT_OP_GDN_STATE_GATHER, // get_rows of the recurrent state, skipped when the update runs in place
    SPACEMIT_OP_GDN_INPLACE,      // gated_delta_net updating the cache row in place
    SPACEMIT_OP_GDN_STATE_STORE,  // cpy of the new state to the cache row, skipped when in place
};

#define SPACEMIT_OP_MAX_KERN_PARAMS 32

struct spacemit_opnode {
    ggml_tensor * node = nullptr;
    ggml_tensor * packed_im2col = nullptr;
    ggml_tensor * elided_pad = nullptr;
    ggml_tensor * prepared_attention = nullptr;
    // in-place recurrent state update: {state gather (GET_ROWS), GATED_DELTA_NET, state store (CPY)}
    ggml_tensor * gdn_state[3] = {nullptr, nullptr, nullptr};

    std::vector<ggml_tensor *> fused;

    spacemit_op_code opcode = SPACEMIT_OP_INVALID;

    std::vector<ggml_tensor *> extra_dsts;

    int32_t kernel_params[SPACEMIT_OP_MAX_KERN_PARAMS] = {0};

    spacemit_opnode(ggml_tensor * node = nullptr, std::vector<ggml_tensor *> fused = {}, spacemit_op_code opcode = SPACEMIT_OP_INVALID, std::vector<ggml_tensor *> extra_dsts = {})
        : node(node), fused(std::move(fused)), opcode(opcode), extra_dsts(std::move(extra_dsts)) {}

    ggml_op op() const {
        return node->op;
    }

    const ggml_tensor * dst() const {
        return fused.empty() ? node : fused.back();
    }

    void add_fused(ggml_tensor * t, bool extra_dst = false) {
        fused.push_back(t);
        if (extra_dst) {
            extra_dsts.push_back(t);
        }
    }

    std::vector<const ggml_tensor *> get_outputs() const {
        std::vector<const ggml_tensor *> res;
        if (extra_dsts.empty()) {
            res.push_back(dst());
        } else {
            res.push_back(node);
            for (const auto * x : extra_dsts) {
                res.push_back(x);
            }
        }
        return res;
    }

    const ggml_tensor * src0() const {
        return node->src[0];
    }

    const ggml_tensor * src1() const {
        return node->src[1];
    }

    bool is_empty() const {
        return ggml_op_is_empty(node->op);
    }

    bool stackable() const {
        if(packed_im2col||elided_pad)return false;
        switch (this->op()) {
            case GGML_OP_MUL_MAT:
            case GGML_OP_MUL_MAT_ID:
                return ggml_is_quantized(this->src0()->type);
            default:
                return false;
        }
    }

    bool same_input(const spacemit_opnode& n) const {
        return n.src1() == this->src1();
    }

    std::vector<const ggml_tensor *> get_inputs() const {
        if(elided_pad)return {node->src[0],elided_pad->src[0]};
        if(packed_im2col)return {node->src[0],packed_im2col->src[0],packed_im2col->src[1]};
        if (fused.empty()) {
            int last_non_null = -1;
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (node->src[i]) {
                    last_non_null = i;
                }
            }
            std::vector<const ggml_tensor *> inputs(last_non_null + 1, nullptr);
            for (int i = 0; i <= last_non_null; i++) {
                inputs[i] = node->src[i];
            }
            return inputs;
        }

        std::vector<const ggml_tensor *> inputs(GGML_MAX_SRC, nullptr);
        std::vector<const ggml_tensor *> outputs;
        outputs.push_back(node);
        for (const auto * f : fused) {
            outputs.push_back(f);
        }

        auto contains = [&](const std::vector<const ggml_tensor *> & vec, const ggml_tensor * t) {
            for (const auto * x : vec) {
                if (x == t) return true;
            }
            return false;
        };

        int count = 0;
        auto add_input = [&](const ggml_tensor * t) {
            if (t && !contains(outputs, t) && !contains(inputs, t)) {
                if (count < (int)inputs.size()) {
                    inputs[count++] = t;
                } else {
                    inputs.push_back(t);
                }
            }
        };

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node->src[i]) {
                add_input(node->src[i]);
            }
        }
        for (const auto * f : fused) {
            for (int i = 0; i < GGML_MAX_SRC; i++) {
                if (f->src[i]) {
                    add_input(f->src[i]);
                }
            }
        }

        inputs.resize(count);
        return inputs;
    }

    std::string op_name() const {
        if(elided_pad)return "PAD+PC8_MUL_MAT";
        if(packed_im2col)return "IM2COL+PACK+MUL_MAT";
        if (fused.empty()) {
            return ggml_op_desc(node);
        }
        std::string name = ggml_op_desc(node);
        for (const auto * f : fused) {
            name += "+";
            name += ggml_op_desc(f);
        }
        return name;
    }
};

namespace ggml::spacemit {
struct context;
}

bool ggml_spacemit_compute_opnode(ggml::spacemit::context & ctx, const spacemit_opnode & node);

#endif // SPACEMIT_OPNODE_H
