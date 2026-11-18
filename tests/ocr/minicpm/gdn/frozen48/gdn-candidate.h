namespace gdn_candidate {
static inline float rvv_gdn_fused_row_f32(float *             dst,
                                          const float *       src,
                                          const vfloat32m8_t kv,
                                          const vfloat32m8_t qv,
                                          const vfloat32m8_t dv,
                                          const float         v_j,
                                          const float         beta,
                                          const float         kq,
                                          const float         scale,
                                          const size_t        vl) {
    vfloat32m8_t s = __riscv_vle32_v_f32m8(src, vl);
    s = __riscv_vfmul_vv_f32m8(s, dv, vl);
    vfloat32m1_t sum_k = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t sum_q = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    sum_k = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(s, kv, vl), sum_k, vl);
    sum_q = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(s, qv, vl), sum_q, vl);
    const float state_k = __riscv_vfmv_f_s_f32m1_f32(sum_k);
    const float state_q = __riscv_vfmv_f_s_f32m1_f32(sum_q);
    const float delta_j = (v_j - state_k) * beta;
    s = __riscv_vfmacc_vf_f32m8(s, delta_j, kv, vl);
    __riscv_vse32_v_f32m8(dst, s, vl);
    return (state_q + delta_j * kq) * scale;
}

// A head that fits in m4 leaves registers for the state and reduction
// temporaries alongside k, q and decay, avoiding whole-vector row-loop spills.
static inline float rvv_gdn_fused_row_f32(float *             dst,
                                          const float *       src,
                                          const vfloat32m4_t kv,
                                          const vfloat32m4_t qv,
                                          const vfloat32m4_t dv,
                                          const float         v_j,
                                          const float         beta,
                                          const float         kq,
                                          const float         scale,
                                          const size_t        vl) {
    vfloat32m4_t s = __riscv_vle32_v_f32m4(src, vl);
    s = __riscv_vfmul_vv_f32m4(s, dv, vl);
    vfloat32m1_t sum_k = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t sum_q = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    sum_k = __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(s, kv, vl), sum_k, vl);
    sum_q = __riscv_vfredusum_vs_f32m4_f32m1(__riscv_vfmul_vv_f32m4(s, qv, vl), sum_q, vl);
    const float state_k = __riscv_vfmv_f_s_f32m1_f32(sum_k);
    const float state_q = __riscv_vfmv_f_s_f32m1_f32(sum_q);
    const float delta_j = (v_j - state_k) * beta;
    s = __riscv_vfmacc_vf_f32m4(s, delta_j, kv, vl);
    __riscv_vse32_v_f32m4(dst, s, vl);
    return (state_q + delta_j * kq) * scale;
}

static inline float rvv_gdn_dot_f32(const float * x, const float * y, int64_t n) {
    const size_t vl = __riscv_vsetvl_e32m8(n);
    const vfloat32m8_t xv = __riscv_vle32_v_f32m8(x, vl);
    const vfloat32m8_t yv = __riscv_vle32_v_f32m8(y, vl);
    const vfloat32m8_t prod = __riscv_vfmul_vv_f32m8(xv, yv, vl);
    vfloat32m1_t sum = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    sum = __riscv_vfredusum_vs_f32m8_f32m1(prod, sum, vl);
    return __riscv_vfmv_f_s_f32m1_f32(sum);
}

struct rvv_gdn_decay_dots_f32 {
    float state_k;
    float state_q;
};

// Two-pass row update of the reference implementation: decay the state row,
// store it back and reduce the two dot products in the same sweep, then apply
// the delta-rule correction in a second sweep over the rows. The early store
// keeps the loads of subsequent rows pipelined on the in-order cores, which
// makes this form the faster one once the chunk has enough tokens that the
// state stays bandwidth-bound (measured crossover vs the fused form: ~8
// tokens/chunk; prefill chunks are 64, kda chunks 16).
static inline rvv_gdn_decay_dots_f32 rvv_gdn_decay_dots_f32_impl(float *       state,
                                                                   const float * decay,
                                                                   float         decay_scalar,
                                                                   const float * k,
                                                                   const float * q,
                                                                   int64_t       n,
                                                                   bool          kda) {
    const size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t s  = __riscv_vle32_v_f32m8(state, vl);
    if (kda) {
        const vfloat32m8_t d = __riscv_vle32_v_f32m8(decay, vl);
        s = __riscv_vfmul_vv_f32m8(s, d, vl);
    } else {
        s = __riscv_vfmul_vf_f32m8(s, decay_scalar, vl);
    }
    __riscv_vse32_v_f32m8(state, s, vl);

    const vfloat32m8_t kv = __riscv_vle32_v_f32m8(k, vl);
    const vfloat32m8_t qv = __riscv_vle32_v_f32m8(q, vl);
    vfloat32m1_t sum_k = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    vfloat32m1_t sum_q = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    sum_k = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(s, kv, vl), sum_k, vl);
    sum_q = __riscv_vfredusum_vs_f32m8_f32m1(__riscv_vfmul_vv_f32m8(s, qv, vl), sum_q, vl);
    return { __riscv_vfmv_f_s_f32m1_f32(sum_k), __riscv_vfmv_f_s_f32m1_f32(sum_q) };
}

static inline void rvv_gdn_update_f32(float * state, const float * k, float delta, int64_t n) {
    const size_t vl = __riscv_vsetvl_e32m8(n);
    vfloat32m8_t s  = __riscv_vle32_v_f32m8(state, vl);
    const vfloat32m8_t kv = __riscv_vle32_v_f32m8(k, vl);
    s = __riscv_vfmacc_vf_f32m8(s, delta, kv, vl);
    __riscv_vse32_v_f32m8(state, s, vl);
}

// Ported from ggml-cpu/spacemit (linpengt/qwen35-opt); the IME variant (off by default there) is not.
// ---- chunkwise gated delta net (prefill) ----
// Per head, with S the k x v state (the op stores its transpose), chunk-local cumulative log decay
// gam_r = sum_{s<=r} g_s, Gam_r = exp(gam_r) and E_rs = exp(gam_r - gam_s) <= 1 for s <= r:
//   U   = (I + L)^-1 * beta (V - Gam K S0),  L_rs = beta_r E_rs (k_r . k_s), s < r   (forward substitution)
//   O   = scale (Gam Q S0 + (E o QK^T)_lower-incl U)
//   S1  = Gam_C S0 + (E_C. o K)^T U
// which equals the token recurrence S_t = e^{g_t} S_{t-1} + k_t beta_t (v_t - e^{g_t} S_{t-1}^T k_t)^T,
// o_t = scale S_t^T q_t, up to float rounding. All products are 4-row blocked RVV matmuls.
// Default chunk 16: everything runs at the A100's fp32 vector rate, so smaller chunks (less C^2 work) are
// faster (Qwen3.6 pp512 GDN ms/token: chunk 8 1.23, 16 1.29, 32 1.53, 64 1.98, 128 2.74; recurrence 2.56).
// KLD vs the recurrence is 0.004-0.005 for all chunk sizes (rounding-level for this MoE model).
constexpr int64_t GDN_CHUNK_SIZE = 16;

// C[m x n] = (C0 ? c0s * C0 : 0) + A[m x k] B[k x n]. A rows strided by lda; B, C, C0 rows contiguous
// with leading dims ldb, ldc, ldc0; n <= vlmax(e32m4). C may alias C0 (row-wise in place).
static void gdn_gemm(int64_t m, int64_t n, int64_t k, const float * A, int64_t lda, const float * B, int64_t ldb,
                     float * C, int64_t ldc, const float * C0, int64_t ldc0, float c0s) {
    const size_t vl = __riscv_vsetvl_e32m4(n);
    int64_t      r  = 0;
    for (; r + 4 <= m; r += 4) {
        vfloat32m4_t c0, c1, c2, c3;
        if (C0) {
            c0 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + (r + 0) * ldc0, vl), c0s, vl);
            c1 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + (r + 1) * ldc0, vl), c0s, vl);
            c2 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + (r + 2) * ldc0, vl), c0s, vl);
            c3 = __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + (r + 3) * ldc0, vl), c0s, vl);
        } else {
            c0 = c1 = c2 = c3 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        }
        const float * a0 = A + (r + 0) * lda;
        const float * a1 = A + (r + 1) * lda;
        const float * a2 = A + (r + 2) * lda;
        const float * a3 = A + (r + 3) * lda;
        for (int64_t p = 0; p < k; ++p) {
            const vfloat32m4_t b = __riscv_vle32_v_f32m4(B + p * ldb, vl);
            c0 = __riscv_vfmacc_vf_f32m4(c0, a0[p], b, vl);
            c1 = __riscv_vfmacc_vf_f32m4(c1, a1[p], b, vl);
            c2 = __riscv_vfmacc_vf_f32m4(c2, a2[p], b, vl);
            c3 = __riscv_vfmacc_vf_f32m4(c3, a3[p], b, vl);
        }
        __riscv_vse32_v_f32m4(C + (r + 0) * ldc, c0, vl);
        __riscv_vse32_v_f32m4(C + (r + 1) * ldc, c1, vl);
        __riscv_vse32_v_f32m4(C + (r + 2) * ldc, c2, vl);
        __riscv_vse32_v_f32m4(C + (r + 3) * ldc, c3, vl);
    }
    for (; r < m; ++r) {
        vfloat32m4_t c = C0 ? __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(C0 + r * ldc0, vl), c0s, vl) :
                              __riscv_vfmv_v_f_f32m4(0.0f, vl);
        const float * a = A + r * lda;
        for (int64_t p = 0; p < k; ++p) {
            c = __riscv_vfmacc_vf_f32m4(c, a[p], __riscv_vle32_v_f32m4(B + p * ldb, vl), vl);
        }
        __riscv_vse32_v_f32m4(C + r * ldc, c, vl);
    }
}

// dst[j * ld_dst + i] = src[i * ld_src + j] for an n x n block (n <= vlmax(e32m4))
static void gdn_transpose(int64_t n_rows, int64_t n_cols, const float * src, int64_t ld_src, float * dst,
                          int64_t ld_dst) {
    const size_t vl = __riscv_vsetvl_e32m4(n_cols);
    for (int64_t i = 0; i < n_rows; ++i) {
        __riscv_vsse32_v_f32m4(dst + i, ld_dst * sizeof(float), __riscv_vle32_v_f32m4(src + i * ld_src, vl), vl);
    }
}

struct gdn_chunk_args {
    const ggml_tensor *src_q, *src_k, *src_v, *src_g, *src_beta;
    int64_t            S_v, H, n_tokens;
    float              scale;
};

// One (head, seq): s_in / s_out are the transposed states (S_v x S_v), attn_data the first token's output.
static void gdn_chunked_head(const gdn_chunk_args & a, int64_t iq1, int64_t ik1, int64_t iv1, int64_t iq3, int64_t ik3,
                             int64_t iv3, const float * s_in, float * s_out, float * attn_data, float * scratch_workspace) {
    const int64_t D  = a.S_v;
    const int64_t CS = GDN_CHUNK_SIZE;

    static thread_local std::vector<float> buf;
    const size_t need = D * D + 7 * CS * D + 3 * CS * CS + 4 * CS + 64;
    if (scratch_workspace == nullptr && buf.size() < need) {
        buf.resize(need);
    }
    float * S   = scratch_workspace != nullptr ? scratch_workspace : buf.data();
    float * Qc  = S + D * D;
    float * Kc  = Qc + CS * D;
    float * U   = Kc + CS * D;  // v rows, then U in place
    float * KT  = U + CS * D;   // D x CS
    float * KTs = KT + CS * D;  // D x CS, columns scaled by exp(gam_C - gam_s)
    float * P   = KTs + CS * D; // K S0, then reused
    float * QS  = P + CS * D;   // Q S0 (rows scaled by Gam)
    float * KK  = QS + CS * D;  // CS x CS
    float * QK  = KK + CS * CS;
    float * E   = QK + CS * CS;
    float * gam = E + CS * CS;
    float * Gam = gam + CS;
    float * bet = Gam + CS;
    float * wst = bet + CS;

    const size_t vl = __riscv_vsetvl_e32m4(D);

    gdn_transpose(D, D, s_in, D, S, D);  // S[i][j] = s_in[j][i]

    GGML_TENSOR_LOCALS(size_t, nbq, a.src_q, nb);
    GGML_TENSOR_LOCALS(size_t, nbk, a.src_k, nb);
    GGML_TENSOR_LOCALS(size_t, nbv, a.src_v, nb);
    GGML_TENSOR_LOCALS(size_t, nbg, a.src_g, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, a.src_beta, nb);

    for (int64_t t0 = 0; t0 < a.n_tokens; t0 += CS) {
        const int64_t Cn = std::min(CS, a.n_tokens - t0);

        float acc_g = 0.0f;
        for (int64_t r = 0; r < Cn; ++r) {
            const int64_t t = t0 + r;
            memcpy(Qc + r * D, (const char *) a.src_q->data + iq3 * nbq3 + t * nbq2 + iq1 * nbq1, D * sizeof(float));
            memcpy(Kc + r * D, (const char *) a.src_k->data + ik3 * nbk3 + t * nbk2 + ik1 * nbk1, D * sizeof(float));
            memcpy(U + r * D, (const char *) a.src_v->data + iv3 * nbv3 + t * nbv2 + iv1 * nbv1, D * sizeof(float));
            acc_g += *(const float *) ((const char *) a.src_g->data + iv3 * nbg3 + t * nbg2 + iv1 * nbg1);
            gam[r] = acc_g;
            Gam[r] = expf(acc_g);
            bet[r] = *(const float *) ((const char *) a.src_beta->data + iv3 * nbb3 + t * nbb2 + iv1 * nbb1);
        }
        for (int64_t r = 0; r < Cn; ++r) {
            for (int64_t c = 0; c <= r; ++c) {
                E[r * CS + c] = expf(gam[r] - gam[c]);
            }
            wst[r] = expf(gam[Cn - 1] - gam[r]);
        }

        gdn_transpose(Cn, D, Kc, D, KT, CS);                     // KT[i][r] = K[r][i]
        gdn_gemm(Cn, Cn, D, Kc, D, KT, CS, KK, CS, nullptr, 0, 0.0f);  // K K^T
        gdn_gemm(Cn, Cn, D, Qc, D, KT, CS, QK, CS, nullptr, 0, 0.0f);  // Q K^T
        gdn_gemm(Cn, D, D, Kc, D, S, D, P, D, nullptr, 0, 0.0f);       // K S0
        gdn_gemm(Cn, D, D, Qc, D, S, D, QS, D, nullptr, 0, 0.0f);      // Q S0

        // U_r = beta_r (v_r - Gam_r P_r) - sum_{s<r} beta_r E_rs KK_rs U_s
        for (int64_t r = 0; r < Cn; ++r) {
            vfloat32m4_t u = __riscv_vle32_v_f32m4(U + r * D, vl);
            u              = __riscv_vfnmsac_vf_f32m4(u, Gam[r], __riscv_vle32_v_f32m4(P + r * D, vl), vl);
            u              = __riscv_vfmul_vf_f32m4(u, bet[r], vl);
            for (int64_t c = 0; c < r; ++c) {
                const float l = bet[r] * E[r * CS + c] * KK[r * CS + c];
                u             = __riscv_vfnmsac_vf_f32m4(u, l, __riscv_vle32_v_f32m4(U + c * D, vl), vl);
            }
            __riscv_vse32_v_f32m4(U + r * D, u, vl);
        }

        // O = scale (Gam Q S0 + A U), A_rs = E_rs QK_rs for s <= r (zero above the diagonal)
        for (int64_t r = 0; r < Cn; ++r) {
            float * qkr = QK + r * CS;
            for (int64_t c = 0; c <= r; ++c) {
                qkr[c] *= E[r * CS + c];
            }
            for (int64_t c = r + 1; c < Cn; ++c) {
                qkr[c] = 0.0f;
            }
            float * qs = QS + r * D;
            __riscv_vse32_v_f32m4(qs, __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(qs, vl), Gam[r], vl), vl);
        }
        gdn_gemm(Cn, D, Cn, QK, CS, U, D, P, D, QS, D, 1.0f);
        for (int64_t r = 0; r < Cn; ++r) {
            __riscv_vse32_v_f32m4(attn_data + (t0 + r) * D * a.H,
                                  __riscv_vfmul_vf_f32m4(__riscv_vle32_v_f32m4(P + r * D, vl), a.scale, vl), vl);
        }

        // S = Gam_C S + KTs U
        for (int64_t i = 0; i < D; ++i) {
            for (int64_t c = 0; c < Cn; ++c) {
                KTs[i * CS + c] = KT[i * CS + c] * wst[c];
            }
        }
        gdn_gemm(D, D, Cn, KTs, CS, U, D, S, D, S, D, Gam[Cn - 1]);
    }

    gdn_transpose(D, D, S, D, s_out, D);
}

static void forward_gated_delta_net_impl(ggml::spacemit::context & ctx, ggml_tensor * op, float * state_rows,
                                         const ggml_tensor * snapshot_store) {
    const ggml_tensor * src_q     = op->src[0];
    const ggml_tensor * src_k     = op->src[1];
    const ggml_tensor * src_v     = op->src[2];
    const ggml_tensor * src_g     = op->src[3];
    const ggml_tensor * src_beta  = op->src[4];
    const ggml_tensor * src_state = op->src[5];

    const int64_t S_v      = src_v->ne[0];
    const int64_t H        = src_v->ne[1];
    const int64_t n_tokens = src_v->ne[2];
    const int64_t n_seqs   = src_v->ne[3];
    const int64_t K        = ggml_get_op_params_i32(op, 0);
    GGML_ASSERT(K >= 1 && S_v <= (int64_t) __riscv_vsetvlmax_e32m8());

    const int64_t nr  = H * n_seqs;
    const int64_t dr  = (nr + ctx.nth - 1) / ctx.nth;
    const int64_t ir0 = dr * ctx.ith;
    const int64_t ir1 = std::min(nr, ir0 + dr);

    const int64_t state_seq_stride = src_state->nb[3] / sizeof(float);
    const int64_t attn_score_elems = S_v * H * n_tokens * n_seqs;
    const int64_t state_size_per_snap = S_v * S_v * H * n_seqs;
    float *       attn_out_base  = (float *) op->data;
    // state_rows != nullptr: in-place update of the cache row (K == 1, one sequence). Every path below
    // reads a state row completely before it stores it, so s_in == s_out is safe.
    GGML_ASSERT(state_rows == nullptr || (K == 1 && n_seqs == 1));
    float *       state_out_base = state_rows ? state_rows : attn_out_base + attn_score_elems;
    const float * state_in_base  = state_rows ? state_rows : (const float *) src_state->data;

    const float scale = 1.0f / sqrtf((float) S_v);
    const bool  kda   = src_g->ne[0] == S_v;
    if (ocr_gdn_rollback_tcm::debug_enabled.load(std::memory_order_relaxed) && ctx.ith == 0) {
        fprintf(stderr, "[gdn] S_v=%lld H=%lld T=%lld K=%lld kda=%d vlmax_m4=%lld vlmax_m8=%lld\n",
                (long long) S_v, (long long) H, (long long) n_tokens, (long long) K, (int) kda,
                (long long) __riscv_vsetvlmax_e32m4(), (long long) __riscv_vsetvlmax_e32m8());
        fprintf(stderr, "[gdn] q.ne=[%lld,%lld,%lld,%lld] v.ne=[%lld,%lld,%lld,%lld]\n",
                (long long) src_q->ne[0], (long long) src_q->ne[1], (long long) src_q->ne[2], (long long) src_q->ne[3],
                (long long) src_v->ne[0], (long long) src_v->ne[1], (long long) src_v->ne[2], (long long) src_v->ne[3]);
    }

    // prefill: chunkwise form (the token recurrence below handles the remaining cases)
    if (K == 1 && !kda && n_tokens >= 16 && S_v <= (int64_t) __riscv_vsetvlmax_e32m4() &&
        src_q->type == GGML_TYPE_F32 && src_k->type == GGML_TYPE_F32 && src_v->type == GGML_TYPE_F32 &&
        src_g->type == GGML_TYPE_F32 && src_beta->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(src_q) &&
        ggml_is_contiguous_rows(src_k) && ggml_is_contiguous_rows(src_v) && ggml_is_contiguous(src_g) &&
        ggml_is_contiguous(src_beta) && src_beta->ne[0] == 1) {
        const int64_t        rq3c = n_seqs / src_q->ne[3];
        const int64_t        rk3c = n_seqs / src_k->ne[3];
        const gdn_chunk_args args = { src_q, src_k, src_v, src_g, src_beta, S_v, H, n_tokens, scale };
        float * gdn_scratch = nullptr;
        if (ocr_gdn_tcm::enabled() && ctx.shared.data != nullptr &&
            reinterpret_cast<uintptr_t>(ctx.shared.data) % alignof(float) == 0 &&
            ocr_gdn_tcm::scratch_bytes(S_v) <= ctx.shared.size) {
            gdn_scratch = static_cast<float *>(ctx.shared.data);
        }
        for (int64_t ir = ir0; ir < ir1; ++ir) {
            const int64_t iv1 = ir % H;
            const int64_t iv3 = ir / H;
            gdn_chunked_head(args, iv1 % src_q->ne[1], iv1 % src_k->ne[1], iv1, iv3 / rq3c, iv3 / rk3c, iv3,
                             state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v,
                             state_out_base + (iv3 * H + iv1) * S_v * S_v,
                             attn_out_base + (iv3 * n_tokens * H + iv1) * S_v, gdn_scratch);
        }
        return;
    }
    // Decode (n_tokens == 1, any n_seqs) and prefill tail chunks up to 4 tokens
    // run the fused single-pass row update: the state is read/written once per
    // token instead of once per phase, which halves the kernel time at
    // n_tokens == 1. Larger chunks keep the reference two-pass form — its early
    // state store keeps row loads pipelined, and with the chunk state L2
    // resident it already runs at the DRAM floor, so the fused form cannot win
    // there (measured crossover ~8 tokens; chunks are 64, or 16 with kda).
    const bool fused = n_tokens <= 4;
    // KDA needs the per-column decay vector for every state row; the fused
    // row update consumes it as a vector, so only this small buffer remains
    // of the reference implementation's scratch space.
    float kda_decay[512];
    GGML_ASSERT(S_v <= (int64_t) (sizeof(kda_decay) / sizeof(kda_decay[0])));
    // K > 1 (chunked prefill snapshots) evolves a private working copy.
    std::vector<float> state_work((fused && K > 1) ? (size_t) (S_v * S_v) : 0);
    // Two-pass scratch: per-row delta and attention sums, plus its own K > 1
    // working copy of the state.
    // Host admitted enough private pinned TCM for this exact K16 shape.
    // Every scratch float read below is fully written first. Arithmetic and
    // snapshot stores retain the original order; only storage is changed.
    float * rollback_tcm = ocr_gdn_rollback_tcm::workspace(op, ctx.shared.data, ctx.shared.size,
        unsigned(ctx.ith), unsigned(ctx.nth), ctx.check_tcm_view);
    std::vector<float> scratch(!fused && !rollback_tcm ?
        (size_t) (3 * S_v + (K > 1 ? S_v * S_v : 0)) : 0);
    float * delta      = !fused ? (rollback_tcm ? rollback_tcm : scratch.data()) : nullptr;
    float * decay      = !fused ? delta + S_v : nullptr;
    float * state_q    = !fused ? decay + S_v : nullptr;
    float * state_work_tp = (!fused && K > 1) ? state_q + S_v : nullptr;

    const int64_t rq3 = src_v->ne[3] / src_q->ne[3];
    const int64_t rk3 = src_v->ne[3] / src_k->ne[3];

    for (int64_t ir = ir0; ir < ir1; ++ir) {
        const int64_t iv1 = ir % H;
        const int64_t iv3 = ir / H;
        const int64_t iq1 = iv1 % src_q->ne[1];
        const int64_t ik1 = iv1 % src_k->ne[1];
        const int64_t iq3 = iv3 / rq3;
        const int64_t ik3 = iv3 / rk3;

        float * s_out = K > 1 ? (fused ? state_work.data() : state_work_tp)
                              : state_out_base + (iv3 * H + iv1) * S_v * S_v;
        const float * s_in = state_in_base + iv3 * state_seq_stride + iv1 * S_v * S_v;
        if (!fused && s_out != s_in) {
            memcpy(s_out, s_in, (size_t) (S_v * S_v) * sizeof(float));
        }
        // In the fused form the first token is fused with the state copy: it
        // reads the input state row and writes the updated row straight to the
        // output, so decode (n_tokens == 1) never materialises an intermediate
        // copy.
        const float * s_cur = s_in;

        float * attn_data = attn_out_base + (iv3 * n_tokens * H + iv1) * S_v;
        for (int64_t t = 0; t < n_tokens; ++t) {
            const float * q = (const float *) ((const char *) src_q->data + iq3 * src_q->nb[3] +
                                                t * src_q->nb[2] + iq1 * src_q->nb[1]);
            const float * k = (const float *) ((const char *) src_k->data + ik3 * src_k->nb[3] +
                                                t * src_k->nb[2] + ik1 * src_k->nb[1]);
            const float * v = (const float *) ((const char *) src_v->data + iv3 * src_v->nb[3] +
                                                t * src_v->nb[2] + iv1 * src_v->nb[1]);
            const float beta = *(const float *) ((const char *) src_beta->data + iv3 * src_beta->nb[3] +
                                                  t * src_beta->nb[2] + iv1 * src_beta->nb[1]);
            const float * g = (const float *) ((const char *) src_g->data + iv3 * src_g->nb[3] +
                                                t * src_g->nb[2] + iv1 * src_g->nb[1]);

            const float kq = rvv_gdn_dot_f32(k, q, S_v);

            if (fused) {
                float decay_scalar = 0.0f;
                if (kda) {
                    for (int64_t i = 0; i < S_v; ++i) kda_decay[i] = expf(g[i]);
                } else {
                    decay_scalar = expf(g[0]);
                }

                if (S_v <= (int64_t) __riscv_vsetvlmax_e32m4()) {
                    const size_t vl = __riscv_vsetvl_e32m4(S_v);
                    const vfloat32m4_t kv = __riscv_vle32_v_f32m4(k, vl);
                    const vfloat32m4_t qv = __riscv_vle32_v_f32m4(q, vl);
                    const vfloat32m4_t dv = kda ? __riscv_vle32_v_f32m4(kda_decay, vl)
                                              : __riscv_vfmv_v_f_f32m4(decay_scalar, vl);
                    for (int64_t j = 0; j < S_v; ++j) {
                        attn_data[j] = rvv_gdn_fused_row_f32(s_out + j * S_v, s_cur + j * S_v,
                                                           kv, qv, dv, v[j], beta, kq, scale, vl);
                    }
                } else {
                    const size_t vl = __riscv_vsetvl_e32m8(S_v);
                    const vfloat32m8_t kv = __riscv_vle32_v_f32m8(k, vl);
                    const vfloat32m8_t qv = __riscv_vle32_v_f32m8(q, vl);
                    const vfloat32m8_t dv = kda ? __riscv_vle32_v_f32m8(kda_decay, vl)
                                              : __riscv_vfmv_v_f_f32m8(decay_scalar, vl);
                    for (int64_t j = 0; j < S_v; ++j) {
                        attn_data[j] = rvv_gdn_fused_row_f32(s_out + j * S_v, s_cur + j * S_v,
                                                           kv, qv, dv, v[j], beta, kq, scale, vl);
                    }
                }
                s_cur = s_out;
            } else {
                float decay_scalar = 0.0f;
                if (kda) {
                    for (int64_t i = 0; i < S_v; ++i) decay[i] = expf(g[i]);
                } else {
                    decay_scalar = expf(g[0]);
                }

                for (int64_t j = 0; j < S_v; ++j) {
                    const auto dots = rvv_gdn_decay_dots_f32_impl(s_out + j * S_v, decay, decay_scalar, k, q, S_v, kda);
                    delta[j] = (v[j] - dots.state_k) * beta;
                    state_q[j] = dots.state_q;
                }
                for (int64_t j = 0; j < S_v; ++j) {
                    rvv_gdn_update_f32(s_out + j * S_v, k, delta[j], S_v);
                    attn_data[j] = (state_q[j] + delta[j] * kq) * scale;
                }
            }
            attn_data += S_v * H;

            if (K > 1) {
                const int64_t target_slot = n_tokens - 1 - t;
                if (target_slot >= 0 && target_slot < K) {
                    float * state_snapshot = snapshot_store != nullptr ?
                        ocr_gdn_snapshot_store::snapshot_pointer(snapshot_store->data, snapshot_store->nb[2], target_slot, iv1) :
                        state_out_base + target_slot * state_size_per_snap + (iv3 * H + iv1) * S_v * S_v;
                    memcpy(state_snapshot, s_out, (size_t) (S_v * S_v) * sizeof(float));
                }
            }
        }
    }
}

void forward_gated_delta_net(ggml::spacemit::context & ctx, ggml_tensor * op) {
    forward_gated_delta_net_impl(ctx, op, nullptr, nullptr);
}

void forward_gated_delta_net_inplace(ggml::spacemit::context & ctx, ggml_tensor * op, float * state_row) {
    forward_gated_delta_net_impl(ctx, op, state_row, nullptr);
}

void forward_gated_delta_net_snapshot_store(ggml::spacemit::context & ctx, ggml_tensor * op,
                                           const ggml_tensor * store) {
    GGML_ASSERT(ocr_gdn_snapshot_store::storage_eligible(op, store));
    forward_gated_delta_net_impl(ctx, op, nullptr, store);
}

}
