#pragma once

// Request opt-in policy for a single DSpark context. No model/cache operations
// occur here; the caller must clear both contexts before committing a plan.
namespace server_request_spec {
struct capabilities {
    int parallel = 0;
    int completions = 0;
    bool completion = false;
    bool child = false;
    bool parent = false;
    bool shared_prompt = false;
    bool has_draft = false;
    bool dspark_only = false;
};
struct plan {
    bool managed = false;
    bool enabled = false;
    bool clear_contexts = false;
    const char * error = nullptr;
};
inline int parse_mode(bool is_boolean, bool enabled) {
    return is_boolean ? int(enabled) : -2;
}
inline plan resolve(int mode, bool was_managed, bool global_enabled, const capabilities & cap) {
    plan out;
    out.managed = was_managed || mode != -1;
    out.enabled = global_enabled && mode != 0;
    out.clear_contexts = out.managed;
    if (!out.managed) return out; // preserve the existing global behavior
    if (mode < -1 || mode > 1) out.error = "speculative.enabled must be a boolean";
    else if (cap.parallel != 1) out.error = "speculative.enabled requires parallel=1";
    else if (cap.completions != 1 || !cap.completion || cap.child || cap.parent || cap.shared_prompt)
        out.error = "speculative.enabled requires one completion without shared/child prompts";
    else if (!global_enabled || !cap.has_draft || !cap.dspark_only)
        out.error = "speculative.enabled requires exactly one initialized DSpark implementation without synthetic acceptance";
    return out;
}
template<class Slot> inline void clear_candidates(Slot & slot) {
    slot.spec_draft.clear();
    slot.spec_draft_q.clear();
    slot.spec_i_batch.clear();
    slot.spec_ckpt.clear();
    slot.spec_prompt.clear();
    slot.spec_is_replay = false;
}
} // namespace server_request_spec
