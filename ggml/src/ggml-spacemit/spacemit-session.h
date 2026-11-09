#ifndef SPACEMIT_SESSION_H
#define SPACEMIT_SESSION_H

#include "spacemit-opnode.h"
#include "spacemit-context.h"
#include "spacemit-resident-scratch.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

struct spacemit_session {
    uint32_t num_cores = 0;
    int64_t  arch_id   = 0;
    int64_t  vlen      = 0;
    int64_t  tcm_size  = 0;
    bool     use_ime1  = false;
    bool     use_ime2  = false;

    std::string name = "SPACEMIT0";

    // Serialize graph_compute calls because the hardware stream is exclusive.
    std::mutex stream_mutex;

    // Opt-in diagnostic sequence for no-callback operator output hashes.
    uint64_t op_hash_trace_sequence = 0;

    void * workspace = nullptr;
    size_t workspace_size = 0;
    std::vector<ggml::spacemit::context> contexts;
    ggml::spacemit::resident_scratch ddr_scratch;
    // Graph calls on this session are serialized by stream_mutex. The counter
    // is reset and synchronized before each dynamic attention dispatch.
    std::atomic<int64_t> attention_work_queue{0};

    struct {
        uint64_t uid = 0;
        std::vector<spacemit_opnode> nodes;
    } cached_graph;

    spacemit_session() = default;
    ~spacemit_session();

    const char * c_name() const { return name.c_str(); }
};

#endif // SPACEMIT_SESSION_H
