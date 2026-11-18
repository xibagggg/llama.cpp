#pragma once
// Diagnostic only: workers store timestamps into host-prefaulted lanes.
// All allocation, tensor inspection, aggregation and file I/O stay on the host.
#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ocr_op_stats {
inline const char * path() {
    static const char * value = std::getenv("GGML_SPACEMIT_OP_STATS_PATH");
    return value && *value ? value : nullptr;
}
inline uint64_t now() {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}
inline std::string quote(const char * value, bool normalize = false) {
    std::string result = "\"";
    bool digit = false;
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(value); *p; ++p) {
        if (normalize && std::isdigit(*p)) {
            if (!digit) result += '#';
            digit = true;
            continue;
        }
        digit = false;
        if (*p == '"' || *p == '\\') result += '\\';
        result += *p < 32 ? '_' : char(*p);
    }
    return result + '"';
}
struct timestamps { uint64_t begin = 0, compute = 0, barrier = 0; };
struct totals {
    uint64_t count = 0, critical_compute = 0, including_barrier = 0;
    uint64_t worker_compute_sum = 0, worker_wait_sum = 0, compute_max = 0;
    uint64_t start_skew = 0, max_one_op = 0;
    std::string example;
};
struct alignas(128) lane { std::vector<timestamps> items; };
class storage {
  public:
    std::array<lane, 6> lanes;
    std::map<std::string, totals> accumulated;
    std::map<std::string, std::array<uint64_t, 2>> phases;
    uint64_t completed_graphs = 0, failed_graphs = 0, invalid_nodes = 0, total_nodes = 0;
    unsigned workers = 0;
    uint64_t graph_begin = 0;
    std::string phase;

    static storage & get() {
        static storage * value = new storage;
        return *value;
    }
    storage() { std::atexit(dump); }
    template<class Nodes> void prepare(unsigned count, const Nodes & nodes) {
        if (count == 0 || count > lanes.size()) throw std::runtime_error("operator stats requires 1..6 workers");
        workers = count;
        for (unsigned w = 0; w < workers; ++w) lanes[w].items.assign(nodes.size(), timestamps{});
        phase = "vision_aux";
        for (const auto & n : nodes) {
            const auto * t = n.node;
            if (!std::strncmp(t->name, "attn_norm-", 10) || !std::strncmp(t->name, "ffn_gate-", 9)) {
                phase = t->ne[1] > 1 ? "prefill" : "decode";
                break;
            }
            if (t->op == GGML_OP_GET_ROWS && !std::strcmp(t->name, "model.input_embed")) phase = "embedding";
        }
        graph_begin = now();
    }
    void record(unsigned worker, size_t index, uint64_t begin, uint64_t compute, uint64_t barrier) {
        // The graph has an immutable node count; each lane/index has one writer.
        if (worker < workers && index < lanes[worker].items.size())
            lanes[worker].items[index] = {begin, compute, barrier};
    }
    template<class Nodes> void finish(const Nodes & nodes, bool succeeded) {
        const uint64_t graph_end = now();
        if (!succeeded) { ++failed_graphs; return; }
        ++completed_graphs;
        auto & phase_total = phases[phase];
        ++phase_total[0]; phase_total[1] += graph_end - graph_begin;
        size_t index = 0;
        for (const auto & n : nodes) {
            uint64_t first = std::numeric_limits<uint64_t>::max(), last_start = 0, last_compute = 0, last_barrier = 0;
            uint64_t compute_sum = 0, wait_sum = 0, max_compute = 0;
            bool valid = true;
            for (unsigned w = 0; w < workers; ++w) {
                const auto & x = lanes[w].items[index];
                valid &= x.begin && x.begin <= x.compute && x.compute <= x.barrier;
                first = std::min(first, x.begin); last_start = std::max(last_start, x.begin);
                last_compute = std::max(last_compute, x.compute); last_barrier = std::max(last_barrier, x.barrier);
                compute_sum += x.compute - x.begin; wait_sum += x.barrier - x.compute;
                max_compute = std::max(max_compute, x.compute - x.begin);
            }
            ++index;
            if (!valid) { ++invalid_nodes; continue; }
            const auto * t = n.node;
            std::ostringstream key;
            key << "\"phase\":" << quote(phase.c_str()) << ",\"name\":" << quote(t->name, true)
                << ",\"op\":" << quote(ggml_op_name(t->op)) << ",\"opcode\":" << int(n.opcode)
                << ",\"dst_type\":" << quote(ggml_type_name(t->type)) << ",\"dst_shape\":[";
            for (unsigned d = 0; d < 4; ++d) key << (d ? "," : "") << t->ne[d];
            key << ']';
            for (unsigned src = 0; src < 2; ++src) {
                const auto * in = t->src[src];
                key << ",\"type" << src << "\":" << quote(in ? ggml_type_name(in->type) : "none")
                    << ",\"shape" << src << "\":[";
                for (unsigned d = 0; d < 4; ++d) key << (d ? "," : "") << (in ? in->ne[d] : 0);
                key << ']';
            }
            auto & value = accumulated[key.str()];
            if (!value.count) value.example = t->name;
            ++value.count; value.critical_compute += last_compute-first;
            value.including_barrier += last_barrier-first;
            value.worker_compute_sum += compute_sum; value.worker_wait_sum += wait_sum;
            value.compute_max += max_compute; value.start_skew += last_start-first;
            value.max_one_op = std::max(value.max_one_op, last_barrier-first);
            ++total_nodes;
        }
    }
    static void dump() {
        auto & s = get();
        FILE * file = std::fopen(path(), "w");
        if (!file) return;
        std::fprintf(file, "{\"kind\":\"summary\",\"workers\":%u,\"completed_graphs\":%llu,\"failed_graphs\":%llu,\"invalid_nodes\":%llu,\"total_nodes\":%llu,\"groups\":%zu,\"scope\":\"all nodes of completed graphs; six-lane timestamp aggregation on host; normalized numeric name components; exact tensor shapes; graph wall excludes host aggregation\"}\n",
            s.workers, (unsigned long long)s.completed_graphs, (unsigned long long)s.failed_graphs,
            (unsigned long long)s.invalid_nodes, (unsigned long long)s.total_nodes, s.accumulated.size());
        for (const auto & p : s.phases) std::fprintf(file,
            "{\"kind\":\"phase\",\"phase\":%s,\"graphs\":%llu,\"graph_wall_ns\":%llu}\n",
            quote(p.first.c_str()).c_str(), (unsigned long long)p.second[0], (unsigned long long)p.second[1]);
        for (const auto & p : s.accumulated) {
            const auto & v = p.second;
            std::fprintf(file, "{\"kind\":\"op\",%s,\"example\":%s,\"count\":%llu,\"critical_compute_ns\":%llu,\"including_barrier_ns\":%llu,\"worker_compute_sum_ns\":%llu,\"worker_wait_sum_ns\":%llu,\"max_worker_compute_sum_ns\":%llu,\"start_skew_sum_ns\":%llu,\"max_one_op_ns\":%llu}\n",
                p.first.c_str(),quote(v.example.c_str()).c_str(),(unsigned long long)v.count,
                (unsigned long long)v.critical_compute,(unsigned long long)v.including_barrier,
                (unsigned long long)v.worker_compute_sum,(unsigned long long)v.worker_wait_sum,
                (unsigned long long)v.compute_max,(unsigned long long)v.start_skew,(unsigned long long)v.max_one_op);
        }
        std::fclose(file);
    }
};
} // namespace ocr_op_stats
