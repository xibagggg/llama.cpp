#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-alloc.h"

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <memory>
#include <string>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

// Linux/K3 only. Each invocation executes one option combination in a fresh
// child process: the backend caches option values, and phase traces are dumped
// by its atexit handler. The parent validates that trace after child exit.
// This is a numerical/path regression; its instrumented timings are not a
// benchmark. No engine-private header or CPU/backend implementation is changed.
namespace {
constexpr int D = 72, H = 16, STEPS = 2;
constexpr uint64_t MAGIC = UINT64_C(0x4131504149524249); // A1PAIRBI
constexpr uint64_t VERSION = 1;
constexpr uint64_t INPUT_VERSION = 1;
constexpr uint64_t FNV_OFFSET = UINT64_C(14695981039346656037);
constexpr uint64_t FNV_PRIME = UINT64_C(1099511628211);
constexpr double MAX_RELATIVE = 0.002, MAX_ABS = 0.002;

const char * const option_names[] = {
    "GGML_SPACEMIT_OCR_FA_FULL_PACK",
    "GGML_SPACEMIT_OCR_FA_LEAN_PREPARE",
    "GGML_SPACEMIT_OCR_FA_FUSED_QUERY",
    "GGML_SPACEMIT_OCR_FA_HALF_PACK",
    "GGML_SPACEMIT_OCR_FA_SOFTMAX_PACK",
    "GGML_SPACEMIT_OCR_FA_COMPACT",
    "GGML_SPACEMIT_OCR_FA_HEAD_K_TCM_2PASS",
    "GGML_SPACEMIT_OCR_FA_Q32",
    "GGML_SPACEMIT_OCR_FA_FIXED_INDEX",
    "GGML_SPACEMIT_OCR_FA_P_STORE64",
    "GGML_SPACEMIT_OCR_FA_QK_SCATTER",
};

enum class scenario : int { regular = 0, reject_k = 1, reject_v = 2, reject_q = 3 };
struct test_case {
    int n;
    int layout;       // 0: [D,N,H], 1: [D,H,N] -> permute [D,N,H]
    scenario kind;
    bool invalid_first;
};

std::vector<test_case> make_cases() {
    std::vector<test_case> result;
    for (int n : {63,64,65,848,1120,1121,1496,1500,1504,1536,1537}) {
        for (int layout = 0; layout < 2; ++layout) {
            result.push_back({n, layout, scenario::regular, false});
        }
    }
    // Whole-head rejection forces the fallback to rebuild from current source
    // tensors when LEAN_PREPARE deliberately skipped the old shared K/V packs.
    // Both valid -> invalid and invalid -> valid run in one graph/allocation.
    for (int invalid_n : {848,1500}) for (scenario kind : {scenario::reject_k, scenario::reject_v, scenario::reject_q}) {
        for (int layout = 0; layout < 2; ++layout) {
            for (bool invalid_first : {false, true}) {
                result.push_back({invalid_n, layout, kind, invalid_first});
            }
        }
    }
    return result;
}

bool invalid_step(const test_case & c, int step) {
    return c.kind != scenario::regular && (c.invalid_first ? step == 0 : step == 1);
}

std::string op_name(size_t id, const test_case & c, int step) {
    char name[64];
    std::snprintf(name, sizeof(name), "pack_c%02zu_n%d_l%d_s%d", id, c.n, c.layout, step);
    return name;
}

uint64_t hash_bytes(uint64_t hash, const void * ptr, size_t size) {
    const auto * bytes = static_cast<const uint8_t *>(ptr);
    for (size_t i = 0; i < size; ++i) {
        hash = (hash ^ bytes[i]) * FNV_PRIME;
    }
    return hash;
}

uint64_t hash_inputs(const std::vector<float> & q, const std::vector<ggml_fp16_t> & k,
                     const std::vector<ggml_fp16_t> & v) {
    uint64_t hash = hash_bytes(FNV_OFFSET, q.data(), q.size()*sizeof(float));
    hash = hash_bytes(hash, k.data(), k.size()*sizeof(ggml_fp16_t));
    return hash_bytes(hash, v.data(), v.size()*sizeof(ggml_fp16_t));
}

uint32_t float_bits(float value) {
    uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

// Versioned deterministic integer generator; no implementation-defined
// standard-library uniform distribution enters fixture identity.
uint32_t random_u32(uint32_t & state) {
    state ^= state << 13; state ^= state >> 17; state ^= state << 5;
    return state;
}

void make_inputs(const test_case & c, int step, std::vector<float> & q,
                 std::vector<ggml_fp16_t> & k, std::vector<ggml_fp16_t> & v) {
    uint32_t state = 0x176a9e7fu ^ uint32_t(c.n*71) ^ uint32_t(step*0x1793) ^
                     uint32_t(int(c.kind)*0x5a17) ^ uint32_t(c.invalid_first*0x15a3);
    for (size_t i = 0; i < q.size(); ++i) {
        q[i] = float(int(random_u32(state) % 131073u) - 65536) * 0x1p-15f;
        k[i] = ggml_fp32_to_fp16(float(int(random_u32(state) % 131073u) - 65536) * 0x1p-15f);
        v[i] = ggml_fp32_to_fp16(float(int(random_u32(state) % 65537u) - 32768) * 0x1p-15f);
    }
    // Signed zero, minimum/max FP16 subnormal and minimum normal survive the
    // actual attention graph. Keep values finite and within the pack bounds.
    const uint16_t edges[] = {0x0000,0x8000,0x0001,0x8001,0x03ff,0x83ff,0x0400,0x8400};
    for (size_t i = 0; i < q.size(); i += 97) {
        const auto h = ggml_fp16_t(edges[(i/97 + size_t(step)) % 8]);
        k[i] = h; v[i] = h;
        q[i] = ggml_fp16_to_fp32(h);
    }
    // Preserve an exact zero query row in step 0 and replace it in step 1.
    // This also exposes incomplete writes and stale packed-query caching.
    if (step == 0) {
        for (int head = 0; head < H; ++head) {
            std::fill_n(q.begin() + size_t(head)*c.n*D, D, 0.f);
        }
    }
    if (c.kind != scenario::regular) {
        // A rejected Q/K coordinate makes no mathematical contribution, so
        // the existing FP64 accuracy threshold remains meaningful at >63.
        for (int row = 0; row < c.n; ++row) {
            if (c.kind == scenario::reject_k) q[(size_t(H-1)*c.n+row)*D+D-1] = 0.f;
            if (c.kind == scenario::reject_q) k[(size_t(H-1)*c.n+row)*D+D-1] = ggml_fp32_to_fp16(0.f);
        }
        if (invalid_step(c, step)) {
            const size_t last = (size_t(H)*c.n)*D-1;
            // Exactly the next finite FP16 value above the accepted |h|<=63.
            const auto outside = ggml_fp32_to_fp16(63.03125f);
            if (c.kind == scenario::reject_k) k[last] = outside;
            if (c.kind == scenario::reject_v) v[last] = outside;
            if (c.kind == scenario::reject_q) q[last] = 64.f * std::sqrt(float(D));
        }
    }
}

// Canonical logical inputs are [H,N,D] regardless of backing layout. A real
// permute is filled through its backing bytes using tensor nb[], never by
// pretending the strided view is canonical contiguous storage.
template<class T>
std::vector<uint8_t> physical_bytes(const ggml_tensor * view, const ggml_tensor * backing,
                                  const std::vector<T> & logical) {
    std::vector<uint8_t> result(ggml_nbytes(backing), 0xa5);
    for (int head = 0; head < H; ++head) {
        for (int row = 0; row < int(view->ne[1]); ++row) {
            for (int d = 0; d < D; ++d) {
                const size_t offset = size_t(head)*view->nb[2] + size_t(row)*view->nb[1] + size_t(d)*view->nb[0];
                const size_t index = (size_t(head)*view->ne[1]+row)*D+d;
                if (offset + sizeof(T) > result.size()) std::abort();
                std::memcpy(result.data()+offset, &logical[index], sizeof(T));
            }
        }
    }
    return result;
}

bool unchanged(const ggml_tensor * backing, const std::vector<uint8_t> & expected, const char * label) {
    std::vector<uint8_t> actual(expected.size());
    ggml_backend_tensor_get(backing, actual.data(), 0, actual.size());
    if (actual != expected) {
        std::fprintf(stderr, "%s input backing was modified\n", label);
        return false;
    }
    return true;
}

// The fixture is endian-tagged, contains an immutable suite/arithmetical
// identity, then per-record shape/layout/step/nb[] and input-hash metadata.
// It cannot accidentally accept a different input generator or record order.
bool fixture_header(FILE * file, bool write, size_t count) {
    const std::array<uint64_t, 12> expected = {{MAGIC, VERSION, 0x01020304u,
        sizeof(float), sizeof(ggml_fp16_t), D, H, STEPS, count, INPUT_VERSION,
        uint64_t(float_bits(1.f/std::sqrt(float(D)))), 2047 /* all eleven production packing flags fixed on */}};
    auto actual = expected;
    if (write) return std::fwrite(expected.data(), sizeof(expected), 1, file) == 1;
    if (std::fread(actual.data(), sizeof(actual), 1, file) != 1 || actual != expected) {
        std::fprintf(stderr, "Fixture suite/header differs or is truncated\n");
        return false;
    }
    return true;
}

bool fixture_record(FILE * file, bool write, size_t id, const test_case & c, int step,
                    const ggml_tensor * q, const ggml_tensor * k, const ggml_tensor * v,
                    uint64_t input_hash, const std::vector<float> & output, double relative, double max_abs) {
    std::array<uint64_t, 25> expected{};
    size_t m = 0;
    for (uint64_t field : {MAGIC, VERSION, uint64_t(id), uint64_t(c.n), uint64_t(c.layout),
                          uint64_t(c.kind), uint64_t(c.invalid_first), uint64_t(step),
                          uint64_t(invalid_step(c, step)), input_hash, uint64_t(output.size()),
                          uint64_t(D), uint64_t(H)}) expected[m++] = field;
    for (const auto * tensor : {q,k,v}) for (size_t stride : tensor->nb) expected[m++] = stride;
    if (m != expected.size()) std::abort();
    if (write) {
        return std::fwrite(expected.data(), sizeof(expected), 1, file) == 1 &&
               std::fwrite(output.data(), sizeof(float), output.size(), file) == output.size() &&
               std::fwrite(&relative,sizeof(relative),1,file)==1 &&
               std::fwrite(&max_abs,sizeof(max_abs),1,file)==1;
    }
    auto actual = expected;
    if (std::fread(actual.data(), sizeof(actual), 1, file) != 1 || actual != expected) {
        std::fprintf(stderr, "Fixture record metadata differs: case=%zu N=%d layout=%d step=%d\n", id,c.n,c.layout,step);
        return false;
    }
    std::vector<float> wanted(output.size());
    if (std::fread(wanted.data(), sizeof(float), wanted.size(), file) != wanted.size()) {
        std::fprintf(stderr, "Fixture record truncated: case=%zu step=%d\n", id,step);
        return false;
    }
    double reference_relative=0,reference_max_abs=0;
    if(std::fread(&reference_relative,sizeof(reference_relative),1,file)!=1||
       std::fread(&reference_max_abs,sizeof(reference_max_abs),1,file)!=1)return false;
    // A1 must equal frozen v3 for every shape; no tolerance exception.
    if(!std::isfinite(reference_relative)||!std::isfinite(reference_max_abs))return false;
    for (size_t i = 0; i < output.size(); ++i) {
        if (float_bits(wanted[i]) != float_bits(output[i])) {
            const size_t d = i % D, head = (i/D) % H, row = i/(D*H);
            std::fprintf(stderr, "F32 bits differ: case=%zu N=%d layout=%d step=%d row=%zu head=%zu d=%zu got=0x%08x want=0x%08x\n",
                id,c.n,c.layout,step,row,head,d,float_bits(output[i]),float_bits(wanted[i]));
            return false;
        }
    }
    return true;
}

bool sampled_fp64(int n, const std::vector<float> & q, const std::vector<ggml_fp16_t> & k,
                  const std::vector<ggml_fp16_t> & v, const std::vector<float> & output,
                  double & relative, double & max_abs) {
    const float scale = 1.f/std::sqrt(float(D));
    double sqerr = 0, sqref = 0;
    max_abs = 0;
    // Keep exactly the established 16 heads x three sampled rows and thresholds.
    for (int head = 0; head < H; ++head) for (int row : {0,n/2,n-1}) {
        std::vector<double> scores(n), reference(D, 0.);
        for (int token = 0; token < n; ++token) {
            double dot = 0;
            for (int d = 0; d < D; ++d) {
                dot += double(q[(size_t(head)*n+row)*D+d]) *
                       ggml_fp16_to_fp32(k[(size_t(head)*n+token)*D+d]);
            }
            scores[token] = dot * scale;
        }
        const double maximum = *std::max_element(scores.begin(), scores.end());
        double sum = 0;
        for (double & score : scores) { score = std::exp(score-maximum); sum += score; }
        for (int token = 0; token < n; ++token) for (int d = 0; d < D; ++d) {
            reference[d] += scores[token] * ggml_fp16_to_fp32(v[(size_t(head)*n+token)*D+d]) / sum;
        }
        for (int d = 0; d < D; ++d) {
            const double error = output[(size_t(row)*H+head)*D+d]-reference[d];
            sqerr += error*error; sqref += reference[d]*reference[d];
            max_abs = std::max(max_abs, std::abs(error));
        }
    }
    relative = std::sqrt(sqerr/std::max(sqref,1e-30));
    return std::isfinite(relative) && std::isfinite(max_abs) &&
           relative < MAX_RELATIVE && max_abs < MAX_ABS;
}

int worker(bool write, const char * path) {
    // O_EXCL is intentional. Never silently replace a previous reference,
    // including after a failed run; retain it and choose a fresh path.
    const int fd = write ? open(path, O_WRONLY|O_CREAT|O_EXCL, 0644) : open(path, O_RDONLY);
    if (fd < 0) { std::perror(path); return 2; }
    std::unique_ptr<FILE, decltype(&std::fclose)> file(fdopen(fd, write ? "wb" : "rb"), &std::fclose);
    if (!file) { close(fd); std::perror("fdopen"); return 2; }
    const auto cases = make_cases();
    if (!fixture_header(file.get(), write, cases.size())) return 2;
    ggml_backend_load_all();
    const auto device = ggml_backend_dev_by_name("SPACEMIT0");
    if (!device) { std::fprintf(stderr,"SPACEMIT0 device missing\n"); return 3; }
    const auto backend = ggml_backend_dev_init(device, nullptr);
    if (!backend) return 3;
    bool ok = true;
    uint64_t output_count = 0;
    for (size_t id = 0; id < cases.size(); ++id) {
        const auto & c = cases[id];
        const auto ctx = ggml_init({32*1024*1024, nullptr, true});
        if (!ctx) return 4;
        auto * q = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, D,c.n,H);
        auto * kb = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D,c.layout ? H : c.n,c.layout ? c.n : H);
        auto * vb = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, D,c.layout ? H : c.n,c.layout ? c.n : H);
        auto * k = c.layout ? ggml_permute(ctx,kb,0,2,1,3) : kb;
        auto * v = c.layout ? ggml_permute(ctx,vb,0,2,1,3) : vb;
        auto * y = ggml_flash_attn_ext(ctx,q,k,v,nullptr,1.f/std::sqrt(float(D)),0,0);
        ggml_set_input(q); ggml_set_input(kb); ggml_set_input(vb); ggml_set_output(y);
        if (k->nb[0] != 2 || v->nb[0] != 2 || k->ne[1] != c.n || k->ne[2] != H ||
            (c.layout && (k->nb[1] != D*H*2 || k->nb[2] != D*2 || ggml_is_contiguous(k)))) {
            std::fprintf(stderr,"Unexpected K/V physical layout\n"); return 4;
        }
        if (!ggml_backend_supports_op(backend,y)) {
            std::fprintf(stderr,"SPACEMIT does not support case=%zu N=%d layout=%d\n",id,c.n,c.layout);
            return 4;
        }
        auto * graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph,y);
        const auto buffer = ggml_backend_alloc_ctx_tensors(ctx,backend);
        if (!buffer) return 4;
        const std::array<const void *,4> addresses = {{q->data,kb->data,vb->data,y->data}};
        const size_t elements = size_t(D)*H*c.n;
        std::vector<float> qs(elements), output(elements), previous;
        std::vector<ggml_fp16_t> ks(elements), vs(elements);
        uint64_t previous_hash = 0;
        for (int step = 0; step < STEPS; ++step) {
            make_inputs(c,step,qs,ks,vs);
            const uint64_t input_hash = hash_inputs(qs,ks,vs);
            const auto qp = physical_bytes(q,q,qs), kp = physical_bytes(k,kb,ks), vp = physical_bytes(v,vb,vs);
            if (step && input_hash == previous_hash) { std::fprintf(stderr,"Input did not change\n"); return 7; }
            previous_hash = input_hash;
            ggml_backend_tensor_set(q,qp.data(),0,qp.size());
            ggml_backend_tensor_set(kb,kp.data(),0,kp.size());
            ggml_backend_tensor_set(vb,vp.data(),0,vp.size());
            // Every output starts as a quiet NaN, so missing tail/head stores
            // cannot accidentally pass because a previous result was finite.
            std::fill(output.begin(),output.end(),std::nanf("176"));
            ggml_backend_tensor_set(y,output.data(),0,output.size()*sizeof(float));
            const std::string name = op_name(id,c,step);
            ggml_set_name(y,name.c_str());
            const auto begin = std::chrono::steady_clock::now();
            const auto status = ggml_backend_graph_compute(backend,graph);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now()-begin).count();
            if (status != GGML_STATUS_SUCCESS) return 5;
            const std::array<const void *,4> current = {{q->data,kb->data,vb->data,y->data}};
            if (current != addresses) { std::fprintf(stderr,"Tensor addresses changed between steps\n"); return 7; }
            ggml_backend_tensor_get(y,output.data(),0,output.size()*sizeof(float));
            bool finite = std::all_of(output.begin(),output.end(),[](float a){return std::isfinite(a);});
            const bool inputs_same = unchanged(q,qp,"Q") && unchanged(kb,kp,"K") && unchanged(vb,vp,"V");
            const bool changed = !step || std::memcmp(output.data(),previous.data(),output.size()*sizeof(float)) != 0;
            double relative = 0, max_abs = 0;
            const bool accuracy = sampled_fp64(c.n,qs,ks,vs,output,relative,max_abs);
            const bool same = fixture_record(file.get(),write,id,c,step,q,k,v,input_hash,output,relative,max_abs);
            const bool pass = finite && inputs_same && changed && accuracy && same;
            ok = ok && pass;
            std::printf("case=%zu N=%d layout=%s scenario=%d invalid_first=%d step=%d invalid=%d input_hash=%016llx relative_l2=%.9g max_abs=%.9g finite=%d inputs_unchanged=%d changed=%d reference=%s seconds_instrumented=%.6f %s\n",
                id,c.n,c.layout?"permuted":"contiguous",int(c.kind),int(c.invalid_first),step,int(invalid_step(c,step)),
                static_cast<unsigned long long>(input_hash),relative,max_abs,int(finite),int(inputs_same),int(changed),
                write?"write":same?"identical":"different",seconds,pass?"PASS":"FAIL");
            std::fflush(stdout);
            previous = output;
            output_count += output.size();
        }
        ggml_backend_buffer_free(buffer); ggml_free(ctx);
    }
    const std::array<uint64_t,3> footer = {{MAGIC,cases.size()*STEPS,output_count}};
    auto actual_footer = footer;
    // A failed writer retains its evidence but never creates a valid footer.
    // A later compare therefore cannot accept a numerically failed fixture.
    const bool io = write ? (ok && std::fwrite(footer.data(),sizeof(footer),1,file.get()) == 1 && std::fflush(file.get()) == 0) :
        (std::fread(actual_footer.data(),sizeof(actual_footer),1,file.get()) == 1 && actual_footer == footer &&
         std::fgetc(file.get()) == EOF && !std::ferror(file.get()));
    ggml_backend_free(backend);
    std::printf("suite cases=%zu executions=%zu output_f32=%llu fixture_io=%d %s\n",cases.size(),cases.size()*STEPS,
        static_cast<unsigned long long>(output_count),int(io),ok&&io?"PASS":"FAIL");
    return ok&&io ? 0 : 1;
}

// A small parser for the exact diagnostic JSONL format emitted by the existing
// backend. Unknown/truncated lines fail closed; this is not a general JSON API.
bool string_field(const std::string & line, const char * field, std::string & value) {
    const std::string key = std::string("\"")+field+"\":\"";
    const auto begin = line.find(key);
    if (begin == std::string::npos) return false;
    const auto first = begin+key.size(), end = line.find('"',first);
    if (end == std::string::npos) return false;
    value = line.substr(first,end-first);
    return true;
}

bool integer_field(const std::string & line, const char * field, uint64_t & value) {
    const std::string key = std::string("\"")+field+"\":";
    const auto begin = line.find(key);
    if (begin == std::string::npos) return false;
    const char * start = line.c_str()+begin+key.size();
    char * end = nullptr; errno = 0;
    const unsigned long long parsed = std::strtoull(start,&end,10);
    if (errno || end == start || (*end != ',' && *end != '}')) return false;
    value = uint64_t(parsed); return true;
}

bool verify_phases(const std::string & directory) {
    const char * configured_workers=std::getenv("GGML_SPACEMIT_WORKERS");
    if(!configured_workers||std::strcmp(configured_workers,"6"))return false;
    constexpr uint64_t workers=6;
    std::ifstream file(directory+"/attention-phases.jsonl");
    if (!file) { std::fprintf(stderr,"Missing attention phase trace in %s\n",directory.c_str()); return false; }
    struct counts { uint64_t prepare=0,fullkv=0,headcache=0,compute=0,serial=0; };
    std::map<std::string,counts> observed;
    const auto cases = make_cases();
    std::map<std::string,int> expected_n;
    for (size_t id=0;id<cases.size();++id) for (int step=0;step<STEPS;++step) expected_n[op_name(id,cases[id],step)] = cases[id].n;
    std::array<bool,8> lanes{};
    std::array<uint64_t,8> used{},records{};
    std::string line;
    bool ok = true;
    while (std::getline(file,line)) {
        std::string kind;
        uint64_t lane=0;
        if (!string_field(line,"kind",kind) || !integer_field(line,"worker",lane) || lane >= lanes.size()) return false;
        if (kind == "lane") {
            uint64_t dropped=0;
            if (lanes[lane] || !integer_field(line,"used",used[lane]) || !integer_field(line,"dropped",dropped) || dropped) return false;
            lanes[lane] = true;
        } else if (kind == "phase") {
            std::string name,stage; uint64_t n=0,start=0,end=0;
            if (!string_field(line,"name",name) || !string_field(line,"stage",stage) ||
                !integer_field(line,"N",n) || !integer_field(line,"start_ns",start) || !integer_field(line,"end_ns",end) ||
                end < start || !expected_n.count(name) || uint64_t(expected_n[name]) != n || lane >= workers) return false;
            auto & count = observed[name];
            if (stage == "prepare") ++count.prepare;
            else if (stage == "fullkv") ++count.fullkv;
            else if (stage == "fullkv-k2") { ++count.fullkv; ++count.headcache; }
            else if (stage == "fullkv-s3") { ++count.fullkv; ++count.serial; }
            else if (stage == "compute") ++count.compute;
            else return false;
            ++records[lane];
        } else return false;
    }
    if (!file.eof()) return false;
    for (size_t lane=0;lane<lanes.size();++lane) if (!lanes[lane] || used[lane] != records[lane]) ok = false;
    for (size_t id=0;id<cases.size();++id) for (int step=0;step<STEPS;++step) {
        const auto & c = cases[id];
        const auto name = op_name(id,c,step);
        const auto count = observed[name];
        const uint64_t tasks = uint64_t(H)*uint64_t((c.n+511)/512);
        uint64_t prepare=workers,fullkv=tasks,compute=0;
        const char *serial_value=std::getenv("GGML_SPACEMIT_OCR_FA_SERIAL_P2");
        const bool serial=serial_value&&(!std::strcmp(serial_value,"1")||!std::strcmp(serial_value,"2"))&&c.n>1120&&c.n<=1536;
        if (c.n < 64) { prepare=0; fullkv=0; }
        else if (c.n > (serial?1536:1120)) { fullkv=0; compute=tasks; }
        else if (invalid_step(c,step)) {
            if(serial){compute=tasks;fullkv=0;}
            else {compute=c.kind==scenario::reject_q?1:uint64_t((c.n+511)/512);fullkv=tasks-compute;}
        }
        const char * flag = std::getenv("GGML_SPACEMIT_OCR_FA_HEAD_K_TCM_2PASS");
        const bool cache = !serial && flag && !std::strcmp(flag,"1") && c.n>=345 && c.n<=936;
        const uint64_t expected_cache = cache ? fullkv : 0;
        const uint64_t expected_serial=serial?fullkv:0;
        const bool pass = count.prepare==prepare && count.fullkv==fullkv && count.compute==compute &&
            count.headcache==expected_cache && count.serial==expected_serial;
        std::printf("serial_path name=%s hits=%llu expected=%llu %s\n",name.c_str(),
            static_cast<unsigned long long>(count.serial),static_cast<unsigned long long>(expected_serial),pass?"PASS":"FAIL");
        std::printf("headcache name=%s hits=%llu expected=%llu\n",name.c_str(),
            static_cast<unsigned long long>(count.headcache),static_cast<unsigned long long>(expected_cache));
        ok = ok && pass;
        std::printf("path name=%s prepare=%llu/%llu fullkv=%llu/%llu compute=%llu/%llu %s\n",name.c_str(),
            static_cast<unsigned long long>(count.prepare),static_cast<unsigned long long>(prepare),
            static_cast<unsigned long long>(count.fullkv),static_cast<unsigned long long>(fullkv),
            static_cast<unsigned long long>(count.compute),static_cast<unsigned long long>(compute),pass?"PASS":"FAIL");
    }
    std::printf("phase_trace=%s/attention-phases.jsonl %s\n",directory.c_str(),ok?"PASS":"FAIL");
    return ok;
}

bool configure(bool write, unsigned & option_mask) {
    option_mask = 0;
    for (size_t i=0;i<sizeof(option_names)/sizeof(option_names[0]);++i) {
        const char * value = "1"; // Freeze every production packing flag on both sides.
        if (std::strcmp(value,"0") && std::strcmp(value,"1")) {
            std::fprintf(stderr,"%s must be 0 or 1; softmax mode 2 is outside this equivalence gate\n",option_names[i]); return false;
        }
        const bool enabled = !std::strcmp(value,"1");
        if (enabled) option_mask |= 1u<<i;
        if (setenv(option_names[i],enabled?"1":"0",1)) return false;
        std::printf("effective %s=%d\n",option_names[i],int(enabled));
    }
    // Both sides freeze the same production packing/precision options.
    // Both sides retain v3 mode2; only the independent A1 flag changes.
    if(setenv("GGML_SPACEMIT_OCR_FA_SERIAL_P2","2",1))return false;
    if(setenv("GGML_SPACEMIT_OCR_FA_SERIAL_P2_PAIR_QK",write?"0":"1",1))return false;
    if(setenv("GGML_SPACEMIT_OCR_FA_SERIAL_P2_PAIR_QK_REPORT","1",1))return false;
    if(setenv("GGML_SPACEMIT_ALLOW_EIGHT_WORKERS","0",1))return false;
    const std::pair<const char *,const char *> fixed[] = {
        {"GGML_SPACEMIT_BACKEND","1"}, {"GGML_SPACEMIT_WORKERS","6"},
        {"SPACEMIT_PERFER_CORE_ID","8,9,10,11,12,13"}, {"GGML_SPACEMIT_SHARED_MODE","PINNED_TCM"},
        {"GGML_SPACEMIT_STRICT_AI_CORES","1"}, {"SPERT_SHARED_MODE","TCM"},
        {"SPERT_BACKEND","spacemit-k3"}, {"SPERT_RT_PRIO","0"},
        {"GGML_SPACEMIT_OCR_FA_PACK","1"}, {"GGML_SPACEMIT_OCR_FA_COMPUTE","3"},
        {"GGML_SPACEMIT_OCR_PV_IME2","2"}, {"GGML_SPACEMIT_OCR_FA_DYNAMIC_ROWS","512"},
        {"GGML_SPACEMIT_OCR_FA_FULL_KV","2"}, {"GGML_SPACEMIT_OCR_FA_DIRECT","0"},
        {"GGML_SPACEMIT_OCR_FA_LAZY_INIT","0"}, {"GGML_SPACEMIT_OCR_FA_TRUST_BOUNDS","0"},
        {"GGML_SPACEMIT_OCR_FA_HEAD_K_TCM","0"}, {"GGML_SPACEMIT_OCR_FA_ALL_K_TCM","0"},
        {"GGML_SPACEMIT_OCR_FA_BATCH_CHECK","0"}, {"GGML_SPACEMIT_OCR_FA_SEPARATE_BARRIER","0"},
        {"GGML_SPACEMIT_OCR_QK_PIPELINE","0"}, {"GGML_SPACEMIT_OCR_PV_PIPELINE","0"},
        {"GGML_SPACEMIT_OCR_PV_TAIL_PIPELINE","0"},
        {"GGML_SPACEMIT_OCR_STRIDED_CAST","0"}, {"GGML_SPACEMIT_PADDLE_CAST_KV_FUSE","0"},
    };
    for (const auto & entry : fixed) if (setenv(entry.first,entry.second,1)) return false;
    // This candidate is six-worker only, including its regression fixture.
    const char *workers=std::getenv("GGML_SPACEMIT_TEST_WORKERS");
    if(workers&&std::strcmp(workers,"6")) {
        std::fprintf(stderr,"This fixture supports exactly six workers\n");return false;
    }
    std::printf("effective AI workers=%s preferred=%s\n",std::getenv("GGML_SPACEMIT_WORKERS"),std::getenv("SPACEMIT_PERFER_CORE_ID"));
    return true;
}
} // namespace

int main(int argc, char ** argv) {
    const bool child = argc==4 && !std::strcmp(argv[3],"--worker");
    if ((!child && argc!=3) || (std::strcmp(argv[1],"--write-reference") && std::strcmp(argv[1],"--compare-reference"))) {
        std::fprintf(stderr,"Usage: %s --write-reference NEW_FILE | --compare-reference FILE\n",argv[0]); return 2;
    }
    const bool write = !std::strcmp(argv[1],"--write-reference");
    if (child) return worker(write,argv[2]);
    unsigned mask = 0;
    if (!configure(write,mask)) return 2;
    struct stat info{};
    if (write && (lstat(argv[2],&info)==0 || errno!=ENOENT)) {
        std::fprintf(stderr,"Refusing to replace existing/unreadable fixture: %s\n",argv[2]); return 2;
    }
    std::string pattern = std::string(argv[2])+".phases-"+(write?"reference":"compare")+"-XXXXXX";
    std::vector<char> path(pattern.begin(),pattern.end()); path.push_back('\0');
    if (!mkdtemp(path.data())) { std::perror("mkdtemp phase directory"); return 2; }
    if (setenv("GGML_SPACEMIT_OCR_FA_PHASES",path.data(),1)) return 2;
    std::printf("option_mask=%u reference=%s phase_directory=%s\n",mask,argv[2],path.data()); std::fflush(stdout);
    std::array<char,4096> executable{};
    const auto length = readlink("/proc/self/exe",executable.data(),executable.size()-1);
    if (length<=0 || size_t(length)>=executable.size()-1) { std::perror("readlink /proc/self/exe"); return 2; }
    executable[size_t(length)]='\0';
    const pid_t pid = fork();
    if (pid<0) { std::perror("fork"); return 2; }
    if (pid==0) {
        char * args[] = {executable.data(),argv[1],argv[2],const_cast<char *>("--worker"),nullptr};
        execv(executable.data(),args); std::perror("execv"); _exit(127);
    }
    int status=0;
    while (waitpid(pid,&status,0)<0) if (errno!=EINTR) { std::perror("waitpid"); return 2; }
    if (!WIFEXITED(status) || WEXITSTATUS(status)!=0) {
        std::fprintf(stderr,"Numerical worker failed status=%d; fixture/trace retained at %s\n",status,path.data()); return 1;
    }
    return verify_phases(path.data()) ? 0 : 1;
}
