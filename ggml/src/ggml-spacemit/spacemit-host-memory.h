#pragma once

#include "spine_mem_pool.h"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ggml::spacemit::host_memory {

using backend = ggml::cpu::riscv64_spacemit::spine_mem_pool_backend;

inline backend parse_backend(const char * text) {
    if (!text || !*text) return backend::transparent_hugepage;
    std::string value(text);
    for (char & ch : value) {
        if (ch >= 'A' && ch <= 'Z') ch = char(ch - 'A' + 'a');
    }
    if (value == "none") return backend::none;
    if (value == "posix") return backend::posix_memalign;
    if (value == "hpage" || value == "hugepage") return backend::transparent_hugepage;
    if (value == "hpage1gb" || value == "hugetlb") return backend::hugetlb_1g;
    throw std::invalid_argument("invalid SPACEMIT_MEM_BACKEND: " + value +
                                "; expected NONE, POSIX, HPAGE or HPAGE1GB");
}

inline const char * effective_backend_name(backend value) {
    // The independent backend's pool has always mapped NONE to HPAGE.
    switch (value) {
        case backend::posix_memalign: return "POSIX";
        case backend::hugetlb_1g: return "HPAGE1GB";
        case backend::none:
        case backend::transparent_hugepage: return "HPAGE";
    }
    return "unknown";
}

inline bool prefault_enabled() {
    static const bool enabled = [] {
        const char * value = std::getenv("GGML_SPACEMIT_HOST_PREFAULT");
        return value && std::strcmp(value, "1") == 0;
    }();
    return enabled;
}

// Only call for newly owned, writable scratch or tensor allocations, on the
// host before dispatch. No activation or weight contents may be live here.
inline void prefault(void * base, size_t bytes, size_t page_size) {
    if (!base || bytes == 0 || page_size == 0) return;
    auto * memory = static_cast<volatile unsigned char *>(base);
    size_t offset = 0;
    for (;;) {
        memory[offset] = 0;
        if (bytes - offset <= page_size) break;
        offset += page_size;
    }
    // An unaligned range can end on a page not covered by the stride above.
    memory[bytes - 1] = 0;
}

} // namespace ggml::spacemit::host_memory
