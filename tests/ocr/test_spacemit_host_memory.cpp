#include "spacemit-host-memory.h"

#include <cstdio>
#include <vector>

#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace hm = ggml::spacemit::host_memory;

#define CHECK(condition) do { if (!(condition)) { \
    std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #condition); return 1; \
} } while (0)

int main() {
    int cases = 0;
    for (const auto text : {"POSIX", "posix", "PoSiX"}) {
        CHECK(hm::parse_backend(text) == hm::backend::posix_memalign); ++cases;
    }
    for (const auto text : {"HPAGE", "hpage", "HUGEPage", ""}) {
        CHECK(hm::parse_backend(text) == hm::backend::transparent_hugepage); ++cases;
    }
    CHECK(hm::parse_backend(nullptr) == hm::backend::transparent_hugepage); ++cases;
    for (const auto text : {"HPAGE1GB", "hpage1gb", "HugeTLB"}) {
        CHECK(hm::parse_backend(text) == hm::backend::hugetlb_1g); ++cases;
    }
    CHECK(hm::parse_backend("NONE") == hm::backend::none); ++cases;
    CHECK(std::strcmp(hm::effective_backend_name(hm::backend::none), "HPAGE") == 0);
    CHECK(std::strcmp(hm::effective_backend_name(hm::parse_backend("POSIX")), "POSIX") == 0);
    for (const auto text : {"", "posix ", "unknown"}) {
        if (!*text) continue;
        bool rejected = false;
        try { (void) hm::parse_backend(text); }
        catch (const std::invalid_argument &) { rejected = true; }
        CHECK(rejected); ++cases;
    }
    for (const size_t page : {size_t(64), size_t(4096), size_t(16384)}) {
        for (const size_t length : {size_t(0), size_t(1), page-1, page, page+1, 2*page+7}) {
            for (const size_t prefix : {size_t(1), size_t(63), page-1}) {
                std::vector<unsigned char> actual(prefix+length+128, 0xa5), expected(actual);
                for (size_t i = 0; i < length; i += page) expected[prefix+i] = 0;
                if (length) expected[prefix+length-1] = 0;
                hm::prefault(actual.data()+prefix, length, page);
                CHECK(actual == expected); ++cases;
            }
        }
    }
    hm::prefault(nullptr, 4096, 4096);
    unsigned char unchanged = 0xa5;
    hm::prefault(&unchanged, 1, 0); CHECK(unchanged == 0xa5);
#if defined(__linux__)
    const size_t page = size_t(sysconf(_SC_PAGESIZE)), count = 9;
    auto * base = static_cast<unsigned char *>(mmap(nullptr, (count+2)*page,
        PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0));
    CHECK(base != MAP_FAILED);
    auto * writable = base+page;
    CHECK(mprotect(writable, count*page, PROT_READ|PROT_WRITE) == 0);
    std::vector<unsigned char> resident(count);
    CHECK(mincore(writable, count*page, resident.data()) == 0);
    for (auto value : resident) CHECK(!(value&1));
    // The range is not page aligned and reaches all nine writable pages.
    // Both neighboring guard pages remain inaccessible throughout the call.
    hm::prefault(writable+page-1, (count-1)*page+1, page);
    CHECK(mincore(writable, count*page, resident.data()) == 0);
    for (auto value : resident) CHECK(value&1);
    CHECK(munmap(base, (count+2)*page) == 0); ++cases;
#endif
    std::printf("PASS cases=%d backend_aliases=exact invalid=rejected prefault=bounded\n", cases);
}
