#pragma once

#include <spert.hpp>

#include <cerrno>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>
#include <sys/mman.h>
#include <unistd.h>

namespace ggml::spacemit {

enum class scratch_mode { worker_tcm, pinned_tcm, locked_ddr };

inline scratch_mode configured_scratch_mode() {
    const char * value = std::getenv("GGML_SPACEMIT_SHARED_MODE");
    if (!value || std::strcmp(value, "TCM") == 0) return scratch_mode::worker_tcm;
    if (std::strcmp(value, "PINNED_TCM") == 0) return scratch_mode::pinned_tcm;
    if (std::strcmp(value, "DDR") == 0) return scratch_mode::locked_ddr;
    throw std::invalid_argument("GGML_SPACEMIT_SHARED_MODE expects TCM, PINNED_TCM or DDR");
}

// Host-owned, page-resident scratch. Each lane has inaccessible guard pages.
// Setup either locks every lane or fails before any tile is published.
class resident_scratch {
  public:
    resident_scratch() = default;
    resident_scratch(const resident_scratch &) = delete;
    resident_scratch & operator=(const resident_scratch &) = delete;
    ~resident_scratch() { clear(); }

    void prepare(size_t lanes, size_t bytes) {
        if (base_ && lanes_ == lanes && bytes_ == bytes) return;
        if (!lanes || lanes > 8 || !bytes) throw std::invalid_argument("invalid DDR scratch shape");
        const long page = ::sysconf(_SC_PAGESIZE);
        if (page <= 0) throw std::runtime_error("invalid system page size");
        const size_t page_bytes = size_t(page);
        if (bytes > std::numeric_limits<size_t>::max() - 3 * page_bytes)
            throw std::overflow_error("DDR scratch size overflow");
        const size_t writable = (bytes + page_bytes - 1) / page_bytes * page_bytes;
        const size_t stride = writable + 2 * page_bytes;
        if (stride > std::numeric_limits<size_t>::max() / lanes)
            throw std::overflow_error("DDR scratch lane overflow");
        const size_t total = stride * lanes;
        auto * memory = static_cast<unsigned char *>(::mmap(nullptr, total, PROT_NONE,
            MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (memory == MAP_FAILED) throw std::system_error(errno, std::generic_category(), "DDR mmap");
        for (size_t lane = 0; lane < lanes; ++lane) {
            auto * data = memory + lane * stride + page_bytes;
            if (::mprotect(data, writable, PROT_READ | PROT_WRITE) != 0) {
                const int error = errno;
                ::munmap(memory, total);
                throw std::system_error(error, std::generic_category(), "DDR mprotect");
            }
            std::memset(data, 0, writable);
            if (::mlock(data, writable) != 0) {
                const int error = errno;
                // munmap also unlocks any earlier successful lane locks.
                ::munmap(memory, total);
                throw std::system_error(error, std::generic_category(), "DDR mlock");
            }
        }
        clear();
        base_ = memory;
        mapped_ = total;
        stride_ = stride;
        page_ = page_bytes;
        lanes_ = lanes;
        bytes_ = bytes;
    }

    spert::SharedBufferView lane(size_t index) const {
        if (!base_ || index >= lanes_) return {};
        return {base_ + index * stride_ + page_, bytes_};
    }

    void clear() {
        if (base_) ::munmap(base_, mapped_);
        base_ = nullptr;
        mapped_ = stride_ = page_ = lanes_ = bytes_ = 0;
    }

  private:
    unsigned char * base_ = nullptr;
    size_t mapped_ = 0, stride_ = 0, page_ = 0, lanes_ = 0, bytes_ = 0;
};

// Reserving the complete pool pins the tile across cooperative barriers and
// preserves the original full-buffer layout. Never refresh a live TCM pointer.
class tcm_lease {
  public:
    tcm_lease(spert::Context & runtime, size_t bytes, bool pin) : runtime_(runtime) {
        if (!bytes || !pin) return;
        view_ = runtime_.alloc_shared(bytes, 128);
        const auto full = runtime_.shared_buffer();
        if (!view_ || view_.data != full.data || view_.size != bytes || full.size < bytes) {
            if (view_) runtime_.free_shared(view_);
            view_ = {};
            throw std::runtime_error("full TCM reservation could not preserve its layout");
        }
    }
    tcm_lease(const tcm_lease &) = delete;
    tcm_lease & operator=(const tcm_lease &) = delete;
    ~tcm_lease() { if (view_) runtime_.free_shared(view_); }

  private:
    spert::Context & runtime_;
    spert::SharedBufferView view_{};
};

} // namespace ggml::spacemit
