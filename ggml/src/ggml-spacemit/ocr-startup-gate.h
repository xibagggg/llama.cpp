#pragma once

// Owned-server host startup may overlap Layout; no accelerator graph may do so.
// The SDK passes a private, inherited Unix socket endpoint only for this mode.
// Waiting occurs before taking execution locks or checking out any AI stream.
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <mutex>
#include <sys/socket.h>
#include <unistd.h>

namespace ocr_startup_gate {

using trace_callback = void (*)(const char *, size_t);

[[noreturn]] inline void fail(const char * message) {
    std::fprintf(stderr, "ggml-spacemit: startup compute gate failed: %s\n", message);
    std::fflush(stderr);
    _exit(125);
}

inline void wait(trace_callback trace = nullptr) {
    static std::once_flag once;
    std::call_once(once, [trace] {
        const char * text = std::getenv("GGML_SPACEMIT_STARTUP_GATE_FD");
        if (text == nullptr) return;
        char * end = nullptr;
        errno = 0;
        const long parsed = std::strtol(text, &end, 10);
        if (errno != 0 || end == text || *end != '\0' || parsed < 0 || parsed > INT_MAX)
            fail("invalid inherited descriptor");
        const int fd = static_cast<int>(parsed);
        int type = 0;
        socklen_t type_size = sizeof(type);
        if (::getsockopt(fd, SOL_SOCKET, SO_TYPE, &type, &type_size) != 0 || type != SOCK_STREAM)
            fail("descriptor is not a connected stream socket");
        const int flags = ::fcntl(fd, F_GETFD);
        if (flags < 0 || ::fcntl(fd, F_SETFD, flags | FD_CLOEXEC) < 0)
            fail("cannot protect inherited descriptor");
        if (trace) trace("startup-gate-wait-begin", 0);
        char token = 0;
        ssize_t count;
        do { count = ::read(fd, &token, 1); }
        while (count < 0 && errno == EINTR);
        const int saved_errno = errno;
        ::close(fd);
        ::unsetenv("GGML_SPACEMIT_STARTUP_GATE_FD");
        if (count != 1 || token != 'G') {
            errno = saved_errno;
            fail(count == 0 ? "owner closed before releasing compute" :
                 count < 0 ? "cannot read release token" : "invalid release token");
        }
        if (trace) trace("startup-gate-wait-end", 0);
    });
}

}  // namespace ocr_startup_gate
