#pragma once
// Host resolves the independent option before dispatching any prepared table.
#include <cstdlib>
#include <cstring>
namespace paddle_vision_rope {
inline bool planar_enabled() {
    static const bool enabled=[] {
        const char * p=std::getenv("GGML_SPACEMIT_PADDLE_ROPE_PLANAR");
        return p&&!std::strcmp(p,"1");
    }();
    return enabled;
}
} // namespace paddle_vision_rope
