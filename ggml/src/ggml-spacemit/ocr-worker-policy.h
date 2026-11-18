#pragma once
#include <cstddef>
#include <cstdint>
namespace ocr_worker_policy {
constexpr bool requested_count(long requested,std::size_t available,bool allow_eight) {
    return requested>=1&&requested<=(allow_eight?8:6)&&std::size_t(requested)<=available;
}
// A zero mask is invalid. Physical CC identifiers are vendor AI IDs8..15.
constexpr std::uint32_t ai_mask(const int*ids,std::size_t count) {
    if(count<1||count>8)return 0;
    std::uint32_t mask=0;
    for(std::size_t i=0;i<count;++i) {
        if(ids[i]<8||ids[i]>15)return 0;
        const auto bit=std::uint32_t(1)<<(ids[i]-8);
        if(mask&bit)return 0;
        mask|=bit;
    }
    return mask;
}
constexpr bool exact_grant(std::size_t requested,std::size_t granted,
                           const int*preferred,std::size_t preferred_count,
                           const int*observed,std::size_t observed_count) {
    return requested==granted&&preferred_count==requested&&observed_count==requested&&
        ai_mask(preferred,preferred_count)&&ai_mask(preferred,preferred_count)==ai_mask(observed,observed_count);
}
} // namespace ocr_worker_policy
