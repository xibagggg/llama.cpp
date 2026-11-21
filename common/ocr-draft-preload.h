#pragma once

// Loading the draft model while the target model loads.
//
// llama-server loads the target model, creates its context and then loads the
// draft model, all on one thread. The draft model does not depend on the target
// until its context is created: it comes from its own file and is loaded with
// its own llama_model_params. loader runs that load, the same
// common_speculative_load_draft_model() call the serial path makes, on a second
// thread that starts before the target load. take() joins the thread and
// returns the model for common_speculative_init_from_params(), which creates
// the draft context on the calling thread at the point it always did.
//
// The model is the result of the same call with the same parameters; what
// changes is the moment its buffers are allocated. blocker() names the setups
// in which that moment can change the result, or in which the two loads would
// share state that is not synchronized:
// - --fit on: the draft is sized together with the target, from the free
//   memory of the devices;
// - no explicit device list, or more than one device for either model: layers
//   are assigned by the free memory of each device, which the other load is
//   changing;
// - a tensor split: the meta devices are kept in an unsynchronized registry;
// - warmup: the target would compute a graph while the draft is still loading.
//
// take() compares the parameters the model was loaded with against the ones
// the caller would load it with at that point. If they differ, or the load
// failed, it returns nullptr and the caller loads the draft model as before.

#include "common.h"
#include "log.h"
#include "speculative.h"

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <future>
#include <string>
#include <system_error>
#include <vector>

namespace ocr_draft_preload {

// OCR_DRAFT_PARALLEL_LOAD=1 turns it on; off by default
inline bool requested() {
    const char * v = std::getenv("OCR_DRAFT_PARALLEL_LOAD");
    return v != nullptr && std::strcmp(v, "1") == 0;
}

// number of devices in a list the way common_params keeps it (null-terminated unless empty)
inline size_t n_devices(const std::vector<ggml_backend_dev_t> & devices) {
    size_t n = 0;
    for (ggml_backend_dev_t dev : devices) {
        if (dev != nullptr) {
            n++;
        }
    }
    return n;
}

// Why the draft model that goes with these target parameters is not loaded
// ahead of time, nullptr if it can be.
inline const char * blocker(const common_params & params) {
    if (!params.speculative.has_dft()) {
        return "there is no draft model";
    }
    if (params.fit_params) {
        return "--fit is on";
    }
    if (params.warmup) {
        return "warmup is on";
    }
    if (params.split_mode == LLAMA_SPLIT_MODE_TENSOR) {
        return "the split mode is tensor";
    }
    if (params.devices.empty()) {
        return "the target model has no explicit device list";
    }
    if (n_devices(params.devices) > 1) {
        return "the target model uses more than one device";
    }
    // an empty draft list inherits the target list
    if (n_devices(params.speculative.draft.devices) > 1) {
        return "the draft model uses more than one device";
    }
    return nullptr;
}

inline bool same_devices(const ggml_backend_dev_t * a, const ggml_backend_dev_t * b) {
    if (a == nullptr || b == nullptr) {
        return a == b;
    }
    for (;; a++, b++) {
        if (*a != *b) {
            return false;
        }
        if (*a == nullptr) {
            return true;
        }
    }
}

inline bool same_buft_overrides(const llama_model_tensor_buft_override * a, const llama_model_tensor_buft_override * b) {
    if (a == nullptr || b == nullptr) {
        return a == b;
    }
    for (;; a++, b++) {
        if (a->pattern == nullptr || b->pattern == nullptr) {
            return a->pattern == b->pattern;
        }
        if (std::strcmp(a->pattern, b->pattern) != 0 || a->buft != b->buft) {
            return false;
        }
    }
}

inline bool same_kv_overrides(const llama_model_kv_override * a, const llama_model_kv_override * b) {
    if (a == nullptr || b == nullptr) {
        return a == b;
    }
    for (;; a++, b++) {
        if (std::strncmp(a->key, b->key, sizeof(a->key)) != 0) {
            return false;
        }
        if (a->key[0] == 0) {
            return true;
        }
        if (a->tag != b->tag) {
            return false;
        }
        switch (a->tag) {
            case LLAMA_KV_OVERRIDE_TYPE_INT:
                if (a->val_i64 != b->val_i64) {
                    return false;
                }
                break;
            case LLAMA_KV_OVERRIDE_TYPE_FLOAT:
                if (std::memcmp(&a->val_f64, &b->val_f64, sizeof(a->val_f64)) != 0) {
                    return false;
                }
                break;
            case LLAMA_KV_OVERRIDE_TYPE_BOOL:
                if (a->val_bool != b->val_bool) {
                    return false;
                }
                break;
            case LLAMA_KV_OVERRIDE_TYPE_STR:
                if (std::strncmp(a->val_str, b->val_str, sizeof(a->val_str)) != 0) {
                    return false;
                }
                break;
            default:
                return false;
        }
    }
}

// True if a and b make llama_model_load_from_file() load a file the same way.
// Lists are compared by content, the progress callback is not compared: it
// reports the load and has no say in what is loaded as long as it returns true.
// Every other field of llama_model_params has to be listed here. A field that
// is missing can only be one the two sides are not known to agree on, so add it.
inline bool same_model_params(const llama_model_params & a, const llama_model_params & b) {
    if (!same_devices(a.devices, b.devices) ||
        !same_buft_overrides(a.tensor_buft_overrides, b.tensor_buft_overrides) ||
        !same_kv_overrides(a.kv_overrides, b.kv_overrides)) {
        return false;
    }
    if (a.tensor_split == nullptr || b.tensor_split == nullptr) {
        if (a.tensor_split != b.tensor_split) {
            return false;
        }
    } else if (std::memcmp(a.tensor_split, b.tensor_split, llama_max_devices() * sizeof(float)) != 0) {
        return false;
    }
    return a.n_gpu_layers    == b.n_gpu_layers    &&
           a.split_mode      == b.split_mode      &&
           a.load_mode       == b.load_mode       &&
           a.lazy_mode       == b.lazy_mode       &&
           a.main_gpu        == b.main_gpu        &&
           a.vocab_only      == b.vocab_only      &&
           a.check_tensors   == b.check_tensors   &&
           a.use_extra_bufts == b.use_extra_bufts &&
           a.no_host         == b.no_host         &&
           a.no_alloc        == b.no_alloc        &&
           a.load_mtp        == b.load_mtp;
}

// One draft model load on a second thread. The object stays where it is from
// start() until the load has been taken or the object is destroyed; the
// destructor waits for a load that was not taken and frees its model.
class loader {
public:
    loader() = default;
    loader(const loader &) = delete;
    loader & operator=(const loader &) = delete;

    // Start loading the draft model that goes with these target parameters.
    // Returns nullptr when the load is running, otherwise the reason it is not.
    const char * start(const common_params & params_tgt) {
        if (model.valid()) {
            return "a draft model load is already running";
        }
        if (const char * reason = blocker(params_tgt)) {
            return reason;
        }

        params = common_base_params_to_speculative(params_tgt);

        // A null callback makes llama print a dot per percent, and the callback
        // of the caller is not written to be called from two threads.
        params.load_progress_callback           = [](float, void *) { return true; };
        params.load_progress_callback_user_data = nullptr;

        // A backend may fill in its buffer type the first time it is asked for
        // it. Ask here, so that the two loads only read it.
        for (ggml_backend_dev_t dev : params.devices) {
            if (dev != nullptr) {
                ggml_backend_dev_buffer_type(dev);
            }
        }

        try {
            model = std::async(std::launch::async, [this]() {
                const int64_t t_start_us = ggml_time_us();
                llama_model_ptr res = common_speculative_load_draft_model(params);
                t_load_us = ggml_time_us() - t_start_us;
                return res;
            });
        } catch (const std::system_error &) {
            return "the load thread could not be started";
        }
        return nullptr;
    }

    bool started() const {
        return model.valid();
    }

    // Wait for the load and return its model if it is the one
    // common_speculative_load_draft_model(params_dft) would load now. Returns
    // nullptr if no load was started, if it failed, or if the parameters differ.
    llama_model_ptr take(common_params & params_dft) {
        if (!model.valid()) {
            return nullptr;
        }

        const int64_t t_start_us = ggml_time_us();
        llama_model_ptr res;
        try {
            res = model.get();
        } catch (const std::exception & e) {
            LOG_WRN("ocr_draft_preload: the parallel load of the draft model failed (%s), loading it again\n", e.what());
            return nullptr;
        }
        const int64_t t_wait_us = ggml_time_us() - t_start_us;

        const std::string & path = params.speculative.draft.mparams.path;
        if (res == nullptr) {
            LOG_WRN("ocr_draft_preload: the parallel load of the draft model '%s' failed, loading it again\n", path.c_str());
            return nullptr;
        }
        if (path != params_dft.speculative.draft.mparams.path ||
            !same_model_params(common_model_params_to_llama(params), common_model_params_to_llama(params_dft))) {
            LOG_WRN("ocr_draft_preload: the draft model parameters changed while the target model loaded, "
                    "loading the draft model '%s' again\n", params_dft.speculative.draft.mparams.path.c_str());
            return nullptr;
        }

        LOG_INF("ocr_draft_preload: the draft model '%s' loaded in %.3f s next to the target model, "
                "the caller waited %.3f s for it\n", path.c_str(), t_load_us / 1e6, t_wait_us / 1e6);
        return res;
    }

private:
    // the future is declared last: it is destroyed first, which joins the thread that reads params
    common_params params;
    int64_t       t_load_us = 0;
    std::future<llama_model_ptr> model;
};

} // namespace ocr_draft_preload
