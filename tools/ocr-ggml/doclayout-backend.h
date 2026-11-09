#pragma once

// Requires the shared require/json definitions; no ORT or model-format-specific execution.
#include <set>
#ifdef OCR_LAYOUT_SPACEMIT
#include "ggml-spacemit.h"
#endif

class layout_backend {
public:
    std::string mode;
    ggml_backend_t cpu=nullptr, accelerator=nullptr;
    ggml_backend_sched_t scheduler=nullptr;
    ggml_context * pending_host_context=nullptr;
    ggml_context * accelerator_context=nullptr;
    ggml_backend_buffer_t host_weights_buffer=nullptr,accelerator_weights_buffer=nullptr;
    std::vector<ggml_backend_buffer_t> external_buffers;
    std::map<std::string,ggml_tensor *> imported;
    std::set<ggml_tensor *> accelerated_weights;
    int accelerated_matrices=0,accelerated_im2cols=0,cpu_nodes=0;
    uint64_t offloaded_im2col_image_bytes=0,offloaded_im2col_patch_bytes=0;
    uint64_t compute_calls=0;

    explicit layout_backend(std::string selected,ggml_backend_t host):mode(std::move(selected)),cpu(host) {
        require(mode=="cpu-sched"||mode=="spacemit","invalid layout scheduler backend");
        if(mode=="spacemit") {
#ifdef OCR_LAYOUT_SPACEMIT
            accelerator=ggml_backend_spacemit_init();
            require(accelerator,"SpacemiT backend unavailable");
#else
            throw std::runtime_error("this layout build has no SpacemiT backend");
#endif
        }
    }

    ~layout_backend() {
        if(scheduler){ggml_backend_sched_synchronize(scheduler);ggml_backend_sched_free(scheduler);}
        for(auto * buffer:external_buffers)ggml_backend_buffer_free(buffer);
        if(host_weights_buffer)ggml_backend_buffer_free(host_weights_buffer);
        if(accelerator_weights_buffer)ggml_backend_buffer_free(accelerator_weights_buffer);
        if(pending_host_context)ggml_free(pending_host_context);
        if(accelerator_context)ggml_free(accelerator_context);
        if(accelerator)ggml_backend_free(accelerator);
    }

    void import_weights(ggml_context *& original,const json & descriptors) {
        const size_t metadata=ggml_tensor_overhead()*(descriptors.size()+16);
        pending_host_context=ggml_init({metadata,nullptr,true});
        require(pending_host_context,"host weight metadata allocation failed");
        if(accelerator) {
            accelerator_context=ggml_init({metadata,nullptr,true});
            require(accelerator_context,"accelerator weight metadata allocation failed");
        }
        for(const auto & item:descriptors) {
            const std::string name=item.at("id");
            auto * source=ggml_get_tensor(original,name.c_str());require(source,"missing source weight: "+name);
            const bool offload=accelerator != nullptr;
            auto * target=ggml_dup_tensor(offload?accelerator_context:pending_host_context,source);
            require(target,"weight metadata allocation failed");ggml_set_name(target,name.c_str());
            imported.emplace(name,target);if(offload && ggml_is_quantized(source->type))accelerated_weights.insert(target);
        }
        host_weights_buffer=ggml_backend_alloc_ctx_tensors(pending_host_context,cpu);
        require(host_weights_buffer || accelerator,"host weight allocation failed");
        if(host_weights_buffer)ggml_backend_buffer_set_usage(host_weights_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        if(accelerator) {
            require(!accelerated_weights.empty(),"SpacemiT layout requires a packed Q4/Q8 convolution model");
            accelerator_weights_buffer=ggml_backend_alloc_ctx_tensors(accelerator_context,accelerator);
            require(accelerator_weights_buffer,"accelerator weight allocation failed");
            ggml_backend_buffer_set_usage(accelerator_weights_buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
        }
        for(const auto & item:imported) {
            auto * source=ggml_get_tensor(original,item.first.c_str());
            auto * target=item.second;
            if(accelerated_weights.count(target))require(target->extra,"SpacemiT has no repack trait for weight: "+item.first);
            // Upload complete tensors atomically: SpacemiT set_tensor performs physical repacking.
            ggml_backend_tensor_set(target,source->data,0,ggml_nbytes(source));
        }
        // Do not retain both loaded GGUF data and the uploaded copies.
        ggml_free(original);original=pending_host_context;pending_host_context=nullptr;
    }

    void allocate(ggml_cgraph * graph,ggml_tensor * input) {
        std::set<ggml_tensor *> im2cols;
        const char * im2col_option=std::getenv("OCR_LAYOUT_OFFLOAD_IM2COL");
        if(accelerator&&im2col_option&&std::string(im2col_option)=="1") {
            for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
                auto * out=ggml_graph_node(graph,i);
                if(out->op!=GGML_OP_MUL_MAT||!accelerated_weights.count(out->src[0]))continue;
                auto * image=out->src[1];while(image&&image->view_src)image=image->view_src;
                if(image&&image->op==GGML_OP_IM2COL)im2cols.insert(image);
            }
        }
        // IM2COL geometry leaves have initialized host storage outside the GGUF.
        // Give them owned, aligned CPU buffers so scheduler copies have a valid owner.
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto * out=ggml_graph_node(graph,i);
            for(int s=0;s<GGML_MAX_SRC;++s) {
                auto * t=out->src[s];if(!t||!t->data||t->buffer||t->view_src)continue;
                auto * original=t->data;
                auto * leaf_backend=accelerator?accelerator:cpu;
                auto * buffer=ggml_backend_alloc_buffer(leaf_backend,ggml_nbytes(t)+ggml_backend_get_alignment(leaf_backend));
                require(buffer,"external leaf allocation failed");external_buffers.push_back(buffer);
                auto allocator=ggml_tallocr_new(buffer);t->data=nullptr;
                require(ggml_tallocr_alloc(&allocator,t)==GGML_STATUS_SUCCESS,"external leaf binding failed");
                ggml_backend_tensor_set(t,original,0,ggml_nbytes(t));
                ggml_backend_buffer_set_usage(buffer,GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
            }
        }
        ggml_backend_t backends[]={accelerator,cpu};
        scheduler=ggml_backend_sched_new(accelerator?backends:backends+1,nullptr,accelerator?2:1,20000,false,false);
        require(scheduler,"layout scheduler initialization failed");
        ggml_backend_sched_set_tensor_backend(scheduler,input,accelerator?accelerator:cpu);
        for(int i=0;i<ggml_graph_n_nodes(graph);++i) {
            auto * out=ggml_graph_node(graph,i);
            if(out->op==GGML_OP_NONE||out->op==GGML_OP_VIEW||out->op==GGML_OP_RESHAPE||
               out->op==GGML_OP_PERMUTE||out->op==GGML_OP_TRANSPOSE)continue;
            const bool matrix=accelerator && out->op==GGML_OP_MUL_MAT && accelerated_weights.count(out->src[0]);
            const bool im2col=accelerator && out->op==GGML_OP_IM2COL;
            const bool offload=accelerator != nullptr;
            auto * selected=offload?accelerator:cpu;
            // supports_op in SpacemiT rejects foreign allocated buffers. The
            // scheduler will copy these inputs; query its post-copy layout.
            auto candidate=*out;ggml_tensor sources[GGML_MAX_SRC];
            if(offload) {
                candidate.buffer=nullptr;candidate.view_src=nullptr;
                for(int s=0;s<GGML_MAX_SRC;++s)if(out->src[s]) {
                    sources[s]=*out->src[s];sources[s].buffer=nullptr;sources[s].view_src=nullptr;
                    candidate.src[s]=&sources[s];
                }
            }
            require(ggml_backend_supports_op(selected,offload?&candidate:out),std::string("layout backend cannot execute ")+ggml_op_name(out->op));
            ggml_backend_sched_set_tensor_backend(scheduler,out,selected);
            if(matrix)++accelerated_matrices;
            else if(im2col) {
                ++accelerated_im2cols;offloaded_im2col_image_bytes+=ggml_nbytes(out->src[1]);
                offloaded_im2col_patch_bytes+=ggml_nbytes(out);
            }else if(!offload) ++cpu_nodes;
        }
        require(!accelerator||accelerated_matrices>0,"no layout matrices assigned to SpacemiT");
        require(ggml_backend_sched_alloc_graph(scheduler,graph),"layout scheduler allocation failed");
    }

    enum ggml_status compute(ggml_cgraph * graph) {
        const auto status=ggml_backend_sched_graph_compute(scheduler,graph);
        if(status==GGML_STATUS_SUCCESS)++compute_calls;
        return status;
    }

    size_t compute_bytes() const {
        size_t bytes=ggml_backend_sched_get_buffer_size(scheduler,cpu);
        if(accelerator)bytes+=ggml_backend_sched_get_buffer_size(scheduler,accelerator);
        return bytes;
    }

    json describe() const {
        return {{"mode",mode},{"scheduler",true},{"compute_calls",compute_calls},
                {"assigned_accelerated_matrices",accelerated_matrices},{"assigned_cpu_nodes",cpu_nodes},
                {"assigned_accelerated_im2cols",accelerated_im2cols},
                {"offloaded_im2col_image_bytes_static",offloaded_im2col_image_bytes},
                {"offloaded_im2col_patch_bytes_static",offloaded_im2col_patch_bytes},
                {"uploaded_quantized_weights",accelerated_weights.size()},
                {"host_weight_buffer_bytes",host_weights_buffer?ggml_backend_buffer_get_size(host_weights_buffer):0},
                {"accelerator_weight_buffer_bytes",accelerator_weights_buffer?ggml_backend_buffer_get_size(accelerator_weights_buffer):0},
                {"cpu_compute_buffer_bytes",scheduler?ggml_backend_sched_get_buffer_size(scheduler,cpu):0},
                {"accelerator_compute_buffer_bytes",scheduler&&accelerator?ggml_backend_sched_get_buffer_size(scheduler,accelerator):0},
                {"graph_splits",scheduler?ggml_backend_sched_get_n_splits(scheduler):0},
                {"strict_ai_inference",accelerator!=nullptr},{"hardware_execution_validated",compute_calls>0&&cpu_nodes==0&&accelerator!=nullptr}};
    }
};
