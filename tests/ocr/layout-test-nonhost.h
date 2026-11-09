#pragma once

// Test only: distinct non-host buffers and real GGML scheduler copies; CPU math.
// This does not emulate SpacemiT repacking, IME2, RVV, TCM or asynchronous DMA.
#include "ggml-backend-impl.h"
#include "ggml-impl.h"

class layout_test_nonhost {
    struct allocation { layout_test_nonhost * owner; ggml_backend_buffer_t backing; size_t size; };
    static allocation * storage(ggml_backend_buffer_t b) { return static_cast<allocation *>(b->context); }
    static layout_test_nonhost * owner(ggml_backend_dev_t d) { return static_cast<layout_test_nonhost *>(d->context); }
public:
    ggml_backend_t executor = nullptr;
    ggml_backend frontend{};
    ggml_backend_device device{};
    ggml_backend_buffer_type buffer_type{};
    size_t uploaded_bytes=0,downloaded_bytes=0,upload_calls=0,download_calls=0,compute_calls=0,live_buffers=0;
    bool (*special_compute)(ggml_tensor *)=nullptr;
    ggml_status (*inspect_graph)(layout_test_nonhost *,ggml_cgraph *)=nullptr;

    explicit layout_test_nonhost(int threads=4) {
        executor=ggml_backend_cpu_init(); require(executor,"test CPU unavailable");
        ggml_backend_cpu_set_n_threads(executor,threads);
        buffer_type.device=&device;buffer_type.context=this;
        buffer_type.iface.get_name=[](ggml_backend_buffer_type_t){return "LAYOUT_TEST_NONHOST";};
        buffer_type.iface.get_alignment=[](ggml_backend_buffer_type_t)->size_t{return 64;};
        buffer_type.iface.is_host=[](ggml_backend_buffer_type_t){return false;};
        buffer_type.iface.alloc_buffer=[](ggml_backend_buffer_type_t type,size_t size) {
            auto * self=static_cast<layout_test_nonhost *>(type->context);
            auto * backing=ggml_backend_alloc_buffer(self->executor,size+128);
            if(!backing)return ggml_backend_buffer_t(nullptr);
            auto * a=new allocation{self,backing,size};++self->live_buffers;
            std::memset(ggml_backend_buffer_get_base(backing),0x5b,size+128);
            ggml_backend_buffer_i api{};
            api.get_base=[](ggml_backend_buffer_t b)->void * {
                return static_cast<char *>(ggml_backend_buffer_get_base(storage(b)->backing))+64;
            };
            api.free_buffer=[](ggml_backend_buffer_t b) {
                auto * a=storage(b);const auto * p=static_cast<const unsigned char *>(ggml_backend_buffer_get_base(a->backing));
                for(size_t i=0;i<64;++i)GGML_ASSERT(p[i]==0x5b&&p[64+a->size+i]==0x5b);
                --a->owner->live_buffers;ggml_backend_buffer_free(a->backing);delete a;
            };
            api.memset_tensor=[](ggml_backend_buffer_t,ggml_tensor * t,uint8_t v,size_t off,size_t n) {
                std::memset(static_cast<char *>(t->data)+off,v,n);
            };
            api.set_tensor=[](ggml_backend_buffer_t b,ggml_tensor * t,const void * p,size_t off,size_t n) {
                auto * s=storage(b)->owner;++s->upload_calls;s->uploaded_bytes+=n;
                std::memcpy(static_cast<char *>(t->data)+off,p,n);
            };
            api.get_tensor=[](ggml_backend_buffer_t b,const ggml_tensor * t,void * p,size_t off,size_t n) {
                auto * s=storage(b)->owner;++s->download_calls;s->downloaded_bytes+=n;
                std::memcpy(p,static_cast<const char *>(t->data)+off,n);
            };
            api.clear=[](ggml_backend_buffer_t b,uint8_t v) {
                auto * a=storage(b);std::memset(static_cast<char *>(ggml_backend_buffer_get_base(a->backing))+64,v,a->size);
            };
            return ggml_backend_buffer_init(type,api,a,size);
        };
        device.iface=ggml_backend_get_device(executor)->iface;
        device.context=this;
        device.iface.get_name=[](ggml_backend_dev_t){return "LAYOUT_TEST_NONHOST";};
        device.iface.get_description=[](ggml_backend_dev_t){return "Test-only non-host buffers; CPU execution";};
        device.iface.get_type=[](ggml_backend_dev_t){return GGML_BACKEND_DEVICE_TYPE_GPU;};
        device.iface.get_buffer_type=[](ggml_backend_dev_t d){return &owner(d)->buffer_type;};
        device.iface.supports_buft=[](ggml_backend_dev_t d,ggml_backend_buffer_type_t b){return b==&owner(d)->buffer_type;};
        device.iface.supports_op=[](ggml_backend_dev_t d,const ggml_tensor * t) {
            return t->op!=GGML_OP_CUSTOM&&ggml_backend_supports_op(owner(d)->executor,t);
        };
        device.iface.get_host_buffer_type=nullptr;
        device.iface.buffer_from_host_ptr=nullptr;
        device.iface.init_backend=nullptr;
        static ggml_guid guid={0xc9,0x18,0x8e,0x77,0x03,0x63,0x41,0x84,0xb6,0xf1,0x61,0xaf,0x39,0xe4,0x05,0x2c};
        frontend.guid=&guid;frontend.device=&device;frontend.context=this;
        frontend.iface.get_name=[](ggml_backend_t){return "LAYOUT_TEST_NONHOST";};
        frontend.iface.free=[](ggml_backend_t){}; // stack-owned; scheduler/application do not own this test device
        frontend.iface.graph_compute=[](ggml_backend_t b,ggml_cgraph * g) {
            auto * s=static_cast<layout_test_nonhost *>(b->context);++s->compute_calls;
            if(s->inspect_graph)return s->inspect_graph(s,g);
            if(!s->special_compute)return ggml_backend_graph_compute(s->executor,g);
            for(int i=0;i<ggml_graph_n_nodes(g);++i) {
                if(s->special_compute(ggml_graph_node(g,i)))continue;
                // Match a graph view without calling its hidden internal symbol.
                auto single=*g;single.size=0;single.n_nodes=1;single.n_leafs=0;
                single.nodes=g->nodes+i;single.leafs=nullptr;single.grads=nullptr;single.grad_accs=nullptr;single.uid=0;
                const auto status=ggml_backend_graph_compute(s->executor,&single);
                if(status!=GGML_STATUS_SUCCESS)return status;
            }
            return GGML_STATUS_SUCCESS;
        };
    }
    ~layout_test_nonhost(){GGML_ASSERT(live_buffers==0);ggml_backend_free(executor);}
    layout_test_nonhost(const layout_test_nonhost &)=delete;
    layout_test_nonhost & operator=(const layout_test_nonhost &)=delete;
};
